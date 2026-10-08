"""Forecast-blend subloop: learn how to combine the as-of win signals into the trader's probability.

    p_home = sigmoid(sum_k w_k * logit(signal_k)),   w_k >= 0, sum_k w_k = 1

A convex combination in logit space, so a one-hot weight on "anchor + M4 shift" is the published
agent's estimate exactly. After each settled game-day the learner refits the weights on settled games
only (exponentiated gradient on log loss, shrunk toward the current weights) and treats the new
weights like a notebook rule: a split gate accepts them only if Brier improves by at least
GATE_BRIER on held-out earlier days that the fit never saw. Every proposal, accepted or rejected,
is written to a blend log with its evidence.

Primary gate metric: forecast Brier (chosen up front; CLV dollars are reported, not gated on).

Day windows mirror agents/graph.py's split gate. At the review after day d, with settled market days
S (all <= d): gate days G = the GATE_DAYS days before the last SELECT_DAYS; the fit uses the other
days in the last FIT_DAYS. Nothing from day > d enters a fit, a gate, or the weights used on day d+1.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIGNALS = ("anchor", "mid", "anchor_m4", "anchor_mlp", "anchor_gru", "m4", "mlp", "gru")
MARKET_SIGNALS = ("anchor", "mid", "anchor_m4", "anchor_mlp", "anchor_gru")  # anchored shifts contain the market
DEFAULT_SIGNAL = "anchor_m4"
SELECT_DAYS = 7
GATE_DAYS = 7
FIT_DAYS = 42
GATE_BRIER = 0.0005
SHRINK = 0.02          # L2 pull of the fitted weights toward the current weights
STEPS = 400
LR = 0.5
EPS = 0.02


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def blend(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """z: (n, k) logits of the signals; w: (k,) simplex weights -> p_home (n,)."""
    return 1 / (1 + np.exp(-(z @ w)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_weights(z: np.ndarray, y: np.ndarray, w0: np.ndarray, shrink=SHRINK, steps=STEPS, lr=LR) -> np.ndarray:
    """Exponentiated gradient on mean log loss + shrink * ||w - w0||^2, staying on the simplex."""
    w0 = np.asarray(w0, float)
    w = np.full_like(w0, 1 / len(w0)) * 0.5 + w0 * 0.5     # start inside the simplex so EG can move any weight
    for _ in range(steps):
        p = blend(z, w)
        grad = z.T @ (p - y) / len(y) + 2 * shrink * (w - w0)
        w = w * np.exp(-lr * grad)
        w /= w.sum()
    return w


@dataclass
class BlendLearner:
    """Daily weight learner with a held-out Brier gate. `signals` picks the inputs (for ablations)."""
    signals: tuple = SIGNALS
    start: str = DEFAULT_SIGNAL
    weights: np.ndarray = None
    log: list = field(default_factory=list)
    history: list = field(default_factory=list)     # (day the weights apply from, weights)

    def __post_init__(self):
        if self.weights is None:
            w = np.zeros(len(self.signals))
            w[self.signals.index(self.start) if self.start in self.signals else 0] = 1.0
            self.weights = w
        self.history.append((None, self.weights.copy()))

    def z(self, frame: pd.DataFrame) -> np.ndarray:
        return logit(frame[list(self.signals)].to_numpy(float))

    def predict(self, frame: pd.DataFrame, weights=None) -> np.ndarray:
        return blend(self.z(frame), self.weights if weights is None else weights)

    def weights_on(self, day: str) -> np.ndarray:
        """Weights the trader uses on `day`: the last ones accepted at a review strictly before `day`."""
        w = self.history[0][1]
        for applies_from, weights in self.history[1:]:
            if applies_from <= day:
                w = weights
        return w

    def review(self, settled: pd.DataFrame, day: str) -> dict:
        """Propose and gate new weights after `day`. `settled` must hold only games with date <= day."""
        if len(settled) and settled.date.max() > day:
            raise ValueError("review saw a game after its own day")
        days = sorted(settled.date.unique())
        entry = {"day": day, "signals": list(self.signals), "current": self.weights.round(4).tolist()}
        if len(days) < SELECT_DAYS + GATE_DAYS + 7:
            self.log.append({**entry, "result": "deferred", "reason": f"{len(days)} settled days, need "
                             f"{SELECT_DAYS + GATE_DAYS + 7}"})
            return self.log[-1]
        gate_days = days[-(SELECT_DAYS + GATE_DAYS):-SELECT_DAYS]
        fit_days = [d for d in days[-FIT_DAYS:] if d not in gate_days]
        fit, test = settled[settled.date.isin(fit_days)], settled[settled.date.isin(gate_days)]
        proposed = fit_weights(self.z(fit), fit.home_win.to_numpy(float), self.weights)
        b_cur = brier(self.predict(test), test.home_win)
        b_new = brier(self.predict(test, proposed), test.home_win)
        ok = b_cur - b_new >= GATE_BRIER
        entry.update(proposed=proposed.round(4).tolist(), fit_games=len(fit), gate_games=len(test),
                     gate_days=[gate_days[0], gate_days[-1]], brier_current=b_cur, brier_proposed=b_new,
                     result="accepted" if ok else "rejected",
                     reason=f"held-out Brier {b_cur:.4f} -> {b_new:.4f} on {len(test)} games "
                            f"({gate_days[0]}..{gate_days[-1]}); need -{GATE_BRIER}")
        if ok:
            self.weights = proposed
            next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            self.history.append((next_day, proposed.copy()))
        self.log.append(entry)
        return entry

    def run(self, games: pd.DataFrame) -> pd.DataFrame:
        """Walk forward over `games` (one row per game: date, signals, home_win). Returns p_blend per game,
        each predicted with the weights accepted before its day."""
        out = games.sort_values(["date", "game_id"]).copy()
        p = np.empty(len(out))
        for day in sorted(out.date.unique()):
            idx = np.flatnonzero(out.date.to_numpy() == day)
            p[idx] = blend(self.z(out.iloc[idx]), self.weights)
            self.review(out[out.date <= day], day)
        out["p_blend"] = p
        return out
