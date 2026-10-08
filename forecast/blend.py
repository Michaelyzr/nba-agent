"""Forecast components and the blends the model-review subloop chooses between.

    python -m forecast.blend          # trains the M4-NN MLP (3 seeds, games before 2026-02-01) -> models/m4nn/mlp.pkl

Every component is as-of at the decision time and every trained model saw only games
before SPLIT (2026-02-01):

    anchor   home market mid 24 h before tip (the agent's base rate)
    mid      home market mid now
    m4       M4 logistic after-minus-before-the-news shift (the default agent's signal)
    mlp      M4-NN MLP after-minus-before shift (3-seed mean; config chosen on validation in win_nn_eval)
    m6       M6 ImpactNet predicted move of the home mid from now to tip (median)

A blend maps components to P(home): either a linear rule
    p = clip(base + w_m4 * m4 + w_mlp * mlp + w_m6 * m6),   base in {anchor, mid}
or "stack", a logistic regression with strong L2 on [logit(anchor), logit(mid) - logit(anchor), m4, mlp, m6],
fit by the reviewer on selection-day decision points only (fit_stack).

M1/M2 player projections are not a separate component: M4's missing-minutes and missing-points
inputs already carry the absent players' usual minutes and points, and the GRU quantiles would
add a slow per-player pass for little new information. The NBA injury-report PDFs are future work.
"""
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MLP_PATH = ROOT / "models" / "m4nn" / "mlp.pkl"
MLP_CONFIG = {"hidden": 16, "lr": 3e-4}          # chosen on the validation slice in evaluation/win_nn_eval.py
MLP_SEEDS = (0, 1, 2)
COMPONENTS = ("anchor", "mid", "m4", "mlp", "m6")
STACK_FEATURES = ("logit_anchor", "logit_drift", "m4", "mlp", "m6")
STACK_C = 0.1                                    # strong L2: inverse regularisation strength
P_MIN, P_MAX = 0.02, 0.98

# Pre-stated candidate set for the model-review subloop (name -> params of a forecast_blend rule).
DEFAULT_BLEND = {"name": "m4", "base": "anchor", "w_m4": 1.0, "w_mlp": 0.0, "w_m6": 0.0}
CANDIDATE_BLENDS = [
    DEFAULT_BLEND,
    {"name": "market", "base": "anchor", "w_m4": 0.0, "w_mlp": 0.0, "w_m6": 0.0},
    {"name": "m4_half", "base": "anchor", "w_m4": 0.5, "w_mlp": 0.0, "w_m6": 0.0},
    {"name": "mlp", "base": "anchor", "w_m4": 0.0, "w_mlp": 1.0, "w_m6": 0.0},
    {"name": "m4_mlp_mean", "base": "anchor", "w_m4": 0.5, "w_mlp": 0.5, "w_m6": 0.0},
    {"name": "m6", "base": "mid", "w_m4": 0.0, "w_mlp": 0.0, "w_m6": 1.0},
    {"name": "stack"},                           # coefficients fit on the selection days
]


def clip(p):
    return np.clip(np.asarray(p, float), P_MIN, P_MAX)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def stack_matrix(frame: pd.DataFrame) -> np.ndarray:
    la = logit(frame.anchor)
    return np.column_stack([la, logit(frame["mid"]) - la, frame.m4.fillna(0), frame.mlp.fillna(0),
                            frame.m6.fillna(0)])


def blend_home(params: dict, frame: pd.DataFrame) -> np.ndarray:
    """P(home) for each component row (columns COMPONENTS, home terms)."""
    if params["name"] == "stack":
        z = stack_matrix(frame) @ np.asarray(params["coef"], float) + params["intercept"]
        return clip(1 / (1 + np.exp(-z)))
    base = frame[params["base"]].to_numpy(float)
    p = base + sum(params[f"w_{k}"] * frame[k].fillna(0).to_numpy(float) for k in ("m4", "mlp", "m6"))
    return clip(p)


def fit_stack(frame: pd.DataFrame) -> dict | None:
    """Logistic stacking on the rows given (the reviewer passes selection-day decision points only)."""
    from sklearn.linear_model import LogisticRegression
    y = frame.home_win.to_numpy(float)
    if len(frame) < 20 or y.min() == y.max():
        return None
    m = LogisticRegression(C=STACK_C, max_iter=1000).fit(stack_matrix(frame), y)
    return {"name": "stack", "coef": [round(float(c), 6) for c in m.coef_[0]],
            "intercept": round(float(m.intercept_[0]), 6), "fit_rows": int(len(frame)),
            "fit_days": sorted(frame.date.unique().tolist())}


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y) -> float:
    p, y = np.clip(np.asarray(p, float), 0.01, 0.99), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------- the MLP component ----------------

class MLPShift:
    """3-seed M4-NN MLP: P(home) from M4's features, with and without tonight's absences."""

    def __init__(self, models: list):
        self.models = models

    def p_home(self, features: dict) -> float:
        from forecast.win import WIN_FEATURES
        row = pd.DataFrame([{k: features[k] for k in WIN_FEATURES}])
        return float(np.mean([m.predict(row)[0] for m in self.models]))


def train_mlp(rows: pd.DataFrame | None = None) -> MLPShift:
    from forecast.win_nn import SPLIT, NeuralWinModel
    if rows is None:
        cache = ROOT / "runs" / "win_nn" / "rows.parquet"
        if cache.exists():
            rows = pd.read_parquet(cache)
        else:
            from forecast.baselines import load_source
            from forecast.history import History
            from forecast.win import training_rows
            player_games, games = load_source("frozen")
            rows = training_rows(History(player_games, games), games)
    train = rows[pd.DatetimeIndex(rows.tip_time) < pd.Timestamp(SPLIT, tz="UTC")].reset_index(drop=True)
    return MLPShift([NeuralWinModel("mlp", seed=s, **MLP_CONFIG).fit(train) for s in MLP_SEEDS])


def load_mlp(path: Path = MLP_PATH, train_if_missing: bool = True) -> MLPShift:
    if Path(path).exists():
        return pickle.loads(Path(path).read_bytes())
    if not train_if_missing:
        raise FileNotFoundError(path)
    mlp = train_mlp()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(mlp))
    return mlp


class Components:
    """As-of component values for one decision, in home terms (the agent caches them per decision)."""

    def __init__(self, base, mlp: MLPShift | None = None, impact=None):
        self.base, self.mlp, self.impact = base, mlp, impact     # base: forecast.api.Forecaster (M4)

    def __call__(self, view, game, out: list, now) -> dict:
        from forecast.win import game_features
        h = self.base.history(view)
        after = game_features(h, game.home_team_id, game.away_team_id, game.tip_time, out)
        before = game_features(h, game.home_team_id, game.away_team_id, game.tip_time, [])
        mlp = None if self.mlp is None else self.mlp.p_home(after) - self.mlp.p_home(before)
        m6 = None
        if self.impact is not None:
            r = self.impact.predict(view, game, now)
            m6 = None if r is None else r["move"]
        return {"mlp": mlp, "m6": m6}


def main():
    from forecast.blend import train_mlp as train      # pickle the class under forecast.blend, not __main__
    mlp = train()
    MLP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLP_PATH.write_bytes(pickle.dumps(mlp))
    print(f"saved {MLP_PATH} ({len(mlp.models)} seeds, best epochs {[m.best_epoch for m in mlp.models]})")


if __name__ == "__main__":
    main()
