"""G: a learned trade/pass policy from full-information replay decision points.

    from forecast.policy import decision_rows, PolicyModel, select, LearnedPolicy
    rows = decision_rows(tables, forecaster, "2025-11-01", "2026-04-12")    # as-of features + labels
    table, best, forced = select(rows[rows.date <= TRAIN[1]])              # validation inside Nov-Jan only
    model = PolicyModel(best["family"], best["params"]).fit(rows[rows.date <= TRAIN[1]])
    Replay(tables, LearnedPolicy(rows, model, best["tau"]).policy).run(*TEST)

At every replay decision time (news updates in the six hours before tip and one
hour before tip) the trader's own trigger -> investigate -> forecast -> analyse
steps give the anchored probability for each game-winner market. Each market
gives two options, buy yes or buy no, so a game has four. For every option we
know the counterfactual: we would fill at the recorded ask (or 1 - bid) and the
mid at tip is recorded for both sides. The label is closing-line value after
the Kalshi fee, per contract. That makes this supervised learning of action
values, not off-policy reinforcement learning.

The policy predicts net CLV for each option, takes the best option in the game
and trades it once, the first time the prediction clears a threshold tau.
Features only use the AsOf view; labels are never passed to the model.
"""
import math

import numpy as np
import pandas as pd

from agents.graph import DEFAULT_MIN_EDGE, DEFAULT_STAKE, MarketAgent, fee_per_contract, minutes_between
from replay import LEAD, VOLUME_LOOKBACK, AsOf, Order, Replay

SEED = 7606
FIT = ("2025-11-01", "2026-01-14")
VALID = ("2026-01-15", "2026-01-31")
TRAIN = ("2025-11-01", "2026-01-31")
TEST = ("2026-02-01", "2026-04-12")
HOLDOUT = ("2026-04-13", "2026-06-14")
NEWS_AGE_CAP = 6.0               # hours; no news counts as the cap with has_news = 0
FEATURES = ["gap", "side_price", "market_move", "model_shift", "news_age_hours", "has_news", "hours_to_tip",
            "spread", "log_volume", "p_side", "is_yes", "is_home"]
LABEL = "net_clv"
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
GRID = ([("ridge", {"alpha": a}) for a in (1.0, 10.0, 100.0)]
        + [("gbm", {"max_depth": d, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 40}) for d in (2, 3)]
        + [("mlp", {"hidden_layer_sizes": h, "alpha": a}) for h, a in (((16,), 1e-2), ((32, 16), 1e-3))])
MIN_FORCED_TRADES = 10           # the "forced to trade" diagnostic needs at least this many validation trades


# ---------------- decision rows ----------------

def decision_rows(tables: dict, forecaster, start: str, end: str, stake: float = DEFAULT_STAKE,
                  agent: MarketAgent | None = None) -> pd.DataFrame:
    """One row per (decision time, market, side) with as-of features and the realised net-CLV label."""
    rp = Replay(tables, policy=None)
    agent = agent or MarketAgent(forecaster=forecaster, learn=False)
    games = tables["games"].dropna(subset=["tip_time"])
    games = games[(games.date >= start) & (games.date <= end) & games.game_id.isin(tables["markets"].game_id)]
    out = []
    for game in games.sort_values("tip_time").itertuples():
        close_view = AsOf(tables, game.tip_time, rp.price_index)
        for now in rp.decision_times(game):
            state = {"view": AsOf(tables, now, rp.price_index), "game": game, "now": now, "tries": 0,
                     "plant": None, "trace": []}
            state.update(agent.trigger(state))
            if state["status"] == "idle":
                continue
            for step in (agent.investigate, agent.forecast, agent.analyse):
                state.update(step(state))
            out += _option_rows(state, close_view, rp, stake)
    rows = pd.DataFrame(out)
    if rows.empty:
        return rows
    best = rows.groupby(["game_id", "as_of", "ticker"]).gap.transform("max")
    rows["rule_act"] = (rows.gap == best) & (rows.gap > DEFAULT_MIN_EDGE)
    return rows.reset_index(drop=True)


def _option_rows(state, close_view, rp, stake) -> list:
    view, game, now, news = state["view"], state["game"], state["now"], state["news"]
    age = minutes_between(news.published_at.max(), now) / 60 if len(news) else None
    rows = []
    for r in state["candidates"]:
        recent = view.prices(r["ticker"])
        volume = recent.loc[recent.ts > now - VOLUME_LOOKBACK, "volume"].sum()
        mid = (r["bid"] + r["ask"]) / 2
        moved = mid - r["before"] if r.get("anchored") else 0.0
        q = close_view.quote(r["ticker"])
        close_yes = (q.bid + q.ask) / 2 if q is not None else np.nan
        outcome = rp.outcomes.get(r["ticker"])
        for side in ("yes", "no"):
            sign = 1 if side == "yes" else -1
            price = r["ask"] if side == "yes" else 1 - r["bid"]
            if not 0 < price < 1:
                continue
            p_side = r["p"] if side == "yes" else 1 - r["p"]
            fee = fee_per_contract(price)
            close = close_yes if side == "yes" else 1 - close_yes
            rows.append({
                "date": game.date, "game_id": game.game_id, "as_of": now, "tip_time": game.tip_time,
                "ticker": r["ticker"], "team": r["team"], "side": side, "p_yes": r["p"], "price": float(price),
                "gap": p_side - price - fee, "side_price": float(price), "market_move": sign * moved,
                "model_shift": sign * r["shift"],
                "news_age_hours": NEWS_AGE_CAP if age is None else min(age, NEWS_AGE_CAP),
                "has_news": float(age is not None), "hours_to_tip": minutes_between(now, game.tip_time) / 60,
                "spread": r["ask"] - r["bid"], "log_volume": math.log1p(max(float(volume), 0.0)),
                "p_side": p_side, "is_yes": float(side == "yes"), "is_home": float(r["team"] == game.home_team),
                "close": close, "clv": close - price, "fee": fee, "net_clv": close - price - fee,
                "contracts": math.floor(stake / price + 1e-9),
                "won": None if outcome is None else float((outcome == 1) == (side == "yes"))})
    return rows


def split(rows: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return rows[(rows.date >= start) & (rows.date <= end)]


# ---------------- value model ----------------

class PolicyModel:
    """Predicts net CLV per contract for one option from FEATURES only (scaled; ridge, boosting or a small MLP)."""

    def __init__(self, family: str, params: dict | None = None, seed: int = SEED):
        self.family, self.params, self.seed = family, dict(params or {}), seed

    def _estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if self.family == "ridge":
            est = Ridge(**self.params)
        elif self.family == "gbm":
            est = HistGradientBoostingRegressor(random_state=self.seed, **self.params)
        elif self.family == "mlp":
            est = MLPRegressor(random_state=self.seed, max_iter=500, early_stopping=True, **self.params)
        else:
            raise ValueError(f"unknown policy family {self.family!r}")
        return make_pipeline(StandardScaler(), est)

    def fit(self, rows: pd.DataFrame):
        rows = rows.dropna(subset=[LABEL])
        self.model = self._estimator().fit(rows[FEATURES].to_numpy(float), rows[LABEL].to_numpy(float))
        self.trained_dates = (rows.date.min(), rows.date.max())
        self.n = len(rows)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[FEATURES].to_numpy(float))

    @property
    def name(self) -> str:
        return f"{self.family}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


def choose(rows: pd.DataFrame, pred: np.ndarray, tau: float) -> pd.DataFrame:
    """Simulated policy on labelled rows: per game, the first decision time whose best prediction clears tau."""
    r = rows.assign(pred=pred).dropna(subset=[LABEL])
    if r.empty:
        return r
    best = r.sort_values(["as_of", "pred"], ascending=[True, False]).groupby(["game_id", "as_of"]).head(1)
    picked = best[best.pred > tau].groupby("game_id").head(1)
    return picked.assign(clv_dollars=picked.clv * picked.contracts, net_dollars=picked.net_clv * picked.contracts)


def select(train: pd.DataFrame, seed: int = SEED) -> tuple:
    """Fit on FIT, score every (config, tau) on VALID by net CLV dollars. Returns (table, best, forced)."""
    if train.date.max() > TRAIN[1]:
        raise ValueError("selection rows must end with the training window")
    fit, valid = split(train, *FIT), split(train, *VALID)
    out = []
    for family, params in GRID:
        m = PolicyModel(family, params, seed).fit(fit)
        pred = m.predict(valid)
        for tau in THRESHOLDS:
            c = choose(valid, pred, tau)
            out.append({"family": family, "params": params, "config": m.name, "tau": tau, "trades": len(c),
                        "net_dollars": float(c.net_dollars.sum()) if len(c) else 0.0,
                        "clv_dollars": float(c.clv_dollars.sum()) if len(c) else 0.0,
                        "mean_net_clv": float(c.net_clv.mean()) if len(c) else np.nan})
    table = pd.DataFrame(out)
    order = table.sort_values(["net_dollars", "trades"], ascending=[False, True])
    top = order.iloc[0].to_dict()
    best = top if top["net_dollars"] > 0 else {**top, "family": "never", "params": {}, "config": "never trade",
                                               "tau": math.inf, "trades": 0, "net_dollars": 0.0}
    trading = order[order.trades >= MIN_FORCED_TRADES]
    forced = trading.iloc[0].to_dict() if len(trading) else None
    return table, best, forced


# ---------------- replay policy ----------------

class LearnedPolicy:
    """Replay policy over precomputed as-of rows: one position per game, the first time the best option clears tau."""

    def __init__(self, rows: pd.DataFrame, model: PolicyModel | None, tau: float, stake: float = DEFAULT_STAKE):
        self.tau, self.stake, self.held = tau, stake, set()
        pred = model.predict(rows) if model is not None and len(rows) else np.full(len(rows), -np.inf)
        self.rows = {k: g for k, g in rows[["game_id", "as_of", "ticker", "side", "p_yes"]]
                     .assign(pred=pred).groupby(["game_id", "as_of"])}
        self.name = "never trade" if model is None else f"learned {model.name} tau={tau}"

    def policy(self, view, game, now) -> list:
        if game.game_id in self.held:
            return []
        opts = self.rows.get((game.game_id, now))
        if opts is None or opts.empty:
            return []
        best = opts.loc[opts.pred.idxmax()]
        if not best.pred > self.tau:
            return []
        self.held.add(game.game_id)
        reason = (f"learned policy ({self.name}): predicted closing-line value after fees "
                  f"{best.pred * 100:+.1f} cents per contract for {best.side} on {best.ticker}")
        return [Order(best.ticker, best.side, float(best.p_yes), self.stake, reason, [], "platform", [],
                      "offline")]


def tip_lead_only(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows at the scheduled decision one hour before tip (useful for quick checks)."""
    return rows[rows.as_of == rows.tip_time - LEAD]
