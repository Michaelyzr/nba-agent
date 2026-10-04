"""M1 baselines: rolling average (M1a) and quantile gradient boosting (M1b).

    python -m forecast.baselines --source synthetic            # prototype data, train before split, test after
    python -m forecast.baselines --source frozen --split 2026-02-01

Both predict points and minutes for a player who plays, as quantiles, so they
share the forecast format of M2. Trained models go to models/m1/.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast.features import FEATURES, TARGETS, make_features

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "m1"


def _qcols(target):
    return [f"{target}_q{int(q * 100):02d}" for q in QUANTILES]


class RollingAverage:
    """M1a: centre on the last-10-game average; spread from training residuals.

    Residuals are scaled by sqrt(centre + 1) because scorers with higher averages
    vary more. Players with no history get the training mean.
    """
    name = "rolling10"

    def fit(self, rows: pd.DataFrame):
        self.fallback, self.scaled_q = {}, {}
        for t in TARGETS:
            centre = rows[f"{t}_avg10"].fillna(rows[t].mean())
            self.fallback[t] = float(rows[t].mean())
            z = (rows[t] - centre) / np.sqrt(centre.clip(lower=0) + 1)
            self.scaled_q[t] = np.quantile(z.dropna(), QUANTILES)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=rows.index)
        for t in TARGETS:
            centre = rows[f"{t}_avg10"].fillna(self.fallback[t]).to_numpy()
            scale = np.sqrt(np.clip(centre, 0, None) + 1)
            for col, z in zip(_qcols(t), self.scaled_q[t]):
                out[col] = np.clip(centre + z * scale, 0, None)
        return out


class GBMForecaster:
    """M1b: one quantile gradient-boosting model per target and quantile."""
    name = "gbm"

    def __init__(self, max_iter=300, learning_rate=0.05, min_samples_leaf=40, random_state=0):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, min_samples_leaf=min_samples_leaf,
                           early_stopping=True, validation_fraction=0.1, random_state=random_state)

    def fit(self, rows: pd.DataFrame):
        # Features with no values (e.g. starters without --advanced) break the binning step.
        self.features = [f for f in FEATURES if rows[f].notna().any()]
        X = rows[self.features].astype(float)
        self.models = {(t, q): HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params).fit(X, rows[t])
                       for t in TARGETS for q in QUANTILES}
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows[self.features].astype(float)
        out = pd.DataFrame(index=rows.index)
        for t in TARGETS:
            preds = np.column_stack([self.models[(t, q)].predict(X) for q in QUANTILES])
            preds = np.clip(np.sort(preds, axis=1), 0, None)    # quantiles must not cross
            for i, col in enumerate(_qcols(t)):
                out[col] = preds[:, i]
        return out


def prob_at_least(qvalues, threshold: float) -> float:
    """P(X >= threshold) for a whole-number stat, from QUANTILES by linear interpolation.

    Tails are extended linearly to probability 0 and 1 at twice the outer
    quantile gaps. For "25+ points" pass threshold=25.
    """
    v = np.asarray(qvalues, dtype=float)
    lo, hi = v[0] - 2 * max(v[1] - v[0], 0.5), v[-1] + 2 * max(v[-1] - v[-2], 0.5)
    xs = np.maximum.accumulate(np.concatenate([[lo], v, [hi]])) + np.arange(len(v) + 2) * 1e-9
    ps = np.concatenate([[0.0], QUANTILES, [1.0]])
    return float(1 - np.interp(threshold - 0.5, xs, ps))


def evaluate(pred: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Pinball loss (lower is better), 10-90% interval coverage (target 0.80) and median error."""
    out = []
    for t in TARGETS:
        y = rows[t].to_numpy()
        q = pred[_qcols(t)].to_numpy()
        diff = y[:, None] - q
        pinball = np.mean(np.maximum(np.array(QUANTILES) * diff, (np.array(QUANTILES) - 1) * diff))
        coverage = np.mean((y >= q[:, 0]) & (y <= q[:, -1]))
        mae = np.mean(np.abs(y - q[:, QUANTILES.index(0.5)]))
        out.append({"target": t, "pinball": pinball, "coverage_10_90": coverage, "median_abs_error": mae})
    return pd.DataFrame(out)


def forecast_json(row: pd.Series, pred: pd.Series, model: str, target: str, lines=(), as_of=None, out=()) -> dict:
    """One forecast in the README section 5 format."""
    q = [float(pred[c]) for c in _qcols(target)]
    return {"game_id": row.game_id, "player_id": int(row.player_id),
            "as_of": None if as_of is None else pd.Timestamp(as_of).isoformat(),
            "model": model, "target": target,
            "quantiles": {f"p{int(p * 100)}": round(v, 2) for p, v in zip(QUANTILES, q)},
            "p_over": {str(line): round(prob_at_least(q, line), 4) for line in lines},
            "p_play": None, "overrides": {"out": list(out)}}


def save(model, path: Path | None = None) -> Path:
    path = path or MODEL_DIR / f"{model.name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    return path


def load(name: str, path: Path | None = None):
    return pickle.loads((path or MODEL_DIR / f"{name}.pkl").read_bytes())


def load_source(source: str):
    if source == "synthetic":
        from forecast.dev_data import synthetic_tables
        return synthetic_tables()
    from data_sources import read_table
    return read_table("player_games"), read_table("games")


def main():
    # Import from the package so pickles record forecast.baselines, not __main__, and load anywhere.
    from forecast.baselines import GBMForecaster, RollingAverage

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen"], default="synthetic")
    ap.add_argument("--split", default=None, help="first test date (default: last 30%% of games)")
    args = ap.parse_args()
    player_games, games = load_source(args.source)
    rows = make_features(player_games, games)
    split = pd.Timestamp(args.split, tz="UTC") if args.split else rows.tip_time.quantile(0.7)
    train, test = rows[rows.tip_time < split], rows[rows.tip_time >= split]
    print(f"{args.source}: {len(train)} training rows, {len(test)} test rows, split at {split:%Y-%m-%d}")
    results = []
    for model in (RollingAverage(), GBMForecaster()):
        model.fit(train)
        scores = evaluate(model.predict(test), test).assign(model=model.name)
        results.append(scores)
        print(f"saved {save(model)}")
    print(pd.concat(results)[["model", "target", "pinball", "coverage_10_90", "median_abs_error"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
