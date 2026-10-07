"""Independent score/time win model. Diffusion is an uncalibrated fallback.

Optional logistic training uses timestamped in-play snapshots, split by games:
    python -m forecast.inplay --training-data snapshots.parquet --split 2026-02-01
The existing pregame M4 model and its weights are never modified.
"""
import argparse
import math
import pickle
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

FEATURES = ["scaled_margin", "scaled_prior", "scaled_news", "scaled_possession"]
NORMAL = NormalDist()


def remaining_seconds(score):
    if score["phase"] == "final":
        return 0.0
    period, clock = score["period"], score["clock_seconds"]
    if period <= 4:
        remaining = max(0, 4 - max(period, 1)) * 720 + clock
    else:
        remaining = clock
    # A tied zero-clock live state awaits overtime; it is not a settled result.
    if remaining == 0 and score["home_score"] == score["away_score"]:
        remaining = 300.0
    return remaining


def features(score, prior, news_margin=0.0):
    if not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("initial probability must be between zero and one")
    fraction = max(remaining_seconds(score) / 2880, 1 / 2880)
    root = math.sqrt(fraction)
    return {"scaled_margin": (score["home_score"] - score["away_score"]) / root,
            "scaled_prior": math.log(prior / (1 - prior)) * root,
            "scaled_news": news_margin / root,
            "scaled_possession": float(score.get("possession_sign") or 0) / root}


class InPlayWinModel:
    def __init__(self, sigma=14.0, fitted=None):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("margin uncertainty must be positive and finite")
        self.sigma, self.fitted = float(sigma), fitted
        self.validation = None

    @property
    def name(self):
        if self.fitted is not None and hasattr(self.fitted, "kind"):
            return f"inplay-{self.fitted.kind}-calibrated"
        return "inplay-logistic" if self.fitted is not None else "inplay-diffusion-prototype"

    def predict(self, score, prior, news_margin=0.0):
        f = features(score, prior, news_margin)
        if self.fitted is not None:
            p = float(self.fitted.predict_proba(pd.DataFrame([f])[FEATURES])[0, 1])
        else:
            fraction = max(remaining_seconds(score) / 2880, 1 / 2880)
            mean = NORMAL.inv_cdf(prior) * self.sigma * fraction
            z = (score["home_score"] - score["away_score"] + mean + news_margin) / (self.sigma * math.sqrt(fraction))
            p = NORMAL.cdf(z)
        return min(max(p, 0.001), 0.999)

    def fit(self, frame):
        if frame.home_win.nunique() != 2 or len(frame) < 20:
            raise ValueError("training requires at least 20 snapshots and both outcomes")
        if not np.isfinite(frame[FEATURES].to_numpy(float)).all():
            raise ValueError("training features must be finite")
        # Give each game equal total weight; long/fast-polled games cannot dominate.
        weight = 1 / frame.groupby("game_id").game_id.transform("size")
        weight *= len(frame) / weight.sum()
        self.fitted = LogisticRegression(C=0.1, max_iter=2000).fit(frame[FEATURES], frame.home_win,
                                                                 sample_weight=weight)
        return self


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-data", type=Path, required=True)
    ap.add_argument("--split", required=True, help="game-tip UTC date separating train/test")
    ap.add_argument("--output", type=Path, default=Path("models/inplay/win.pkl"))
    args = ap.parse_args()
    frame = pd.read_parquet(args.training_data)
    required = set(FEATURES + ["game_id", "tip_time", "final_at", "as_of", "home_win"])
    if not required <= set(frame):
        ap.error(f"missing training columns: {sorted(required - set(frame))}")
    for column in ("tip_time", "final_at", "as_of"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if ((frame.as_of < frame.tip_time) | (frame.as_of >= frame.final_at)).any():
        ap.error("training snapshots must be observed during the game, before its outcome")
    if (frame.groupby("game_id")[["home_win", "tip_time", "final_at"]].nunique() != 1).any().any():
        ap.error("inconsistent outcome or game times within a game")
    if not set(frame.home_win) <= {0, 1}:
        ap.error("home_win must contain binary completed-game labels")
    split = pd.to_datetime(args.split, utc=True)
    train = frame[(frame.tip_time < split) & (frame.final_at < split)]
    test = frame[frame.tip_time >= split]
    if test.empty:
        ap.error("chronological holdout must not be empty")
    model = InPlayWinModel().fit(train)
    p = model.fitted.predict_proba(test[FEATURES])[:, 1]
    model.validation = {"train_games": int(train.game_id.nunique()), "test_games": int(test.game_id.nunique()),
                        "brier": brier_score_loss(test.home_win, p), "log_loss": log_loss(test.home_win, p, labels=[0, 1])}
    print(model.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(model))


if __name__ == "__main__":
    main()
