"""Game-grouped, chronological train/calibration/test for independent in-play models.

python -m forecast.inplay_training --training-data snapshots.parquet \
    --train-end 2026-02-01 --calibration-end 2026-03-01 --candidate logistic
Synthetic fixtures validate mechanics only; no sample weights are fitted for the UI.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from forecast.inplay import FEATURES, InPlayWinModel


def game_weights(frame):
    w = 1 / frame.groupby("game_id").game_id.transform("size")
    return np.asarray(w * len(frame)/w.sum())


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


class CalibratedPredictor:
    def __init__(self, base, calibration, kind):
        self.base, self.calibration, self.kind = base, calibration, kind

    def predict_proba(self, frame):
        p = self.base.predict_proba(frame)[:, 1]
        return self.calibration.predict_proba(logit(p))


def chronological_parts(frame, train_end, calibration_end):
    frame = frame.copy()
    required = set(FEATURES + ["game_id", "tip_time", "final_at", "as_of", "home_win"])
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("training fields are missing or null")
    for name in ("tip_time", "final_at", "as_of"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if not np.isfinite(frame[FEATURES].to_numpy(float)).all() or not set(frame.home_win) <= {0, 1}:
        raise ValueError("invalid training features or outcomes")
    if ((frame.as_of < frame.tip_time) | (frame.as_of >= frame.final_at)).any():
        raise ValueError("snapshot must precede the final result")
    if (frame.groupby("game_id")[["home_win", "tip_time", "final_at"]].nunique() != 1).any().any():
        raise ValueError("inconsistent game labels or times")
    a, b = pd.to_datetime(train_end, utc=True), pd.to_datetime(calibration_end, utc=True)
    if a >= b:
        raise ValueError("calibration end must follow train end")
    train = frame[(frame.tip_time < a) & (frame.final_at < a)]
    calibration = frame[(frame.tip_time >= a) & (frame.final_at < b)]
    test = frame[frame.tip_time >= b]
    for part in (train, calibration, test):
        if part.game_id.nunique() < 2 or part.home_win.nunique() != 2:
            raise ValueError("each split needs independent games and both outcomes")
    return train, calibration, test


def fit_calibrated(frame, train_end, calibration_end, candidate="logistic"):
    train, calibration, test = chronological_parts(frame, train_end, calibration_end)
    if candidate == "logistic":
        base = LogisticRegression(C=.1, max_iter=2000)
    elif candidate == "hgb":
        base = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=10,
                                             l2_regularization=2, monotonic_cst=[1, 1, 1, 1], random_state=0)
    else:
        raise ValueError("unknown candidate model")
    base.fit(train[FEATURES], train.home_win, sample_weight=game_weights(train))
    cal = LogisticRegression(C=1, max_iter=2000).fit(logit(base.predict_proba(calibration[FEATURES])[:, 1]),
                                                  calibration.home_win, sample_weight=game_weights(calibration))
    if cal.coef_[0, 0] < 0:
        raise ValueError("calibration reversed probability ordering; do not publish this model")
    predictor = CalibratedPredictor(base, cal, candidate)
    p = predictor.predict_proba(test[FEATURES])[:, 1]
    raw = base.predict_proba(test[FEATURES])[:, 1]
    weight = game_weights(test)
    reliability = []
    for low in np.arange(0, 1, .1):
        mask = (p >= low) & (p < low+.1)
        if mask.any():
            reliability.append({"bin": round(float(low), 1), "predicted": float(np.average(p[mask], weights=weight[mask])),
                                "observed": float(np.average(test.home_win.to_numpy()[mask], weights=weight[mask])),
                                "game_weight": float(weight[mask].sum()/weight.sum())})
    model = InPlayWinModel(fitted=predictor)
    model.validation = {"calibrated": True, "candidate": candidate,
                        "train_games": int(train.game_id.nunique()), "calibration_games": int(calibration.game_id.nunique()),
                        "test_games": int(test.game_id.nunique()), "train_end": str(train_end), "calibration_end": str(calibration_end),
                        "brier": float(brier_score_loss(test.home_win, p, sample_weight=weight)),
                        "log_loss": float(log_loss(test.home_win, p, sample_weight=weight, labels=[0, 1])),
                        "raw_brier": float(brier_score_loss(test.home_win, raw, sample_weight=weight)),
                        "raw_log_loss": float(log_loss(test.home_win, raw, sample_weight=weight, labels=[0, 1])),
                        "ece": sum(r['game_weight']*abs(r['predicted']-r['observed']) for r in reliability),
                        "reliability": reliability, "development_only": True,
                        "note": "训练入口仅供研究；上线还需独立市场成本回测、跨赛季检验和样本充分性评审。"}
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--calibration-end", required=True)
    parser.add_argument("--candidate", choices=("logistic", "hgb"), default="logistic")
    parser.add_argument("--output", type=Path, default=Path("models/inplay/calibrated.pkl"))
    args = parser.parse_args()
    model = fit_calibrated(pd.read_parquet(args.training_data), args.train_end, args.calibration_end, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(model))
    args.output.with_suffix('.validation.json').write_text(json.dumps(model.validation, ensure_ascii=False, indent=2))
    print(model.validation)


if __name__ == "__main__":
    main()
