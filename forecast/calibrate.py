"""Post-hoc calibration of the M4 win model (Platt scaling or isotonic regression).

    from forecast.calibrate import Calibrator, CalibratedWinModel, fit_before, month_starts, out_of_sample
    oos = out_of_sample(rows, month_starts("2023-11-01", "2026-06-01"))   # rolling-origin M4 probabilities
    cal = fit_before(oos, "2026-02-01", "platt")                          # only games before the cut-off
    Forecaster(CalibratedWinModel(win, cal), play, points)                # same interface as the raw model

A calibrator must only see predictions M4 made on games it was not trained on,
so out_of_sample refits M4 before every month on all earlier games and
predicts that month.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from forecast.win import WinModel

EPS = 1e-4
METHODS = ("platt", "isotonic")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Monotone map from raw M4 probability to a calibrated one, strictly inside (0, 1).

    platt:    sigmoid(slope * logit(p) + intercept); slope > 1 stretches a model compressed toward 50%.
    isotonic: non-decreasing step function, clipped to [0.01, 0.99].
    """

    def __init__(self, method="platt"):
        if method not in METHODS:
            raise ValueError(f"unknown calibration method {method!r}; use one of {METHODS}")
        self.method = method

    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y, float)
        if self.method == "platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
            self.slope, self.intercept = float(m.coef_[0, 0]), float(m.intercept_[0])
        else:
            self.iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
        self.n = len(p)
        return self

    def __call__(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "platt":
            return 1 / (1 + np.exp(-(self.slope * logit(p) + self.intercept)))
        return self.iso.predict(p)

    def params(self) -> dict:
        if self.method == "platt":
            return {"slope": self.slope, "intercept": self.intercept, "n": self.n}
        return {"n": self.n, "steps": int(len(self.iso.X_thresholds_))}


class CalibratedWinModel:
    """A WinModel whose probabilities pass through a calibrator; drop-in for forecast.api.Forecaster."""

    def __init__(self, base: WinModel, calibrator: Calibrator):
        self.base, self.calibrator = base, calibrator
        self.name = f"{base.name}+{calibrator.method}"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.calibrator(self.base.predict(rows))

    def coefficients(self) -> dict:
        return {**self.base.coefficients(), **{f"cal_{k}": v for k, v in self.calibrator.params().items()}}


def month_starts(start: str, end: str) -> list:
    """First day of every month from start to end inclusive, as YYYY-MM-DD strings."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="MS")]


def out_of_sample(rows: pd.DataFrame, months, min_train=50) -> pd.DataFrame:
    """M4 refit on every game before each month start, then scored on that month only.

    rows: forecast.win.training_rows(...) plus a `date` column (YYYY-MM-DD).
    Returns date, game_id, p_raw (out-of-sample probability), home_win and fit_before.
    """
    starts = sorted(months)
    out = []
    for start, end in zip(starts, starts[1:] + ["9999-12-31"]):
        train, test = rows[rows.date < start], rows[(rows.date >= start) & (rows.date < end)]
        if len(train) < min_train or test.empty:
            continue
        out.append(pd.DataFrame({"date": test.date.to_numpy(), "game_id": test.game_id.to_numpy(),
                                 "p_raw": WinModel().fit(train).predict(test),
                                 "home_win": test.home_win.to_numpy(), "fit_before": start}))
    cols = ["date", "game_id", "p_raw", "home_win", "fit_before"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def fit_before(oos: pd.DataFrame, cutoff: str, method="platt") -> Calibrator:
    """Calibrator fit only on out-of-sample predictions for games dated before `cutoff`."""
    part = oos[oos.date < cutoff]
    if len(part) < 50:
        raise ValueError(f"only {len(part)} out-of-sample predictions before {cutoff}")
    return Calibrator(method).fit(part.p_raw, part.home_win)
