"""Fixed-complexity quantile boosting and a shared monotone points calibrator."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from forecast.baselines import QUANTILES, TARGETS, _qcols, prob_at_least


class PlayerBoost:
    """Same model capacity for baseline/enriched features; no random early stopping."""
    def __init__(self, features, iterations=150, name='experiment-gbm'):
        self.feature_names = tuple(features)
        self.iterations, self.name = iterations, name

    def fit(self, rows):
        self.features = [f for f in self.feature_names if rows[f].notna().any()]
        x = rows[self.features].astype(float)
        self.models = {}
        for target in TARGETS:
            for q in QUANTILES:
                self.models[target, q] = HistGradientBoostingRegressor(
                    loss='quantile', quantile=q, max_iter=self.iterations,
                    max_leaf_nodes=15, min_samples_leaf=40, learning_rate=.05,
                    l2_regularization=1., early_stopping=False, random_state=7606,
                ).fit(x, rows[target])
        return self

    def predict(self, rows):
        x = rows[self.features].astype(float)
        result = pd.DataFrame(index=rows.index)
        for target in TARGETS:
            q = np.sort(np.column_stack([self.models[target, p].predict(x) for p in QUANTILES]), axis=1)
            result[_qcols(target)] = np.maximum(q, 0)
        return result


def points_probabilities(pred, thresholds):
    """P(integer points >= threshold | played); shares the legacy quantile CDF."""
    values = pred[_qcols('pts')].to_numpy()
    return np.array([[prob_at_least(q, float(np.ceil(t))) for t in thresholds] for q in values])


class PointsCalibrator:
    """A single positive-slope sigmoid mapping shared by all points thresholds.

    This preserves event ordering. It calibrates event probabilities, not the
    raw quantile predictions. Fit only after the forecaster is frozen.
    """
    epsilon = 1e-5

    def fit(self, probabilities, outcomes, sample_weight=None):
        p, y = np.asarray(probabilities), np.asarray(outcomes)
        if p.shape != y.shape or p.ndim != 2 or len(p) < 20:
            raise ValueError('Matching 2D arrays with >=20 calibration rows required')
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Probabilities must be finite and in [0, 1]')
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
            raise ValueError('Binary outcomes with both classes required')
        # Each player-game receives one unit of weight across threshold events.
        weights = np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight)
        weights = np.repeat(weights / p.shape[1], p.shape[1])
        x = logit(np.clip(p.ravel(), self.epsilon, 1-self.epsilon)).reshape(-1, 1)
        m = LogisticRegression(C=1000., max_iter=1000).fit(x, y.ravel(), sample_weight=weights)
        self.slope, self.intercept = float(m.coef_[0, 0]), float(m.intercept_[0])
        self.identity_fallback = self.slope <= 0
        if self.identity_fallback:
            self.slope, self.intercept = 1., 0.
        return self

    def predict(self, probabilities):
        p = np.asarray(probabilities)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Probabilities must be finite and in [0, 1]')
        return expit(self.slope * logit(np.clip(p, self.epsilon, 1-self.epsilon)) + self.intercept)
