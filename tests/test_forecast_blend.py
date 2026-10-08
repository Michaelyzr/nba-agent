import numpy as np
import pandas as pd
import pytest

from agents.forecast_blend import SIGNALS, BlendLearner, fit_weights, logit


def toy_games(days=30, per_day=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2026-02-01", periods=days).strftime("%Y-%m-%d"):
        for i in range(per_day):
            truth = rng.uniform(0.2, 0.8)
            row = {"game_id": f"{d}-{i}", "date": d, "home_win": float(rng.uniform() < truth)}
            for s in SIGNALS:
                noise = 0.02 if s == "anchor_mlp" else 0.2
                row[s] = float(np.clip(truth + rng.normal(0, noise), 0.03, 0.97))
            rows.append(row)
    return pd.DataFrame(rows)


def test_default_weights_reproduce_agent_estimate():
    g = toy_games(days=3)
    learner = BlendLearner()
    assert learner.weights[SIGNALS.index("anchor_m4")] == 1.0
    np.testing.assert_allclose(learner.predict(g), g.anchor_m4.clip(0.02, 0.98), atol=1e-9)


def test_fitted_weights_stay_on_simplex():
    g = toy_games(days=10)
    z = logit(g[list(SIGNALS)].to_numpy())
    w0 = np.eye(len(SIGNALS))[SIGNALS.index("anchor_m4")]
    w = fit_weights(z, g.home_win.to_numpy(), w0)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    y = g.home_win.to_numpy()
    loss = lambda w: -np.mean(y * np.log(1 / (1 + np.exp(-z @ w))) + (1 - y) * np.log(1 / (1 + np.exp(z @ w))))
    assert loss(w) < loss(w0)


def test_review_rejects_future_games():
    g = toy_games(days=5)
    with pytest.raises(ValueError):
        BlendLearner().review(g, "2026-02-02")


def test_walk_forward_uses_no_future_outcomes():
    g = toy_games(days=30)
    base = BlendLearner().run(g).set_index("game_id").p_blend
    flipped = g.copy()
    late = flipped.date >= "2026-02-25"
    flipped.loc[late, "home_win"] = 1 - flipped.loc[late, "home_win"]
    alt = BlendLearner().run(flipped).set_index("game_id").p_blend
    early = g.loc[g.date <= "2026-02-25", "game_id"]
    np.testing.assert_allclose(base[early], alt[early])     # day 25 is predicted before its outcomes are known
