"""Day-clustered bootstrap, paired comparisons and the CLV > 0 test on small hand-made fills."""
import numpy as np
import pandas as pd
import pytest

from evaluation.stats import bootstrap, clv_positive_test, day_table, paired, p_greater

GAMES = pd.DataFrame({"game_id": [f"g{i}" for i in range(6)],
                      "date": ["d1", "d1", "d2", "d3", "d4", "d5"]})
DAYS = ["d1", "d2", "d3", "d4", "d5"]


def fills(clv, pnl, games=("g0", "g1", "g2", "g3", "g4", "g5"), price=0.5, contracts=40):
    n = len(clv)
    return pd.DataFrame({"game_id": list(games)[:n], "clv": clv, "pnl": pnl, "price": [price] * n,
                         "contracts": [contracts] * n})


def test_day_table_sums_per_day_and_keeps_empty_days():
    t = day_table(fills([0.01, 0.03, -0.02], [5.0, -3.0, 1.0]), GAMES, DAYS)
    assert list(t.index) == DAYS
    assert t.loc["d1", "trades"] == 2 and t.loc["d1", "clv_sum"] == pytest.approx(0.04)
    assert t.loc["d1", "clv_dollars"] == pytest.approx(0.04 * 40)
    assert t.loc["d2", "pnl"] == 1.0 and t.loc["d2", "staked"] == 20.0
    assert (t.loc[["d3", "d4", "d5"]] == 0).all().all()


def test_day_table_rejects_fills_outside_window():
    with pytest.raises(ValueError):
        day_table(fills([0.01, 0.0, 0.0], [1.0, 0.0, 0.0]), GAMES, ["d2", "d3"])


def test_point_estimates_match_plain_formulas():
    f = fills([0.02, -0.01, 0.03, 0.00, 0.01, -0.02], [4.0, -6.0, 9.0, -2.0, 3.0, -5.0])
    r = bootstrap(day_table(f, GAMES, DAYS), reps=500, seed=1)
    assert r.loc["mean_clv", "estimate"] == pytest.approx(f.clv.mean())
    assert r.loc["clv_dollars", "estimate"] == pytest.approx((f.clv * f.contracts).sum())
    assert r.loc["pnl", "estimate"] == pytest.approx(f.pnl.sum())
    assert r.loc["roi", "estimate"] == pytest.approx(f.pnl.sum() / (f.price * f.contracts).sum())
    for m in r.index:
        assert r.loc[m, "ci_low"] <= r.loc[m, "estimate"] <= r.loc[m, "ci_high"]


def test_bootstrap_is_deterministic_for_a_seed():
    t = day_table(fills([0.02, -0.01, 0.03, 0.0], [4.0, -6.0, 9.0, -2.0]), GAMES, DAYS)
    pd.testing.assert_frame_equal(bootstrap(t, reps=300, seed=3), bootstrap(t, reps=300, seed=3))


def test_never_trade_has_zero_pnl_and_undefined_clv():
    r = bootstrap(day_table(pd.DataFrame(), GAMES, DAYS), reps=200)
    assert r.loc["pnl", "estimate"] == 0 and r.loc["pnl", "ci_low"] == 0 and r.loc["pnl", "ci_high"] == 0
    assert np.isnan(r.loc["mean_clv", "estimate"])


def test_clv_test_rejects_only_for_a_clear_positive_edge():
    days = [f"d{i}" for i in range(40)]
    games = pd.DataFrame({"game_id": [f"g{i}" for i in range(40)], "date": days})
    rng = np.random.default_rng(0)
    good = fills(list(0.03 + 0.01 * rng.standard_normal(40)), [1.0] * 40, games.game_id)
    noise = fills(list(0.03 * rng.standard_normal(40) - 0.002), [1.0] * 40, games.game_id)
    assert clv_positive_test(day_table(good, games, days))["p_value"] < 0.01
    assert clv_positive_test(day_table(noise, games, days))["p_value"] > 0.05


def test_paired_difference_and_one_sided_p():
    a = day_table(fills([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], [5.0] * 6), GAMES, DAYS)
    b = day_table(fills([-0.02] * 6, [-5.0] * 6), GAMES, DAYS)
    d = paired(a, b, reps=500, seed=2)
    assert d.loc["pnl", "estimate"] == pytest.approx(60.0)
    assert d.loc["mean_clv", "estimate"] == pytest.approx(0.04)
    assert d.loc["pnl", "p_a_gt_b"] < 0.01
    assert paired(b, a, reps=500, seed=2).loc["pnl", "p_a_gt_b"] > 0.99


def test_paired_against_never_trade_equals_the_setup_itself():
    a = day_table(fills([0.01, -0.03, 0.02, 0.0], [3.0, -8.0, 2.0, 1.0]), GAMES, DAYS)
    never = day_table(pd.DataFrame(), GAMES, DAYS)
    d, solo = paired(a, never, reps=400, seed=5), bootstrap(a, reps=400, seed=5)
    assert d.loc["pnl", "estimate"] == pytest.approx(solo.loc["pnl", "estimate"])
    assert d.loc["pnl", "ci_low"] == pytest.approx(solo.loc["pnl", "ci_low"])


def test_paired_needs_the_same_days():
    a = day_table(pd.DataFrame(), GAMES, DAYS)
    with pytest.raises(ValueError):
        paired(a, day_table(pd.DataFrame(), GAMES, DAYS[:-1]))


def test_p_greater_bounds():
    assert p_greater(1.0, np.full(99, 1.0)) == pytest.approx(0.01)
    assert p_greater(-1.0, np.full(99, -1.0)) == 1.0
    assert np.isnan(p_greater(0.0, np.zeros(99)))
