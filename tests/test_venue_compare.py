"""Odds conversion, vig removal, fees, arbitrage maths, quote lookup and cross-venue game matching."""
import numpy as np
import pandas as pd
import pytest

from evaluation.venue_compare import (_last, all_in, american_to_prob, arbitrage, devig, kalshi_fee,
                                      match_games, polymarket_fee, sharpness)


def test_american_odds_to_implied_probability():
    assert american_to_prob(-150) == pytest.approx(0.6)
    assert american_to_prob(150) == pytest.approx(0.4)
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(-110) == pytest.approx(110 / 210)
    assert np.isnan(american_to_prob(float("nan")))


def test_devig_is_proportional_and_sums_to_one():
    h, a = american_to_prob(-110), american_to_prob(-110)
    assert h + a == pytest.approx(1.0476, abs=1e-4)
    assert devig(h, a) == pytest.approx((0.5, 0.5))
    h, a = devig(american_to_prob(-135), american_to_prob(114))
    assert h + a == pytest.approx(1.0)
    assert h / a == pytest.approx(american_to_prob(-135) / american_to_prob(114))


def test_fees_and_all_in_cost():
    assert kalshi_fee(0.5) == pytest.approx(0.0175)
    assert polymarket_fee(0.5) == pytest.approx(0.0125)
    assert kalshi_fee(0.3) == pytest.approx(kalshi_fee(0.7))
    assert all_in("kalshi", 0.6) == pytest.approx(0.6 + 0.07 * 0.24)
    assert all_in("polymarket", 0.6) == pytest.approx(0.6 + 0.05 * 0.24)
    assert all_in("draftkings", 0.6) == 0.6


def test_arbitrage_edge():
    assert arbitrage(0.48, 0.50) == pytest.approx(0.02)
    assert arbitrage(all_in("kalshi", 0.50), all_in("polymarket", 0.49)) < 0.01
    assert arbitrage(american_to_prob(-110), american_to_prob(-110)) < 0


def test_last_quote_is_strictly_before_and_fresh():
    ts = np.array(["2026-02-01T10:00", "2026-02-01T11:00"], dtype="datetime64[ns]")
    series = (ts, np.array([0.40, 0.42]), np.array([0.41, 0.43]))
    at = lambda s: np.datetime64(s, "ns")
    assert _last(series, at("2026-02-01T11:00"), pd.Timedelta(hours=2)) == (0.40, 0.41)
    assert _last(series, at("2026-02-01T11:30"), pd.Timedelta(hours=2)) == (0.42, 0.43)
    assert _last(series, at("2026-02-01T15:00"), pd.Timedelta(hours=2)) is None
    assert _last(series, at("2026-02-01T09:00"), pd.Timedelta(hours=2)) is None


def test_match_games_ignores_order_and_tolerates_a_day():
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "date": ["2026-02-20", "2026-02-21", "2026-03-01"],
                          "home_team": ["CHA", "BOS", "CHA"], "away_team": ["CLE", "MIA", "CLE"]})
    events = pd.DataFrame({"date": ["2026-02-20", "2026-02-22", "2026-03-01", "2026-02-25", "2026-02-20"],
                           "team_a": ["CLE", "BOS", "CHA", "CHA", "LAL"],
                           "team_b": ["CHA", "MIA", "CLE", "CLE", "GSW"]})
    assert list(match_games(events, games)) == ["g1", "g2", "g3", None, None]


def test_sharpness_prefers_the_better_forecast():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    table = pd.DataFrame({"date": np.repeat([f"d{i}" for i in range(20)], 10), "home_win": y,
                          "kalshi_close_p_home": np.where(y == 1, 0.7, 0.3),
                          "draftkings_close_p_home": np.full(200, 0.5)})
    level, diff = sharpness(table, reps=200)
    brier = level[level.metric == "brier"].set_index("venue").value
    assert brier["kalshi"] == pytest.approx(0.09)
    assert brier["draftkings"] == pytest.approx(0.25)
    row = diff[diff.metric == "brier"].iloc[0]
    assert row.value < 0 and row.ci_high < 0
