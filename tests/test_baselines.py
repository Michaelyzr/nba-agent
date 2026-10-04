"""M1 baselines and shared features on hand-made seasons (no network, no downloads)."""
import numpy as np
import pandas as pd
import pytest

from forecast.baselines import (QUANTILES, GBMForecaster, RollingAverage, evaluate, forecast_json, load,
                                prob_at_least, save)
from forecast.features import FEATURES, make_features

START = pd.Timestamp("2025-10-21T23:30:00Z")
STAR, WING, BENCH, OPP1, OPP2 = 1, 2, 3, 11, 12


def season(n_games=60, star_out=lambda i: i % 4 == 3, seed=0):
    """Team 100 hosts team 200 every other day. When the star sits, the wing plays 12 more minutes."""
    rng = np.random.default_rng(seed)
    games, pg = [], []
    for i in range(n_games):
        gid, tip = f"g{i:03d}", START + pd.Timedelta(days=2 * i)
        games.append({"game_id": gid, "date": str(tip.date()), "tip_time": tip,
                      "final_at": tip + pd.Timedelta(hours=3), "home_team_id": 100, "away_team_id": 200,
                      "home_team": "HOM", "away_team": "AWY", "home_pts": 110 + i % 7, "away_pts": 105})
        out = star_out(i)
        lineup = [] if out else [(STAR, 100, 34)]
        lineup += [(WING, 100, 24 + (12 if out else 0)), (BENCH, 100, 14), (OPP1, 200, 33), (OPP2, 200, 30)]
        for pid, team, mins in lineup:
            m = float(np.clip(mins + rng.normal(0, 2), 1, 48))
            pg.append({"game_id": gid, "player_id": pid, "team_id": team, "min": m,
                       "pts": int(round(m * 0.6 + rng.normal(0, 2))), "fga": int(m / 3), "fta": 2,
                       "usage": np.nan, "started": pd.NA})
    return pd.DataFrame(pg), pd.DataFrame(games)


def test_features_use_only_earlier_games():
    pg, games = season(n_games=8, star_out=lambda i: False)
    rows = make_features(pg, games)
    wing = rows[rows.player_id == WING].reset_index(drop=True)
    actual = pg[pg.player_id == WING].reset_index(drop=True)
    assert np.isnan(wing.pts_avg5[0]) and wing.games_prior[0] == 0
    assert wing.pts_avg5[3] == pytest.approx(actual.pts[:3].mean())
    assert wing.pts_avg10[7] == pytest.approx(actual.pts[:7].mean())
    assert wing.rest_days[1] == pytest.approx(2.0) and wing.back_to_back[1] == 0


def test_changing_a_later_game_does_not_change_earlier_features():
    pg, games = season(n_games=10)
    before = make_features(pg, games)
    pg.loc[pg.game_id == "g009", "pts"] = 99
    after = make_features(pg, games)
    early = before.game_id < "g009"
    pd.testing.assert_frame_equal(before[early][FEATURES], after[early][FEATURES])


def test_teammates_out_counts_the_missing_star_only_when_he_sits():
    pg, games = season(n_games=12)
    rows = make_features(pg, games).set_index(["game_id", "player_id"])
    assert rows.loc[("g003", WING), "teammates_out_min"] == pytest.approx(
        pg[(pg.player_id == STAR) & (pg.game_id < "g003")]["min"].mean())
    assert rows.loc[("g004", WING), "teammates_out_min"] == 0
    assert rows.loc[("g003", OPP1), "teammates_out_min"] == 0


def test_traded_player_is_not_counted_as_missing():
    pg, games = season(n_games=20, star_out=lambda i: False)
    pg = pg[~((pg.player_id == STAR) & (pg.game_id >= "g010"))]       # star leaves the team after game 9
    rows = make_features(pg, games).set_index(["game_id", "player_id"])
    assert rows.loc[("g010", WING), "teammates_out_min"] > 0           # first game without him: counted
    assert rows.loc[("g019", WING), "teammates_out_min"] == 0          # weeks later: no longer rotation


def test_prediction_rows_match_training_rows_and_take_out_list():
    pg, games = season(n_games=12, star_out=lambda i: False)
    train = make_features(pg, games).set_index(["game_id", "player_id"])
    target = pd.DataFrame({"game_id": ["g011"], "player_id": [WING], "team_id": [100], "out": [[STAR]]})
    pred = make_features(pg[pg.game_id < "g011"], games, targets=target).iloc[0]
    same = [f for f in FEATURES if f != "teammates_out_min"]
    np.testing.assert_allclose(pred[same].astype(float), train.loc[("g011", WING), same].astype(float))
    assert pred.teammates_out_min == pytest.approx(pg[(pg.player_id == STAR) & (pg.game_id < "g011")]["min"].tail(10).mean())
    assert np.isnan(pred["pts"])


def test_rolling_average_quantiles_are_ordered_and_centred():
    pg, games = season()
    rows = make_features(pg, games).dropna(subset=["pts_avg10"])
    pred = RollingAverage().fit(rows).predict(rows)
    q = pred[[f"pts_q{int(p * 100):02d}" for p in QUANTILES]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert np.median(q[:, 2] - rows.pts_avg10) == pytest.approx(0, abs=1.5)


def test_gbm_learns_that_minutes_rise_when_the_star_sits():
    pg, games = season(n_games=120)
    rows = make_features(pg, games)
    wing = rows[(rows.player_id == WING) & (rows.games_prior >= 5)]
    model = GBMForecaster(max_iter=150, min_samples_leaf=5).fit(rows)
    pred = model.predict(wing)
    star_out = wing.teammates_out_min > 0
    assert pred.loc[star_out, "min_q50"].mean() - pred.loc[~star_out, "min_q50"].mean() > 6
    q = pred[[f"min_q{int(p * 100):02d}" for p in QUANTILES]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    gbm = evaluate(pred, wing).set_index("target").loc["min", "median_abs_error"]
    rolling = evaluate(RollingAverage().fit(rows).predict(wing), wing).set_index("target").loc["min", "median_abs_error"]
    assert gbm < rolling


def test_gbm_skips_empty_features_and_round_trips(tmp_path):
    pg, games = season()
    rows = make_features(pg, games)
    model = GBMForecaster(max_iter=20).fit(rows)
    assert "started_avg5" not in model.features and "teammates_out_min" in model.features
    restored = load("gbm", save(model, tmp_path / "gbm.pkl"))
    pd.testing.assert_frame_equal(model.predict(rows), restored.predict(rows))


def test_prob_at_least():
    q = [10, 13, 15, 18, 22]
    probs = [prob_at_least(q, t) for t in range(0, 40)]
    assert all(0 <= p <= 1 for p in probs) and all(a >= b for a, b in zip(probs, probs[1:]))
    assert prob_at_least(q, 15.5) == pytest.approx(0.5)
    assert prob_at_least(q, 0) == 1.0 and prob_at_least(q, 40) == 0.0


def test_evaluate_and_forecast_json():
    rows = pd.DataFrame({"pts": [10.0, 30.0], "min": [20.0, 20.0], "game_id": ["g1", "g1"], "player_id": [1, 2]})
    pred = pd.DataFrame({f"{t}_q{int(p * 100):02d}": [v, v] for t in ("pts", "min")
                         for p, v in zip(QUANTILES, [8, 10, 12, 14, 16])})
    scores = evaluate(pred, rows).set_index("target")
    assert scores.loc["pts", "coverage_10_90"] == 0.5 and scores.loc["min", "coverage_10_90"] == 0.0
    f = forecast_json(rows.iloc[0], pred.iloc[0], "gbm", "pts", lines=[12.5], out=[7])
    assert f["quantiles"] == {"p10": 8, "p25": 10, "p50": 12, "p75": 14, "p90": 16}
    assert set(f) == {"game_id", "player_id", "as_of", "model", "target", "quantiles", "p_over", "p_play", "overrides"}
    assert 0.4 < f["p_over"]["12.5"] < 0.6 and f["overrides"] == {"out": [7]}
