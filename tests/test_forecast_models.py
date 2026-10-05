"""History index, M3 play classifier, M4 win model, M2 GRU, forecast API and channel briefs on a small fake league."""
import numpy as np
import pandas as pd
import pytest

from forecast.history import History

T0 = pd.Timestamp("2026-01-01T00:00Z")


def league(n_games=60, star_out_from=40, seed=0):
    """Two teams; team 1's star (player 10, 36 minutes, 30 points) misses every game from star_out_from on.

    Team 1 wins by 10 with the star and loses by 6 without him.
    """
    rng = np.random.default_rng(seed)
    games, pg = [], []
    for i in range(n_games):
        gid, tip = f"g{i:03d}", T0 + pd.Timedelta(days=2 * i)
        home, away = (1, 2) if i % 2 == 0 else (2, 1)
        star = i < star_out_from
        margin1 = (10 if star else -6) + rng.normal(0, 2)
        pts1, pts2 = 105 + max(margin1, -40) / 2, 105 - margin1 / 2
        games.append({"game_id": gid, "date": str(tip.date()), "tip_time": tip, "final_at": tip + pd.Timedelta(hours=3),
                      "home_team_id": home, "away_team_id": away, "home_team": f"T{home}", "away_team": f"T{away}",
                      "home_pts": round(pts1 if home == 1 else pts2), "away_pts": round(pts2 if home == 1 else pts1)})
        for team, roster in ((1, [10, 11, 12, 13, 14]), (2, [20, 21, 22, 23, 24])):
            for p in roster:
                if p == 10 and not star:
                    continue
                minutes = 36.0 if p == 10 else (30.0 if star or team == 2 else 37.0)
                pg.append({"game_id": gid, "player_id": p, "team_id": team, "min": minutes + rng.normal(0, 1),
                           "pts": 30 if p == 10 else 12 + int(rng.integers(0, 6)), "fga": 15, "fta": 4,
                           "usage": np.nan, "started": True})
    return pd.DataFrame(pg), pd.DataFrame(games)


def test_history_only_sees_earlier_games():
    pg, games = league()
    h = History(pg, games)
    tip = games.tip_time.iloc[10]
    assert len(h.player_rows(10, tip)) == 10
    assert h.team_games(1, tip) == list(games.game_id.iloc[:10])
    assert set(h.rotation(1, tip)) == {10, 11, 12, 13, 14}
    later = games.tip_time.iloc[50]                       # star has not played for 10 games (20 days)
    assert 10 not in h.rotation(1, later)


def test_play_model_learns_that_a_missing_streak_predicts_absence():
    from forecast.play import PlayModel, roster_rows, training_rows
    pg, games = league()
    h = History(pg, games)
    rows = training_rows(h, games)
    model = PlayModel().fit(rows)
    tip = games.tip_time.iloc[43]                         # star missed the last three games
    r = roster_rows(h, 1, tip)
    p = dict(zip(r.player_id, model.predict(r)))
    assert p[10] < 0.5 < p[11]


def test_win_model_moves_when_the_star_is_ruled_out():
    from forecast.win import WinModel, training_rows
    pg, games = league(star_out_from=30)
    pg = pg[~((pg.player_id == 10) & pg.game_id.isin([f"g{i:03d}" for i in range(0, 30, 3)]))]   # star rests sometimes
    h = History(pg, games)
    model = WinModel().fit(training_rows(h, games))
    game = next(games.iloc[[25]].itertuples())
    with_star, without = model.p_home(h, game), model.p_home(h, game, out=[10])
    if game.home_team_id != 1:
        with_star, without = 1 - with_star, 1 - without
    assert without < with_star


def test_gru_outputs_monotone_quantiles():
    from forecast.baselines import _qcols
    from forecast.features import make_features
    from forecast.gru import GRUForecaster
    pg, games = league()
    h = History(pg, games)
    rows = make_features(pg, games)
    model = GRUForecaster(epochs=2, batch=64).fit(rows, h, verbose=False)
    pred = model.predict(rows.tail(20), h)
    for t in ("pts", "min"):
        q = pred[_qcols(t)].to_numpy()
        assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()


@pytest.fixture
def forecaster():
    from forecast.api import Forecaster
    from forecast.baselines import GBMForecaster
    from forecast.features import make_features
    from forecast.play import PlayModel, training_rows as play_rows
    from forecast.win import WinModel, training_rows as win_rows
    pg, games = league(star_out_from=30)
    pg = pg[~((pg.player_id == 10) & pg.game_id.isin([f"g{i:03d}" for i in range(0, 30, 3)]))]
    h = History(pg, games)
    f = Forecaster(WinModel().fit(win_rows(h, games)), PlayModel().fit(play_rows(h, games)),
                   GBMForecaster(max_iter=20, min_samples_leaf=5).fit(make_features(pg, games)))
    return f, pg, games


def view(pg, games, now):
    from replay import AsOf
    tables = {"games": games, "player_games": pg,
              "news": pd.DataFrame(columns=["news_id", "published_at", "game_id", "player_id", "status", "text"])}
    return AsOf(tables, now, {})


def test_api_reads_only_the_as_of_view_and_formats_section5(forecaster):
    f, pg, games = forecaster
    game = next(games.iloc[[20]].itertuples())
    v = view(pg, games, game.tip_time - pd.Timedelta(hours=1))
    w = f.win(v, game, out=[10])
    assert 0.02 <= w["p_home"] <= 0.98 and w["overrides"] == {"out": [10]}
    assert w["missing"]["home" if game.home_team_id == 1 else "away"] == [10]
    full = f.win(v, game)["p_home"]
    half = f.win(v, game, availability={10: 0.5})
    assert min(full, w["p_home"]) <= half["p_home"] <= max(full, w["p_home"])
    assert half["overrides"]["availability"] == {10: 0.5}
    fc = f.player(v, game, 11, "pts", lines=[14.5], out=[10])
    assert set(fc) >= {"game_id", "player_id", "as_of", "model", "target", "quantiles", "p_over", "p_play", "overrides"}
    assert fc["quantiles"]["p10"] <= fc["quantiles"]["p50"] <= fc["quantiles"]["p90"]
    assert f.player(v, game, 10, "pts", lines=[20.5], out=[10])["p_play"] == 0.02
    assert len(f.history(v).team_games(1, game.tip_time)) == 20            # nothing after the decision


def test_channel_briefs_keep_market_language_out_of_media_and_team(forecaster):
    from agents.briefs import MARKET_WORDS, channel_briefs, outlook
    f, pg, games = forecaster
    game = next(games.iloc[[20]].itertuples())
    v = view(pg, games, game.tip_time - pd.Timedelta(hours=1))
    news = pd.DataFrame({"news_id": ["n1"], "published_at": [game.tip_time - pd.Timedelta(hours=2)],
                         "game_id": [game.game_id], "player_id": [10], "status": ["out"], "text": ["Star out: knee"]})
    b = channel_briefs(game, outlook(f, v, game, [10]), news, {10: "Star"}, {"bid": 0.55, "ask": 0.57},
                       "Buying no at 45% for a 60% chance.")
    assert not MARKET_WORDS.search(b["media"]) and not MARKET_WORDS.search(b["team"])
    from agents.graph import BANNED
    assert "Confirm" in b["retail"] and not BANNED.search(b["retail"])
    assert "Star" in b["media"]


def test_scorer_policy_tests_all_blocked():
    from evaluation.scorer import policy_tests
    assert policy_tests().blocked.all()
