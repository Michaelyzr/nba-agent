"""League on the tiny slate: no look-ahead, fills like paper_trade, CLV ranking and badges, bots, caps, storage."""
import pandas as pd
import pytest

from agents import league
from agents.coach import paper_trade
from agents.graph import MarketAgent, record_forecaster
from evaluation.ablations import plain_policy
from replay import Replay
from tests.test_agent_graph import NEWS_AT, TIP2, make_tables
from tests.test_coach import FakeForecaster


@pytest.fixture
def rp():
    return Replay(make_tables(), lambda *a: [])


def at(when):
    return [{"game_id": "g2", "as_of": pd.Timestamp(when).isoformat()}]


def joined(rp, when=NEWS_AT, **kw):
    return league.join(league.create("test-league", rp, slate=at(when), **kw), "alice")


def info(rp, lg, idx=0):
    return league.decision_info(FakeForecaster(), rp, lg, idx, {7: "Player Seven"})


def test_seeded_slate_uses_real_decision_points(rp):
    a = league.choose_slate(rp, "test", 5, seed=league.seed_for("x"))
    assert a == league.choose_slate(rp, "test", 5, seed=league.seed_for("x"))
    assert [s["game_id"] for s in a] == ["g2"]                       # g1 has no market
    assert pd.Timestamp(a[0]["as_of"]) == NEWS_AT                   # news time preferred over tip - 1 h
    quiet = Replay(make_tables(with_news=False), lambda *a: [])
    assert pd.Timestamp(league.choose_slate(quiet, "test", 5)[0]["as_of"]) == TIP2 - pd.Timedelta(hours=1)
    assert league.create("x", rp)["slate"] == a
    assert league.choose_slate(rp, "holdout", 5) == []


def test_decision_info_has_no_look_ahead(rp):
    early = NEWS_AT - pd.Timedelta(minutes=30)
    i = info(rp, joined(rp, early))
    assert i["news"].empty and i["snapshot"]["news"] == [] and i["snapshot"]["shift"] == 0
    assert all(p.ts.max() <= early for p in i["prices"].values())
    assert pd.isna(i["game"].home_pts) and pd.isna(i["game"].away_pts)      # result hidden before tip
    later = info(rp, joined(rp, NEWS_AT))
    assert later["news"].news_id.tolist() == ["n_out"] and later["snapshot"]["out"] == ["Player Seven"]
    assert all(p.ts.max() == NEWS_AT for p in later["prices"].values())


def test_pick_fills_and_settles_like_paper_trade(rp):
    lg = joined(rp)
    s = info(rp, lg)["snapshot"]
    t = league.pick(lg, "alice", 0, "away", 20, rp, s)
    game = next(g for g in rp.t["games"].itertuples() if g.game_id == "g2")
    expected = league._clean(paper_trade(rp, game, NEWS_AT, s["sides"]["AWY"], 20, s))
    assert {k: v for k, v in t.items() if k not in ("index", "choice")} == expected
    assert t["price"] == pytest.approx(0.40) and t["contracts"] == 50 and t["clv"] == pytest.approx(-0.01)
    assert t["pnl"] == pytest.approx(-20 - t["fee"]) and t["won"] is False
    assert league.balance(lg, "alice") == pytest.approx(league.BANKROLL + t["pnl"])
    assert league.next_index(lg, "alice") is None


def test_pass_is_recorded_and_counts_toward_pass_rate(rp):
    lg = joined(rp)
    league.pick(lg, "alice", 0, "pass", 0, rp, info(rp, lg)["snapshot"])
    s = league.summary(lg["players"]["alice"]["picks"], rp.t["games"])
    assert s["decisions"] == 1 and s["trades"] == 0 and s["pass_rate"] == 1.0 and s["pnl"] == 0


def test_stake_caps_and_order_are_enforced(rp):
    lg = joined(rp)
    s = info(rp, lg)["snapshot"]
    for stake in (league.STAKE_MIN - 1, league.STAKE_MAX + 1):
        with pytest.raises(ValueError, match="stake"):
            league.pick(lg, "alice", 0, "home", stake, rp, s)
    with pytest.raises(ValueError, match="choice"):
        league.pick(lg, "alice", 0, "both", 20, rp, s)
    poor = joined(rp, bankroll=10.0)
    with pytest.raises(ValueError, match="stake"):
        league.pick(poor, "alice", 0, "home", 20, rp, s)
    league.pick(lg, "alice", 0, "home", 20, rp, s)
    with pytest.raises(ValueError, match="in order"):
        league.pick(lg, "alice", 0, "home", 20, rp, s)
    with pytest.raises(ValueError):
        league.join(lg, "Agent")
    with pytest.raises(ValueError):
        league.join(lg, "../etc")


def test_bots_play_the_same_slate_and_appear_on_the_leaderboard(rp):
    lg = joined(rp)
    league.add_bots(lg, rp, MarketAgent(learn=False).policy, plain_policy(record_forecaster))
    assert set(lg["bots"]) == set(league.BOTS)
    assert lg["bots"]["Never trade"]["picks"]["0"]["choice"] == "pass"
    agent = lg["bots"]["Agent"]["picks"]["0"]
    assert agent["filled"] and agent["team"] == "AWY" and agent["as_of"] == NEWS_AT.isoformat()
    league.pick(lg, "alice", 0, "home", 20, rp, info(rp, lg)["snapshot"])
    board = league.leaderboard(lg, rp.t["games"])
    assert set(board.name) == {"alice", *league.BOTS}
    assert board.set_index("name").loc["Never trade", "pnl"] == 0
    cmp = league.compare(lg, "alice", rp.t["games"])
    assert cmp.name.tolist()[0] == "alice" and len(cmp) == 4


def synthetic(clvs, day0=0):
    return {str(i): {"filled": True, "choice": "home", "game_id": f"s{day0 + i}", "clv": c, "contracts": 40,
                     "clv_dollars": c * 40, "pnl": 40 * c - 0.5, "fee": 0.5, "price": 0.5}
            for i, c in enumerate(clvs)}


def test_leaderboard_ranks_by_clv_with_minimum_trades_and_badges():
    games = pd.DataFrame({"game_id": [f"s{i}" for i in range(30)],
                          "date": [f"2026-02-{i + 1:02d}" if i < 28 else f"2026-03-0{i - 27}" for i in range(30)]})
    lg = {"players": {"sharp": {"picks": synthetic([0.03, 0.02, 0.04, 0.03, 0.025, 0.035, 0.03, 0.02])},
                      "fish": {"picks": synthetic([-0.03, -0.02, -0.04, -0.03, -0.025, -0.035], 10)},
                      "coin": {"picks": synthetic([0.05, -0.05, 0.04, -0.04, 0.01, -0.02], 20)},
                      "newbie": {"picks": synthetic([0.20, 0.10], 27)}},
          "bots": {"Never trade": {"picks": {"0": {"filled": False, "choice": "pass"}}}}}
    board = league.leaderboard(lg, games).set_index("name")
    assert board.index.tolist() == ["sharp", "coin", "fish", "newbie", "Never trade"]
    assert board.loc["sharp", "badge"] == "skill" and board.loc["fish", "badge"] == "costs"
    assert board.loc["coin", "badge"] == "too early to tell"
    assert board.loc["newbie", "badge"] == "too early to tell" and board.loc["newbie", "rank"] == "-"
    assert board.loc["sharp", "rank"] == "1" and board.loc["sharp", "beat_close"] == 1.0
    assert board.loc["sharp", "fees"] == pytest.approx(4.0) and board.loc["Never trade", "badge"] == "no trades"


def test_league_round_trips_through_json(rp, tmp_path):
    lg = joined(rp)
    league.add_bots(lg, rp, MarketAgent(learn=False).policy)
    league.pick(lg, "alice", 0, "away", 20, rp, info(rp, lg)["snapshot"])
    p = league.save(lg, tmp_path)
    assert p.name == "test-league.json" and league.list_leagues(tmp_path) == ["test-league"]
    back = league.load("test-league", tmp_path)
    assert back == lg and "Play money only" in back["notice"]
    assert league.leaderboard(back, rp.t["games"]).equals(league.leaderboard(lg, rp.t["games"]))
