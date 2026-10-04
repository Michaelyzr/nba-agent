"""Replay, fills, settlement and leakage on a two-game synthetic slate (no network)."""
import pandas as pd
import pytest

import replay
from replay import AsOf, Order, Replay, cited_after_decision, kalshi_fee, summary

TIP1 = pd.Timestamp("2026-02-01T00:30:00Z")
TIP2 = pd.Timestamp("2026-02-02T00:30:00Z")
TICKER = "KXNBAGAME-26FEB01AWYHOM-HOM"


def make_tables(volume=1000.0):
    games = pd.DataFrame({
        "game_id": ["g1", "g2"], "date": ["2026-01-31", "2026-02-01"], "tip_time": [TIP1, TIP2],
        "final_at": [TIP1 + pd.Timedelta(hours=3), TIP2 + pd.Timedelta(hours=3)],
        "home_team_id": [1, 1], "away_team_id": [2, 2], "home_team": ["HOM", "HOM"],
        "away_team": ["AWY", "AWY"], "home_pts": [110, 101], "away_pts": [100, 99]})
    player_games = pd.DataFrame({"game_id": ["g1", "g2"], "player_id": [7, 7], "team_id": [1, 1],
                                 "min": [34.0, 30.0], "pts": [25, 18], "fga": [18, 15], "fta": [6, 4],
                                 "usage": [0.3, 0.28], "started": [True, True]})
    news = pd.DataFrame({
        "news_id": ["n_early", "n_future"], "published_at": [TIP2 - pd.Timedelta(hours=2), TIP2 + pd.Timedelta(hours=1)],
        "game_id": ["g2", "g2"], "player_id": [9, 9], "status": ["questionable", "out"],
        "source": ["test", "test"], "url": ["", ""], "text": ["listed questionable", "listed out"]})
    markets = pd.DataFrame({"venue": ["kalshi"], "market_ticker": [TICKER], "kind": ["game"], "game_id": ["g2"],
                            "team": ["HOM"], "player_id": [None], "line": [None], "title": ["test"]})
    ts = pd.date_range(TIP2 - pd.Timedelta(hours=3), TIP2 + pd.Timedelta(hours=1), freq="1min")
    prices = pd.DataFrame({"venue": "kalshi", "market_ticker": TICKER, "ts": ts,
                           "bid": 0.38, "ask": 0.40, "volume": volume})
    prices.loc[prices.ts > TIP2, ["bid", "ask"]] = [0.90, 0.92]
    settlements = pd.DataFrame({"market_ticker": [TICKER], "settled_at": [TIP2 + pd.Timedelta(hours=3)],
                                "outcome": [1]})
    return {"games": games, "player_games": player_games, "news": news, "markets": markets,
            "prices": prices, "settlements": settlements}


def buy_home(stake=20.0, channel="platform", reason="test edge", citations=()):
    def policy(view, game, now):
        if now != game.tip_time - replay.LEAD:
            return []
        return [Order(TICKER, "yes", 0.6, stake, reason, list(citations), channel)]
    return policy


def test_as_of_hides_the_future():
    t = make_tables()
    r = Replay(t, buy_home())
    view = AsOf(t, TIP2 - pd.Timedelta(minutes=30), r.price_index)
    assert view.news().news_id.tolist() == ["n_early"]
    g = view.games().set_index("game_id")
    assert g.loc["g1", "home_pts"] == 110 and pd.isna(g.loc["g2", "home_pts"])
    assert view.player_games().game_id.tolist() == ["g1"]
    assert view.prices(TICKER).ts.max() <= view.now
    assert view.quote(TICKER).ask == 0.40
    assert not hasattr(view, "settlements")


def test_policy_never_receives_future_news():
    seen = []

    def spy(view, game, now):
        seen.append(view.news().published_at.max())
        return []

    Replay(make_tables(), spy).run("2026-02-01", "2026-02-01")
    assert seen and all(pd.isna(s) or s <= TIP2 for s in seen)


def test_fill_at_ask_with_fee_and_settlement():
    decisions, fills = Replay(make_tables(), buy_home()).run("2026-02-01", "2026-02-01")
    f = fills.iloc[0]
    assert (f.price, f.contracts) == (0.40, 50)
    assert f.fee == kalshi_fee(50, 0.40) == 0.84
    assert f.pnl == pytest.approx(50 * 0.60 - 0.84)
    assert f.clv == pytest.approx(0.39 - 0.40)
    assert decisions.iloc[0].risk_result == "approved"
    s = summary(fills)
    assert s["fills"] == 1 and s["pnl_after_fees"] == pytest.approx(29.16)


def test_size_capped_by_recent_volume():
    _, fills = Replay(make_tables(volume=2.0), buy_home()).run("2026-02-01", "2026-02-01")
    assert fills.iloc[0].contracts == int(60 * 2.0 * replay.VOLUME_SHARE)


@pytest.mark.parametrize("kwargs, expected", [
    ({"stake": 500.0}, "over_order_cap"),
    ({"channel": "team"}, "channel_cannot_order"),
    ({"reason": "  "}, "missing_reason"),
])
def test_risk_blocks_bad_orders(kwargs, expected):
    decisions, fills = Replay(make_tables(), buy_home(**kwargs)).run("2026-02-01", "2026-02-01")
    assert fills.empty
    assert decisions.iloc[0].risk_result == expected and decisions.iloc[0].fill_result == "blocked"


def test_post_tip_and_stale_quotes_do_not_fill():
    t = make_tables()
    r = Replay(t, buy_home())
    order = Order(TICKER, "yes", 0.6, 20, "late")
    assert r._fill(order, TIP2 + pd.Timedelta(minutes=1), TIP2) == (None, "post_tip")
    t["prices"] = t["prices"][t["prices"].ts < TIP2 - pd.Timedelta(hours=2)]
    assert Replay(t, buy_home())._fill(order, TIP2 - replay.LEAD, TIP2) == (None, "stale_quote")


def test_planted_future_citation_is_flagged():
    decisions, _ = Replay(make_tables(), buy_home(citations=["n_early", "n_future"])).run("2026-02-01", "2026-02-01")
    leaks = cited_after_decision(decisions.iloc[0].to_dict(), make_tables()["news"])
    assert leaks == ["n_future"]


def test_record_baseline_runs_end_to_end():
    decisions, fills = Replay(make_tables(), replay.record_baseline()).run("2026-01-31", "2026-02-01")
    assert len(decisions) == len(fills) == 1
    assert fills.iloc[0].side == "yes"
