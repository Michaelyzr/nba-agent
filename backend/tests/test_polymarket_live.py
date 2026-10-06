"""Unified live Polymarket provider and AsOf bridge tests; no network."""
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from agents.graph import MarketAgent
from data_sources.polymarket_live import PolymarketLiveProvider
from evaluation.fixtures import make_tables
from tests.test_polymarket_adapter import NOW, market, snapshot


class Client:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def fetch_live_nba(self):
        self.calls += 1
        return self.result


def test_refresh_and_for_game_use_current_api_snapshot(tmp_path):
    client = Client(snapshot([market()]))
    context = PolymarketLiveProvider(client, tmp_path / "history.parquet").refresh()
    game = SimpleNamespace(game_id="local", date="2026-10-06", home_team="GSW", away_team="LAL",
                           tip_time=pd.Timestamp("2026-10-07T02:00:00Z"))
    current = context.for_game(game)
    assert client.calls == 1
    assert current["is_live"] and current["usable_for_future_analysis"]
    assert current["fetched_at"] == NOW.isoformat()
    assert {row["token_id"] for row in current["markets"]} == {"lal-token", "gsw-token"}
    assert {row["ask"] for row in current["markets"]} == {0.40, 0.61}


def test_game_match_requires_the_same_date(tmp_path):
    context = PolymarketLiveProvider(Client(snapshot([market()])), tmp_path / "none.parquet").refresh()
    wrong_day = SimpleNamespace(game_id="local", date="2026-10-08", home_team="GSW", away_team="LAL",
                                tip_time=pd.Timestamp("2026-10-09T02:00:00Z"))
    result = context.for_game(wrong_day)
    assert not result["usable_for_future_analysis"]
    assert result["markets"] == []
    assert "matched this game" in result["reasons"][-1]


def test_live_view_implements_existing_market_agent_boundary(tmp_path):
    context = PolymarketLiveProvider(Client(snapshot([market()])), tmp_path / "none.parquet").refresh()
    view = context.view(make_tables())
    game = next(view.games()[view.games().game_id == "polymarket:e1"].itertuples(index=False))
    markets = view.markets(game.game_id)
    assert set(markets.team) == {"LAL", "GSW"}
    assert view.quote("PM-m1-LAL").ask == 0.40
    assert view.polymarket_quality.usable_for_future_analysis


def test_existing_market_agent_decide_reads_live_polymarket_quotes(tmp_path):
    context = PolymarketLiveProvider(Client(snapshot([market()])), tmp_path / "none.parquet").refresh()
    view = context.view(make_tables())
    game = next(view.games()[view.games().game_id == "polymarket:e1"].itertuples(index=False))
    result = MarketAgent(learn=False).decide(view, game, context.now)
    assert {row["ticker"] for row in result["candidates"]} == {"PM-m1-LAL", "PM-m1-GSW"}
    quotes = {row["ticker"]: (row["bid"], row["ask"]) for row in result["candidates"]}
    assert quotes == {"PM-m1-LAL": (0.39, 0.40), "PM-m1-GSW": (0.60, 0.61)}
    assert all(not ticker.startswith("KX") for ticker in quotes)


def test_history_extends_chart_but_current_refresh_wins_quote(tmp_path):
    path = tmp_path / "history.parquet"
    old = snapshot([replace(market(), fetched_at=NOW - pd.Timedelta(minutes=1))])
    old_market = replace(
        old.markets[0],
        outcomes=(replace(old.markets[0].outcomes[0], best_bid=0.20, best_ask=0.21),
                  replace(old.markets[0].outcomes[1], best_bid=0.79, best_ask=0.80)),
    )
    from data_sources.polymarket import LiveNBASnapshot, write_snapshot_history
    write_snapshot_history(LiveNBASnapshot((old_market,), old.quality, "old"), path)
    context = PolymarketLiveProvider(Client(snapshot([market()])), path).refresh()
    prices = context.price_history()
    assert len(prices[prices.market_ticker == "PM-m1-LAL"]) == 2
    view = context.view(make_tables())
    assert view.quote("PM-m1-LAL").ask == 0.40


def test_failed_refresh_never_reuses_a_previous_or_cached_quote(tmp_path):
    good_client = Client(snapshot([market()]))
    provider = PolymarketLiveProvider(good_client, tmp_path / "history.parquet")
    assert provider.refresh(save_history=True).usable_for_live_analysis
    provider.client = Client(snapshot([market()], usable=False))
    failed = provider.refresh()
    assert not failed.usable_for_live_analysis
    assert failed.price_history().empty
    assert failed.for_game(SimpleNamespace(game_id="polymarket:e1"))["markets"] == []
