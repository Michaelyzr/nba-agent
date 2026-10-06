"""Read-only Polymarket-to-agent table adapter tests; no network."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from data_sources import SCHEMAS
from data_sources.polymarket import (DataQualityStatus, LiveNBASnapshot, MarketSnapshot,
                                     OutcomeSnapshot)
from data_sources.polymarket_adapter import adapt_live_nba_snapshot


NOW = datetime(2026, 10, 6, 7, 30, tzinfo=timezone.utc)


def outcome(name, token, bid, ask):
    return OutcomeSnapshot(name, token, (bid + ask) / 2, bid, ask, (bid + ask) / 2, bid, ask - bid)


def market(event_id="e1", slug="nba-lal-gsw-2026-10-06", market_id="m1", market_type="moneyline",
           outcomes=None):
    return MarketSnapshot(
        fetched_at=NOW, source="polymarket", event_id=event_id, event_slug=slug,
        event_title="Lakers vs. Warriors", market_id=market_id, market_slug=f"market-{market_id}",
        question="Lakers vs. Warriors", market_type=market_type,
        market_start_time=datetime(2026, 10, 7, 2, tzinfo=timezone.utc), market_end_time=None,
        outcomes=tuple(outcomes or [outcome("Lakers", "lal-token", 0.39, 0.40),
                                    outcome("Warriors", "gsw-token", 0.60, 0.61)]),
        liquidity=1000.0, total_volume=9000.0, volume_24h=300.0,
        active=True, closed=False, accepting_orders=True,
        polymarket_url="https://polymarket.com/event/nba-lal-gsw-2026-10-06",
    )


def snapshot(markets=(), usable=True):
    quality = DataQualityStatus(
        "OK" if usable else "WARNING",
        "SOURCE DATA AVAILABLE" if usable else "SOURCE DATA INVALID",
        NOW, 0.0, (), usable,
    )
    return LiveNBASnapshot(tuple(markets), quality, "test")


def test_moneyline_maps_to_existing_agent_table_contract():
    adapted = adapt_live_nba_snapshot(snapshot([market()]))
    assert adapted.usable_for_live_analysis
    assert set(SCHEMAS["games"]) <= set(adapted.games)
    assert set(SCHEMAS["markets"]) <= set(adapted.markets)
    assert set(SCHEMAS["prices"]) <= set(adapted.prices)
    game = adapted.games.iloc[0]
    assert (game.away_team, game.home_team) == ("LAL", "GSW")
    assert game.game_id == "polymarket:e1"
    assert set(adapted.markets.team) == {"LAL", "GSW"}
    assert set(adapted.markets.token_id) == {"lal-token", "gsw-token"}
    assert adapted.prices.set_index("market_ticker").bid.to_dict() == {
        "PM-m1-LAL": 0.39, "PM-m1-GSW": 0.60,
    }


def test_adapter_never_treats_cumulative_volume_as_replay_interval_volume():
    adapted = adapt_live_nba_snapshot(snapshot([market()]))
    assert adapted.prices.volume.isna().all()
    assert set(adapted.prices.total_volume) == {9000.0}
    assert set(adapted.prices.volume_24h) == {300.0}


def test_non_moneyline_markets_stay_out_of_game_winner_agent_input():
    adapted = adapt_live_nba_snapshot(snapshot([
        market(),
        market(market_id="m2", market_type="totals",
               outcomes=[outcome("Over", "over", 0.49, 0.51), outcome("Under", "under", 0.49, 0.51)]),
    ]))
    assert adapted.usable_for_live_analysis
    assert adapted.skipped_markets == 1
    assert set(adapted.markets.market_id) == {"m1"}


def test_quality_gate_returns_empty_tables_even_when_markets_are_present():
    adapted = adapt_live_nba_snapshot(snapshot([market()], usable=False))
    assert not adapted.usable_for_live_analysis
    assert adapted.games.empty and adapted.markets.empty and adapted.prices.empty
    assert "quality gate" in adapted.reasons[0]


def test_unmatched_or_incomplete_outcomes_reject_the_whole_moneyline():
    bad_team = market(outcomes=[outcome("Lakers", "lal", 0.39, 0.40),
                                outcome("Celtics", "bos", 0.60, 0.61)])
    missing_quote = market(market_id="m2", outcomes=[outcome("Lakers", "lal2", 0.39, 0.40),
                                                      outcome("Warriors", "gsw2", 0.60, 0.61)])
    missing_quote = replace(
        missing_quote,
        outcomes=(missing_quote.outcomes[0], replace(missing_quote.outcomes[1], best_ask=None)),
    )
    adapted = adapt_live_nba_snapshot(snapshot([bad_team, missing_quote]))
    assert not adapted.usable_for_live_analysis
    assert adapted.markets.empty
    assert adapted.skipped_markets == 2
    assert any("could not map both team outcomes" in reason for reason in adapted.reasons)


def test_started_market_is_never_sent_to_the_pregame_agent():
    started = replace(market(), market_start_time=NOW - timedelta(minutes=1))
    adapted = adapt_live_nba_snapshot(snapshot([started]))
    assert not adapted.usable_for_live_analysis
    assert adapted.markets.empty
    assert any("already started" in reason for reason in adapted.reasons)


def test_multiple_games_have_stable_event_and_token_identity():
    second = market(
        event_id="e2", slug="nba-bos-nyk-2026-10-08", market_id="m2",
        outcomes=[outcome("Celtics", "bos-token", 0.51, 0.52),
                  outcome("Knicks", "nyk-token", 0.48, 0.49)],
    )
    adapted = adapt_live_nba_snapshot(snapshot([market(), second]))
    assert adapted.game_count == 2
    assert adapted.quote_count == 4
    assert set(adapted.games.game_id) == {"polymarket:e1", "polymarket:e2"}
    assert set(adapted.markets.token_id) == {"lal-token", "gsw-token", "bos-token", "nyk-token"}
    assert isinstance(adapted.prices.ts.dtype, pd.DatetimeTZDtype)
