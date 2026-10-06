"""Live Polymarket normalization and failure handling, with no network."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import requests
import pytest

from data_sources.polymarket import (LiveNBASnapshot, PolymarketClient, is_nba_event,
                                     normalize_market, snapshot_frame, validate_data_quality,
                                     write_snapshot_history)

NOW = datetime(2026, 10, 6, 6, 30, tzinfo=timezone.utc)


def raw_market(market_id="m1", tokens=("token-a", "token-b"), prices=("0.41", "0.59"), **overrides):
    row = {
        "id": market_id,
        "slug": f"market-{market_id}",
        "question": "Lakers vs. Warriors",
        "sportsMarketType": "moneyline",
        "gameStartTime": "2026-10-07T02:00:00Z",
        "endDate": "2026-10-07T02:00:00Z",
        "outcomes": '["Lakers", "Warriors"]',
        "outcomePrices": list(prices),
        "clobTokenIds": list(tokens),
        "liquidityNum": 1250.5,
        "volumeNum": 9000,
        "volume24hr": 350,
        "active": True,
        "closed": False,
        "acceptingOrders": True,
    }
    row.update(overrides)
    return row


def raw_event(markets=None, **overrides):
    row = {
        "id": "event-1",
        "slug": "nba-lal-gsw-2026-10-06",
        "title": "Lakers vs. Warriors",
        "active": True,
        "closed": False,
        "tags": [{"id": "745", "slug": "nba"}],
        "markets": markets if markets is not None else [raw_market()],
    }
    row.update(overrides)
    return row


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, events=None, fail=False):
        self.events = [] if events is None else events
        self.fail = fail
        self.posts = []

    def get(self, url, params=None, timeout=None):
        if self.fail:
            raise requests.Timeout("offline")
        if url.endswith("/sports"):
            return FakeResponse([{"sport": "nba", "name": "NBA", "series": "10345", "primaryTagId": 745}])
        if url.endswith("/events"):
            return FakeResponse(self.events if (params or {}).get("offset", 0) == 0 else [])
        raise AssertionError(url)

    def post(self, url, json=None, timeout=None):
        if self.fail:
            raise requests.Timeout("offline")
        self.posts.append((url, json))
        tokens = list(dict.fromkeys(row["token_id"] for row in json))
        if url.endswith("/prices"):
            return FakeResponse({token: {"BUY": "0.40", "SELL": "0.42"} for token in tokens})
        if url.endswith("/midpoints"):
            return FakeResponse({token: "0.41" for token in tokens})
        if url.endswith("/last-trades-prices"):
            return FakeResponse([{"token_id": token, "price": "0.405", "side": "BUY"} for token in tokens])
        raise AssertionError(url)


def client(session):
    return PolymarketClient(session=session, now=lambda: NOW)


def test_nba_event_filter_requires_exact_metadata_or_strict_two_team_slug():
    assert is_nba_event(raw_event())
    assert is_nba_event({"slug": "nba-bos-nyk-2026-10-06", "tags": []})
    assert not is_nba_event({"slug": "wnba-nyl-lva-2026-10-06", "tags": [{"slug": "basketball"}]})
    assert not is_nba_event({"slug": "college-basketball-final", "tags": [{"slug": "basketball"}]})


def test_market_normalization_preserves_generic_outcomes_and_token_quotes():
    market = raw_market(
        outcomes='["Over", "Under"]',
        outcomePrices='["0.53", "0.47"]',
        clobTokenIds='["over-token", "under-token"]',
        sportsMarketType="totals",
    )
    quotes = {
        "over-token": {"best_bid": "0.51", "best_ask": "0.55", "midpoint": "0.53",
                       "last_trade_price": "0.52"},
        "under-token": {"best_bid": "0.45", "best_ask": "0.49", "midpoint": "0.47",
                        "last_trade_price": "0.48"},
    }
    normalized = normalize_market(raw_event(), market, NOW, quotes)
    assert normalized.market_type == "totals"
    assert [(o.outcome, o.token_id, o.price) for o in normalized.outcomes] == [
        ("Over", "over-token", 0.53), ("Under", "under-token", 0.47)]
    assert normalized.outcomes[0].spread == pytest.approx(0.04)
    assert normalized.polymarket_url == "https://polymarket.com/event/nba-lal-gsw-2026-10-06"


def test_outcome_token_parsing_keeps_missing_values_visible_to_validation():
    market = raw_market(outcomes='["Lakers", "Warriors"]', clobTokenIds='["only-one"]',
                        outcomePrices='["0.4", "0.6"]')
    normalized = normalize_market(raw_event(), market, NOW)
    assert len(normalized.outcomes) == 2
    assert normalized.outcomes[1].outcome == "Warriors"
    assert normalized.outcomes[1].token_id is None
    quality = validate_data_quality([normalized], NOW, True, now=NOW)
    assert not quality.usable_for_future_analysis
    assert quality.source_state == "SOURCE DATA INVALID"


def test_invalid_and_missing_prices_are_not_usable():
    invalid = normalize_market(raw_event(), raw_market(prices=("1.2", None)), NOW)
    quality = validate_data_quality([invalid], NOW, True, now=NOW)
    assert quality.status == "WARNING"
    assert quality.source_state == "SOURCE DATA INVALID"
    assert not quality.usable_for_future_analysis
    assert any("invalid outcome price" in reason for reason in quality.reasons)


def test_empty_nba_market_list_is_truthful_and_not_demo_data():
    snapshot = client(FakeSession(events=[])).fetch_live_nba()
    assert snapshot.markets == ()
    assert snapshot.quality.status == "WARNING"
    assert not snapshot.quality.usable_for_future_analysis
    assert snapshot.message == "No active NBA markets are currently available."


def test_active_event_with_no_open_markets_is_also_an_empty_result():
    snapshot = client(FakeSession(events=[raw_event(markets=[])])).fetch_live_nba()
    assert snapshot.markets == ()
    assert snapshot.quality.status == "WARNING"
    assert snapshot.message == "No active NBA markets are currently available."


def test_api_failure_returns_unavailable_result_instead_of_raising():
    snapshot = client(FakeSession(fail=True)).fetch_live_nba()
    assert snapshot.markets == ()
    assert snapshot.quality.status == "UNAVAILABLE"
    assert snapshot.quality.source_state == "SOURCE DATA UNAVAILABLE"
    assert snapshot.message == "Polymarket data temporarily unavailable."


def test_quality_rejects_stale_and_duplicate_outcome_tokens():
    market = normalize_market(raw_event(), raw_market(), NOW, {
        "token-a": {"best_bid": 0.40, "best_ask": 0.42},
        "token-b": {"best_bid": 0.58, "best_ask": 0.60},
    })
    duplicate = replace(market, market_id="m2")
    quality = validate_data_quality(
        [market, duplicate], NOW, True, now=NOW + timedelta(minutes=5), stale_after_seconds=90,
    )
    assert quality.status == "WARNING"
    assert quality.source_state == "SOURCE DATA INVALID"
    assert not quality.usable_for_future_analysis
    assert any("Duplicate" in reason for reason in quality.reasons)
    assert any("stale" in reason for reason in quality.reasons)


def test_multiple_markets_use_batched_clob_pricing_and_are_reusable():
    session = FakeSession(events=[raw_event(markets=[
        raw_market("m1", ("token-a", "token-b")),
        raw_market("m2", ("token-c", "token-d"), question="Spread: Warriors (-3.5)",
                   sportsMarketType="spreads"),
    ])])
    snapshot = client(session).fetch_live_nba()
    assert len(snapshot.markets) == 2
    assert snapshot.quality.status == "OK"
    assert snapshot.quality.usable_for_future_analysis
    assert {row.token_id for market in snapshot.markets for row in market.outcomes} == {
        "token-a", "token-b", "token-c", "token-d"}
    assert len(session.posts) == 3


def test_snapshot_history_is_flattened_and_deduplicated(tmp_path):
    session = FakeSession(events=[raw_event()])
    snapshot = client(session).fetch_live_nba()
    path = tmp_path / "snapshots.parquet"
    write_snapshot_history(snapshot, path)
    write_snapshot_history(snapshot, path)
    frame = snapshot_frame(LiveNBASnapshot(snapshot.markets, snapshot.quality, snapshot.message))
    assert len(frame) == 2
    import pandas as pd
    assert len(pd.read_parquet(path)) == 2
