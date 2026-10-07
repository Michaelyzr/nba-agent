"""Executable-cost math, counterfactual EV and stale/incompatible market blocking."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.market_analysis import compare_routes, walk_asks, synthetic_quotes
from data_sources.polymarket_books import match_market, normalize_book, fee_parameters, PolymarketReader
from test_inplay import NOW, agent, Feed, event


def state_and_quotes(tmp_path):
    s = agent(tmp_path, Feed([event()])).poll(NOW)["snapshot"]
    game = SimpleNamespace(game_id="g2", home_team="BOS", away_team="LAL")
    q = synthetic_quotes(s, game)
    for quote in q.values():
        quote["synthetic"] = False
    s["report"]["synthetic"] = False
    s["model_trained"] = True
    s["calibration_status"] = "calibrated_heldout"
    return s, q


def test_depth_costs_fees_and_unused_cash_are_not_top_of_book_only():
    quote = {"asks": [{"price": .6, "size": 50}, {"price": .4, "size": 10}], "fee_rate": .05, "fee_exponent": 1}
    fill = walk_asks(quote, 100, .002)
    assert fill["shares"] == 60
    assert fill["notional"] == 34
    assert fill["fees"] == pytest.approx(.72)
    assert fill["cost"] == pytest.approx(34.84)
    assert fill["unused_budget"] == pytest.approx(65.16)
    assert fill["effective_price"] > fill["vwap"] > .4
    assert fill["depth_limited"]


def test_negative_edges_wait_and_new_event_ev_uses_the_same_book(tmp_path):
    s, q = state_and_quotes(tmp_path)
    result = compare_routes(s, q, uncertainty_pp=0)
    row = result["routes"][0]
    assert row["ev"] == pytest.approx(row["fill"]["shares"]*s["p_home"]-row["fill"]["cost"])
    assert row["new_news_ev_change"] == pytest.approx(row["fill"]["shares"]*s["report"]["news_effect"]["home_delta_pp"]/100)
    assert result["candidate"] == "away"
    assert compare_routes(s, q, uncertainty_pp=30)["candidate"] == "wait"


@pytest.mark.parametrize("field,value", [
    ("game_id", "wrong"), ("rules_verified", False), ("includes_overtime", False),
    ("kind", "spread"), ("active", False), ("fee_verified", False),
    ("token_id", ""), ("condition_id", ""), ("side", "away"), ("bid", .999),
    ("updated_at", "NaT"),
    ("updated_at", (NOW-pd.Timedelta(seconds=16)).isoformat()),
    ("observed_at", (NOW+pd.Timedelta(seconds=1)).isoformat()),
    ("asks", []), ("asks", [{"price": .5, "size": float('nan')}]),
])
def test_invalid_quotes_never_produce_a_route(tmp_path, field, value):
    s, q = state_and_quotes(tmp_path)
    q["home"][field] = value
    row = compare_routes(s, q)["routes"][0]
    assert row["problems"] and row["ev"] is None and not row["eligible"]


def test_pair_identity_and_development_model_cannot_execute(tmp_path):
    s, q = state_and_quotes(tmp_path)
    q["away"]["condition_id"] = "other"
    assert all(r["problems"] for r in compare_routes(s, q)["routes"])
    q["away"]["condition_id"] = q["home"]["condition_id"]
    s["heldout_validation"] = {"development_only": True}
    assert compare_routes(s, q, uncertainty_pp=0)["decision"] == "wait"
    s["heldout_validation"] = None
    s["report"]["synthetic"] = True
    assert compare_routes(s, q, uncertainty_pp=0)["decision"] == "wait"


def test_final_or_missing_predictions_have_no_trade_ev(tmp_path):
    s, q = state_and_quotes(tmp_path)
    s["quote_state"] = "final"
    assert all(r["ev"] is None for r in compare_routes(s, q)["routes"])


def market():
    return {"sportsMarketType": "moneyline", "outcomes": '["Celtics", "Lakers"]',
            "clobTokenIds": '["boston-token", "lakers-token"]', "conditionId": "condition",
            "gameStartTime": NOW.isoformat(), "feesEnabled": True,
            "feeSchedule": {"rate": .05, "exponent": 1}}


def test_market_matches_token_order_by_team_and_refuses_dates_duplicates_or_props():
    game = SimpleNamespace(home_team="BOS", away_team="LAL", tip_time=NOW)
    m = market()
    found, tokens = match_market({"markets": [m]}, game)
    assert tokens == {"home": "boston-token", "away": "lakers-token"}
    for bad in [{**m, "sportsMarketType": "spread"}, {**m, "gameStartTime": (NOW+pd.Timedelta(days=1)).isoformat()},
                {**m, "outcomes": '["Yes", "No"]'}]:
        with pytest.raises(ValueError):
            match_market({"markets": [bad]}, game)
    with pytest.raises(ValueError):
        match_market({"markets": [m, deepcopy(m)]}, game)


def test_raw_books_ms_timestamp_depth_and_fee_parameters():
    raw = {"asset_id": "token", "market": "condition", "timestamp": str(int(NOW.timestamp()*1000)),
           "bids": [{"price": ".4", "size": "50"}], "asks": [{"price": ".6", "size": "20"}, {"price": ".5", "size": "10"}]}
    q = normalize_book(raw, "token", "condition", NOW)
    assert q["ask"] == .5 and q["bid"] == .4 and q["mid"] == .45
    assert q["updated_at"] == NOW.isoformat()
    with pytest.raises(ValueError):
        normalize_book(raw, "other-token", "condition", NOW)
    assert fee_parameters(market(), {})["fee_rate"] == .05
    assert not fee_parameters({"feesEnabled": True}, {"fd": {"r": .05, "e": 2}})["fee_verified"]
    assert not fee_parameters({"feesEnabled": True}, {})["fee_verified"]
    assert fee_parameters({"feesEnabled": False}, {})["fee_rate"] == 0


def test_reader_only_calls_public_get_and_retains_unverified_rules():
    calls = []
    class Response:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            if '/events/' in url:
                return Response({"markets": [{**market(), "active": True, "closed": False, "acceptingOrders": True}]})
            if '/clob-markets/' in url:
                return Response({"fd": {"r": .05, "e": 1}})
            token = kwargs["params"]["token_id"]
            return Response({"asset_id": token, "market": "condition", "timestamp": str(int(NOW.timestamp()*1000)),
                             "bids": [{"price": ".4", "size": "50"}], "asks": [{"price": ".5", "size": "50"}]})
    game = SimpleNamespace(game_id="g2", home_team="BOS", away_team="LAL", tip_time=NOW)
    result = PolymarketReader(Session(), lambda: NOW).fetch("nba-bos-lal-demo", game)
    assert len(result["quotes"]) == 2 and len(calls) == 4
    assert not result["quotes"]["home"]["rules_verified"]
    assert not result["quotes"]["home"]["includes_overtime"]
