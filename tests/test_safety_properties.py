"""Property-based safety invariants for the risk step, kill switch and as-of view.

Uses Hypothesis when installed; otherwise a seeded random loop. These are the
claims in docs/preregistration_gate.md §kill switch and the review's option B.
"""
import random
from dataclasses import replace

import pandas as pd
import pytest

from agents.graph import BANNED, MAX_ORDER, MarketAgent, auto_confirm, pretrade_risk
from replay import AsOf, Order, Replay, basic_risk
from tests.test_replay import TIP2, TICKER, make_tables

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HAS_HYP = True
except ImportError:  # pragma: no cover
    HAS_HYP = False


MAX_GAME, MAX_DAY = 100.0, 300.0
SEED = 7606


def _order(stake=20.0, channel="platform", side="yes", reason="cited edge", citations=None):
    return Order(TICKER, side, 0.6, stake, reason, list(citations or ["n_early"]), channel)


def _ctx(spent_game=0.0, spent_day=0.0, killed=False, now=None):
    tip = TIP2
    return {"now": tip - pd.Timedelta(hours=1) if now is None else now, "tip_time": tip,
            "spent_game": spent_game, "spent_day": spent_day, "kill_switch_tripped": killed,
            "realised_day": -150.0 if killed else 0.0}


# ---------------- order / day / game caps ----------------

def test_no_order_exceeds_caps_over_random_stakes():
    rng = random.Random(SEED)
    for _ in range(200):
        stake = rng.uniform(0.1, 200.0)
        spent_g, spent_d = rng.uniform(0, MAX_GAME), rng.uniform(0, MAX_DAY)
        order = _order(stake=stake)
        ok_g, why_g = pretrade_risk(order, _ctx())
        ok_r, why_r = basic_risk(order, _ctx(spent_g, spent_d))
        if stake > MAX_ORDER:
            assert not ok_g and why_g == "over_order_cap"
            assert not ok_r and why_r == "over_order_cap"
        elif spent_g + stake > MAX_GAME:
            assert not ok_r and why_r == "over_game_cap"
        elif spent_d + stake > MAX_DAY:
            assert not ok_r and why_r == "over_day_cap"
        else:
            assert ok_r


def test_at_most_one_position_per_game():
    agent = MarketAgent(learn=False, anchor=False)
    decisions, fills = Replay(make_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    assert fills.game_id.nunique() == len(fills) or fills.empty
    if len(decisions):
        by_game = decisions[decisions.fill_result == "filled"].groupby("game_id").size()
        assert (by_game <= 1).all()


def test_no_order_after_kill_switch_trip():
    for stake in (20.0, 50.0, 1.0):
        ok, why = basic_risk(_order(stake=stake), _ctx(killed=True))
        assert not ok and why == "kill_switch"
        ok, why = pretrade_risk(_order(stake=stake), _ctx(killed=True))
        assert not ok and why == "kill_switch"


def test_banned_words_blocked_in_delivered_text():
    from tests.test_agent_graph import make_tables as trade_tables

    for text in ("This is a lock", "guaranteed win", "can't lose tonight", "sure thing", "risk-free edge"):
        assert BANNED.search(text), text
    # Slate with an Out listing so the agent proposes; plant injects "lock" into the brief.
    agent = MarketAgent(plant="lock_wording", anchor=False)
    _, fills = Replay(trade_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    assert fills.empty
    brief = next(t["brief"] for t in agent.traces if t.get("phase") == "decide")
    assert "Order blocked" in brief


def test_as_of_never_exposes_future_news_or_prices():
    t = make_tables()
    r = Replay(t, lambda *a: [])
    rng = random.Random(SEED)
    tips = t["games"].tip_time
    for _ in range(50):
        now = tips.min() + pd.Timedelta(minutes=rng.randint(-180, 180))
        view = AsOf(t, now, r.price_index)
        news = view.news()
        assert news.empty or (news.published_at <= now).all()
        prices = view.prices(TICKER)
        assert prices.empty or (prices.ts <= now).all()
        g = view.games()
        pending = g.final_at > now
        assert g.loc[pending, "home_pts"].isna().all()


def test_retail_auto_confirm_is_empty():
    assert auto_confirm("x", [_order(channel="retail")]) == []
    assert len(auto_confirm("x", [_order(channel="platform")])) == 1


if HAS_HYP:
    @given(stake=st.floats(0.01, 500, allow_nan=False, allow_infinity=False),
           spent_g=st.floats(0, 150, allow_nan=False, allow_infinity=False),
           spent_d=st.floats(0, 400, allow_nan=False, allow_infinity=False),
           killed=st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_hypothesis_caps_and_kill_switch(stake, spent_g, spent_d, killed):
        order = _order(stake=float(stake))
        ok, why = basic_risk(order, _ctx(float(spent_g), float(spent_d), killed))
        if killed:
            assert not ok and why == "kill_switch"
            return
        if stake > MAX_ORDER:
            assert not ok and why == "over_order_cap"
        elif spent_g + stake > MAX_GAME:
            assert not ok and why == "over_game_cap"
        elif spent_d + stake > MAX_DAY:
            assert not ok and why == "over_day_cap"
        else:
            assert ok
