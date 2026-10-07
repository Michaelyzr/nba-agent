"""I: episodic memory only recalls settled episodes and can veto a trade."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.memory import FEATURES, EpisodicMemory, MemoryAgent, features
from agents.notebook import Notebook
from replay import Replay
from tests.test_agent_graph import make_tables


def episode(i, settled, as_of=None, clv=-0.02, **sit):
    base = {"side_price": 0.4, "gap": 0.05, "market_move": 0.0, "model_shift": 0.02,
            "hours_to_tip": 1.0, "news_age_minutes": 20.0, "market_kind": "game", "team": "HOM"}
    base.update(sit)
    settled_at = pd.Timestamp(settled)
    decision = pd.Timestamp(as_of) if as_of is not None else settled_at - pd.Timedelta(hours=3)
    return {"situation": base, "settled_at": settled_at, "as_of": decision,
            "market_ticker": f"M{i}", "side": "yes", "price": 0.4, "contracts": 50, "clv": clv,
            "close_price": 0.4 + clv}


def filled(mem, n=50, start="2026-01-01"):
    for i in range(n):
        day = pd.Timestamp(start) + pd.Timedelta(days=i)
        mem.add(**episode(i, day + pd.Timedelta(hours=3), clv=(-0.03 if i % 2 else 0.01),
                          side_price=0.3 + 0.01 * (i % 5)))
    return mem


def test_features_are_finite_and_ordered():
    x = features({"side_price": 0.4, "gap": 0.05, "market_move": -0.01, "model_shift": 0.02,
                  "hours_to_tip": 1.5, "news_age_minutes": 30})
    assert x.shape == (len(FEATURES),) and np.isfinite(x).all()


def test_recall_never_returns_unsettled_or_future_episodes():
    mem = EpisodicMemory()
    now = pd.Timestamp("2026-02-01T06:00:00Z")
    mem.add(**episode(0, "2026-01-31T06:00:00Z", clv=-0.05, side_price=0.25))   # settled
    mem.add(**episode(1, "2026-02-01T08:00:00Z", as_of="2026-02-01T01:00:00Z",
                      clv=-0.05, side_price=0.25))                              # settles later
    mem.add(**episode(2, "2026-02-02T06:00:00Z", as_of="2026-02-01T01:00:00Z",
                      clv=-0.05, side_price=0.25))                              # future game
    near = mem.recall({"side_price": 0.25, "gap": 0.05, "market_move": 0.0, "model_shift": 0.02,
                       "hours_to_tip": 1.0, "news_age_minutes": 20.0}, now, k=10)
    assert {e["episode_id"] for e in near} == {"e00001"}
    assert all(e["settled_at"] <= now for e in near)
    assert all(e["settled_at"] <= now for e in mem.visible(now))


def test_settled_after_decision_is_required():
    mem = EpisodicMemory()
    with pytest.raises(ValueError, match="settle before"):
        mem.add(**episode(0, "2026-01-01T00:00:00Z", as_of="2026-01-01T03:00:00Z"))


def test_recall_ranks_by_similarity_among_visible_episodes():
    mem = filled(EpisodicMemory(), n=30)
    now = pd.Timestamp("2026-01-20T06:00:00Z")
    q = {"side_price": 0.30, "gap": 0.05, "market_move": 0.0, "model_shift": 0.02,
         "hours_to_tip": 1.0, "news_age_minutes": 20.0}
    near = mem.recall(q, now, k=5)
    assert len(near) == 5
    assert all(e["settled_at"] <= now for e in near)
    assert [e["distance"] for e in near] == sorted(e["distance"] for e in near)
    assert abs(near[0]["situation"]["side_price"] - 0.30) <= 0.02


def test_memory_agent_vetoes_when_similar_episodes_lost_and_stores_a_shadow(monkeypatch):
    """Build a memory of long-shot losers, then put the agent on a long-shot that would trade."""
    from agents import graph as g

    tables = make_tables(home_quote=(0.22, 0.24))
    mem = EpisodicMemory()
    for i in range(50):
        mem.add(**episode(i, f"2026-01-{1 + i % 28:02d}T06:00:00Z", clv=-0.04, side_price=0.24 + 0.001 * i,
                          gap=0.06, market_move=-0.01, model_shift=0.03, hours_to_tip=1.0, news_age_minutes=20))
    agent = MemoryAgent(memory=mem, learn=False, min_episodes=20, k=10, skip_clv=-0.01)

    def forever_fresh(self, view, ticker, now):
        return view.quote(ticker)

    def big_edge(self, view, game, now):
        return 0.50   # force a trade; the memory should veto it

    monkeypatch.setattr(g.MarketAgent, "_fresh_quote", forever_fresh)
    monkeypatch.setattr(g.MarketAgent, "_anchor_mid", big_edge)
    # Make the forecaster produce a large news shift so analyse wants to act.
    agent.forecaster = lambda view, game, markets, overrides: {
        m.market_ticker: (0.55 if "HOM" in m.market_ticker else 0.45) for m in markets.itertuples()}
    decisions, fills = Replay(tables, agent.policy, on_day_end=agent.on_day_end).run("2026-02-01", "2026-02-01")
    assert agent.recalls >= 1 or len(agent.vetoes) >= 1 or len(fills) == 0
    # If it would have traded without memory, the veto path must have fired.
    without = MemoryAgent(memory=EpisodicMemory(), learn=False, min_episodes=10_000)
    without.forecaster = agent.forecaster
    d2, f2 = Replay(tables, without.policy).run("2026-02-01", "2026-02-01")
    if len(f2):
        assert len(fills) == 0 and agent.vetoes
        assert all(v["neighbours"] >= 1 for v in agent.vetoes)
        tip = tables["games"].loc[tables["games"].game_id == "g2", "final_at"].iloc[0]
        shadows = [e for e in agent.memory.episodes if e.get("shadow")]
        assert shadows and all(e["settled_at"] == tip for e in shadows)
        assert not any(e.get("shadow") for e in agent.memory.visible(tip - pd.Timedelta(seconds=1)))
        assert any(e.get("shadow") for e in agent.memory.visible(tip))


def test_memory_agent_does_not_veto_before_min_episodes():
    agent = MemoryAgent(memory=EpisodicMemory(), learn=False, min_episodes=100)
    tables = make_tables()
    Replay(tables, agent.policy, on_day_end=agent.on_day_end).run("2026-02-01", "2026-02-01")
    assert agent.vetoes == []
