"""Adaptive edge: templates validate, the min_edge mask raises and lowers, and the defaults are unchanged."""
import pandas as pd
import pytest

from agents.graph import (ADAPTIVE_TEMPLATES, DEFAULT_MIN_EDGE, EDGE_LEVELS, EDGE_TEMPLATES, REVIEW_TEMPLATES,
                          MarketAgent, effective_edge, is_global_edge)
from agents.notebook import MAX_EDGE, Notebook, validate
from tests.test_agent_graph import make_tables
from replay import Replay

T0 = pd.Timestamp("2026-02-01T00:00:00Z")


def rule(edge, when=None, rid="r001", valid_from=T0, status="active"):
    return {"rule_id": rid, "kind": "trader", "status": status, "when": {"market_kind": "game", **(when or {})},
            "do": {"action": "min_edge", "params": {"edge": edge}}, "valid_from": valid_from,
            "expires_after_days": 45}


def test_every_edge_template_validates():
    agent = MarketAgent()
    for when, do, _, _ in EDGE_TEMPLATES:
        assert validate(agent._rule({"market_kind": "game", **when}, do, "model_wrong", "x",
                                    pd.DataFrame({"decision_id": []}))) == []
    assert {do["params"]["edge"] for when, do, _, _ in EDGE_TEMPLATES if not when} == set(EDGE_LEVELS)
    assert validate(rule(MAX_EDGE + 0.01)) and validate(rule(0.0)) and not validate(rule(0.02))


def test_published_templates_are_unchanged():
    edges = [t for t in REVIEW_TEMPLATES if t[1]["action"] == "min_edge"]
    assert len(edges) == 1 and edges[0][1]["params"]["edge"] == pytest.approx(0.07)
    assert DEFAULT_MIN_EDGE == 0.04
    assert all(t in ADAPTIVE_TEMPLATES for t in REVIEW_TEMPLATES if t[1]["action"] != "min_edge")
    agent = MarketAgent()
    assert agent.adaptive_edge is False and agent.min_edge == DEFAULT_MIN_EDGE


def test_default_threshold_is_max_and_adaptive_lets_newest_global_rule_set_it():
    low, high_old = rule(0.02, rid="r002", valid_from=T0 + pd.Timedelta(days=2)), rule(0.07, rid="r001")
    dog = rule(0.08, {"side_price_max": 0.35}, rid="r003")
    assert effective_edge(0.04, [low, high_old]) == pytest.approx(0.07)          # default: rules only raise
    assert effective_edge(0.04, [low, high_old], adaptive=True) == pytest.approx(0.02)   # newest sets it
    assert effective_edge(0.04, [low, dog], adaptive=True) == pytest.approx(0.08)  # slice rules only raise
    trial = {**rule(0.10, rid="r009"), "valid_from": None}
    assert effective_edge(0.04, [low, trial], adaptive=True) == pytest.approx(0.10)  # gate trial wins
    assert is_global_edge(low) and not is_global_edge(dog)


def test_mask_raising_covers_trades_between_current_and_new_edge():
    sit = pd.DataFrame({"gap": [0.045, 0.06, 0.09], "side_price": [0.3, 0.5, 0.3]})
    raise_ = MarketAgent.template_mask(sit, {}, {"action": "min_edge", "params": {"edge": 0.07}}, current=0.04)
    assert raise_.tolist() == [True, True, False]
    dog = MarketAgent.template_mask(sit, {"side_price_max": 0.35}, {"action": "min_edge", "params": {"edge": 0.08}},
                                    current=0.04)
    assert dog.tolist() == [True, False, False]                 # only underdogs below 8 points


def test_mask_lowering_covers_candidates_the_current_threshold_blocked():
    blocked = pd.DataFrame({"gap": [0.005, 0.025, 0.035, 0.04]})
    lower = MarketAgent.template_mask(blocked, {}, {"action": "min_edge", "params": {"edge": 0.02}}, current=0.04)
    assert lower.tolist() == [False, True, True, True]


def test_published_mask_is_unchanged_without_current():
    sit = pd.DataFrame({"gap": [0.05, 0.08], "side_price": [0.2, 0.2]})
    m = MarketAgent.template_mask(sit, {"side_price_max": 0.1}, {"action": "min_edge", "params": {"edge": 0.07}})
    assert m.tolist() == [True, False]                          # condition ignored, as published


def test_superseded_global_rule_stops_at_valid_until():
    old = {**rule(0.07, rid="r001"), "valid_until": T0 + pd.Timedelta(days=3)}
    nb = Notebook([old, rule(0.02, rid="r002", valid_from=T0 + pd.Timedelta(days=3))])
    assert [r["rule_id"] for r in nb.active(T0 + pd.Timedelta(days=1))] == ["r001"]
    assert [r["rule_id"] for r in nb.active(T0 + pd.Timedelta(days=4))] == ["r002"]
    trial = nb.with_candidate({**old})
    assert all(r.get("valid_until") is None for r in trial.rules if r["rule_id"] == "r001")


def test_low_base_edge_trades_more_and_default_agent_matches_explicit_defaults():
    tables = make_tables(home_quote=(0.66, 0.68))
    _, base = Replay(tables, MarketAgent().policy).run("2026-02-01", "2026-02-01")
    _, same = Replay(tables, MarketAgent(min_edge=0.04, adaptive_edge=False).policy).run("2026-02-01", "2026-02-01")
    _, adaptive = Replay(tables, MarketAgent(adaptive_edge=True).policy).run("2026-02-01", "2026-02-01")
    _, low = Replay(tables, MarketAgent(min_edge=0.0001, adaptive_edge=True).policy).run("2026-02-01", "2026-02-01")
    assert len(base) == len(same) == len(adaptive)              # no rules yet: the flag alone changes nothing
    assert len(low) >= len(base)


def test_save_rule_supersedes_previous_global_threshold():
    agent = MarketAgent(adaptive_edge=True)
    agent.notebook.record(rule(0.07, rid="r001"))
    later = T0 + pd.Timedelta(days=5)
    proposal = {k: v for k, v in rule(0.03, rid="r002").items() if k not in ("valid_from", "status")}
    agent.save_rule({"proposal": {**proposal, "status": "proposed"}, "gate": {"decided_at": later}})
    first, second = agent.notebook.rules
    assert first["valid_until"] == later and second["supersedes"] == "r001"
    now = later + pd.Timedelta(hours=1)
    assert effective_edge(0.04, agent.notebook.active(now, "trader"), adaptive=True) == pytest.approx(0.03)
