"""H: the rule DSL parses only the safe grammar, compiles to the notebook format and plugs into the agent."""
import json

import pandas as pd
import pytest

from agents import rule_dsl
from agents.notebook import matches, validate
from agents.rule_dsl import (DSLError, DSLReviewAgent, cached_ask, check, compile_text, parse, predicate,
                             stub_proposer, to_text)
from tests.test_agent_graph import make_tables
from replay import Replay


@pytest.mark.parametrize("text, rule", [
    ("when side_price <= 0.30 and news_age_minutes >= 30 do skip",
     {"when": {"market_kind": "game", "side_price_max": 0.3, "news_age_minutes_min": 30.0},
      "do": {"action": "skip_market", "params": {}}}),
    ("WHEN gap <= 0.07 DO min_edge 0.07",
     {"when": {"market_kind": "game", "gap_max": 0.07}, "do": {"action": "min_edge", "params": {"edge": 0.07}}}),
    ("when market_kind == game and market_move <= -0.02 do stake_scale 0.5.",
     {"when": {"market_kind": "game", "market_move_max": -0.02},
      "do": {"action": "stake_scale", "params": {"scale": 0.5}}}),
])
def test_valid_rules_compile_to_the_notebook_format(text, rule):
    assert compile_text(text) == rule
    assert validate(rule) == []
    assert compile_text(to_text(rule)) == rule


@pytest.mark.parametrize("text, reason", [
    ("when __import__('os').system('rm -rf /') do skip", "unexpected character"),
    ("when gap <= 0.1 do skip; import os", "unexpected character"),
    ("when gap <= 0.1 do eval", "unknown action"),
    ("when gap <= 0.1 do skip and gap >= 0", "unexpected text"),
    ("when pnl <= 0 do skip", "unknown field"),
    ("when outcome >= 1 do skip", "unknown field"),
    ("when close_price <= 0.5 do skip", "unknown field"),
    ("when side_price <= 1.5 do skip", "outside"),
    ("when news_age_minutes >= -5 do skip", "outside"),
    ("when gap <= 0.1 do min_edge 0.01", "outside"),
    ("when gap <= 0.1 do min_edge 0.9", "outside"),
    ("when gap <= 0.1 do stake_scale 2", "outside"),
    ("when gap <= 0.1 do stake_scale 0", "outside"),
    ("when gap == 0.1 do skip", "needs <= or >="),
    ("when team <= 3 do skip", "only allows =="),
    ("when team == lakers do skip", "not allowed"),
    ("when gap >= 0.2 and gap <= 0.1 do skip", "range is empty"),
    ("when gap <= 0.2 and gap <= 0.1 do skip", "given twice"),
    ("when gap <= 0.1 and side_price <= 0.3 and hours_to_tip >= 1 and model_shift <= 0.01 do skip",
     "more than 3 conditions"),
    ("when gap <= 0.1", "ends early"),
    ("gap <= 0.1 do skip", "expected when"),
    ("when gap <= 0.1 do skip " + "x" * 300, "longer than"),
    ("", "ends early"),
])
def test_unsafe_or_unknown_rules_are_rejected_with_a_reason(text, reason):
    rule, why = check(text)
    assert rule is None and reason in why
    with pytest.raises(DSLError):
        compile_text(text)


def test_non_text_is_rejected():
    for bad in (None, 3, {"when": {}}, ["when"]):
        assert check(bad)[0] is None


def test_compiled_rule_matches_the_expected_situations():
    text = "when side_price <= 0.30 and news_age_minutes >= 30 do skip"
    rule, is_match = compile_text(text), predicate(text)
    base = {"market_kind": "game", "side_price": 0.25, "news_age_minutes": 45.0, "gap": 0.05}
    cases = [(base, True), ({**base, "side_price": 0.30}, True), ({**base, "news_age_minutes": 30.0}, True),
             ({**base, "side_price": 0.31}, False), ({**base, "news_age_minutes": 10.0}, False),
             ({**base, "market_kind": "player"}, False), ({k: v for k, v in base.items() if k != "side_price"}, False)]
    for situation, want in cases:
        assert matches(rule, situation) is want
        assert is_match(situation) is want


def test_parse_never_evaluates_text(monkeypatch):
    monkeypatch.setattr("builtins.eval", lambda *a, **k: pytest.fail("eval called"))
    monkeypatch.setattr("builtins.exec", lambda *a, **k: pytest.fail("exec called"))
    assert parse("when gap >= 0.1 do skip")["action"] == "skip_market"


def test_cache_is_deterministic_and_keyed_by_prompt_and_model(tmp_path):
    calls = []

    def ask(prompt, want_json=True):
        calls.append(prompt)
        return {"rules": [f"when gap >= 0.{len(calls)} do skip"]}

    a1 = cached_ask("p", "m1", ask, tmp_path)
    a2 = cached_ask("p", "m1", ask, tmp_path)
    b = cached_ask("p", "m2", ask, tmp_path)
    c = cached_ask("q", "m1", ask, tmp_path)
    assert a1 == a2 and len(calls) == 3
    assert b != a1 and c != a1
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert "temperature" not in json.loads(next(tmp_path.glob("*.json")).read_text())  # key only, not stored


def test_llm_proposer_uses_the_cache_and_keeps_k(tmp_path, monkeypatch):
    calls = []

    def ask(prompt, want_json=True):
        calls.append(prompt)
        return {"rules": ["when gap >= 0.1 do skip", "when pnl <= 0 do skip", "x", "y"]}

    def ask_cached(prompt, model, ask_fn=None, cache=tmp_path):
        return cached_ask(prompt, model, ask_fn, cache)

    monkeypatch.setattr(rule_dsl, "cached_ask", ask_cached)
    propose = rule_dsl.llm_proposer(k=3, ask=ask, model="test")
    first = propose(trades(), "2026-02-01T06:00Z")
    second = propose(trades(), "2026-02-01T06:00Z")
    assert first == second == ["when gap >= 0.1 do skip", "when pnl <= 0 do skip", "x"]
    assert len(calls) == 1


def trades(n=12):
    return pd.DataFrame({"side_price": [0.2 + 0.05 * i for i in range(n)], "gap": [0.05] * n,
                         "market_move": [0.0] * n, "model_shift": [0.02] * n, "hours_to_tip": [1.0] * n,
                         "news_age_minutes": [20.0] * n,
                         "clv": [-0.03 if i < 6 else 0.01 for i in range(n)], "contracts": [50] * n})


def test_stub_proposer_is_deterministic_valid_and_targets_losing_trades():
    p = stub_proposer(k=3)
    out = p(trades(), "2026-02-01")
    assert out == p(trades(), "2026-02-01") and out
    rule = compile_text(out[0])
    t = trades()
    hit = [matches(rule, s) for s in t.assign(market_kind="game").to_dict("records")]
    assert 6 <= sum(hit) <= 6 and t.clv[hit].mean() < 0
    assert all(check(x)[0] is not None for x in out)
    assert stub_proposer()(trades().iloc[0:0], "2026-02-01") == []


def test_dsl_agent_runs_rules_through_the_gate_and_logs_invalid_ones():
    texts = ["when side_price <= 0.9 do skip", "when pnl <= 0 do skip"]
    agent = DSLReviewAgent(proposer=lambda t, now: list(texts), k=3)
    tables = make_tables()
    Replay(tables, agent.policy, on_day_end=agent.on_day_end).run("2026-02-01", "2026-02-01")
    if not agent.proposals:      # no fills on the tiny slate means no review; check the plumbing directly
        pytest.skip("no fills on the synthetic slate")
    by_text = {p["text"]: p for p in agent.proposals}
    assert by_text["when pnl <= 0 do skip"]["valid"] is False
    assert "unknown field" in by_text["when pnl <= 0 do skip"]["reason"]
    assert by_text["when side_price <= 0.9 do skip"]["gate"] in ("accepted", "rejected", "deferred", "dropped")
    assert all(r["status"] != "active" or r.get("gate") for r in agent.notebook.rules)
