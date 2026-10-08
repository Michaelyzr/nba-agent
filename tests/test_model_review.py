"""Model review: forecast rules validate and are gated, never apply before valid_from, the stack is fit on
selection days only, and the defaults are unchanged."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.graph import MarketAgent
from agents.notebook import Notebook, validate
from forecast.blend import CANDIDATE_BLENDS, DEFAULT_BLEND, blend_home, fit_stack
from replay import Replay
from tests.test_agent_graph import TIP2, make_tables

T0 = pd.Timestamp("2026-02-01T00:00:00Z")


def fake_components(view, game, out, now):
    return {"mlp": 0.01 * len(out), "m6": 0.0}


def blend_rule(params, valid_from=T0, rid="f001", status="active"):
    return {"rule_id": rid, "kind": "forecast", "status": status, "when": {"market_kind": "game"},
            "do": {"action": "forecast_blend", "params": params}, "valid_from": valid_from, "expires_after_days": 45}


MARKET_ONLY = next(b for b in CANDIDATE_BLENDS if b["name"] == "market")


def test_forecast_rules_validate():
    for b in CANDIDATE_BLENDS:
        if b["name"] != "stack":
            assert validate(blend_rule(b)) == []
    assert validate(blend_rule({"name": "stack", "coef": [1.0, 0.5, 0.0, 0.0, 0.0], "intercept": 0.0})) == []
    assert validate(blend_rule({"name": "stack", "coef": [1.0], "intercept": 0.0}))
    assert validate(blend_rule({**DEFAULT_BLEND, "base": "model"}))
    assert validate(blend_rule({**DEFAULT_BLEND, "w_m4": 3.0}))


def test_defaults_unchanged():
    nodes = set(MarketAgent().graph.get_graph().nodes)
    assert not {"review_model", "gate_model", "save_model", "reject_model"} & nodes
    tables = make_tables()
    _, base = Replay(tables, MarketAgent().policy).run("2026-02-01", "2026-02-01")
    logged = MarketAgent(components=fake_components)
    _, same = Replay(tables, logged.policy).run("2026-02-01", "2026-02-01")
    assert base[["market_ticker", "side", "p_model", "price"]].equals(same[["market_ticker", "side", "p_model", "price"]])
    assert len(logged.component_log) > 0
    with pytest.raises(ValueError):
        MarketAgent(model_review=True)


def test_model_review_graph_has_the_subloop():
    nodes = set(MarketAgent(model_review=True, components=fake_components).graph.get_graph().nodes)
    assert {"review_model", "gate_model", "save_model", "reject_model"} <= nodes


def test_blend_rule_is_not_used_before_valid_from():
    later = TIP2 + pd.Timedelta(days=1)
    agent = MarketAgent(Notebook([blend_rule(MARKET_ONLY, valid_from=later)]), model_review=True,
                        components=fake_components)
    assert agent.active_blend(TIP2)["name"] == "m4"
    assert agent.active_blend(later)["name"] == "market"
    _, fills = Replay(make_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    _, base = Replay(make_tables(), MarketAgent().policy).run("2026-02-01", "2026-02-01")
    assert len(fills) == len(base) > 0                       # the future rule changes nothing today


def test_active_blend_changes_the_forecast():
    agent = MarketAgent(Notebook([blend_rule(MARKET_ONLY)]), model_review=True, components=fake_components)
    _, fills = Replay(make_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    _, base = Replay(make_tables(), MarketAgent().policy).run("2026-02-01", "2026-02-01")
    assert len(base) > 0 and len(fills) == 0                 # anchor only: the news shift is no longer traded
    assert {r["blend"] for r in agent.component_log.values()} == {"market"}


def component_rows(days, flip=False, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for i in range(n):
            anchor = rng.uniform(0.3, 0.7)
            win = float(rng.uniform() < anchor)
            m4 = (0.05 if win else -0.05) * (-1 if flip else 1)
            rows.append({"game_id": f"{d}-{i}", "as_of": pd.Timestamp(d, tz="UTC"), "date": d, "anchor": anchor,
                         "mid": anchor, "m4": m4, "mlp": 0.0, "m6": 0.0, "p_used": anchor + m4, "blend": "m4",
                         "home_win": win})
    return rows


def test_stack_is_fit_on_selection_days_only(monkeypatch):
    sel = [f"2026-03-{d:02d}" for d in range(1, 8)]
    other = [f"2026-02-{d:02d}" for d in range(10, 25)]
    good, poison = component_rows(sel), component_rows(other, flip=True, seed=1)
    agent = MarketAgent(model_review=True, components=fake_components)
    agent.component_log = {(r["game_id"], r["as_of"]): {k: v for k, v in r.items() if k != "home_win"}
                           for r in good + poison}
    games = pd.DataFrame([{"game_id": r["game_id"], "date": r["date"], "home_pts": 100 + r["home_win"],
                           "away_pts": 100, "final_at": r["as_of"] + pd.Timedelta(hours=3)} for r in good + poison])
    rp = SimpleNamespace(t={"games": games})
    monkeypatch.setattr(MarketAgent, "model_windows", staticmethod(lambda rp, day: (sel, other[-14:])))
    monkeypatch.setattr("forecast.blend.CANDIDATE_BLENDS", [DEFAULT_BLEND, {"name": "stack"}])
    seen = {}
    real_fit = fit_stack

    def spy(frame):
        seen["dates"] = set(frame.date)
        return real_fit(frame)
    monkeypatch.setattr("forecast.blend.fit_stack", spy)
    out = agent.review_model({"replay": rp, "day": sel[-1]})
    assert seen["dates"] == set(sel)
    expected = real_fit(pd.DataFrame(good))
    if out["model_proposal"] is not None:
        assert out["model_proposal"]["do"]["params"]["coef"] == expected["coef"]
        assert out["model_proposal"]["status"] == "proposed"      # only the gate can make it active


def test_proposed_blend_goes_through_the_gate(monkeypatch):
    agent = MarketAgent(model_review=True, components=fake_components)
    rule = blend_rule(MARKET_ONLY, valid_from=None, status="proposed")
    games = pd.DataFrame({"game_id": ["g"], "date": ["2026-03-01"], "final_at": [T0]})
    rp = SimpleNamespace(t={"games": games})
    monkeypatch.setattr(MarketAgent, "model_windows", staticmethod(lambda rp, day: (["2026-03-01"], ["2026-02-20",
                                                                                                    "2026-02-21"])))
    empty = pd.DataFrame(columns=["clv", "contracts", "pnl", "edge"])
    child = SimpleNamespace(component_log={})
    monkeypatch.setattr(MarketAgent, "backtest", lambda self, rp, nb, a, b, return_agent=False:
                        (empty, child) if return_agent else empty)
    out = agent.gate_model({"replay": rp, "day": "2026-03-01", "model_proposal": rule})
    assert out["model_gate"]["result"] == "rejected"         # no CLV gain: the gate refuses it
    agent.reject_model({"model_gate": out["model_gate"], "model_proposal": rule})
    assert agent.notebook.rules[-1]["status"] == "rejected" and agent.active_blend(T0)["name"] == "m4"


def test_blend_home_linear_and_stack():
    f = pd.DataFrame({"anchor": [0.5], "mid": [0.55], "m4": [0.04], "mlp": [0.02], "m6": [0.01]})
    assert blend_home(DEFAULT_BLEND, f)[0] == pytest.approx(0.54)
    mean = next(b for b in CANDIDATE_BLENDS if b["name"] == "m4_mlp_mean")
    assert blend_home(mean, f)[0] == pytest.approx(0.53)
    m6 = next(b for b in CANDIDATE_BLENDS if b["name"] == "m6")
    assert blend_home(m6, f)[0] == pytest.approx(0.56)
    stack = {"name": "stack", "coef": [1.0, 0.0, 0.0, 0.0, 0.0], "intercept": 0.0}
    assert blend_home(stack, f)[0] == pytest.approx(0.5)
