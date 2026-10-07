"""The hosted demo's precomputed traces: they load, have the required steps, hold no secrets or look-ahead."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

TRACES = Path(__file__).resolve().parents[1] / "demo" / "traces"
SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|api[_-]?key)", re.I)

pytestmark = pytest.mark.skipif(not (TRACES / "index.json").exists(), reason="demo traces not built")


def load(name):
    return json.loads((TRACES / f"{name}.json").read_text())


def test_index_points_at_existing_traces():
    index = load("index")
    assert len(index["scenarios"]) >= 4
    for sc in index["scenarios"]:
        for name in sc["agents"].values():
            assert (TRACES / f"{name}.json").exists(), name


def test_no_secrets_and_small():
    total = 0
    for f in TRACES.glob("*.json"):
        text = f.read_text()
        total += len(text)
        assert not SECRET.search(text), f.name
    assert total < 5_000_000


def test_trader_traces_have_required_steps_and_no_look_ahead():
    for f in TRACES.glob("trader_*.json"):
        t = json.loads(f.read_text())
        assert t["path"][:4] == ["trigger", "investigate", "forecast", "analyse"]
        assert "checks" in t["path"] and t["path"][-1] == "deliver"
        now = pd.Timestamp(t["as_of"])
        assert now < pd.Timestamp(t["tip_time"]) <= pd.Timestamp(t["settlement"]["revealed_at"])
        for s in t["steps"]:
            assert "close" not in json.dumps(s).lower(), f"{f.name}: settlement data in a decision step"
            for n in s.get("news", []):
                assert pd.Timestamp(n["published_at"]) <= now


def test_cle_por_headline_numbers():
    t = load("trader_cle_por")
    assert t["status"] == "sent"
    fill = t["settlement"]["fills"][0]
    assert fill["side"] == "no" and fill["price"] == 0.55 and fill["clv"] == pytest.approx(0.075)
    assert load("trader_cle_por_blocked")["status"] == "blocked"


def test_app_renders_every_scenario_and_agent():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(TRACES.parents[1] / "demo_app.py"), default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Scenarios").run()
    for title in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value(title).run()
        for agent in at.sidebar.radio[1].options:
            at.sidebar.radio[1].set_value(agent).run()
            assert not at.exception, (title, agent, at.exception)


def test_stub_llm_traces_are_labelled():
    for f in TRACES.glob("tool_*.json"):
        t = json.loads(f.read_text())
        assert ("stub" in t["llm"].lower()) == t["stub"]
