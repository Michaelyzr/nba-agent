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
            text = json.dumps(s).lower()
            for word in ("close", "game_outcome", "home_pts", "away_pts", "settled", "pnl", "winner"):
                assert word not in text, f"{f.name}: outcome/settlement data ({word}) in a decision step"
            for n in s.get("news", []):
                assert pd.Timestamp(n["published_at"]) <= now


def real_game_traces():
    for f in TRACES.glob("*.json"):
        t = json.loads(f.read_text())
        if t.get("agent") in ("trader", "tool_agent", "coach", "inplay"):
            yield f.name, t, [t.get("game_outcome")]
        elif t.get("agent") == "orchestrator":
            yield f.name, t, [(t.get("game_outcomes") or {}).get(g["game_id"]) for g in t["night"]["games"]]
        elif t.get("agent") == "pregame":
            yield f.name, t, [g.get("game_outcome") for g in t["games"].values()]


def test_every_real_game_trace_has_an_outcome():
    seen = 0
    for name, t, outcomes in real_game_traces():
        for go in outcomes:
            seen += 1
            assert go, name
            assert go["home_pts"] is not None and go["away_pts"] is not None, name
            assert go["winner"] in (go["home"], go["away"]) and go["margin"] == abs(go["home_pts"] - go["away_pts"])
            assert go["markets"], f"{name}: no Kalshi settlement"
            for p in go["positions"]:
                assert p["result"] in ("won", "lost") and p["pnl"] is not None
        if t.get("agent") == "trader" and t["status"] != "sent":
            assert all(p["counterfactual"] for p in t["game_outcome"]["positions"]), name
    assert seen >= 15
    for sc in load("index")["scenarios"]:
        assert sc.get("game_outcome") and sc["game_outcome"]["headline"], sc["id"]


def test_cle_por_outcome_matches_data():
    go = load("trader_cle_por")["game_outcome"]
    frozen = Path(__file__).resolve().parents[1] / "data" / "frozen" / "games.parquet"
    if frozen.exists():
        g = pd.read_parquet(frozen).set_index("game_id").loc["401810565"]
        assert (go["home_pts"], go["away_pts"]) == (int(g.home_pts), int(g.away_pts))
    assert (go["home"], go["away"], go["home_pts"], go["away_pts"]) == ("POR", "CLE", 111, 130)
    assert go["winner"] == "CLE" and go["margin"] == 19
    p = go["positions"][0]
    assert (p["team"], p["side"], p["price"], p["settled_value"], p["result"], p["counterfactual"]) == \
        ("POR", "no", 0.55, 1.0, "won", False)
    assert p["pnl"] == pytest.approx(15.57) and p["clv"] == pytest.approx(0.075)
    assert {m["team"]: m["settled_yes"] for m in go["markets"]} == {"CLE": 1, "POR": 0}
    kill = load("trader_cle_por_kill")["game_outcome"]["positions"][0]
    assert kill["counterfactual"] and kill["pnl"] == pytest.approx(15.57)
    assert "SYNTHETIC" in load("inplay_cle_por")["game_outcome"]["note"]


def test_cle_por_headline_numbers():
    t = load("trader_cle_por")
    assert t["status"] == "sent"
    fill = t["settlement"]["fills"][0]
    assert fill["side"] == "no" and fill["price"] == 0.55 and fill["clv"] == pytest.approx(0.075)
    assert load("trader_cle_por_blocked")["status"] == "blocked"


APP = TRACES.parents[1] / "demo_app.py"
GRADED = "Graded by the market (revealed at tip-off)"


def panel_text(at):
    texts = [e.value for e in at.main if hasattr(e, "value") and isinstance(e.value, str)]
    texts += [f"{m.label} {m.value}" for m in at.metric]
    start = next(i for i, t in enumerate(texts) if t == GRADED)
    return "\n".join(texts[start:])


def test_graded_panel_shows_cle_por_outcome():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Scenarios").run()
    title = next(t for t in at.sidebar.selectbox[0].options if t.startswith("CLE @ POR"))
    at.sidebar.selectbox[0].set_value(title).run()
    assert at.sidebar.toggle[0].value is False
    for agent in ("trader", "tool_agent", "coach", "orchestrator", "pregame", "inplay"):
        at.sidebar.radio[1].set_value(agent).run()
        assert not at.exception, agent
        assert any(s.value == GRADED for s in at.subheader), agent
        text = panel_text(at)
        assert "CLE 130 – 111 POR" in text, agent
        assert "Winner: **CLE** by 19" in text, agent
    at.sidebar.radio[1].set_value("trader").run()
    text = panel_text(at)
    assert "POR NO won → paid $1/contract" in text and "+$15.57" in text and "+7.5c" in text
    at.sidebar.radio[1].set_value("inplay").run()
    assert "SYNTHETIC" in panel_text(at)


def test_graded_panel_labels_counterfactual_for_pass():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Scenarios").run()
    index = load("index")
    sc = next(s for s in index["scenarios"] if s["agents"].get("trader") == "trader_pass")
    at.sidebar.selectbox[0].set_value(sc["title"]).run()
    at.sidebar.radio[1].set_value("trader").run()
    text = panel_text(at)
    assert "No trade placed. Counterfactual" in text and "would have lost $21.18" in text


def test_outcome_not_in_decision_steps_before_panel():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Scenarios").run()
    at.sidebar.radio[1].set_value("trader").run()
    texts = [e.value for e in at.main if hasattr(e, "value") and isinstance(e.value, str)]
    before = "\n".join(texts[:texts.index(GRADED)])
    assert "130 – 111" not in before and "15.57" not in before


def test_app_renders_every_scenario_and_agent():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Scenarios").run()
    for title in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value(title).run()
        for agent in at.sidebar.radio[1].options:
            at.sidebar.radio[1].set_value(agent).run()
            assert not at.exception, (title, agent, at.exception)
    at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[0]).run()
    at.sidebar.radio[1].set_value("trader").run()
    assert any(s.value == GRADED for s in at.subheader)
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any(b.label == "Reveal game outcome" for b in at.button)
    assert "CLE 130 – 111 POR" not in panel_text(at)
    next(b for b in at.button if b.label == "Reveal game outcome").click().run()
    assert not at.exception and not any(b.label == "Reveal game outcome" for b in at.button)
    assert "CLE 130 – 111 POR" in panel_text(at)


def test_stub_llm_traces_are_labelled():
    for f in TRACES.glob("tool_*.json"):
        t = json.loads(f.read_text())
        assert ("stub" in t["llm"].lower()) == t["stub"]
