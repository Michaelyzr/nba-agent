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
    total = sum(f.stat().st_size for f in TRACES.glob("*.json"))
    total += sum(f.stat().st_size for f in (TRACES.parent / "figures").glob("*.png"))
    for f in TRACES.glob("*.json"):
        assert not SECRET.search(f.read_text()), f.name
    assert total < 3_000_000, total


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


APP = TRACES.parents[1] / "research_demo_app.py"
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


OUTCOME_KEYS = ("close_price", "clv", "pnl", "outcome_yes", "home_pts", "away_pts", "winner", "settled_yes",
                "settled_value", "game_outcome", "settlement", "final_at")


def deepseek_traces():
    return [(f.name, json.loads(f.read_text())) for f in sorted(TRACES.glob("tool_ds_*.json"))]


def keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from keys(v)


def test_deepseek_traces_are_real_as_of_and_complete():
    ds = deepseek_traces()
    assert len(ds) >= 6
    names = {n for n, _ in ds}
    assert {"tool_ds_kept_1.json", "tool_ds_kept_2.json", "tool_ds_chat.json", "tool_ds_budget.json",
            "tool_ds_plain.json"} <= names and sum(n.startswith("tool_ds_veto") for n in names) >= 2
    for name, t in ds:
        assert t["source"] == "deepseek" and not t["stub"] and t["model"] in ("deepseek-chat", "deepseek-reasoner")
        assert t["model_label"] in ("DeepSeek chat", "DeepSeek reasoner (V4.1-Flash, thinking)")
        assert not SECRET.search(json.dumps(t)), name
        now = pd.Timestamp(t["as_of"])
        assert now < pd.Timestamp(t["tip_time"]) <= pd.Timestamp(t["settlement"]["revealed_at"])
        decision = {k: v for k, v in t.items() if k not in ("settlement", "game_outcome")}
        leaked = set(keys(decision)) & set(OUTCOME_KEYS)
        assert not leaked, f"{name}: outcome data {leaked} in the decision steps"
        assert t["market"] and all("anchor_mid" in m for m in t["market"])
        if t["setup"] != "plain":
            calls = [r for turn in t["turns"] for r in turn["tool_results"]]
            assert calls and all(r["tool"] and "output" in r for r in calls), name
        if t["status"] in ("order", "sceptic_reject") and t["setup"] != "plain":
            assert t["validation"]["ok"] and t["gap_after_fees"] > t["min_edge"], name
        if t["setup"] == "tool_sceptic" and t["status"] in ("order", "sceptic_reject"):
            want = "approve" if t["status"] == "order" else "reject"
            assert t["sceptic"]["verdict"]["verdict"] == want, name
        if t["model"] == "deepseek-reasoner":
            assert t["reasoning"]["chars"] > 0 and t["reasoning"]["excerpt"] is None
        go = t["game_outcome"]
        assert go["positions"] and all(p["counterfactual"] == (t["status"] != "order") for p in go["positions"])


def test_deepseek_scenarios_match_the_run():
    kept = sorted((load(n)["game_outcome"]["positions"][0]["pnl"]) for n in ("tool_ds_kept_1", "tool_ds_kept_2"))
    assert kept == [pytest.approx(-20.71), pytest.approx(120.92)]
    vetoes = [t for n, t in deepseek_traces() if n.startswith("tool_ds_veto")]
    results = {t["game_outcome"]["positions"][0]["result"] for t in vetoes}
    assert results == {"won", "lost"}, "show a veto that would have lost and one that would have won"
    b = load("tool_ds_budget")
    assert b["status"] == "invalid" and b["out_of_reasoning_budget"] and b["validation"]["out_of_reasoning_budget"]
    assert load("tool_ds_plain")["turns"] == [] and load("tool_ds_plain")["plain_info"]
    ids = {s["id"] for s in load("index")["scenarios"]}
    assert "sceptic" not in ids and {"ds_kept_1", "ds_veto_1", "ds_chat", "ds_budget", "ds_plain"} <= ids


def test_llm_results_parsed_from_results_file():
    r = load("llm_results")
    rsn = r["models"]["deepseek-reasoner"]
    d = next(x for x in rsn["scoreboard"] if x["Setup"].startswith("D."))
    assert d["Trades"] == "2" and d["CLV $ [95% CI]"] == "+4 [-3, +15]"
    a = next(x for x in rsn["scoreboard"] if x["Setup"].startswith("A."))
    assert a["Trades"] == "53" and a["CLV $ [95% CI]"] == "-19 [-36, -1]"
    sc = next(x for x in rsn["paired"] if x["Comparison"] == "tool_sceptic − tool" and x["Metric"] == "CLV $")
    assert sc["Difference [95% CI]"] == "+17 [+4, +32]" and sc["p"] == "0.010"
    assert "95%" in r["punchline"] and r["reasoning_budget_calls"] == 21 and r["cost_usd"] == 3.49


def test_deepseek_scenarios_render_and_gate_outcome():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception
    assert any("LLM agent vs original agent" in h.value for h in at.header)
    assert any("finds no edge" in str(i.value) for i in at.info)
    at.sidebar.radio[0].set_value("Scenarios").run()
    index = load("index")
    for sc in index["scenarios"]:
        if not sc["id"].startswith("ds_"):
            continue
        t = load(sc["agents"]["tool_agent"])
        at.sidebar.selectbox[0].set_value(sc["title"]).run()
        at.sidebar.radio[1].set_value("tool_agent").run()
        assert not at.exception, sc["id"]
        texts = [e.value for e in at.main if hasattr(e, "value") and isinstance(e.value, str)]
        assert any(t["model_label"] in x for x in texts), sc["id"]
        before = "\n".join(texts[:texts.index(GRADED)])
        go = t["game_outcome"]
        assert f"{go['away_pts']} – {go['home_pts']}" not in before, sc["id"]
        assert f"{go['away']} {go['away_pts']} – {go['home_pts']} {go['home']}" in panel_text(at), sc["id"]
        if t["status"] == "sceptic_reject":
            assert any(x.value.startswith("VETO") for x in at.error) and "Counterfactual" in panel_text(at)
        at.sidebar.toggle[0].set_value(True).run()
        assert any(b.label == "Reveal game outcome" for b in at.button), sc["id"]
        assert f"{go['away']} {go['away_pts']} – {go['home_pts']} {go['home']}" not in panel_text(at)
        at.sidebar.toggle[0].set_value(False).run()


def test_results_json_has_todays_numbers():
    r = load("results")
    by = {w["predictor"]: w for w in r["win_models"]}
    assert by["Always 55% home"]["brier"] == pytest.approx(0.2468, abs=5e-4)
    assert by["Always 55% home"]["accuracy"] == pytest.approx(0.557, abs=5e-4)
    assert by["M4 (logistic regression, absences known)"]["brier"] == pytest.approx(0.1827, abs=5e-4)
    assert by["M4 (logistic regression, absences known)"]["accuracy"] == pytest.approx(0.741, abs=5e-4)
    assert by["M4-NN MLP (3-seed mean)"]["brier"] == pytest.approx(0.1842, abs=5e-4)
    assert by["M4-NN GRU (3-seed mean)"]["brier"] == pytest.approx(0.1864, abs=5e-4)
    assert by["Anchor: market 24 h before tip"]["brier"] == pytest.approx(0.1676, abs=5e-4)
    assert by["Anchor + M4 shift (the agent's estimate)"]["brier"] == pytest.approx(0.1663, abs=5e-4)
    assert by["Anchor + MLP shift"]["brier"] == pytest.approx(0.1632, abs=5e-4)
    assert by["Kalshi 1 h before tip"]["brier"] == pytest.approx(0.1641, abs=5e-4)
    assert by["Kalshi at tip"]["brier"] == pytest.approx(0.1633, abs=5e-4)
    assert by["Kalshi close (venue comparison)"]["brier"] == pytest.approx(0.1614, abs=5e-4)
    assert by["Polymarket close"]["brier"] == pytest.approx(0.1618, abs=5e-4)
    assert by["DraftKings close (de-vigged)"]["brier"] == pytest.approx(0.1629, abs=5e-4)
    assert by["Learned blend (all inputs, walk-forward)"]["brier"] == pytest.approx(0.1654, abs=5e-4)
    assert all(w["games"] in (501, 498) for w in r["win_models"])
    m3 = next(x for x in r["players"]["m3"] if x["predictor"] == "M3 model")
    assert m3["brier"] == pytest.approx(0.1433, abs=5e-4)
    assert r["players"]["m6_agent_trades_test"] == "0"
    eg = r["edge_gate"]
    assert [s["trades"] for s in eg["sweep"]] == [188, 145, 108, 74, 53]
    assert [s["clv_dollars"] for s in eg["sweep"]] == [-82, -64, -45, -28, -19]
    assert eg["a_paired_clv"] == "-63 [-104, -32]"
    assert eg["d0"]["vetoes"] == "107" and eg["d0"]["calls"] == "110" and eg["d0"]["trades"] == "3"
    ae = r["adaptive_edge"]
    assert ae["vs_none"] == "+21 [+4, +38]" and ae["vs_split"] == "+3 [+0, +6]"
    assert [s["accepted"] for s in ae["story"]] == [False, True, False]
    assert any("10¢" in t for t in ae["timeline"]) and any("4¢ from 2026-04-12" in t for t in ae["timeline"])
    fb = r["forecast_blend"]
    assert fb["counts"] == {"accepted": 6, "rejected": 38, "deferred": 20}
    assert fb["blend_vs_agent_clv"].startswith("+18") and "−$176" not in fb["nomarket_clv"]
    assert fb["nomarket_clv"].startswith("-176") and fb["nomarket_pnl"].startswith("-2003")
    assert len(fb["trajectory"]) == 64 and len(fb["accepted"]) == 6
    lv = r["llm_vs_original"]
    assert lv["random_keep_pct_c"] == "54%" and lv["random_keep_pct_d"] == "87%"
    assert any(x["LLM − mid"] == "+9.4¢" and x["A − mid"] == "+4.5¢" for x in lv["overconfidence"])
    assert len(lv["veto_reasons"]) >= 3 and len(r["takeaways"]) >= 4
    assert "full" in r["agents"] and "subsample" in r["agents"]
    assert any("Raw M4" in a["setup"] for a in r["agents"]["full"])
    assert any("no edge gate" in a["setup"] for a in r["agents"]["subsample"])
    assert not SECRET.search(json.dumps(r))
    text = json.dumps(r)
    assert "NaN" not in text and '"nan"' not in text.lower()
    for name in ("edge_gate.png", "adaptive_edge.png", "forecast_blend.png", "llm_vs_original.png"):
        assert (TRACES.parent / "figures" / name).exists(), name


def test_results_and_learning_pages_render():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert list(at.sidebar.radio[0].options) == ["Overview", "Results", "Models", "Workflows", "Learning",
                                                 "Scenarios"]
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception
    assert any("Key takeaways" in s.value for s in at.subheader)
    assert any("Architecture" in s.value for s in at.subheader)
    texts = "\n".join(str(getattr(e, "value", "")) for e in at.main)
    assert "nan" not in texts.lower().split()
    at.sidebar.radio[0].set_value("Results").run()
    assert not at.exception
    assert any("Win probability" in h.value for h in at.header)
    assert any("Edge gate" in h.value for h in at.header)
    assert any("Agents: trading results" in h.value for h in at.header)
    assert any("Player models" in h.value for h in at.header)
    assert any(m.label.startswith("Paired difference") and "-63" in str(m.value) for m in at.metric)
    assert len(at.dataframe) >= 4 and len(at.image) >= 1
    texts = "\n".join(str(getattr(e, "value", "")) for e in at.main)
    assert "nan" not in texts.lower().split()
    at.sidebar.radio[0].set_value("Learning").run()
    assert not at.exception
    assert any("Learned edge threshold" in h.value for h in at.header)
    assert any("Model blend" in h.value for h in at.header)
    assert any(m.label.startswith("Learned threshold") and "+21" in str(m.value) for m in at.metric)
    assert any(m.label.startswith("Blend − agent") for m in at.metric)
    assert len(at.image) >= 1
    texts = "\n".join(str(getattr(e, "value", "")) for e in at.main)
    assert "nan" not in texts.lower().split()
    assert any("Model review" in h.value for h in at.header)
    assert any("Better Brier ≠ better trading" in str(e.value) for e in at.error)
    at.sidebar.radio[0].set_value("Overview").run()
    warn_info = [str(getattr(w, "value", "")) for w in list(at.warning) + list(at.info)]
    assert any("trading less" in t.lower() or "no better than random" in t.lower() for t in warn_info)


def test_models_review_pnl_workflow_json():
    r = load("results")
    cards = {c["name"]: c for c in r["models"]["cards"]}
    assert r["models"]["m4_auc"] == pytest.approx(0.801, abs=1e-3)
    m4 = cards["M4 win model (logistic regression)"]
    assert "0.1827" in m4["metric"] and "74.1%" in m4["metric"] and "0.1633" in m4["metric"]
    assert "+0.234" in m4["architecture"] and "-0.288" in m4["architecture"]
    assert "1.553" in cards["M2 GRU (deep learning)"]["metric"] and "80.9%" in cards["M2 GRU (deep learning)"]["metric"]
    assert "0.1433" in cards["M3 play model"]["metric"] and "0.1686" in cards["M3 play model"]["metric"]
    assert "σ = 14" in cards["In-play model"]["architecture"]
    assert len(cards) >= 9
    mr = r["model_review"]
    assert mr["proposed"] == 6 and mr["accepted"] == 0
    assert mr["story"][0]["clv_without"] == pytest.approx(-5.70) and mr["story"][0]["clv_with"] == pytest.approx(-18.24)
    assert mr["review_rules"]["Trades"] == "53" and mr["review_rules"]["CLV $ [95% CI]"] == "-18 [-34, -2]"
    assert mr["review_edge"]["Trades"] == "42"
    p = r["pnl"]
    assert p["trades"] == {"Raw M4, no agent": 366, "Anchor agent, no learning": 129, "Split-gate learning": 53,
                           "Adaptive edge, base 4¢": 42, "Never trade": 0}
    assert round(p["final"]["Raw M4, no agent"]) == -2087 and round(p["final"]["Anchor agent, no learning"]) == -247
    assert round(p["final"]["Split-gate learning"]) == -96 and round(p["final"]["Adaptive edge, base 4¢"]) == -102
    assert all(len(v) == len(p["days"]) for v in p["series"].values())
    fw = [f["framework"] for f in r["frameworks"]]
    assert fw[0].startswith("Raw M4") and fw[-1] == "Never trade" and any("Model review" in f for f in fw)
    assert {f["window"] for f in r["frameworks"]} >= {"full test, full agent replay",
                                                      "full test, simplified offline trader",
                                                      "40% subsample (304 decision points)"}
    w = r["workflow"]
    assert len(w["tools"]) == 10 and {"citation_after_decision", "banned_wording", "rule_not_active"} <= set(w["checks"])
    assert "kill_switch" in w["risk_reasons"] and w["caps"] == [50, 100, 300] and w["max_quote_age_min"] == 5
    assert any("Better Brier" in t for t in r["takeaways"])
    text = json.dumps(r).lower().replace("spending", "")
    assert "pending" not in text and not re.search(r"\bnan\b", text)
    for name in ("model_review.png", "model_review_loop.png", "pnl_compare.png"):
        assert (TRACES.parent / "figures" / name).exists(), name


def test_models_and_workflows_pages_render():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Models").run()
    assert not at.exception
    assert any("How each framework trades" in h.value for h in at.header)
    texts = "\n".join(str(getattr(e, "value", "")) for e in at.main)
    assert "AUC 0.801" in texts and "M6 ImpactNet" in texts and not re.search(r"\bnan\b", texts.lower())
    at.sidebar.radio[0].set_value("Workflows").run()
    assert not at.exception
    subs = [s.value for s in at.subheader]
    assert sum(s[0].isdigit() for s in subs) >= 7
    texts = "\n".join(str(getattr(e, "value", "")) for e in at.main)
    assert "rejected 6 / 6" in texts and "citation_after_decision" in texts and "kill_switch" in texts
    at.sidebar.radio[0].set_value("Results").run()
    assert not at.exception
    assert any("Cumulative P&L" in str(getattr(e, "value", "")) for e in at.main)
