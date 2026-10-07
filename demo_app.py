"""Hosted demo: the thought process of every agent, replayed from precomputed traces.

    streamlit run demo_app.py

Reads only demo/traces/*.json (built by demo/build_traces.py from as-of data), so it needs no data,
models or API keys and runs on Streamlit Community Cloud.
"""
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
TRACES = ROOT / "demo" / "traces"
FIGURE = ROOT / "presentation" / "figures" / "agent_architecture.png"

AGENT_NAMES = {"trader": "Trader (LangGraph MarketAgent)", "tool_agent": "LLM tool agent + priced-in sceptic",
               "coach": "Coach agent", "orchestrator": "Orchestrator (one night)", "review": "Trader review loop",
               "pregame": "Pregame news agent", "inplay": "In-play agent"}
EXTRA = {"pregame": "pregame_night", "inplay": "inplay_cle_por"}

st.set_page_config(page_title="NBA market agents: thought process", layout="wide")


@st.cache_data
def load(name: str):
    path = TRACES / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def cents(x, d=1):
    return "n/a" if x is None else f"{x * 100:+.{d}f}c"


def badge(ok: bool, text: str):
    (st.success if ok else st.error)(("PASS  " if ok else "FAIL  ") + text)


def graph(nodes, edges, path, title=""):
    taken = set(zip(path, path[1:])) | ({("START", path[0])} if path else set())
    if path and path[-1] in ("deliver", "save_rule", "reject_rule", "review", "trigger"):
        taken.add((path[-1], "END"))
    visited = set(path)
    lines = ["digraph G {", 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             'START [shape=circle, fillcolor="#dddddd", label="start"]; END [shape=doublecircle, fillcolor="#dddddd", label="end"];']
    for n in nodes:
        fill = "#ffd27f" if n == "blocked" and n in visited else "#9be7a1" if n in visited else "#f2f2f2"
        lines.append(f'"{n}" [fillcolor="{fill}"];')
    for a, b in edges:
        on = (a, b) in taken
        lines.append(f'"{a}" -> "{b}" [color="{"#d62728" if on else "#bbbbbb"}", penwidth={3 if on else 1}];')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), width="stretch")


def settlement_panel(s, proposed_label="the order"):
    st.subheader("Graded by the market (revealed at tip-off)")
    st.caption(f"Nothing below was visible to the agent. Revealed at {s.get('revealed_at')}.")
    reveal = st.toggle("Reveal closing price, CLV and P&L", value=False, key=f"rev{id(s)}{proposed_label}")
    if not reveal:
        return
    fills = s.get("fills") or []
    if not fills and s.get("counterfactual"):
        st.info("No order was placed. Counterfactual: what the blocked or rejected order would have done.")
        fills = s["counterfactual"]
    if not fills:
        st.info("No order, so nothing to grade: passing costs nothing.")
    for f in fills:
        if "price" not in f:
            st.write(f"{f.get('ticker')} {f.get('side')}: not filled ({f.get('fill')})")
            continue
        c = st.columns(5)
        c[0].metric("Bought", f"{f['side']} @ {f['price'] * 100:.1f}c", f"{f['contracts']} contracts")
        c[1].metric("Close (tip-off mid)", f"{f['close_price'] * 100:.1f}c")
        c[2].metric("CLV", cents(f["clv"]))
        c[3].metric("P&L", f"${f['pnl']:+.2f}" if f.get("pnl") is not None else "n/a", f"fee ${f['fee_dollars']:.2f}")
        c[4].metric("Contract outcome", "yes" if f.get("outcome_yes") == 1 else "no")
    fin = s.get("final") or {}
    if fin:
        st.write(f"Final score: {fin['away']} {fin['away_pts']} @ {fin['home']} {fin['home_pts']}")


# ---------------- trader ----------------

def trader_step(s):
    node = s["node"]
    if node == "trigger":
        st.write(f"Markets with fresh quotes: {', '.join(s.get('markets_with_fresh_quotes', [])) or 'none'}")
        if s.get("news"):
            st.markdown("**Inputs: injury news public at the decision time**")
            st.dataframe([{k: n[k] for k in ("published_at", "player", "status", "text")} for n in s["news"]],
                         hide_index=True, width="stretch")
    elif node == "investigate":
        st.write("Players ruled out: " + (", ".join(p["player"] for p in s.get("players_out", [])) or "none"))
        st.write(f"Rules applied (forecast): {s.get('rules_applied') or 'none'}; note: {s.get('note')}")
        st.caption("Citations: " + ", ".join(s.get("citations", [])))
    elif node == "forecast":
        st.markdown("**M4 win model before and after the news**")
        st.dataframe([{"team": t, "before news": pct(v["before_news"]), "after news": pct(v["after_news"]),
                       "shift": f"{v['shift'] * 100:+.1f} pts"} for t, v in s.get("m4", {}).items()],
                     hide_index=True, width="stretch")
    elif node == "analyse":
        st.markdown("**Anchor + news shift vs the price, after fees**")
        st.dataframe([{"team": c["team"], "anchor (24h before)": pct(c["anchor_24h"]),
                       "news shift": f"{c['news_shift'] * 100:+.1f} pts", "estimate": pct(c["estimate_p"]),
                       "bid-ask": f"{c['bid'] * 100:.0f}-{c['ask'] * 100:.0f}c", "side": c["side"],
                       "price paid": f"{c['price_paid'] * 100:.0f}c", "fee": f"{c['fee'] * 100:.2f}c",
                       "gap after fees": f"{c['gap_after_fees'] * 100:+.1f}c", "min edge": f"{c['min_edge'] * 100:.0f}c",
                       "rules": ", ".join(c["rules"]) or "-", "act": "YES" if c["act"] else "no",
                       "why not": c["why_not"] or "-"} for c in s.get("candidates", [])],
                     hide_index=True, width="stretch")
    elif node in ("propose", "no_action"):
        for o in s.get("orders", []):
            st.markdown(f"**Order:** buy `{o['side']}` on `{o['market_ticker']}`, stake ${o['stake']:.0f}, "
                        f"p_model {o['p_model']:.3f}")
            st.caption(o["reason"])
        if s.get("brief"):
            st.markdown("**Brief**")
            st.code(s["brief"], language=None)
    elif node == "checks":
        st.markdown(f"**Guardrail checks (try {s.get('tries', 0) + (0 if all(c['pass'] for c in s['checks']) else 0)})**")
        for c in s.get("checks", []):
            badge(c["pass"], c["check"].replace("_", " ") + (f": {c['detail']}" if c["detail"] else ""))
    elif node == "risk":
        st.write(f"Orders kept after risk limits: {s.get('kept_orders') or 'none'}")
    elif node in ("confirm", "blocked", "deliver"):
        if node == "blocked":
            st.error(f"Blocked: {' '.join(s.get('log', []))}")
        st.write(f"Status: **{s.get('status')}**")


def show_trader(t):
    st.markdown(f"### {t['label']}")
    c = st.columns(4)
    c[0].metric("Game", t["matchup"])
    c[1].metric("Decision time (UTC)", t["as_of"][5:16].replace("T", " "))
    c[2].metric("Tip-off (UTC)", t["tip_time"][5:16].replace("T", " "))
    c[3].metric("Outcome", t["status"])
    if t.get("plant"):
        st.warning(f"Fault injected for the demo: {t['plant']}")
    if t.get("kill_switch_tripped"):
        st.warning("Kill switch already tripped today (realised losses over the limit).")
    if t.get("rule"):
        r = t["rule"]
        st.info(f"Learned rule {r['rule_id']} (active from {r.get('valid_from')}): when {r['when']} do "
                f"{r['do']['action']}. Reviewer's rationale: {r.get('rationale')}")
    st.markdown("**LangGraph path taken** (green = visited, red = edges taken)")
    graph(t["nodes"], t["edges"], t["path"])
    st.markdown("**Step-by-step**")
    for i, s in enumerate(t["steps"]):
        fail = s["node"] == "blocked" or (s["node"] == "checks" and not all(c["pass"] for c in s["checks"]))
        icon = "🛑" if fail else "✅" if s["node"] in ("checks", "confirm") else "▶"
        with st.expander(f"{icon} {i + 1}. {s['node']}: {' | '.join(s.get('log', []))[:140]}",
                         expanded=s["node"] in ("analyse", "checks", "blocked")):
            trader_step(s)
            with st.popover("raw JSON"):
                st.json(s)
    settlement_panel(t["settlement"], t["label"])


def show_review(r):
    st.markdown("### Review loop: settle → review → gate → save or reject")
    st.caption(f"Real review-phase traces from {r['source']}. The gate back-tests each proposed rule on earlier "
               "days only and keeps it only if it helps.")
    for key, title in (("save_rule", "Rule accepted"), ("reject_rule", "Rule rejected")):
        ex = r.get(key)
        if not ex:
            continue
        st.markdown(f"#### {title} (review after {ex['day']})")
        graph(r["nodes"], r["edges"], ex["path"])
        for s in ex["steps"]:
            with st.expander(s["node"], expanded=s["node"] in ("gate", "review")):
                for line in s["log"]:
                    st.code(line, language=None)


# ---------------- tool agent ----------------

def show_tool(t):
    st.markdown(f"### {t['label']}")
    if t.get("stub"):
        st.warning("**stub LLM**: every LLM reply below comes from the deterministic `heuristic_reply` stub, not "
                   "Gemini. The tools, validation, edge and risk code are the real ones.")
    else:
        st.success(f"Real LLM: {t['llm']}.")
    c = st.columns(4)
    c[0].metric("Game", t["matchup"])
    c[1].metric("Decision time (UTC)", t["as_of"][5:16].replace("T", " "))
    c[2].metric("Outcome", t["status"])
    c[3].metric("Gap after fees", cents(t.get("gap_after_fees")) if t.get("gap_after_fees") is not None else "n/a")
    st.markdown("**1. Analyst LLM ↔ as-of tools**")
    for turn in t["turns"]:
        reply = turn.get("reply") or {}
        with st.expander(f"Turn {turn['turn']}: " + ("tool calls" if "tool_calls" in reply else "final proposal"),
                         expanded=True):
            st.markdown("LLM reply:")
            st.json(reply)
            for r in turn.get("tool_results", []):
                st.markdown(f"`{r.get('id')}` **{r.get('tool')}**({json.dumps(r.get('args'))}) →")
                st.json(r.get("output"), expanded=False)
    st.markdown("**2. Proposal**")
    st.json(t.get("proposal") or {})
    st.markdown("**3. Validation in code (number matching and grounding)**")
    v = t.get("validation")
    if v:
        for c_ in v["checks"]:
            badge(c_["pass"], c_["check"])
        if v.get("detail"):
            st.caption(v["detail"])
    else:
        st.info("The analyst passed, so there was nothing to validate. Passing costs nothing.")
    st.markdown("**4. Edge (code): estimate minus all-in cost must exceed 4c**")
    if t.get("gap_after_fees") is not None:
        badge(t["gap_after_fees"] > 0.04, f"gap after fees {cents(t['gap_after_fees'])}")
    st.markdown("**5. Priced-in sceptic**")
    sc = t.get("sceptic")
    if sc:
        verdict = (sc.get("verdict") or {}).get("verdict")
        badge(verdict == "approve", f"{verdict}: {(sc.get('verdict') or {}).get('reason')}")
    else:
        st.info("Sceptic not reached (no candidate trade)" if t.get("sceptic_enabled") else "Sceptic not enabled in this run.")
    st.markdown("**6. Result**")
    st.write(f"{t['status']}: {t.get('reason')}")
    if t.get("order"):
        st.json(t["order"])
    if t.get("usage"):
        st.caption(f"LLM usage: {t['usage']}")
    settlement_panel(t["settlement"], t["label"])


# ---------------- coach ----------------

def show_coach(c):
    st.markdown(f"### {c['label']}")
    s = c["snapshot"]
    st.write(f"Decision time {c['as_of']}. Model p(home {s['home']}): {pct(s['p_home_before'])} before news → "
             f"{pct(s['p_home_after'])} after (out: {s.get('out')}).")
    with st.expander("Coach's game explanation", expanded=False):
        for line in c["explanation"]:
            st.write("• " + line)
    labels = [f"User wants {p['team']} ({p['history']})" for p in c["picks"]]
    pick = c["picks"][labels.index(st.radio("Pending pick", labels))]
    nudge = pick["nudge"]
    {"pass": st.error, "caution": st.warning, "none": st.success}[nudge](f"Nudge: **{nudge}**")
    st.markdown("**Checks the Coach chose to run, and why**")
    for s_ in pick["steps"]:
        finding = "warning" if s_["severe"] else "note" if s_["flag"] else "ok"
        with st.expander(f"{'🛑' if s_['severe'] else '⚠️' if s_['flag'] else '✅'} {s_['step']}. "
                         f"{s_['check'].replace('_', ' ')}: {finding}", expanded=True):
            st.write(f"Why: {s_['why']}")
            st.json(s_["result"], expanded=False)
    if pick["skipped"]:
        st.caption("Skipped (not relevant to this pick): " + ", ".join(pick["skipped"]))
    st.markdown(f"**Lesson:** {pick['lesson']['title']}")
    st.caption(pick["lesson"]["body"])
    st.markdown("**Message to the user**")
    st.info(pick["message"])


# ---------------- orchestrator ----------------

def show_orchestrator(o):
    n = o["night"]
    st.markdown(f"### Orchestrator: night of {n['date']}")
    st.caption("A rule-based LangGraph supervisor routes pregame → trader → Coach → briefs. All workers see the "
               "same as-of moment; a failing worker becomes an error message and later agents degrade.")
    nodes = ["supervisor", "pregame", "trader", "coach", "briefs"]
    lines = ['digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#9be7a1", fontname="Helvetica"];',
             'START [shape=circle, label="start"]; END [shape=doublecircle, label="end"]; START -> supervisor;']
    for w in nodes[1:]:
        lines.append(f'supervisor -> {w} [color="#d62728"]; {w} -> supervisor;')
    lines.append("supervisor -> END; }")
    st.graphviz_chart("\n".join(lines))
    st.write("Games: " + ", ".join(f"{g['matchup']} (decides {g['as_of'][11:16]} UTC)" for g in n["games"]))
    st.markdown("**Shared-state message log (handoffs)**")
    for i, m in enumerate(n["messages"]):
        kind = m["kind"]
        head = f"{i + 1}. {m['sender']} → {m['receiver']} [{kind}]" + (f" game {m['game_id']}" if m.get("game_id") else "")
        with st.expander(head, expanded=kind in ("error", "degraded")):
            st.json(m["content"]) if isinstance(m["content"], (dict, list)) else st.write(m["content"])
    for gid, t in (n.get("trader") or {}).items():
        with st.expander(f"Trader detail for game {gid}: {t.get('status')} ({t.get('input')})"):
            st.code(t.get("brief", ""), language=None)
            st.json(t.get("orders"))
    for gid, b in (n.get("briefs") or {}).items():
        with st.expander(f"Channel briefs for game {gid}"):
            st.json(b)


# ---------------- pregame / in-play ----------------

def show_pregame(p):
    st.markdown(f"### {p['label']}")
    st.caption("LangGraph per poll: baseline → retrieve news → extract factors → reforecast → record. Polls the "
               "news table hourly up to the decision time.")
    for gid, g in p["games"].items():
        st.markdown(f"#### {g['matchup']} (tip {g['tip_time'][11:16]} UTC)")
        st.dataframe([{"as_of": q["as_of"][5:16], "p_home": pct(q["p_home"]),
                       "baseline": pct(q["baseline_p_home"]),
                       "delta": f"{(q['delta_from_baseline'] or 0) * 100:+.1f} pts",
                       "factors": "; ".join(f"{f['player']} {f['status']}" for f in q["factors"]) or "-",
                       "news health": q["news_health"]} for q in g["polls"]], hide_index=True, width="stretch")


def show_inplay(p):
    st.markdown(f"### {p['label']}")
    st.warning("SYNTHETIC in-game score and news script over historical rosters (data_sources/inplay_demo.py); "
               "not historical play-by-play.")
    st.dataframe([{"as_of": q["as_of"][11:19], "state": q["quote_state"],
                   "score (home-away)": f"{q['score'].get('home_score')}-{q['score'].get('away_score')}",
                   "period": q["score"].get("period"), "p_home": pct(q["p_home"]),
                   "news margin (pts)": q.get("news_margin"), "new events": "; ".join(q["new_events"]) or "-",
                   "factors": "; ".join(f"{f['player']} {f['status']}" for f in q["factors"]) or "-"}
                  for q in p["polls"]], hide_index=True, width="stretch")


RENDER = {"trader": show_trader, "tool_agent": show_tool, "coach": show_coach, "orchestrator": show_orchestrator,
          "review": show_review, "pregame": show_pregame, "inplay": show_inplay}


# ---------------- pages ----------------

def overview():
    st.title("NBA prediction-market agents: how they think")
    st.write("Every panel replays a real agent run on as-of data from the 2025-26 test period. Decision steps show "
             "only what was public at the decision time; the market's verdict (closing price, CLV, P&L) is revealed "
             "separately.")
    if FIGURE.exists():
        st.image(str(FIGURE), caption="Agent architecture", width="stretch")
    st.markdown("""
| Agent | What it does | Where its thinking shows |
|---|---|---|
| Trader (`agents/graph.py`) | LangGraph: news → M4 forecast → anchor + shift → gap after fees → checks → risk → order | Trader tab: graph path, every node, guardrails |
| Reviewer + gate | Learns rules from settled trades; the gate keeps a rule only if it helps on earlier days | Trader review loop tab |
| LLM tool agent + sceptic (`agents/tool_agent.py`) | LLM calls as-of tools, proposes; code validates numbers; sceptic asks "already priced in?" | LLM tool agent tab |
| Coach (`agents/coach_agent.py`) | Before a user's pick: chooses checks, teaches a lesson, nudges pass or caution | Coach tab |
| Orchestrator (`agents/orchestrator.py`) | Supervisor routes pregame → trader → Coach → briefs for one night | Orchestrator tab |
| Pregame / in-play (`agents/pregame.py`, `agents/inplay.py`) | News polling and fair odds before and during games | Pregame and In-play tabs |
""")


def main():
    index = load("index")
    if not index:
        st.error("No traces found in demo/traces/. Run: PYTHONPATH=. python demo/build_traces.py")
        return
    page = st.sidebar.radio("Page", ["Overview", "Scenarios"])
    if page == "Overview":
        overview()
        return
    scenarios = index["scenarios"]
    titles = [s["title"] for s in scenarios]
    sc = scenarios[titles.index(st.sidebar.selectbox("Scenario / game", titles))]
    agents = dict(sc["agents"])
    for k, v in EXTRA.items():
        if load(v):
            agents.setdefault(k, v)
    keys = list(agents)
    agent = st.sidebar.radio("Agent", keys, format_func=lambda k: AGENT_NAMES.get(k, k))
    st.sidebar.caption(index.get("note", ""))
    st.title(sc["title"])
    st.write(sc["summary"])
    data = load(agents[agent])
    if data is None:
        st.error("Trace missing for this agent.")
        return
    RENDER[agent](data)


main()
