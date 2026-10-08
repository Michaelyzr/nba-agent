"""Hosted demo: the thought process of every agent, replayed from precomputed traces.

    streamlit run research_demo_app.py

Reads only demo/traces/*.json (built by demo/build_traces.py from as-of data), so it needs no data,
models or API keys and runs on Streamlit Community Cloud.
"""
import json
from pathlib import Path

import streamlit as st

import pandas as pd

ROOT = Path(__file__).resolve().parent
TRACES = ROOT / "demo" / "traces"
FIGURES = ROOT / "demo" / "figures"

AGENT_NAMES = {"trader": "Trader (LangGraph MarketAgent)", "tool_agent": "LLM tool agent + priced-in sceptic",
               "coach": "Coach agent", "orchestrator": "Orchestrator (one night)", "review": "Trader review loop",
               "pregame": "Pregame news agent", "inplay": "In-play agent"}
EXTRA = {"pregame": "pregame_night", "inplay": "inplay_cle_por"}

st.set_page_config(page_title="NBA market agents: thought process", layout="wide")


@st.cache_data
def _load(name: str, mtime: float):
    return json.loads((TRACES / f"{name}.json").read_text())


def load(name: str):
    path = TRACES / f"{name}.json"
    return _load(name, path.stat().st_mtime) if path.exists() else None


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def dollars(x):
    return "n/a" if x is None else f"{'-' if x < 0 else '+'}${abs(x):.2f}"


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


# ---------------- graded by the market (outcome + settlement) ----------------

def game_outcomes(agent, data):
    if agent == "orchestrator":
        return list((data.get("game_outcomes") or {}).values())
    if agent == "pregame":
        return [g["game_outcome"] for g in data.get("games", {}).values() if g.get("game_outcome")]
    return [data["game_outcome"]] if data.get("game_outcome") else []


def final_score(go):
    return f"{go['away']} {go['away_pts']} – {go['home_pts']} {go['home']}"


def outcome_block(go, synthetic=False):
    with st.container(border=True):
        ot = go.get("overtime_periods")
        ot_txt = "" if not ot else f" ({ot}OT)" if ot > 1 else " (OT)"
        winner = go.get("winner")
        loser = go["home"] if winner == go["away"] else go["away"]
        st.markdown(f"#### {go['matchup']}: final score **{final_score(go)}**{ot_txt}")
        st.markdown(f"Winner: **{winner}** by {go.get('margin')}{ot_txt}" if winner else "Result: tie")
        mk = go.get("markets") or []
        if mk:
            st.write("Kalshi settlement: " + "; ".join(
                f"{m['team']} YES paid ${m['settled_yes']}, NO paid ${1 - m['settled_yes']}" for m in mk))
        pos = go.get("positions") or []
        for p in pos:
            won = p["result"] == "won"
            settle = (f"{p['team']} {p['side'].upper()} {p['result']} → paid ${p['settled_value']:.0f}/contract")
            if p["counterfactual"]:
                st.info(f"No trade placed. Counterfactual (not traded): buying {p['team']} {p['side'].upper()} at "
                        f"{p['price'] * 100:.0f}c ({p['contracts']} contracts) would have {p['result']} "
                        f"${abs(p['pnl']):.2f} after fees ({settle}; CLV {cents(p.get('clv'))}).")
            else:
                (st.success if won else st.error)(f"Bet {'WON' if won else 'LOST'}: {settle}.")
            label = " (counterfactual)" if p["counterfactual"] else ""
            c = st.columns(4)
            c[0].metric("Counterfactual bet (not placed)" if p["counterfactual"] else "Agent's bet",
                        f"{p['team']} {p['side']} @ {p['price'] * 100:.0f}c", f"{p['contracts']} contracts",
                        delta_color="off")
            c[1].metric("Settled", f"${p['settled_value']:.0f}/contract", "WON" if won else "LOST",
                        delta_color="normal" if won else "inverse")
            c[2].metric("P&L after fees" + label, dollars(p["pnl"]))
            c[3].metric("CLV" + label, cents(p.get("clv")))
        if not pos:
            st.info("No trade placed, so no P&L: passing costs nothing.")
            if winner:
                st.write(f"A pick on **{winner}** would have won; a pick on **{loser}** would have lost.")
        if synthetic:
            st.warning("The in-play score script on this page is SYNTHETIC (not historical play-by-play); "
                       "this is the real final result of the game.")
        elif go.get("note"):
            st.caption(go["note"])
        st.caption(f"Final at {go.get('final_at', '')[:16].replace('T', ' ')} UTC. Source: {go.get('source')}.")


def fills_table(s):
    fills = s.get("fills") or []
    counterfactual = not fills and bool(s.get("counterfactual"))
    if counterfactual:
        fills = s["counterfactual"]
    if not fills:
        return
    st.markdown("**Closing price and CLV**" + (" (counterfactual: the order was not placed)" if counterfactual else ""))
    for f in fills:
        if "price" not in f:
            st.write(f"{f.get('ticker')} {f.get('side')}: not filled ({f.get('fill')})")
            continue
        c = st.columns(5)
        c[0].metric("Bought" + (" (counterfactual)" if counterfactual else ""),
                    f"{f['side']} @ {f['price'] * 100:.1f}c", f"{f['contracts']} contracts")
        c[1].metric("Close (tip-off mid)", f"{f['close_price'] * 100:.1f}c")
        c[2].metric("CLV", cents(f["clv"]))
        c[3].metric("P&L after fees", dollars(f.get("pnl")),
                    f"fee ${f['fee_dollars']:.2f}")
        c[4].metric("Contract outcome", "yes" if f.get("outcome_yes") == 1 else "no")


def graded_panel(outcomes, settlement, key, synthetic=False):
    if not outcomes and not settlement:
        return
    st.divider()
    st.subheader("Graded by the market (revealed at tip-off)")
    revealed = (settlement or {}).get("revealed_at")
    st.caption("Nothing below was visible to the agent at decision time"
               + (f". Revealed at {revealed}." if revealed else "."))
    if st.session_state.get("present") and not st.session_state.get(key):
        st.caption("Hidden in presentation mode. The agent decided before this was known.")
        if st.button("Reveal game outcome", key=f"btn_{key}"):
            st.session_state[key] = True
            st.rerun()
        return
    for go in outcomes:
        outcome_block(go, synthetic)
    if settlement:
        fills_table(settlement)


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
    with st.expander("Where this run sits in the full agent: decide loop, learn loop, blend, LLM add-ons"):
        arch_diagram()
        st.caption("This replay is the published trader: anchor + M4 shift, fixed 4¢ edge plus any notebook rules, "
                   "no LLM and no blend.")
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


def show_review(r):
    st.markdown("### Review loop: settle → review → gate → save or reject")
    st.caption(f"Real review-phase traces from {r['source']}. The gate back-tests each proposed rule on earlier "
               "days only and keeps it only if it helps. With the new edge-learning templates the same loop also "
               "learns the edge threshold itself: see the Learning page.")
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

STATUS_TEXT = {"order": "order placed", "sceptic_reject": "vetoed by the sceptic", "pass": "pass",
               "invalid": "invalid output (no trade)", "gap": "gap too small (no trade)"}


def show_deepseek(t):
    st.markdown(f"### {t['label']}")
    st.success(f"Real LLM: **{t['model_label']}** — {t['setup_label']}. Every reply below is the model's real "
               f"response, replayed from the run's response cache (`{t['run']}`); tools, validation, edge, sceptic "
               "wiring and risk are the production code.")
    c = st.columns(4)
    c[0].metric("Game", t["matchup"])
    c[1].metric("Decision time (UTC, as-of)", t["as_of"][5:16].replace("T", " "))
    c[2].metric("Tip-off (UTC)", t["tip_time"][5:16].replace("T", " "))
    c[3].metric("Outcome", STATUS_TEXT.get(t["status"], t["status"]))
    st.markdown("**Market at the decision time (quote and 24h anchor)**")
    st.dataframe([{"team": m["team"], "bid-ask": f"{m['bid'] * 100:.0f}-{m['ask'] * 100:.0f}c",
                   "mid": pct(m.get("mid")), "anchor (24h before tip)": pct(m.get("anchor_mid")),
                   "move since anchor": cents(m.get("move_since_anchor")),
                   "volume last hour": f"{m.get('volume_last_hour') or 0:,.0f}"} for m in t["market"] if "bid" in m],
                 hide_index=True, width="stretch")
    rs = t.get("reasoning")
    step = 1
    if t["setup"] == "plain":
        st.markdown(f"**{step}. One LLM call, no tools: the information it was given**")
        with st.expander("Information shown to the LLM (as-of)", expanded=False):
            st.json(t.get("plain_info") or {}, expanded=False)
    else:
        st.markdown(f"**{step}. Analyst LLM ↔ as-of tools** (in call order)")
        for turn in t["turns"]:
            reply = turn.get("reply")
            kind = "tool calls" if reply and "tool_calls" in reply else "final proposal" if reply else "no answer"
            thought = f" — thought for {turn['reasoning_chars']:,} characters" if rs else ""
            with st.expander(f"Turn {turn['turn']}: {kind}{thought}", expanded=True):
                if reply is None:
                    st.error(f"Empty or unparseable reply ({turn.get('tokens_out', 0):,} output tokens, "
                             f"{turn.get('reasoning_tokens', 0):,} of them hidden reasoning). Raw text: "
                             f"{turn.get('raw_text') or '(empty)'!r}")
                for r in turn.get("tool_results", []):
                    st.markdown(f"`{r.get('id')}` **{r.get('tool')}**(`{json.dumps(r.get('args'))}`) →")
                    st.json(r.get("output"), expanded=False)
                if reply and "final" in reply:
                    st.json(reply, expanded=False)
    step += 1
    if rs:
        st.caption(f"Hidden reasoning (thinking mode): {rs['chars']:,} characters, {rs['tokens']:,} tokens over "
                   f"this decision. {rs['excerpt_note']}")
    st.markdown(f"**{step}. Proposal**")
    p = t.get("proposal")
    if p and t["setup"] == "plain":
        st.write(f"Estimate p(home wins) = **{pct(p.get('p_home'))}**. Rationale: {p.get('rationale')}")
    elif p:
        st.write(f"Decision **{str(p.get('decision', '')).upper()}**"
                 + (f" {p.get('team')}, estimate **{pct(p.get('estimate_p'))}**" if p.get("team") else ""))
        if p.get("citations"):
            st.dataframe([{"call": x.get("call_id"), "field": x.get("field"), "value cited": x.get("value")}
                          for x in p["citations"] if isinstance(x, dict)], hide_index=True, width="stretch")
        st.caption(f"Reasoning note: {p.get('rationale')}")
    else:
        st.info("No proposal: the model did not return a usable final answer.")
    step += 1
    st.markdown(f"**{step}. Validation in code**")
    v = t.get("validation")
    if v:
        for c_ in v["checks"]:
            badge(c_["pass"], c_["check"])
        if v.get("detail"):
            (st.warning if not v["ok"] else st.caption)(v["detail"])
    elif t["setup"] == "plain":
        st.caption("Plain LLM: only the probability is parsed; no tool citations to match.")
    else:
        st.info("The analyst passed, so there was nothing to validate. Passing costs nothing.")
    step += 1
    st.markdown(f"**{step}. Edge (code): gap after fees must exceed {t.get('min_edge', 0.04) * 100:.0f}c**")
    if t.get("gap_after_fees") is not None:
        badge(t["gap_after_fees"] > t.get("min_edge", 0.04), f"gap after fees {cents(t['gap_after_fees'])}")
    else:
        st.caption("Not reached.")
    step += 1
    st.markdown(f"**{step}. Priced-in sceptic**")
    sc = t.get("sceptic")
    if sc:
        verdict = (sc.get("verdict") or {}).get("verdict")
        reason = (sc.get("reply") or {}).get("reason") or (sc.get("verdict") or {}).get("reason")
        thought = f" (thought for {sc['reasoning_chars']:,} characters)" if rs else ""
        if verdict == "approve":
            st.success(f"ALLOW{thought}: {reason}")
        else:
            st.error(f"VETO{thought}: {reason}")
    else:
        st.caption("Sceptic not enabled in this run." if not t.get("sceptic_enabled") else
                   "Sceptic not reached (no valid candidate trade).")
    step += 1
    st.markdown(f"**{step}. Pre-trade risk and final action**")
    if t.get("risk"):
        badge(t["risk"]["risk_result"] == "approved", f"pre-trade risk: {t['risk']['risk_result']}")
    if t.get("order"):
        o = t["order"]
        st.success(f"ORDER: buy `{o['side']}` on `{o['ticker']}` at {o['price'] * 100:.0f}c "
                   f"(estimate {pct(o['p_model'])}, gap after fees {cents(o['gap'])}).")
    else:
        st.info(f"No order: {STATUS_TEXT.get(t['status'], t['status'])}. Passing costs nothing.")
    if t.get("usage"):
        u = t["usage"]
        st.caption(f"LLM usage: {u.get('calls')} calls, {u.get('tokens_in'):,} tokens in / {u.get('tokens_out'):,} "
                   f"out, about ${u.get('cost', 0):.4f}.")


def show_tool(t):
    if t.get("source") == "deepseek":
        return show_deepseek(t)
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

ARCH = """digraph A {
rankdir=LR; bgcolor="transparent"; compound=true; newrank=true;
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fillcolor="#f2f2f2"];
edge [fontname="Helvetica", fontsize=9];
subgraph cluster_decide { label="DECIDE LOOP (every decision point, 30 min before tip)"; style="rounded"; color="#2c7fb8";
  news [label="as-of news + quotes"]; fc [label="M4 news shift"]; est [label="estimate p\\n= 24 h anchor + shift"];
  edge_ [label="edge gate\\ngap after fees > min edge"]; chk [label="guardrail checks"]; risk [label="risk limits\\n+ kill switch"];
  order [label="order or brief", fillcolor="#9be7a1"];
  news -> fc -> est -> edge_ -> chk -> risk -> order; }
subgraph cluster_learn { label="LEARN LOOP (after each settled day)"; style="rounded"; color="#d95f0e";
  settle [label="settle trades"]; review [label="reviewer proposes a rule\\n(skip templates +\\nEDGE-LEARNING templates: 2-10c)"];
  gate [label="split gate: back-test on\\nheld-out earlier days"]; save [label="save rule", fillcolor="#9be7a1"];
  reject [label="reject rule", fillcolor="#ffb3b3"]; nb [label="notebook\\n(rules + min edge)", fillcolor="#fff2b3"];
  settle -> review -> gate; gate -> save [label="helps"]; gate -> reject [label="doesn't"]; save -> nb; }
subgraph cluster_blend { label="FORECAST-BLEND SUBLOOP (results use the simplified trader)"; style="rounded,dashed"; color="#756bb1";
  sig [label="signals: anchor, 1 h mid,\\nanchor+M4/MLP/GRU, M4/MLP/GRU"]; w [label="weights (refit on\\nsettled days)"];
  bg [label="gate: held-out Brier\\nmust improve"]; bp [label="blend p(home)"];
  sig -> w -> bg -> bp; }
subgraph cluster_llm { label="LLM ARMS (40% subsample only; published trader results use NO LLM)"; style="rounded,dashed"; color="#888888";
  an [label="LLM analyst <-> as-of tools"]; val [label="code validates citations"]; sc [label="priced-in sceptic"];
  an -> val -> sc; }
order -> settle [style=dashed, label="next day"];
nb -> edge_ [color="#d95f0e", penwidth=2, label="min edge, skip rules"];
bp -> est [style=dashed, color="#756bb1", label="replaces p (offline test)"];
val -> edge_ [style=dashed, color="#888888", label="LLM estimate"]; sc -> risk [style=dashed, color="#888888", label="veto or allow"];
}"""


def arch_diagram():
    st.graphviz_chart(ARCH, width="stretch")
    st.caption("Solid = the published trader (no LLM). Orange = what the learn loop writes back. Dashed = tested "
               "add-ons: the forecast blend was scored with a simplified offline trader, the LLM analyst and sceptic "
               "on the 40% subsample of 304 decision points.")


def overview():
    st.title("NBA prediction-market agents: how they think")
    st.write("Every panel replays a real agent run on as-of data from the 2025-26 test period. Decision steps show "
             "only what was public at the decision time; the market's verdict (closing price, CLV, P&L) is revealed "
             "separately.")
    res = load("results")
    if res:
        with st.container(border=True):
            st.subheader("Key takeaways")
            for t in res["takeaways"]:
                st.markdown("- " + t)
    st.subheader("Architecture: decide, learn, blend")
    arch_diagram()
    st.markdown("""
| Agent | What it does | Where its thinking shows |
|---|---|---|
| Trader (`agents/graph.py`) | LangGraph: news → M4 forecast → anchor + shift → gap after fees → checks → risk → order | Trader tab: graph path, every node, guardrails |
| Reviewer + gate | Learns rules (skip rules and edge thresholds) from settled trades; the gate keeps a rule only if it helps on held-out earlier days | Trader review loop tab; Learning page |
| Forecast blend (`agents/forecast_blend.py`) | Learns weights over market and model signals; a gate on held-out Brier accepts each update | Learning page |
| LLM tool agent + sceptic (`agents/tool_agent.py`) | LLM calls as-of tools, proposes; code validates numbers; sceptic asks "already priced in?" | LLM tool agent tab |
| Coach (`agents/coach_agent.py`) | Before a user's pick: chooses checks, teaches a lesson, nudges pass or caution | Coach tab |
| Orchestrator (`agents/orchestrator.py`) | Supervisor routes pregame → trader → Coach → briefs for one night | Orchestrator tab |
| Pregame / in-play (`agents/pregame.py`, `agents/inplay.py`) | News polling and fair odds before and during games | Pregame and In-play tabs |
""")
    llm_results_panel()


def find_row(rows, key, value, **match):
    return next((r for r in rows if r.get(key, "").startswith(value)
                 and all(r.get(k) == v for k, v in match.items())), {})


def llm_results_panel():
    r = load("llm_results")
    if not r:
        return
    st.divider()
    st.header("LLM agent vs original agent (test period, 304 decision points)")
    st.info(f"**{r['punchline']}**")
    chat, rsn = r["models"]["deepseek-chat"], r["models"]["deepseek-reasoner"]
    cols = ("Trades", "Mean CLV [95% CI]", "CLV $ [95% CI]", "P&L after fees [95% CI]")
    rows = [("A. Deterministic agent (original)", find_row(rsn["scoreboard"], "Setup", "A."))]
    for m, lab in ((chat, "DeepSeek chat"), (rsn, "DeepSeek reasoner")):
        for k, name in (("B.", "B. plain LLM"), ("C.", "C. tool agent"), ("D.", "D. tool agent + sceptic")):
            rows.append((f"{name} ({lab})", find_row(m["scoreboard"], "Setup", k)))
    rows.append(("Never trade", find_row(rsn["scoreboard"], "Setup", "Never")))
    st.markdown("**Scoreboard** (CLV $ and P&L vs never trading; 95% day-clustered bootstrap CIs)")
    st.dataframe([{"setup": n, **{c: x.get(c, "") for c in cols}} for n, x in rows], hide_index=True, width="stretch")
    paired = [("D − A (reasoner + sceptic vs original)", find_row(rsn["paired"], "Comparison", "tool_sceptic − anchor",
                                                                  Metric="CLV $")),
              ("D − C sceptic effect (reasoner)", find_row(rsn["paired"], "Comparison", "tool_sceptic − tool",
                                                           Metric="CLV $")),
              ("D − C sceptic effect (chat)", find_row(chat["paired"], "Comparison", "tool_sceptic − tool",
                                                       Metric="CLV $"))]
    st.markdown("**Paired differences (CLV $, same game-days)**")
    st.dataframe([{"comparison": n, "difference [95% CI]": x.get("Difference [95% CI]", ""), "p": x.get("p", "")}
                  for n, x in paired], hide_index=True, width="stretch")
    st.markdown("**The sceptic: what it vetoed vs what it kept**")
    diag = []
    for m, lab in ((chat, "DeepSeek chat"), (rsn, "DeepSeek reasoner")):
        d = find_row(m["diagnostics"], "Setup", "tool_sceptic")
        diag.append({"model": lab, "sceptic vetoes / calls": d.get("Sceptic rejects / calls", ""),
                     "vetoed vs kept CLV (per contract)": d.get("Vetoed vs kept CLV", "").replace(
                         "+nan", "n/a (none kept)"),
                     "invalid outputs": d.get("Invalid (rate)", ""), "cost (D arm)": d.get("Cost", "")})
    st.dataframe(diag, hide_index=True, width="stretch")
    extra = []
    if r.get("reasoning_budget_calls"):
        extra.append(f"{r['reasoning_budget_calls']} reasoner calls used the whole {r['reasoning_budget_tokens']}-token "
                     "budget on hidden reasoning and returned no answer (counted as invalid, so a pass)")
    if r.get("cost_usd"):
        extra.append(f"total DeepSeek spend ${r['cost_usd']:.2f}")
    if extra:
        st.caption("; ".join(extra) + ".")
    st.caption(f"Source: {r['source']}. The kept-trade CLV rests on 2 trades: not evidence of skill.")
    llm_decision_panel()


def table(rows, cols=None, rename=None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if rename:
        df = df.rename(columns=rename)
    st.dataframe(df.fillna("–"), hide_index=True, width="stretch")


def figure(name, caption):
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def llm_decision_panel():
    res = load("results")
    if not res:
        return
    lv, eg = res["llm_vs_original"], res["edge_gate"]
    st.markdown("#### Decision by decision: what the LLM actually changes")
    st.warning(f"**The gain comes from trading less, not from picking better trades.** The LLM tool agent's filter "
               f"on A's trades is no better than random: a random keep of the same size does at least as well "
               f"{lv['random_keep_pct_c']} of the time for the reasoner tool agent (C) and "
               f"{lv['random_keep_pct_d']} for the tool agent + sceptic (D).")
    c = st.columns(2)
    with c[0]:
        st.markdown("**What the LLM improves**")
        for b in lv["improves"]:
            st.markdown("- " + b)
    with c[1]:
        st.markdown("**What it does not improve**")
        for b in lv["worse"]:
            st.markdown("- " + b)
    st.markdown("**The reasoner is overconfident**: on its buy proposals, its estimate sits further above the "
                "market mid than A's at the same decision points, while the close barely moved.")
    table(lv["overconfidence"], ["Model", "Buy proposals", "LLM − mid", "A − mid", "Close − mid (what happened)"])
    st.markdown("**Why the sceptic vetoed (reasoner D)**")
    table(lv["veto_reasons"])
    if lv.get("vetoes_wrong"):
        st.caption(f"Vetoes the sceptic got wrong (positive counterfactual CLV): {lv['vetoes_wrong']}")
    d0 = eg["d0"]
    st.info(f"**Without the 4¢ edge gate** the reasoner proposes {d0['calls']} orders; the sceptic vetoes "
            f"{d0['vetoes']} of them (low-edge calls / vetoes: {d0['low_edge'].split(' (')[0]}), leaving "
            f"{d0['trades']} trades. The sceptic holds the line on its own.")
    figure(lv["figure"], "Kept vs filtered vs added trades, and the random-filter baseline")
    st.caption("Reproducibility: every LLM reply is cached, so these runs replay exactly. The chat model ran at "
               "temperature 0; the reasoner (thinking mode) ignores temperature and samples its answers, so a fresh, "
               f"uncached run could differ. Source: {lv['source']}.")


# ---------------- results ----------------

def fmt(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


def results_page():
    res = load("results")
    st.title("Results scoreboard")
    if not res:
        st.error("results.json missing. Run: python demo/build_results.py")
        return
    st.header("Win probability: models vs the market")
    st.caption("All rows scored on the same 501 test games (1 Feb – 12 Apr 2026), except the venue-close rows "
               "(498 games with prices on every venue). Lower Brier and log loss are better.")
    rows = [{"predictor": r["predictor"], "type": r["group"], "Brier": fmt(r["brier"]),
             "log loss": fmt(r["log_loss"]), "accuracy": "–" if r["accuracy"] is None else pct(r["accuracy"]),
             "games": r["games"], "source": r["source"]} for r in res["win_models"]]
    table(rows)
    chart = pd.DataFrame([{"predictor": r["predictor"], "Brier": r["brier"]} for r in res["win_models"]
                          if r["group"] != "baseline"]).set_index("predictor")
    st.bar_chart(chart, horizontal=True, height=420)
    st.caption("Our models alone (M4, calibrated M4, M4-NN) lose clearly to the market; adding the market as the "
               "base (anchor + shift, blend) closes the gap but never beats Kalshi at tip.")

    pm = res["players"]
    st.header("Player models")
    c = st.columns(2)
    with c[0]:
        st.markdown("**M1 baselines vs M2 GRU: points and minutes** (held-out; pinball loss lower is better, "
                    "10–90% interval coverage should be near 80%)")
        table([{"model": r["model"], "target": r["target"], "pinball": fmt(r["pinball"], 3),
                "10–90% coverage": pct(r["coverage_10_90"]), "median abs. error": fmt(r["median_abs_error"], 2)}
               for r in pm["m2"]])
    with c[1]:
        st.markdown("**M3: will the player play?** (Brier, lower is better)")
        table([{"predictor": r["predictor"], "Brier": fmt(r["brier"]), "rows": f"{r['rows']:,}",
                "base rate": pct(r["base_rate"])} for r in pm["m3"]])
    st.markdown("**M6 ImpactNet (predicted price move): a null result.** On the test decision times it is no better "
                "than predicting zero move, and the M6 agent placed "
                f"{pm['m6_agent_trades_test']} trades.")
    table([{"model": r["model"], "rows": r["rows"], "MAE (¢)": fmt(r["mae_c"], 3),
            "MAE − zero move (¢) [95% CI]": f"{r['mae_minus_zero_c']:+.3f} {r['ci']}"} for r in pm["m6"]])

    ag = res["agents"]
    st.header("Agents: trading results")
    cols = ["setup", "trades", "mean_clv", "clv_dollars", "pnl", "source"]
    names = {"setup": "setup", "trades": "trades", "mean_clv": "mean CLV / contract [95% CI]",
             "clv_dollars": "CLV $ [95% CI]", "pnl": "P&L after fees $ [95% CI]", "source": "source"}
    st.subheader(ag["full_label"])
    table(ag["full"], cols, names)
    st.subheader(ag["subsample_label"])
    st.caption("A different, smaller window: do not compare these numbers with the full-period table above.")
    table(ag["subsample"], cols, names)
    st.caption("CLV is per contract against the mid at tip; CLV $ = CLV × contracts. CIs: day-clustered bootstrap, "
               "2000 replicates. Never trade = $0 by definition.")
    edge_gate_panel(res["edge_gate"])
    st.caption("Sources (evaluation/results/): " + ", ".join(res["sources"]))


def edge_gate_panel(eg):
    st.header("Edge gate: what if the agent trades on any positive edge?")
    st.caption("40% subsample, 304 decision points; deterministic anchor agent (A), no learning.")
    st.write(eg["summary"])
    df = pd.DataFrame(eg["sweep"])
    c = st.columns([3, 2])
    with c[0]:
        table([{"min edge": r["min_edge"], "trades": r["trades"], "CLV $ [95% CI]": r["clv_ci"],
                "mean CLV [95% CI]": r["mean_clv"], "P&L [95% CI]": r["pnl"], "win rate": r["win_rate"]}
               for r in eg["sweep"]])
    with c[1]:
        st.bar_chart(df.set_index("min_edge")[["clv_dollars"]].rename(columns={"clv_dollars": "CLV $"}), height=250)
    st.metric("Paired difference, edge 0 − 4¢ (CLV $, same game-days)", eg["a_paired_clv"])
    st.write("Per contract, the extra low-edge trades lose to the close at the same rate as the normal ones, so "
             "removing the gate just multiplies the loss by the number of trades.")
    table(eg["split"], ["Arm", "Trades", "n", "Mean CLV [95% CI]", "CLV $ [95% CI]"])
    d0 = eg["d0"]
    st.info(f"LLM sceptic at edge 0: vetoes {d0['vetoes']} of {d0['calls']} proposals "
            f"({d0['low_edge'].split(' (')[0]} low-edge calls / vetoes); the reasoner + sceptic makes {d0['trades']} "
            "trades.")
    figure(eg["figure"], "CLV vs minimum edge")


# ---------------- learning ----------------

def learning_page():
    res = load("results")
    st.title("How the agents learn")
    if not res:
        st.error("results.json missing. Run: python demo/build_results.py")
        return
    adaptive_panel(res["adaptive_edge"])
    st.divider()
    blend_panel(res["forecast_blend"])


def adaptive_panel(ae):
    st.header("Learned edge threshold")
    st.write("The reviewer may now propose new edge thresholds (2, 3, 5, 7 or 10¢ market-wide, higher on underdogs, "
             "long shots or stale news). The split gate back-tests each proposal on 14 earlier, held-out market days "
             "and accepts it only if CLV $ improves by at least $2 over at least 3 changed trades. The newest "
             "accepted market-wide rule sets the threshold; rules expire after 45 days.")
    st.markdown("**Threshold timeline (base 4¢): " + " → ".join(ae["timeline"]) + "**")
    st.markdown("#### Step by step: proposed → gate evidence → decision")
    for i, s in enumerate(ae["story"]):
        ok = s["accepted"]
        with st.container(border=True):
            c = st.columns([2, 3, 2])
            c[0].markdown(f"**{i + 1}. {s['decided']}**  \nRule {s['rule']}: {s['proposal']}")
            c[1].markdown(f"Gate evidence on held-out days {s['gate_days']}:  \nCLV $ without the rule "
                          f"**{s['clv_without']:+.2f}** → with it **{s['clv_with']:+.2f}** "
                          f"({s['changed_trades']} changed trades)")
            (c[2].success if ok else c[2].error)("ACCEPTED: saved to the notebook" if ok else "REJECTED")
    if ae.get("expiry"):
        st.caption(f"The accepted 10¢ rule expired after its {ae['expiry']['days']} days on {ae['expiry']['on']}, "
                   "so the threshold returned to the 4¢ base.")
    st.markdown("**The four setups (full test period, 1 Feb – 12 Apr 2026)**")
    table(ae["setups"])
    c = st.columns(2)
    c[0].metric("Learned threshold − no learning (CLV $)", ae["vs_none"], f"P&L {ae['vs_none_pnl']}", delta_color="off")
    c[1].metric("Learned threshold − split-gate learning (CLV $)", ae["vs_split"], f"P&L {ae['vs_split_pnl']}",
                delta_color="off")
    with st.expander("All paired differences"):
        table(ae["paired"])
    st.write(ae["verdict"])
    figure(ae["figure"], "Threshold over time and cumulative CLV")


def blend_panel(fb):
    st.header("Model blend: learning how much to trust each signal")
    st.markdown(
        "**How it works.** Each day the agent has several probability signals for the home team: the market price "
        "24 h before tip (the anchor), the market 1 h before tip, the anchor shifted by each model's news effect "
        "(M4, MLP, GRU), and the raw models. The blend is a weighted average of those signals (on the log-odds "
        "scale). It starts with all the weight on *anchor + M4 shift*, which is exactly the published agent. After "
        "each settled day it proposes new weights fitted on settled games only, and a gate accepts them only if they "
        "lower the Brier score on earlier held-out days by at least 0.0005. The blended probability then replaces "
        "the trader's estimate.")
    n = fb["counts"]
    c = st.columns(3)
    c[0].metric("Updates accepted", n["accepted"])
    c[1].metric("Rejected by the gate", n["rejected"])
    c[2].metric("Deferred (warm-up)", n["deferred"])
    st.markdown("**Weight trajectory** (weight on each signal, per test day)")
    traj = pd.DataFrame(fb["trajectory"]).set_index("day")
    traj = traj[[c_ for c_ in traj.columns if traj[c_].max() >= 0.02]]
    st.area_chart(traj, height=300)
    st.caption(f"Final weights: {fb['final_weights']}.")
    st.markdown("**Each accepted update: held-out Brier before → after**")
    table([{"day": a["day"], "held-out Brier before": fmt(a["brier_before"]),
            "after": fmt(a["brier_after"]), "gate games": a["gate_games"], "gate days": a["gate_days"],
            "new weights": a["weights"]} for a in fb["accepted"]])
    st.markdown("**Forecast accuracy (all 501 test games; blends are walk-forward)**")
    table(fb["forecast"], ["Predictor", "Brier", "Log loss", "Brier − agent [95% CI]",
                           "Brier − market at tip [95% CI]"])
    st.markdown("**Trading with each forecaster** (simplified offline trader, identical for every row)")
    table(fb["trading"])
    c = st.columns(2)
    c[0].metric("Blend − agent: CLV $", fb["blend_vs_agent_clv"],
                f"blend {fb['blend_clv'].split()[0]} vs agent {fb['agent_clv'].split()[0]}", delta_color="off")
    c[1].metric("Blend − agent: P&L $", fb["blend_vs_agent_pnl"], "no detectable difference", delta_color="off")
    st.error(f"**Ablation: without market prices the blend does much worse**: CLV $ {fb['nomarket_clv']}, P&L "
             f"{fb['nomarket_pnl']}. The market is the base, our models are the correction.")
    with st.expander("All paired comparisons"):
        table(fb["paired"])
    st.warning("Caveats: the trading numbers come from a simplified offline trader (one decision per game on the "
               "home market, $20 stake, no notebook rules), not the full agent replay. Part of the CLV gain may come "
               "from the 1 h mid input pulling the estimate toward the traded price (fewer, smaller disagreements "
               "with the market). The forecast gain over the agent is not significant.")
    figure(fb["figure"], "Blend weights over the test period")
    st.caption(f"Source: {fb['source']}, forecast_blend_log.json.")


PAGES = {"Overview": overview, "Results": results_page, "Learning": learning_page}


def main():
    index = load("index")
    if not index:
        st.error("No traces found in demo/traces/. Run: PYTHONPATH=. python demo/build_traces.py")
        return
    page = st.sidebar.radio("Page", ["Overview", "Results", "Learning", "Scenarios"])
    if page in PAGES:
        PAGES[page]()
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
    st.sidebar.toggle("Presentation mode: hide outcome until revealed", value=False, key="present")
    st.sidebar.caption(index.get("note", ""))
    st.title(sc["title"])
    st.write(sc["summary"])
    data = load(agents[agent])
    if data is None:
        st.error("Trace missing for this agent.")
        return
    RENDER[agent](data)
    graded_panel(game_outcomes(agent, data), data.get("settlement"), f"reveal_{sc['id']}_{agent}",
                 synthetic=agent == "inplay")


main()
