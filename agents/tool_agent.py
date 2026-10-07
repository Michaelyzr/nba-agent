"""Tool-using LLM trading agent with a "priced-in sceptic" step (docs/agentic_review.md, option A).

    python -m evaluation.llm_agent_eval --help         # runs and scores it on the replay

At each replay decision time (same `policy(view, game, now)` interface as
MarketAgent, so Replay, caps and fills are unchanged):

    analyst (LLM) <-> as-of tools (agents/tools.py), up to MAX_TOOL_CALLS calls in MAX_TURNS turns
      -> proposal {decision: buy|pass, team, estimate_p, rationale, citations}
      -> validation in code: schema, cited numbers match tool outputs, rationale numbers come from tools,
         estimate within GROUND_BAND of a tool-derived probability, banned wording, news public by now
      -> code gap after fees > min edge (same rule as MarketAgent)
      -> sceptic (LLM, optional): is the edge already priced in? approve | reject
      -> order on the cheapest route -> pretrade_risk -> Replay caps and fills

Any invalid output, LLM error or rejection is a pass with a logged reason.
PlainLLMAgent is the README's "ChatGPT baseline": one call with the same
information as text, no tools and no sceptic, then the same trading rule.

Leakage: tools only read the AsOf view, prompts carry relative times only
(plus the game date, which is not after `now`). The LLM's pretraining may
still know 2025-26 results; the anonymised setting (TEAM_A/TEAM_B, no date)
tests for that.
"""
import json
import math
import re

import pandas as pd

from agents.graph import BANNED, DEFAULT_MIN_EDGE, DEFAULT_STAKE, pretrade_risk
from agents.notebook import Notebook
from agents.tools import TOOL_SPECS, AsOfTools, lookup, numbers
from replay import Order

MAX_TOOL_CALLS = 8
MAX_TURNS = 4
GROUND_BAND = 0.03              # estimate_p must sit within this of a tool-derived probability
CITE_TOL = 0.0051               # cited numbers may be rounded to two decimals
P_MIN, P_MAX = 0.01, 0.99
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?")
INVALID = ("invalid_json", "bad_schema", "unknown_team", "no_citations", "number_mismatch", "ungrounded_estimate",
           "banned_wording", "citation_after_decision", "budget_exhausted")

ANALYST_SYSTEM = """You are a trading analyst for Kalshi NBA game-winner markets. A contract on a team pays $1 if that \
team wins and $0 otherwise. At this decision time you decide whether to buy one team's contract or pass.

You can call tools that read only data available right now. Code computes every number; never invent one.
Tools (name(args): what it returns):
{tools}

Reply with exactly one JSON object, either
  {{"tool_calls": [{{"tool": "get_quote", "args": {{"team": "XXX"}}}}, ...]}}
or, when you are done,
  {{"final": {{"decision": "buy" or "pass", "team": team label or null, "estimate_p": your probability that the \
team wins (null for pass), "rationale": two or three sentences, "citations": [{{"call_id": "c1", "field": "mid", \
"value": 0.55}}, ...]}}}}

Rules:
- Base estimate_p on tool outputs, for example the market anchor plus the M4 news shift, or the current mid adjusted \
for news the market has not absorbed. Cite every number estimate_p relies on, by call id, field path and value.
- Only numbers that appear in tool outputs may appear in the rationale.
- A trade is placed only if estimate_p minus the all-in cost (price plus fee) is more than {min_edge} \
(the edge tool computes this). Passing costs nothing.
- Never promise an outcome. You have a budget of {budget} tool calls."""

SCEPTIC_SYSTEM = """You are a sceptical risk reviewer for Kalshi NBA game-winner trades. An analyst proposes buying \
a team's contract because its probability estimate beats the all-in cost. Argue whether that edge is already priced \
in: compare the price move since the anchor 24 hours before tip with the news and its timing (news published earlier \
has usually been absorbed), check whether the model's news shift is large relative to the spread and fees, and \
whether the analyst's estimate double-counts information the price already reflects. Approve only if the edge \
plausibly survives after this scrutiny.

Reply with exactly one JSON object: {"verdict": "approve" or "reject", "reason": one or two sentences}. \
Do not introduce new numbers."""

PLAIN_SYSTEM = """You estimate the probability that the home team wins an NBA game, for a Kalshi game-winner market. \
You are given the information available at this decision time. Reply with exactly one JSON object: \
{"p_home": probability between 0 and 1, "rationale": one or two sentences}."""


def tool_menu() -> str:
    return "\n".join(f"- {name}({args}): {what}" for name, (args, what) in TOOL_SPECS.items())


def header(tools: AsOfTools, game, anonymise: bool) -> str:
    when = "" if anonymise else f", game date {game.date}"
    return (f"Game: {tools.away} (away) at {tools.home} (home){when}. Tip-off in {tools.minutes_to_tip():.0f} minutes. "
            f"Team labels you may use: {', '.join(tools.teams)}. Teams with a fresh quote: "
            f"{', '.join(tools.fresh_teams()) or 'none'}.")


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


# ---------------- validation (code, not the LLM) ----------------

def _close(a: float, b: float, tol: float = CITE_TOL) -> bool:
    return abs(a - b) <= max(tol, 0.01 * abs(b))


def rationale_numbers_ok(text: str, allowed: list) -> list:
    """Numbers in the rationale that are not a tool number (as a value, a percentage or probability points)."""
    bad = []
    for raw, pct in NUMBER.findall(text or ""):
        t = float(raw)
        if not pct and t.is_integer() and t <= 10:
            continue                                   # small counts: "two players", "1 hour"
        ok = any(_close(t / 100, v, 0.006) for v in allowed if abs(v) <= 1.5)
        if not pct:
            ok = ok or any(abs(t - v) <= max(0.006, 0.02 * abs(v)) for v in allowed)
        if not ok:
            bad.append(raw + (pct or "").strip())
    return bad


def validate_proposal(final, tools: AsOfTools, results: dict, view) -> tuple:
    """(ok, reason, detail); results maps call id -> {"tool", "args", "output"} for calls actually executed."""
    if not isinstance(final, dict):
        return False, "bad_schema", "final is not an object"
    decision = str(final.get("decision", "")).lower()
    if decision == "pass":
        return True, "agent_pass", str(final.get("rationale", ""))[:300]
    if decision != "buy":
        return False, "bad_schema", f"decision {final.get('decision')!r}"
    team = str(final.get("team") or "").strip().upper()
    if team not in tools.teams:
        return False, "unknown_team", f"team {final.get('team')!r}"
    try:
        p = float(final.get("estimate_p"))
    except (TypeError, ValueError):
        return False, "bad_schema", "estimate_p is not a number"
    if not math.isfinite(p) or not 0 <= p <= 1:
        return False, "bad_schema", f"estimate_p {p} is not a probability"
    rationale = str(final.get("rationale", ""))
    if BANNED.search(rationale):
        return False, "banned_wording", BANNED.search(rationale).group(0)
    cites = final.get("citations") or []
    if not isinstance(cites, list) or not cites:
        return False, "no_citations", "a buy must cite the tool numbers it uses"
    for c in cites:
        if not isinstance(c, dict) or c.get("call_id") not in results:
            return False, "number_mismatch", f"citation {c!r} does not name an executed call"
        try:
            value = float(c.get("value"))
        except (TypeError, ValueError):
            return False, "number_mismatch", f"citation {c!r} has no numeric value"
        out = results[c["call_id"]]["output"]
        at = lookup(out, c.get("field", "")) if c.get("field") else None
        if isinstance(at, (int, float)) and not isinstance(at, bool):
            if not _close(value, float(at)):
                return False, "number_mismatch", f"{c['call_id']}.{c.get('field')} is {at}, cited {value}"
        elif not any(_close(value, v) for v in numbers(out)):
            return False, "number_mismatch", f"{value} is not in the output of {c['call_id']}"
    allowed = [v for r in results.values() for v in numbers(r["output"])] + [p, tools.min_edge, 0.07, 24.0]
    allowed += [abs(v) for v in allowed]
    bad = rationale_numbers_ok(rationale, allowed)
    if bad:
        return False, "number_mismatch", f"rationale numbers not in any tool output: {bad[:3]}"
    refs = grounding_refs(team, tools, results)
    if not any(abs(p - r) <= GROUND_BAND for r in refs):
        return False, "ungrounded_estimate", f"estimate {p:.3f} is not within {GROUND_BAND} of {sorted(refs)}"
    public = set(view.news().news_id)
    cited_news = re.findall(r"\b(espn-[\w-]+)\b", rationale)
    if any(n not in public for n in cited_news):
        return False, "citation_after_decision", str([n for n in cited_news if n not in public])
    return True, "proposal", ""


def grounding_refs(team: str, tools: AsOfTools, results: dict) -> list:
    """Tool-derived win probabilities for `team`: market mids, anchor, anchor + M4 shift, M4, M6 estimate."""
    opp = tools.other(team)
    refs = []
    q = tools.get_quote(team)
    if q.get("mid") is not None:
        refs.append(q["mid"])
    if opp in tools.tickers:
        qo = tools.get_quote(opp)
        if qo.get("mid") is not None:
            refs.append(1 - qo["mid"])
    anchor = tools.get_anchor(team).get("anchor_mid")
    if anchor is not None:
        refs.append(anchor)
    for r in results.values():
        out = r["output"]
        if r["tool"] == "m4_win_prob" and "after" in out:
            refs.append(out["after"][team])
            if anchor is not None:
                refs.append(anchor + out["shift"][team])
        if r["tool"] == "m6_predicted_move" and "home_mid_at_tip_estimate" in out:
            est = out["home_mid_at_tip_estimate"]
            refs.append(est if team == tools.home else 1 - est)
    return [float(x) for x in refs if x is not None]


# ---------------- agents ----------------

class _Base:
    setup = "base"

    def __init__(self, llm, forecaster=None, impact=None, players=None, min_edge=DEFAULT_MIN_EDGE,
                 stake=DEFAULT_STAKE, anonymise=False, channel="platform"):
        self.llm, self.forecaster, self.impact, self.players = llm, forecaster, impact, players or {}
        self.min_edge, self.stake, self.anonymise, self.channel = min_edge, stake, anonymise, channel
        self.notebook = Notebook()          # unused (no learning); lets graph.save_run write a run folder
        self.held = set()
        self.traces = []

    @property
    def llm_name(self):
        return f"{self.setup}@{getattr(self.llm, 'model', 'none')}"

    def tools(self, view, game, now) -> AsOfTools:
        return AsOfTools(view, game, now, self.forecaster, self.impact, self.players, self.anonymise, self.min_edge)

    def policy(self, view, game, now) -> list:
        trace = {"setup": self.setup, "game_id": game.game_id, "as_of": str(now), "status": None, "reason": "",
                 "tool_calls": [], "turns": 0}
        start = len(self.llm.log) if self.llm is not None else 0
        try:
            orders = self._decide(view, game, now, trace)
        except Exception as exc:                       # never let one decision kill a replay
            trace.update(status="error", reason=f"{exc.__class__.__name__}: {str(exc)[:200]}")
            orders = []
        calls = self.llm.log[start:] if self.llm is not None else []
        trace["llm"] = {"calls": len(calls), "cached": sum(c["cached"] for c in calls),
                        "errors": sum(bool(c["error"]) for c in calls),
                        "tokens_in": sum(c["tokens_in"] for c in calls), "tokens_out": sum(c["tokens_out"] for c in calls),
                        "cost": sum(c["cost"] for c in calls), "latency": sum(c["latency"] for c in calls),
                        "by_role": {r: sum(c["role"] == r for c in calls) for r in {c["role"] for c in calls}}}
        self.traces.append(trace)
        return orders

    def _gate(self, view, game, now, trace):
        """Skip without an LLM call: game already held, or no fresh quote. Returns tools or None."""
        if game.game_id in self.held:
            trace.update(status="held", reason="one position per game")
            return None
        tools = self.tools(view, game, now)
        if not tools.fresh_teams():
            trace.update(status="idle", reason="no fresh quote")
            return None
        return tools

    def _order(self, tools, team, p, reason, trace, now, game):
        route = tools.route(team, p)
        p_yes = p if route["side"] == "yes" else 1 - p
        order = Order(route["ticker"], route["side"], float(p_yes), self.stake, reason, trace.get("citations", []),
                      self.channel, [], self.llm_name)
        ok, why = pretrade_risk(order, {"now": now, "tip_time": game.tip_time})
        if not ok:
            trace.update(status="risk_block", reason=why)
            return []
        self.held.add(game.game_id)
        trace.update(status="order", reason=f"{route['route']} at {route['price']:.2f}, gap {route['gap_after_fees']:+.3f}",
                     order={"ticker": route["ticker"], "side": route["side"], "price": route["price"],
                            "p_model": p_yes, "gap": route["gap_after_fees"]})
        return [order]


class ToolAgent(_Base):
    """Analyst LLM with as-of tools, code validation, optional priced-in sceptic."""

    def __init__(self, llm, sceptic=True, max_tool_calls=MAX_TOOL_CALLS, max_turns=MAX_TURNS, **kw):
        super().__init__(llm, **kw)
        self.sceptic = sceptic
        self.max_tool_calls, self.max_turns = max_tool_calls, max_turns
        self.setup = ("tool_sceptic" if sceptic else "tool") + ("_anon" if self.anonymise else "")
        self.system = ANALYST_SYSTEM.format(tools=tool_menu(), min_edge=self.min_edge, budget=max_tool_calls)

    def _prompt(self, head, transcript, left, force=False):
        lines = ["DECISION CONTEXT", head, f"Tool calls left: {left}.", "", "TRANSCRIPT"]
        lines += transcript or ["(no tool calls yet)"]
        lines += ["", "Return your final JSON object now; no more tool calls are allowed." if force or left <= 0
                  else "Reply with the next JSON object."]
        return "\n".join(lines)

    def _decide(self, view, game, now, trace):
        tools = self._gate(view, game, now, trace)
        if tools is None:
            return []
        head = header(tools, game, self.anonymise)
        transcript, results, final = [], {}, None
        for turn in range(self.max_turns):
            left = self.max_tool_calls - len(results)
            force = turn == self.max_turns - 1
            reply = self.llm.ask("analyst", self.system, self._prompt(head, transcript, left, force),
                                 {"turn": turn, "tools": tools, "results": results})
            trace["turns"] = turn + 1
            if reply["error"]:
                trace.update(status="llm_error", reason=reply["error"])
                return []
            msg = reply["json"]
            if msg is None:
                trace.update(status="invalid", reason="invalid_json", detail=reply["text"][:200])
                return []
            if "final" in msg:
                final = msg["final"]
                break
            calls = msg.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                trace.update(status="invalid", reason="bad_schema", detail=reply["text"][:200])
                return []
            shown = []
            for c in calls:
                if len(results) >= self.max_tool_calls:
                    shown.append({"dropped": c, "why": "tool budget exhausted"})
                    continue
                cid = f"c{len(results) + 1}"
                name, args = (c.get("tool"), c.get("args") or {}) if isinstance(c, dict) else (None, {})
                out = tools.call(str(name), args if isinstance(args, dict) else {})
                results[cid] = {"tool": name, "args": args, "output": out}
                trace["tool_calls"].append({"id": cid, "tool": name, "args": args, "error": "error" in out})
                shown.append({"id": cid, "tool": name, "args": args, "output": out})
            transcript.append(f"[turn {turn + 1}] tool results: {dumps(shown)}")
        if final is None:
            trace.update(status="invalid", reason="budget_exhausted")
            return []
        trace["proposal"] = final
        ok, why, detail = validate_proposal(final, tools, results, view)
        if not ok:
            trace.update(status="invalid", reason=why, detail=detail)
            return []
        if why == "agent_pass":
            trace.update(status="pass", reason="agent passed", detail=detail)
            return []
        team, p = str(final["team"]).strip().upper(), min(max(float(final["estimate_p"]), P_MIN), P_MAX)
        route = tools.route(team, p)
        trace["gap"] = route.get("gap_after_fees")
        if "error" in route or not route["clears_min_edge"]:
            trace.update(status="gap", reason=route.get("error") or f"gap {route['gap_after_fees']:+.3f} not above "
                                                                   f"{self.min_edge}")
            return []
        trace["citations"] = [n for n in tools.news_rows().news_id]
        if self.sceptic:
            verdict = self._sceptic(tools, head, final, route, results)
            trace["sceptic"] = verdict
            if verdict["verdict"] != "approve":
                trace.update(status="sceptic_reject", reason=verdict["reason"],
                             candidate={"ticker": route["ticker"], "side": route["side"], "price": route["price"]})
                return []
        reason = (f"{team} win: LLM estimate {p:.0%} vs all-in cost {route['all_in_cost']:.0%} "
                  f"({route['route']}). {str(final.get('rationale', ''))[:240]}")
        return self._order(tools, team, p, reason, trace, now, game)

    def _sceptic(self, tools, head, final, route, results) -> dict:
        team = route["team"]
        evidence = {"proposal": {k: final.get(k) for k in ("team", "estimate_p", "rationale")},
                    "code_computed_edge": {k: v for k, v in route.items() if k != "ticker"},
                    "anchor": tools.get_anchor(team), "price_history_6h": tools.get_price_history(team, 6),
                    "news": tools.get_news(), "analyst_tool_results": {k: v["output"] for k, v in results.items()}}
        prompt = "\n".join(["DECISION CONTEXT", head, "", "EVIDENCE", dumps(evidence), "",
                            "Reply with your JSON verdict."])
        reply = self.llm.ask("sceptic", SCEPTIC_SYSTEM, prompt, {"tools": tools, "route": route, "results": results})
        msg = reply["json"] or {}
        verdict = str(msg.get("verdict", "")).lower()
        if reply["error"] or verdict not in ("approve", "reject"):
            return {"verdict": "reject", "reason": "sceptic output invalid or failed (fail closed)",
                    "invalid": True}
        reason = str(msg.get("reason", ""))[:300]
        if BANNED.search(reason):
            return {"verdict": "reject", "reason": "sceptic used banned wording (fail closed)", "invalid": True}
        return {"verdict": verdict, "reason": reason}


class PlainLLMAgent(_Base):
    """One LLM call with the decision-time information as text; no tools, no sceptic, same trading rule."""

    def __init__(self, llm, **kw):
        super().__init__(llm, **kw)
        self.setup = "plain" + ("_anon" if self.anonymise else "")

    def info(self, tools) -> dict:
        out = tools.out_from_news()
        return {"quotes": {t: tools.get_quote(t) for t in tools.teams},
                "anchors": {t: tools.get_anchor(t) for t in tools.teams},
                "news": tools.get_news(), "m4_win_model": tools.m4_win_prob(out) if self.forecaster else None,
                "fee_rule": "Kalshi fee per contract is 0.07 x price x (1 - price)"}

    def _decide(self, view, game, now, trace):
        tools = self._gate(view, game, now, trace)
        if tools is None:
            return []
        info = self.info(tools)
        prompt = "\n".join(["DECISION CONTEXT", header(tools, game, self.anonymise),
                            f"The home team is {tools.home}.", "", "INFORMATION", dumps(info), "",
                            "Reply with your JSON estimate."])
        reply = self.llm.ask("plain", PLAIN_SYSTEM, prompt, {"tools": tools, "info": info})
        trace["turns"] = 1
        if reply["error"]:
            trace.update(status="llm_error", reason=reply["error"])
            return []
        msg = reply["json"]
        try:
            p_home = float(msg["p_home"])
            assert math.isfinite(p_home) and 0 <= p_home <= 1
        except Exception:
            trace.update(status="invalid", reason="invalid_json" if msg is None else "bad_schema",
                         detail=reply["text"][:200])
            return []
        p_home = min(max(p_home, P_MIN), P_MAX)
        trace["proposal"] = {"p_home": p_home, "rationale": str(msg.get("rationale", ""))[:300]}
        best = None
        for team in tools.teams:
            p = p_home if team == tools.home else 1 - p_home
            r = tools.route(team, p)
            if "error" not in r and (best is None or r["gap_after_fees"] > best[1]["gap_after_fees"]):
                best = (team, r, p)
        if best is None:
            trace.update(status="idle", reason="no route")
            return []
        team, route, p = best
        trace["gap"] = route["gap_after_fees"]
        if not route["clears_min_edge"]:
            trace.update(status="gap", reason=f"gap {route['gap_after_fees']:+.3f} not above {self.min_edge}")
            return []
        trace["citations"] = [n for n in tools.news_rows().news_id]
        reason = f"{team} win: plain LLM estimate {p:.0%} vs all-in cost {route['all_in_cost']:.0%} ({route['route']})"
        return self._order(tools, team, p, reason, trace, now, game)


# ---------------- heuristic stub (dry runs and tests, no network) ----------------

def heuristic_reply(role, system, prompt, meta):
    """Plays the protocol like a disciplined analyst: anchor + M4 news shift, sceptic rejects moved prices."""
    tools = meta["tools"]
    if role == "plain":
        info = meta["info"]
        anchor = info["anchors"][tools.home].get("anchor_mid") or info["quotes"][tools.home]["mid"]
        shift = (info["m4_win_model"] or {}).get("shift", {}).get(tools.home, 0.0)
        return {"p_home": round(anchor + shift, 4), "rationale": "Anchor plus the model's news shift."}
    if role == "sceptic":
        team = meta["route"]["team"]
        move = tools.get_anchor(team).get("move_since_anchor") or 0.0
        if move >= 0.03:
            return {"verdict": "reject", "reason": "The price already moved toward the trade since the anchor."}
        return {"verdict": "approve", "reason": "The move since the anchor is small relative to the news shift."}
    results, turn = meta["results"], meta["turn"]
    if turn == 0:
        calls = [{"tool": "get_news", "args": {}}]
        calls += [{"tool": t, "args": {"team": team}} for team in tools.teams for t in ("get_quote", "get_anchor")]
        return {"tool_calls": calls}
    if turn == 1 and not any(r["tool"] == "m4_win_prob" for r in results.values()):
        news = next(r["output"] for r in results.values() if r["tool"] == "get_news")
        out = [i["player_id"] for i in news["items"] if i["status"] in ("out", "doubtful")]
        return {"tool_calls": [{"tool": "m4_win_prob", "args": {"out": out}}]}
    by_tool = {}
    for cid, r in results.items():
        by_tool.setdefault((r["tool"], (r["args"] or {}).get("team")), (cid, r["output"]))
    m4_id, m4 = by_tool[("m4_win_prob", None)]
    best = None
    for team in tools.fresh_teams():
        a_id, a = by_tool[("get_anchor", team)]
        if a.get("anchor_mid") is None:
            continue
        p = round(a["anchor_mid"] + m4["shift"][team], 4)
        e = tools.edge(team, min(max(p, P_MIN), P_MAX))
        if "error" not in e and (best is None or e["gap_after_fees"] > best[0]):
            best = (e["gap_after_fees"], team, p, a_id, a, e)
    if best is None or not best[5]["clears_min_edge"]:
        return {"final": {"decision": "pass", "team": None, "estimate_p": None,
                          "rationale": "Anchor plus news shift does not clear the cost.", "citations": []}}
    _, team, p, a_id, a, _ = best
    return {"final": {"decision": "buy", "team": team, "estimate_p": p,
                      "rationale": f"Anchor {a['anchor_mid']} plus the M4 shift {m4['shift'][team]} gives {p}.",
                      "citations": [{"call_id": a_id, "field": "anchor_mid", "value": a["anchor_mid"]},
                                    {"call_id": m4_id, "field": f"shift.{team}", "value": m4["shift"][team]}]}}
