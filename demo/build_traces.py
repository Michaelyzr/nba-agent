"""Precompute compact, as-of agent traces for the hosted demo (demo_app.py reads only demo/traces/).

    PYTHONPATH=. python demo/build_traces.py                    # all scenarios -> demo/traces/*.json
    PYTHONPATH=. python demo/build_traces.py --live 401810565 "2026-02-02 01:30"   # one trader run -> stdout JSON

Needs the full local setup (data/frozen, models/, langgraph). Every decision step is built from the
AsOf view at the decision time; closing prices, CLV and P&L are computed separately and stored under
"settlement" with revealed_at = tip-off, which the app shows only after the decision steps.
"""
import argparse
import json
import math
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import replay  # noqa: E402
from agents.graph import MarketAgent, ANCHOR_LEAD, DEFAULT_MIN_EDGE  # noqa: E402
from agents.notebook import Notebook  # noqa: E402
from replay import LEAD, AsOf, Replay  # noqa: E402

OUT = ROOT / "demo" / "traces"
CLE_POR = "401810565"
CLE_POR_AT = "2026-02-02 01:30:00+00:00"
NIGHT = "2026-02-01"
NIGHT_LEAD = pd.Timedelta(minutes=30)     # CLE @ POR tips 02:00 UTC, so the night decides at 01:30 like the trade
RULE_RUN = ROOT / "runs" / "results" / "test-full"
SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|api[_-]?key)", re.I)


# ---------------- JSON helpers ----------------

def clean(x, depth=0):
    """JSON-safe, compact: rounds floats, drops DataFrames to short records, timestamps to ISO strings."""
    if depth > 8:
        return str(x)
    if isinstance(x, dict):
        return {str(k): clean(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v, depth + 1) for v in x]
    if isinstance(x, pd.DataFrame):
        return clean(x.head(20).to_dict("records"), depth + 1)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else round(x, 4)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if hasattr(x, "__dataclass_fields__"):
        return clean(asdict(x), depth + 1)
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


# ---------------- shared state ----------------

class Ctx:
    def __init__(self):
        from forecast.api import Forecaster
        self.tables = replay.load_tables()
        self.forecaster = Forecaster.load()
        players = self.tables.get("players")
        if players is None:
            players = pd.read_parquet(ROOT / "data" / "frozen" / "players.parquet")
        self.players = players
        self.names = dict(zip(players.player_id.astype(int), players.player_name))
        self.rp = Replay(self.tables, policy=None)
        self.games = self.tables["games"].set_index("game_id", drop=False)

    def game(self, gid):
        return next(self.tables["games"][self.tables["games"].game_id == gid].itertuples(index=False))

    def view(self, now, tripped=False):
        return AsOf(self.tables, pd.Timestamp(now), self.rp.price_index, 0.0, tripped)

    def name(self, pid):
        try:
            return self.names.get(int(pid), str(pid))
        except (TypeError, ValueError):
            return str(pid)


def news_rows(ctx, news: pd.DataFrame) -> list:
    if news is None or news.empty:
        return []
    latest = news.sort_values("published_at")
    return [{"news_id": r.news_id, "published_at": pd.Timestamp(r.published_at).isoformat(),
             "player": ctx.name(r.player_id), "player_id": clean(r.player_id), "status": r.status,
             "text": str(r.text)[:200]} for r in latest.itertuples()]


# ---------------- trader (MarketAgent LangGraph) ----------------

TRADER_NODES = ["trigger", "investigate", "forecast", "analyse", "propose", "no_action", "checks", "risk",
                "confirm", "blocked", "deliver"]
REVIEW_NODES = ["settle", "review", "gate", "save_rule", "reject_rule"]
TRADER_EDGES = [("START", "trigger"), ("trigger", "investigate"), ("trigger", "END"), ("investigate", "forecast"),
                ("forecast", "analyse"), ("analyse", "propose"), ("analyse", "no_action"), ("propose", "checks"),
                ("no_action", "checks"), ("checks", "risk"), ("checks", "deliver"), ("checks", "investigate"),
                ("checks", "blocked"), ("risk", "confirm"), ("risk", "blocked"), ("confirm", "deliver"),
                ("blocked", "deliver"), ("deliver", "END")]
REVIEW_EDGES = [("START", "settle"), ("settle", "review"), ("review", "gate"), ("review", "END"),
                ("gate", "save_rule"), ("gate", "reject_rule"), ("save_rule", "END"), ("reject_rule", "END")]

CHECK_NAMES = ["citation_after_decision", "number_mismatch", "missing_reason", "banned_wording", "rule_not_active"]


def node_payload(ctx, node, upd, state):
    """Inputs / numbers / decision for one node update (small)."""
    p = {}
    if node == "trigger":
        p["news"] = news_rows(ctx, upd.get("news"))
        m = upd.get("markets")
        p["markets_with_fresh_quotes"] = [] if m is None else m.market_ticker.tolist()
        p["status"] = upd.get("status")
    elif node == "investigate":
        inv = upd["investigation"]
        p["players_out"] = [{"player_id": clean(x), "player": ctx.name(x)} for x in inv["overrides"].get("out", [])]
        p["citations"] = inv["citations"]
        p["rules_applied"] = inv["rules_applied"]
        p["note"] = inv.get("note")
    elif node == "forecast":
        p["m4"] = {t.rsplit("-", 1)[-1]: {"before_news": round(f["before"], 4), "after_news": round(f["after"], 4),
                                          "shift": round(f["after"] - f["before"], 4)}
                   for t, f in upd["forecasts"].items()}
    elif node == "analyse":
        rows = []
        for r in upd["candidates"]:
            price = r["ask"] if r["side"] == "yes" else 1 - r["bid"]
            fee = 0.07 * price * (1 - price)
            rows.append({"team": r["team"], "ticker": r["ticker"], "anchor_24h": round(r["before"], 4),
                         "news_shift": round(r["shift"], 4), "estimate_p": round(r["p"], 4), "bid": r["bid"],
                         "ask": r["ask"], "side": r["side"], "price_paid": round(price, 4), "fee": round(fee, 4),
                         "gap_after_fees": round(r["gap"], 4), "min_edge": r["min_edge"], "rules": r["rules"],
                         "act": bool(r["act"]), "why_not": r["why_not"], "stake": r["stake"]})
        p["candidates"] = rows
    elif node in ("propose", "no_action"):
        p["orders"] = [clean(asdict(o)) for o in upd.get("orders", [])]
        p["brief"] = upd.get("brief", "")
    elif node == "checks":
        fails = upd.get("failures", [])
        failed = {a for a, _ in fails}
        p["checks"] = [{"check": c, "pass": c not in failed,
                        "detail": "; ".join(b for a, b in fails if a == c)} for c in CHECK_NAMES]
        p["tries"] = upd.get("tries", state.get("tries", 0))
    elif node == "risk":
        p["kept_orders"] = [o.market_ticker for o in upd.get("orders", [])]
    elif node in ("confirm", "blocked", "deliver"):
        p["status"] = upd.get("status")
        if "brief" in upd:
            p["brief"] = upd["brief"]
    return p


def settle_orders(ctx, orders, now, game):
    out = []
    for o in orders:
        fill, why = ctx.rp._fill(o, pd.Timestamp(now), game.tip_time)
        if fill is None:
            out.append({"ticker": o.market_ticker, "side": o.side, "fill": why})
            continue
        s = ctx.rp._settle({**fill, "market_ticker": o.market_ticker, "side": o.side, "game_id": game.game_id}, game.tip_time)
        out.append(clean({"ticker": o.market_ticker, "side": o.side, "price": s["price"], "contracts": s["contracts"],
                          "fee_dollars": s["fee"], "close_price": s["close_price"], "clv": s["clv"],
                          "outcome_yes": s["outcome"], "pnl": s["pnl"]}))
    return out


def final_score(game):
    return {"home": game.home_team, "away": game.away_team, "home_pts": clean(game.home_pts),
            "away_pts": clean(game.away_pts), "final_at": pd.Timestamp(game.final_at).isoformat()}


def run_trader(ctx, gid, now, plant=None, notebook=None, tripped=False, label=""):
    game, now = ctx.game(gid), pd.Timestamp(now)
    agent = MarketAgent(notebook or Notebook(), forecaster=ctx.forecaster, learn=False, plant=plant)
    view = ctx.view(now, tripped)
    state = {"phase": "decide", "view": view, "game": game, "now": now, "tries": 0, "plant": plant, "trace": []}
    steps, merged = [], dict(state)
    for chunk in agent.graph.stream(state, stream_mode="updates"):
        for node, upd in chunk.items():
            upd = upd or {}
            merged.update({k: v for k, v in upd.items() if k != "trace"})
            steps.append({"node": node, "log": [t["detail"] for t in upd.get("trace", [])],
                          **node_payload(ctx, node, upd, merged)})
    orders = merged.get("orders", []) if merged.get("status") == "sent" else []
    # what the agent would have bought if not blocked (for the "blocked" panel)
    proposed = next((s["orders"] for s in steps if s["node"] == "propose" and s.get("orders")), [])
    return {"agent": "trader", "label": label, "game_id": gid, "matchup": f"{game.away_team} @ {game.home_team}",
            "tip_time": pd.Timestamp(game.tip_time).isoformat(), "as_of": now.isoformat(), "plant": plant,
            "kill_switch_tripped": tripped, "status": merged.get("status"), "brief": merged.get("brief", ""),
            "path": [s["node"] for s in steps], "steps": steps, "proposed_orders": proposed,
            "nodes": TRADER_NODES, "edges": TRADER_EDGES,
            "settlement": {"revealed_at": pd.Timestamp(game.tip_time).isoformat(),
                           "fills": settle_orders(ctx, orders, now, game),
                           "counterfactual": settle_orders(ctx, [o for o in merged.get("orders", [])] or
                                                           _orders_from(proposed), now, game) if not orders else [],
                           "final": final_score(game)}}


def _orders_from(proposed):
    from replay import Order
    keep = {"market_ticker", "side", "p_model", "stake", "reason", "citations", "channel", "rules_applied", "llm"}
    return [Order(**{k: v for k, v in o.items() if k in keep}) for o in proposed]


def find_rule_decision(ctx):
    nb = Notebook.load(RULE_RUN / "notebook.json")
    tried = 0
    for line in (RULE_RUN / "trace.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("phase") != "decide" or "(rule " not in d.get("brief", ""):
            continue
        tried += 1
        t = run_trader(ctx, d["game_id"], d["as_of"], notebook=Notebook.load(RULE_RUN / "notebook.json"),
                       label="Rule applied: a learned rule vetoes a trade")
        cands = next(s["candidates"] for s in t["steps"] if s["node"] == "analyse")
        if any(c["gap_after_fees"] > c["min_edge"] and c["why_not"].startswith("rule") for c in cands):
            rid = next(c["rules"][0] for c in cands if c["rules"])
            rule = next(r for r in nb.rules if r["rule_id"] == rid)
            t["rule"] = clean({k: rule.get(k) for k in ("rule_id", "when", "do", "rationale", "blame_category",
                                                         "valid_from", "status")})
            return t
        if tried >= 25:
            break
    return None


def review_examples():
    """Real review-phase traces (settle -> review -> gate -> save/reject) from the published test run."""
    out = {"save_rule": None, "reject_rule": None}
    for line in (RULE_RUN / "trace.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("phase") != "review":
            continue
        steps = [t["step"] for t in d["trace"]]
        for k in out:
            if out[k] is None and k in steps:
                out[k] = {"day": d["day"], "path": steps,
                          "steps": [{"node": t["step"], "log": [t["detail"]]} for t in d["trace"]]}
    return out


def night_games(ctx, date):
    g = ctx.tables["games"].dropna(subset=["tip_time"])
    return list(g[(g.date == date) & g.game_id.isin(ctx.tables["markets"].game_id)].sort_values("tip_time")
                .itertuples(index=False))


# ---------------- tool agent + sceptic ----------------

class Recorder:
    """Wraps CachedLLM.ask and keeps each request's role, the tool results it saw, and the reply."""

    def __init__(self, llm):
        self.llm, self.calls = llm, []
        self.model = llm.model
        self.log = llm.log

    def ask(self, role, system, prompt, meta=None):
        out = self.llm.ask(role, system, prompt, meta)
        self.calls.append({"role": role, "prompt": prompt, "reply": out.get("json"), "text": out.get("text", "")[:600],
                           "cached": out.get("cached"), "error": out.get("error")})
        return out


class CacheOnly:
    model = "gemini-3.5-flash-lite"

    def __init__(self, model):
        self.model = model

    def generate(self, system, prompt, meta=None):
        raise RuntimeError("cache miss (demo builder never calls the API)")


def tool_turns(calls):
    """Per analyst turn: the reply (tool calls or final) and the tool results returned to it."""
    turns = []
    for i, c in enumerate(calls):
        if c["role"] != "analyst":
            continue
        turn = {"turn": len(turns) + 1, "reply": c["reply"], "tool_results": []}
        nxt = next((d for d in calls[i + 1:] if d["role"] == "analyst"), None)
        if nxt is not None:
            m = re.findall(r"\[turn (\d+)\] tool results: (.*)", nxt["prompt"])
            if m:
                try:
                    turn["tool_results"] = json.loads(m[-1][1])
                except json.JSONDecodeError:
                    pass
        turns.append(turn)
    return turns


def run_tool_agent(ctx, gid, now, backend_kind, sceptic, impact, label):
    from agents.llm_client import CachedLLM, StubBackend
    from agents.tool_agent import ToolAgent, heuristic_reply, validate_proposal
    if backend_kind == "heuristic":
        llm = CachedLLM(StubBackend(heuristic_reply, model="heuristic-stub"))
    else:
        llm = CachedLLM(CacheOnly(backend_kind))
    rec = Recorder(llm)
    agent = ToolAgent(rec, sceptic=sceptic, forecaster=ctx.forecaster, impact=impact, players=ctx.names)
    game, now = ctx.game(gid), pd.Timestamp(now)
    view = ctx.view(now)
    orders = agent.policy(view, game, now)
    tr = agent.traces[-1]
    if any(c["error"] for c in rec.calls):
        return None
    tools = agent.tools(view, game, now)
    validation = None
    if tr.get("proposal") and str(tr["proposal"].get("decision", "")).lower() == "buy":
        # re-run the code validators step by step for display
        results = {}
        for t in tool_turns(rec.calls):
            for r in t["tool_results"]:
                if "id" in r:
                    results[r["id"]] = {"tool": r["tool"], "args": r["args"], "output": r["output"]}
        ok, why, detail = validate_proposal(tr["proposal"], tools, results, view)
        validation = {"ok": ok, "result": why, "detail": detail,
                      "checks": [{"check": "schema and team label", "pass": why not in ("bad_schema", "unknown_team")},
                                 {"check": "cites tool numbers", "pass": why != "no_citations"},
                                 {"check": "cited numbers match tool outputs", "pass": why != "number_mismatch"},
                                 {"check": "estimate grounded (within 3 pts of a tool probability)",
                                  "pass": why != "ungrounded_estimate"},
                                 {"check": "no banned wording", "pass": why != "banned_wording"},
                                 {"check": "news public at decision time", "pass": why != "citation_after_decision"}]}
    sceptic_call = next((c for c in rec.calls if c["role"] == "sceptic"), None)
    settled = settle_orders(ctx, orders, now, game)
    cand = tr.get("candidate")
    counter = []
    if cand and not orders:
        from replay import Order
        counter = settle_orders(ctx, [Order(cand["ticker"], cand["side"], 0.5, 20.0, "counterfactual", [],
                                            "platform", [], "x")], now, game)
    stub = backend_kind == "heuristic"
    return clean({"agent": "tool_agent", "label": label, "game_id": gid,
                  "matchup": f"{game.away_team} @ {game.home_team}", "tip_time": pd.Timestamp(game.tip_time).isoformat(),
                  "as_of": now.isoformat(), "llm": "stub LLM (deterministic heuristic_reply, not Gemini)" if stub
                  else f"Gemini ({backend_kind}), replayed from the response cache", "stub": stub,
                  "sceptic_enabled": sceptic, "status": tr["status"], "reason": tr.get("reason"),
                  "detail": tr.get("detail"), "turns": tool_turns(rec.calls), "proposal": tr.get("proposal"),
                  "validation": validation, "gap_after_fees": tr.get("gap"), "order": tr.get("order"),
                  "sceptic": None if sceptic_call is None else {"verdict": tr.get("sceptic"),
                                                                 "reply": sceptic_call["reply"]},
                  "usage": tr.get("llm"),
                  "settlement": {"revealed_at": pd.Timestamp(game.tip_time).isoformat(), "fills": settled,
                                 "counterfactual": counter, "final": final_score(game)}})


def gemini_candidates():
    """(game_id, as_of, status, run) for Gemini tool-agent decisions in saved runs."""
    out = []
    for meta in sorted((ROOT / "runs" / "llm_agent").glob("*/*/meta.json")):
        m = json.loads(meta.read_text())
        if m.get("backend") != "gemini" or not str(m.get("setup", "")).startswith("tool"):
            continue
        for line in (meta.parent / "trace.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("llm", {}).get("calls"):
                out.append((len(d.get("tool_calls", [])), d["game_id"], d["as_of"], d["status"], d.get("reason"),
                            m["model"], m["setup"]))
    return [o[1:] for o in sorted(out, key=lambda o: -o[0])]


# ---------------- coach, orchestrator, pregame, in-play ----------------

def run_coach(ctx, gid, now, label):
    from agents import coach
    from agents.coach_agent import advise
    game, now = ctx.game(gid), pd.Timestamp(now)
    s = coach.snapshot(ctx.forecaster, ctx.view(now), game, ctx.names)
    chaser = [{"team": "X", "choice": "buy", "price": 0.3, "move": 0.05, "gap": -0.01}] * 4
    picks = []
    for team in [game.away_team, game.home_team]:
        for hist_label, hist in [("new user", []), ("user who chases moves (4 earlier buys after price jumps)", chaser)]:
            a = advise(s, team, hist)
            picks.append({"team": team, "history": hist_label, "steps": a["steps"], "skipped": a["skipped"],
                          "nudge": a["nudge"], "lesson": {k: a["lesson"].get(k) for k in ("id", "title", "body")},
                          "message": a["message"], "habits": a["habits"], "mode": a["mode"]})
    snap = {k: s.get(k) for k in ("home", "away", "p_home_before", "p_home_after", "shift", "out", "pick", "as_of")}
    snap["sides"] = s.get("sides")
    snap["news"] = s.get("news")
    return clean({"agent": "coach", "label": label, "game_id": gid, "matchup": f"{game.away_team} @ {game.home_team}",
                  "tip_time": pd.Timestamp(game.tip_time).isoformat(), "as_of": now.isoformat(),
                  "snapshot": snap, "explanation": coach.explain(s), "picks": picks})


def run_night(ctx, date, game_ids):
    from agents.orchestrator import NightOrchestrator
    out_dir = Path(tempfile.mkdtemp(prefix="demo_orch_"))

    class Pick(NightOrchestrator):
        def slate(self, d):
            return [g for g in super().slate(d) if g.game_id in game_ids]

    orch = Pick(ctx.tables, ctx.forecaster, ctx.players, NIGHT_LEAD, "pregame", out_dir)
    out = orch.run(date)
    print("  pregame snapshot keys:", sorted(json.loads(next((out_dir / "pregame").glob("*/snapshots.jsonl"))
                                                        .read_text().splitlines()[-1]).keys()), flush=True)
    pregame = {}
    for gid in game_ids:
        f = out_dir / "pregame" / str(gid) / "snapshots.jsonl"
        if not f.exists():
            continue
        polls = []
        for line in f.read_text().splitlines():
            sn = json.loads(line)
            polls.append({"as_of": sn.get("as_of"), "p_home": sn.get("p_home"),
                          "baseline_p_home": (sn.get("baseline") or {}).get("p_home"),
                          "delta_from_baseline": sn.get("delta_home_from_baseline"),
                          "expected_lost_share": sn.get("expected_lost_share"),
                          "factors": [{"player": ctx.name(x.get("player_id")), "status": x.get("status"),
                                       "news_id": x.get("news_id"), "published_at": x.get("published_at"),
                                       "source": x.get("source")} for x in sn.get("factors", [])],
                          "new_news_ids": sn.get("new_news_ids") or sn.get("new_items"),
                          "news_health": sn.get("news_health"), "fair_odds": sn.get("fair_odds") or
                          {k: sn.get(k) for k in ("home_decimal_odds", "away_decimal_odds") if k in sn}})
        g = ctx.game(gid)
        pregame[gid] = {"matchup": f"{g.away_team} @ {g.home_team}", "tip_time": pd.Timestamp(g.tip_time).isoformat(),
                        "polls": polls}
    trader = {gid: {k: v for k, v in r.items() if k != "trace"} | {"trace": r["trace"]}
              for gid, r in (out.get("trader") or {}).items()}
    keep = {"date": date, "games": out.get("games"), "done": out.get("done"), "failed": out.get("failed"),
            "messages": out.get("messages"), "trader": trader, "coach": out.get("coach"), "briefs": out.get("briefs"),
            "pregame_summary": out.get("pregame")}
    return clean({"agent": "orchestrator", "label": f"One night: {date}", "night": keep}), clean(pregame)


def run_inplay(ctx, gid):
    from agents.inplay import InPlayAgent
    from data_sources.inplay import TableInPlay
    from data_sources.inplay_demo import synthetic_game
    from data_sources.news_registry import NewsRegistry
    game = ctx.game(gid)
    tables = {"games": ctx.tables["games"], "player_games": ctx.tables["player_games"], "news": pd.DataFrame()}
    scores, events = synthetic_game(tables, ctx.players, game)
    provider = TableInPlay(scores, events, NewsRegistry.load())
    agent = InPlayAgent(tables, ctx.players, game, provider, Path(tempfile.mkdtemp(prefix="demo_inplay_")),
                        prior_model=ctx.forecaster)
    times = sorted(set(pd.to_datetime(scores.observed_at, utc=True)) | set(pd.to_datetime(events.observed_at, utc=True)))
    polls = []
    for t in times:
        sn = agent.poll(t)["snapshot"] or {}
        sc = sn.get("score") or {}
        polls.append({"as_of": sn.get("as_of"), "quote_state": sn.get("quote_state"),
                      "score": {k: sc.get(k) for k in ("home_score", "away_score", "period", "clock_seconds", "phase")},
                      "p_home": sn.get("p_home"), "home_decimal_odds": sn.get("home_decimal_odds"),
                      "away_decimal_odds": sn.get("away_decimal_odds"), "prior": sn.get("prior"),
                      "news_margin": sn.get("news_margin"), "player_effects": [
                          {**e, "player": ctx.name(e.get("player_id"))} for e in sn.get("player_effects", []) or []],
                      "new_events": [r.text for r in events.itertuples() if r.event_id in (sn.get("new_event_ids") or [])],
                      "factors": [{"player": ctx.name(f.get("player_id")), "status": f.get("status")}
                                  for f in sn.get("factors", [])],
                      "conflicts": sn.get("conflicts"), "freshness": sn.get("freshness"),
                      "quote_quality": sn.get("quote_quality"), "errors": sn.get("errors")})
        if agent.stopped:
            break
    return clean({"agent": "inplay", "label": "In-play agent (SYNTHETIC game script over real rosters)",
                  "game_id": gid, "matchup": f"{game.away_team} @ {game.home_team}",
                  "tip_time": pd.Timestamp(game.tip_time).isoformat(), "synthetic": True, "polls": polls})


# ---------------- main ----------------

def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(obj), indent=1, default=str)
    if SECRET.search(text):
        raise ValueError(f"{name}: looks like a secret in the trace")
    (OUT / f"{name}.json").write_text(text)
    print(f"  {name}.json {len(text) / 1024:.0f} KB", flush=True)


def build_all():
    ctx = Ctx()
    print("context loaded", flush=True)
    scenarios = []

    def add(sid, title, summary, agents):
        scenarios.append({"id": sid, "title": title, "summary": summary, "agents": agents})

    # 1. CLE @ POR, Avdija out: the headline trade
    t = run_trader(ctx, CLE_POR, CLE_POR_AT, label="Trade: Avdija out, buy POR 'no'")
    write("trader_cle_por", t)
    # 2. same decision with a planted future citation: checks catch it, the retry recovers
    write("trader_cle_por_retry", run_trader(ctx, CLE_POR, CLE_POR_AT, plant="future_citation",
                                             label="Planted fault: cited future news -> checks retry"))
    # 3. planted 'lock' wording every try: blocked after the retry
    write("trader_cle_por_blocked", run_trader(ctx, CLE_POR, CLE_POR_AT, plant="lock_wording",
                                               label="Planted violation: 'lock' wording -> blocked"))
    # 4. kill switch tripped
    write("trader_cle_por_kill", run_trader(ctx, CLE_POR, CLE_POR_AT, tripped=True,
                                            label="Kill switch tripped: risk blocks the order"))
    print("trader CLE@POR done", flush=True)
    # 5. a pass on the same night
    night = night_games(ctx, NIGHT)
    pass_t = None
    for g in night:
        if g.game_id == CLE_POR:
            continue
        t2 = run_trader(ctx, g.game_id, g.tip_time - LEAD, label="Pass: already priced in")
        if t2["status"] == "brief_only" and t2["steps"][-1]["node"] == "deliver":
            pass_t = t2
            break
    if pass_t:
        write("trader_pass", pass_t)
    # 6. a learned rule vetoes a trade
    rule_t = find_rule_decision(ctx)
    if rule_t:
        write("trader_rule", rule_t)
    rev = review_examples()
    write("review", {"agent": "review", "source": "runs/results/test-full (published test-period run)",
                     "nodes": REVIEW_NODES, "edges": REVIEW_EDGES, **rev})
    print("trader done", flush=True)

    # tool agent: Gemini replays from cache, else stub
    try:
        from forecast.impact import load_impact
        impact = load_impact()
    except Exception:
        impact = None
    gem = []
    seen = set()
    for gid, at, status, reason, model, setup in gemini_candidates():
        if (gid, at, model) in seen or len(gem) >= 3:
            continue
        seen.add((gid, at, model))
        want = status in ("invalid", "gap", "pass") and sum(g["status"] == status for g in gem) < 3
        if not want:
            continue
        tr = run_tool_agent(ctx, gid, at, model, False, impact,
                            f"Gemini tool agent: {status} ({reason})")
        if tr:
            gem.append(tr)
    for i, tr in enumerate(gem):
        write(f"tool_gemini_{i + 1}", tr)
    print(f"gemini replays: {len(gem)}", flush=True)
    write("tool_stub_cle_por", run_tool_agent(ctx, CLE_POR, CLE_POR_AT, "heuristic", True, impact,
                                              "Stub LLM tool agent + sceptic: CLE @ POR"))
    rej = None
    stub_run = ROOT / "runs" / "llm_agent" / "test-stub" / "tool_sceptic" / "trace.jsonl"
    if stub_run.exists():
        for line in stub_run.read_text().splitlines():
            d = json.loads(line)
            if d.get("status") == "sceptic_reject":
                rej = run_tool_agent(ctx, d["game_id"], d["as_of"], "heuristic", True, impact,
                                     "Stub LLM tool agent: sceptic rejects (already priced in)")
                if rej and rej["status"] == "sceptic_reject":
                    break
    if rej:
        write("tool_stub_sceptic_reject", rej)
    print("tool agent done", flush=True)

    write("coach_cle_por", run_coach(ctx, CLE_POR, CLE_POR_AT, "Coach: CLE @ POR"))
    if pass_t:
        write("coach_pass", run_coach(ctx, pass_t["game_id"], pass_t["as_of"], f"Coach: {pass_t['matchup']}"))
    print("coach done", flush=True)

    ids = [CLE_POR] + ([pass_t["game_id"]] if pass_t else [])
    orch, pregame = run_night(ctx, NIGHT, ids)
    write("orchestrator_night", orch)
    write("pregame_night", {"agent": "pregame", "label": f"Pregame news agent, night of {NIGHT}", "games": pregame})
    print("orchestrator done", flush=True)
    write("inplay_cle_por", run_inplay(ctx, CLE_POR))
    print("inplay done", flush=True)

    cle = {"trader": "trader_cle_por", "tool_agent": "tool_stub_cle_por", "coach": "coach_cle_por",
           "orchestrator": "orchestrator_night", "pregame": "pregame_night", "inplay": "inplay_cle_por",
           "review": "review"}
    add("cle_por", "CLE @ POR, 1 Feb 2026: Avdija out (trade, CLV +7.5c)",
        "Portland's Deni Avdija is ruled out. The model shifts Portland's chance down, the agent buys POR 'no' at "
        "55c and the close at 62.5c grades it +7.5c of closing-line value.", cle)
    add("retry", "Planted fault: future news citation (checks -> retry -> pass)",
        "A fault injected for the demo makes the investigator cite news published after the decision. The checks "
        "reject it, the graph loops back once, and the clean retry passes.", {"trader": "trader_cle_por_retry"})
    add("blocked", "Planted violation: 'lock' wording (blocked)",
        "The brief says 'This one is a lock.' on every try. Checks fail twice, the order is blocked and logged.",
        {"trader": "trader_cle_por_blocked"})
    add("kill", "Kill switch tripped (risk blocks)",
        "Same trade, but the day's realised losses already exceeded the limit: the risk node blocks every order.",
        {"trader": "trader_cle_por_kill"})
    if pass_t:
        add("pass", f"{pass_t['matchup']}: pass (priced in)",
            "No edge after fees: the agent writes a brief and places nothing. Passing is a decision.",
            {"trader": "trader_pass", "coach": "coach_pass", "orchestrator": "orchestrator_night"})
    if rule_t:
        add("rule", f"{rule_t['matchup']}: learned rule {rule_t.get('rule', {}).get('rule_id', '')} vetoes a trade",
            "The gap clears the threshold but a rule the reviewer learned and the gate accepted (valid from before "
            "this decision) says skip.", {"trader": "trader_rule", "review": "review"})
    if rej:
        add("sceptic", f"{rej['matchup']}: priced-in sceptic rejects (stub LLM)",
            "The analyst finds an edge, code validates it, and the sceptic rejects because the price already moved.",
            {"tool_agent": "tool_stub_sceptic_reject"})
    for i, tr in enumerate(gem):
        add(f"gemini{i + 1}", f"{tr['matchup']}: real Gemini tool agent ({tr['status']})",
            f"Replayed from the cached Gemini responses ({tr['llm']}). Outcome: {tr['status']} - {tr.get('reason')}.",
            {"tool_agent": f"tool_gemini_{i + 1}"})
    write("index", {"scenarios": scenarios, "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "note": "Decision steps use only data available at the decision time; settlement is revealed at "
                            "tip-off."})


def live(gid, at):
    ctx = Ctx()
    print(json.dumps(clean(run_trader(ctx, gid, pd.Timestamp(at, tz="UTC") if pd.Timestamp(at).tzinfo is None
                                      else pd.Timestamp(at), label="Live run")), default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", nargs=2, metavar=("GAME_ID", "AS_OF_UTC"))
    a = ap.parse_args()
    if a.live:
        live(*a.live)
    else:
        build_all()
