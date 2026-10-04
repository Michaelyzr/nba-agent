"""A6: the agent loop as a LangGraph, wired into replay.py.

    python -m agents.graph --draw                                          # Mermaid of the compiled graph
    python -m agents.graph --source synthetic --start 2026-02-01 --end 2026-02-28 --name agent-feb
    python -m agents.graph --source frozen --start 2026-02-01 --end 2026-02-28 --llm

One graph, two phases. "decide" runs at every replay decision time:

    trigger -> news investigator (LLM) -> forecast models -> market analyst (LLM)
            -> propose | no_action -> checks (one retry) -> risk -> confirm | blocked -> deliver

"review" runs after each replay day:

    settle -> reviewer (LLM) -> gate (back-test on earlier days only) -> save_rule | reject_rule

LLM steps fall back to offline rules when no key is set or --llm is off, so the
loop runs with no network. The LLMs choose and explain; code computes every
number, and an LLM can only drop a candidate trade, never add one.
"""
import argparse
import json
import math
import operator
import re
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

import llm
import replay
from agents.notebook import LIMITS, Notebook, validate
from replay import MAX_QUOTE_AGE, NEWS_WINDOW, Order, Replay

DEFAULT_MIN_EDGE = 0.04          # probability gap after fees needed to trade
DEFAULT_STAKE = 20.0
MAX_RETRIES = 1                  # checks give the agent one retry, then block
OUT_STATUSES = {"out", "doubtful"}
OUT_MINUTE_VALUE = 0.003         # win probability per usual minute of a missing player (until M4)
HOME_EDGE = 0.03
NO_NEWS_AGE = 9999.0             # news_age_minutes when a game has no news yet
STALE_NEWS_MINUTES = 30.0
GATE_DAYS = 14
GATE_THRESHOLD = 0.005           # mean closing-line value must improve by this much
BANNED = re.compile(r"\block\b|guarantee|can'?t lose|sure thing|risk[- ]free", re.I)
PCT = re.compile(r"(\d+)%")
FAULTS = ("future_citation", "lock_wording", "number_mismatch")


class State(TypedDict, total=False):
    phase: str
    plant: Optional[str]
    # decide
    view: Any
    game: Any
    now: pd.Timestamp
    news: pd.DataFrame
    markets: pd.DataFrame
    investigation: dict
    forecasts: dict
    candidates: list
    orders: list
    brief: str
    failures: list
    tries: int
    status: str
    # review
    day: str
    replay: Any
    proposal: Optional[dict]
    gate: dict
    trace: Annotated[list, operator.add]


def log(step, detail):
    return [{"step": step, "detail": detail}]


def fee_per_contract(price: float) -> float:
    return 0.07 * price * (1 - price)


def clip(p: float) -> float:
    return float(min(max(p, 0.02), 0.98))


def minutes_between(a, b) -> float:
    return pd.Timedelta(b - a).total_seconds() / 60


# ---------------- placeholder forecaster (until forecast/api.py, M5) ----------------

def win_rate(games: pd.DataFrame, team_id) -> float:
    """Season win rate so far, shrunk toward 0.5 with five wins and five losses."""
    home, away = games[games.home_team_id == team_id], games[games.away_team_id == team_id]
    wins = (home.home_pts > home.away_pts).sum() + (away.away_pts > away.home_pts).sum()
    return (wins + 5) / (len(home) + len(away) + 10)


def log5_home(h: float, a: float) -> float:
    return h * (1 - a) / (h * (1 - a) + a * (1 - h)) + HOME_EDGE


def usual_minutes(player_games: pd.DataFrame, player_id) -> tuple:
    """(team_id, mean minutes over the last 10 games) for a player; player_games is in game order."""
    rows = player_games[player_games.player_id == player_id]
    if rows.empty:
        return None, 0.0
    return rows.team_id.iloc[-1], float(rows["min"].tail(10).mean())


def record_forecaster(view, game, markets: pd.DataFrame, overrides: dict) -> dict:
    """P(yes) per game-winner market: win rates to date, log5, home edge, minus missing minutes."""
    done = view.games().dropna(subset=["home_pts"])
    p_home = log5_home(win_rate(done, game.home_team_id), win_rate(done, game.away_team_id))
    pg = view.player_games()
    for pid in overrides.get("out", []):
        team, minutes = usual_minutes(pg, pid)
        if team == game.home_team_id:
            p_home -= OUT_MINUTE_VALUE * minutes
        elif team == game.away_team_id:
            p_home += OUT_MINUTE_VALUE * minutes
    p_home = clip(p_home)
    return {m.market_ticker: p_home if m.team == game.home_team else 1 - p_home
            for m in markets.itertuples() if m.kind == "game"}


# ---------------- risk and confirmation defaults ----------------

def pretrade_risk(order: Order, ctx: dict, max_order=50.0):
    """Stateless limits checked inside the graph; replay.py enforces game and day caps again."""
    if ctx["now"] >= ctx["tip_time"]:
        return False, "post_tip"
    if order.channel in ("team", "media"):
        return False, "channel_cannot_order"
    if not order.reason.strip():
        return False, "missing_reason"
    if order.side not in ("yes", "no"):
        return False, "bad_side"
    if order.stake > max_order:
        return False, "over_order_cap"
    return True, "approved"


def auto_confirm(brief: str, orders: list) -> list:
    """Replay stands in for the user and confirms every approved order."""
    return orders


class MarketAgent:
    def __init__(self, notebook: Notebook | None = None, forecaster: Callable = record_forecaster,
                 risk: Callable = pretrade_risk, confirm: Callable = auto_confirm, use_llm: bool = False,
                 learn: bool = True, stake: float = DEFAULT_STAKE, channel: str = "platform", plant=None):
        self.notebook = notebook or Notebook()
        self.forecaster = forecaster
        self.risk = risk
        self.confirm = confirm
        self.use_llm = use_llm and llm.available()
        self.learn = learn
        self.stake = stake
        self.channel = channel
        self.plant = plant
        self.situations = {}            # (market_ticker, as_of) -> situation at decision time
        self.held = set()               # game_ids with a confirmed order
        self.traces = []
        self.graph = self.build()

    @property
    def llm_name(self):
        return f"gemini/{llm.MODEL}" if self.use_llm else "offline"

    # ---------------- decide phase ----------------

    def trigger(self, state):
        view, game, now = state["view"], state["game"], state["now"]
        news = view.news(game.game_id)
        news = news[news.published_at >= game.tip_time - NEWS_WINDOW]
        markets = view.markets(game.game_id)
        live = [m for m in markets.market_ticker if self._fresh_quote(view, m, now) is not None]
        markets = markets[markets.market_ticker.isin(live)]
        status = "running" if len(markets) else "idle"
        return {"news": news, "markets": markets, "status": status, "orders": [], "failures": [],
                "trace": log("trigger", f"{game.away_team}@{game.home_team} {now:%H:%M} UTC: "
                                        f"{len(news)} news items, {len(markets)} markets with fresh quotes")}

    def investigate(self, state):
        news, now = state["news"], state["now"]
        latest = news.sort_values("published_at").groupby("player_id").tail(1)
        offline = {"out": [p for p, s in zip(latest.player_id, latest.status) if str(s).lower() in OUT_STATUSES],
                   "citations": latest.news_id.tolist(), "note": "official statuses"}
        found, how = offline, "offline rules"
        if self.use_llm and len(latest):
            try:
                found, how = self._investigate_llm(latest), self.llm_name
            except Exception as exc:
                how = f"offline rules ({self.llm_name} failed: {exc.__class__.__name__})"
        situations = [{"ruled_out_player": p, "status": "out",
                       "hours_to_tip": minutes_between(now, state["game"].tip_time) / 60} for p in found["out"]]
        rules = {r["rule_id"]: r for s in situations for r in self.notebook.matching(s, now, "forecast")}
        overrides = {"out": list(found["out"])}
        for r in rules.values():
            overrides.setdefault(r["do"]["action"], []).append(r["do"]["params"])
        citations = list(found["citations"])
        trace = log("investigate", f"[{how}] out={overrides['out']} rules={list(rules)}")
        if self._faulty(state, "future_citation"):
            citations.append("planted-future-news")
            trace += log("fault", "injected for the demo: cited news published after the decision")
        return {"investigation": {"overrides": overrides, "citations": citations, "note": found.get("note", ""),
                                  "rules_applied": list(rules)}, "trace": trace}

    def _investigate_llm(self, latest):
        rows = latest[["news_id", "player_id", "status", "text"]].astype(str).to_dict("records")
        prompt = ("You read NBA injury news for a forecasting system. Return JSON with keys out (player_id strings "
                  "of players who will not play), citations (news_id strings you relied on) and note (one sentence). "
                  "Use only these rows; do not guess.\n" + json.dumps(rows))
        raw = llm.ask(prompt, want_json=True)
        ids = {str(p): p for p in latest.player_id}
        news_ids = set(latest.news_id)
        out = [ids[str(p)] for p in raw.get("out", []) if str(p) in ids]
        cites = [c for c in raw.get("citations", []) if c in news_ids]
        if len(out) != len(raw.get("out", [])) or len(cites) != len(raw.get("citations", [])):
            raise ValueError("LLM named a player or news item that is not in the rows")
        return {"out": out, "citations": cites, "note": str(raw.get("note", ""))[:200]}

    def forecast(self, state):
        view, game, markets = state["view"], state["game"], state["markets"]
        overrides = state["investigation"]["overrides"]
        before = self.forecaster(view, game, markets, {"out": []})
        after = self.forecaster(view, game, markets, overrides)
        forecasts = {t: {"before": before[t], "after": after[t]} for t in after}
        shown = ", ".join(f"{t.rsplit('-', 1)[-1]} {f['before']:.2f}->{f['after']:.2f}" for t, f in forecasts.items())
        return {"forecasts": forecasts, "trace": log("forecast", shown or "no market the models cover")}

    def analyse(self, state):
        view, game, now, news = state["view"], state["game"], state["now"], state["news"]
        markets = state["markets"].set_index("market_ticker")
        age = minutes_between(news.published_at.max(), now) if len(news) else NO_NEWS_AGE
        rows = []
        for ticker, f in state["forecasts"].items():
            m, q = markets.loc[ticker], self._fresh_quote(view, ticker, now)
            if q is None:
                continue
            situation = {"team": m.team, "opponent": game.away_team if m.team == game.home_team else game.home_team,
                         "market_kind": m.kind, "hours_to_tip": minutes_between(now, game.tip_time) / 60,
                         "news_age_minutes": float(age)}
            self.situations[(ticker, now)] = situation
            rules = self.notebook.matching(situation, now, "trader")
            p = f["after"]
            yes_gap = p - q.ask - fee_per_contract(q.ask)
            no_gap = (1 - p) - (1 - q.bid) - fee_per_contract(1 - q.bid)
            side, gap = ("yes", yes_gap) if yes_gap >= no_gap else ("no", no_gap)
            min_edge = max([DEFAULT_MIN_EDGE] + [r["do"]["params"]["edge"] for r in rules
                                                 if r["do"]["action"] == "min_edge"])
            scale = min([1.0] + [r["do"]["params"]["scale"] for r in rules if r["do"]["action"] == "stake_scale"])
            skip = [r["rule_id"] for r in rules if r["do"]["action"] == "skip_market"]
            rows.append({"ticker": ticker, "team": m.team, "kind": m.kind, "p": p, "before": f["before"],
                         "bid": float(q.bid), "ask": float(q.ask), "side": side, "gap": float(gap),
                         "min_edge": min_edge, "scale": scale, "rules": [r["rule_id"] for r in rules],
                         "act": gap > min_edge and not skip, "why_not": "rule " + ", ".join(skip) if skip else
                         ("" if gap > min_edge else "gap below threshold")})
        # Both teams' markets price the same game: hold at most one position per game.
        best = max((r for r in rows if r["act"]), key=lambda r: r["gap"], default=None)
        for r in rows:
            if r["act"] and (game.game_id in self.held or r is not best):
                r.update(act=False, why_not="one position per game")
        how = "offline rules"
        if self.use_llm and any(r["act"] for r in rows):
            try:
                self._analyse_llm(state, rows)
                how = self.llm_name
            except Exception as exc:
                how = f"offline rules ({self.llm_name} failed: {exc.__class__.__name__})"
        acted = [r["ticker"].rsplit("-", 1)[-1] for r in rows if r["act"]]
        return {"candidates": rows, "trace": log("analyse", f"[{how}] act on {acted or 'nothing'}; "
                                                          + "; ".join(f"{r['team']} gap {r['gap']:+.3f}" for r in rows))}

    def _analyse_llm(self, state, rows):
        """The analyst may drop a candidate (e.g. price already moved) and explain; it cannot add one."""
        table = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                  if k in ("ticker", "team", "p", "before", "bid", "ask", "side", "gap", "act")} for r in rows]
        prompt = ("You are a sports-market analyst. Code computed these model probabilities and gaps after fees. "
                  "For each row with act=true decide whether the market has already priced the news in. Return JSON "
                  "{\"decisions\": [{\"ticker\": ..., \"act\": true|false, \"why\": one sentence, no numbers}]}. "
                  "Never promise an outcome.\nNews: " + json.dumps(state["investigation"]["note"])
                  + "\nRows: " + json.dumps(table))
        raw = llm.ask(prompt, want_json=True)
        by_ticker = {d.get("ticker"): d for d in raw.get("decisions", [])}
        for r in rows:
            d = by_ticker.get(r["ticker"])
            if r["act"] and d is not None:
                r["act"] = bool(d.get("act")) and r["act"]
                r["why"] = str(d.get("why", ""))[:200]
                if not r["act"]:
                    r["why_not"] = "analyst: " + r["why"]

    def _line(self, r):
        price = r["ask"] if r["side"] == "yes" else 1 - r["bid"]
        p_side = r["p"] if r["side"] == "yes" else 1 - r["p"]
        moved = f" ({r['before']:.0%} before the news)" if f"{r['before']:.0%}" != f"{r['p']:.0%}" else ""
        return (f"{r['team']} win: model {r['p']:.0%}{moved}, market "
                f"{r['bid']:.0%}-{r['ask']:.0%}; buying {r['side']} at {price:.0%} for a {p_side:.0%} chance")

    def _headline(self, state):
        news = state["news"]
        if news.empty:
            return "No injury news for this game yet."
        latest = news.sort_values("published_at").groupby("player_id").tail(1)
        return " ".join(f"{t} ({pd.Timestamp(ts):%H:%M} UTC)." for t, ts in zip(latest.text, latest.published_at))

    def propose(self, state):
        inv = state["investigation"]
        orders, lines = [], [self._headline(state)]
        for r in state["candidates"]:
            if not r["act"]:
                lines.append(f"{r['team']} win: model {r['p']:.0%}, market {r['bid']:.0%}-{r['ask']:.0%}; "
                             f"no action ({r['why_not']}).")
                continue
            reason = self._line(r) + (f". {r['why']}" if r.get("why") else "")
            orders.append(Order(r["ticker"], r["side"], r["p"], round(self.stake * r["scale"], 2), reason,
                                list(inv["citations"]), self.channel, inv["rules_applied"] + r["rules"],
                                self.llm_name))
            lines.append(reason + ". Possible value; confirm?")
        trace = log("propose", f"{len(orders)} orders")
        if self._faulty(state, "number_mismatch") and orders:
            orders[0].p_model = clip(orders[0].p_model + 0.1)
            trace += log("fault", "injected for the demo: order probability differs from the model")
        if state.get("plant") == "lock_wording":
            lines.append("This one is a lock.")
            trace += log("fault", "injected for the demo: 'lock' wording in the brief")
        return {"orders": orders, "brief": "\n".join(lines), "trace": trace}

    def no_action(self, state):
        lines = [self._headline(state)]
        for r in state["candidates"]:
            lines.append(f"{r['team']} win: model {r['p']:.0%}, market {r['bid']:.0%}-{r['ask']:.0%}; "
                         f"already priced in, no action ({r['why_not']}).")
        return {"orders": [], "brief": "\n".join(lines), "trace": log("no_action", "brief only")}

    def checks(self, state):
        view, now, inv = state["view"], state["now"], state["investigation"]
        failures = []
        public = set(view.news().news_id)
        cited = set(inv["citations"]) | {c for o in state["orders"] for c in o.citations}
        for c in sorted(cited - public):
            failures.append(("citation_after_decision", f"{c} was not public at {now}"))
        for o in state["orders"]:
            f = state["forecasts"].get(o.market_ticker)
            if f is None or not math.isclose(o.p_model, f["after"], abs_tol=1e-9):
                failures.append(("number_mismatch", f"{o.market_ticker} p_model {o.p_model} is not the model's"))
            if not o.reason.strip():
                failures.append(("missing_reason", o.market_ticker))
        text = state["brief"] + " ".join(o.reason for o in state["orders"])
        if BANNED.search(text):
            failures.append(("banned_wording", BANNED.search(text).group(0)))
        allowed = {f"{v:.0%}" for r in state["candidates"] for v in
                   (r["p"], r["before"], 1 - r["p"], r["bid"], r["ask"], 1 - r["bid"], 1 - r["ask"])}
        for token in PCT.findall(text):
            if f"{token}%" not in allowed:
                failures.append(("number_mismatch", f"{token}% in the brief is not a model or market number"))
        active = {r["rule_id"] for r in self.notebook.active(now)}
        for rid in {rid for o in state["orders"] for rid in o.rules_applied} | set(inv["rules_applied"]):
            if rid not in active:
                failures.append(("rule_not_active", rid))
        if not failures:
            return {"failures": [], "trace": log("checks", "passed")}
        tries = state.get("tries", 0) + 1
        detail = f"rejected (try {tries} of {MAX_RETRIES + 1}): " + "; ".join(f"{a}: {b}" for a, b in failures)
        return {"failures": failures, "tries": tries, "trace": log("checks", detail)}

    def risk_limits(self, state):
        ctx = {"now": state["now"], "tip_time": state["game"].tip_time}
        kept, notes = [], []
        for o in state["orders"]:
            ok, why = self.risk(o, ctx)
            notes.append(f"{o.market_ticker.rsplit('-', 1)[-1]} {why}")
            if ok:
                kept.append(o)
        return {"orders": kept, "trace": log("risk", "; ".join(notes))}

    def confirm_orders(self, state):
        orders = self.confirm(state["brief"], state["orders"])
        if orders:
            self.held.add(state["game"].game_id)
        return {"orders": orders, "status": "sent", "trace": log("confirm", f"{len(orders)} confirmed")}

    def blocked(self, state):
        why = "checks failed twice" if state.get("failures") else "risk limits"
        brief = state.get("brief", "") + f"\nOrder blocked ({why}); reason logged."
        return {"orders": [], "status": "blocked", "brief": brief, "trace": log("blocked", why)}

    def deliver(self, state):
        status = state.get("status") if state.get("status") in ("sent", "blocked") else "brief_only"
        return {"status": status, "trace": log("deliver", status)}

    # ---------------- review phase ----------------

    def settle(self, state):
        rp, day = state["replay"], state["day"]
        today = pd.DataFrame(rp.fills)
        today = today[today.game_id.isin(rp.t["games"].loc[rp.t["games"].date == day, "game_id"])] if len(today) else today
        s = replay.summary(today)
        return {"trace": log("settle", f"{day}: {json.dumps({k: round(v, 3) for k, v in s.items()})}")}

    def review(self, state):
        fills = pd.DataFrame(state["replay"].fills)
        if fills.empty:
            return {"proposal": None, "trace": log("review", "no settled trades yet")}
        fills["news_age"] = [self.situations.get((t, a), {}).get("news_age_minutes", np.nan)
                             for t, a in zip(fills.market_ticker, fills.as_of)]
        proposal, how = self._review_offline(fills), "offline rules"
        if self.use_llm:
            try:
                proposal, how = self._review_llm(fills), self.llm_name
            except Exception as exc:
                how = f"offline rules ({self.llm_name} failed: {exc.__class__.__name__})"
        games = state["replay"].t["games"]
        now = games.loc[games.date == state["day"], "final_at"].max()
        if proposal and (validate(proposal) or self.notebook.seen(proposal, now)):
            why = "; ".join(validate(proposal)) or "already proposed before"
            return {"proposal": None, "trace": log("review", f"[{how}] dropped proposal: {why}")}
        detail = f"[{how}] " + (json.dumps({k: proposal[k] for k in ("when", "do", "blame_category")})
                                if proposal else "no pattern worth a rule")
        return {"proposal": proposal, "trace": log("review", detail)}

    def _review_offline(self, fills):
        stale = fills[fills.news_age >= STALE_NEWS_MINUTES]
        fresh = fills[fills.news_age < STALE_NEWS_MINUTES]
        if len(stale) >= LIMITS["min_cases"] and stale.clv.mean() < 0:
            return self._rule({"market_kind": "game", "news_age_minutes_min": STALE_NEWS_MINUTES},
                              {"action": "skip_market", "params": {}}, "market_priced_in",
                              f"{len(stale)} trades without fresh news averaged {stale.clv.mean():+.3f} closing-line "
                              "value: with no new information the market price is already right.", stale)
        if len(fresh) >= LIMITS["min_cases"] and fresh.clv.mean() < 0:
            return self._rule({"market_kind": "game"},
                              {"action": "min_edge", "params": {"edge": DEFAULT_MIN_EDGE + 0.02}}, "model_wrong",
                              f"{len(fresh)} news trades averaged {fresh.clv.mean():+.3f} closing-line value.", fresh)
        return None

    def _review_llm(self, fills):
        bad = fills[fills.clv < 0]
        sample = bad[["market_ticker", "side", "p_model", "price", "close_price", "clv", "pnl", "news_age"]].round(3)
        prompt = ("You review losing sports-market trades. Pick one blame_category from news_misread, minutes_wrong, "
                  "model_wrong, market_priced_in, and propose at most one rule. Return JSON {\"rule\": null or "
                  "{\"when\": {...}, \"do\": {\"action\": ..., \"params\": {...}}}, \"blame_category\": ..., "
                  "\"rationale\": one sentence}. Allowed when fields: market_kind, team, opponent, hours_to_tip_min, "
                  "hours_to_tip_max, news_age_minutes_min, news_age_minutes_max. Allowed actions: skip_market (no "
                  "params), min_edge ({\"edge\": >0.04}), stake_scale ({\"scale\": 0-1}).\nTrades: "
                  + sample.to_json(orient="records"))
        raw = llm.ask(prompt, want_json=True)
        if not raw.get("rule"):
            return None
        return self._rule(raw["rule"].get("when", {}), raw["rule"].get("do", {}), raw.get("blame_category"),
                          str(raw.get("rationale", ""))[:300], bad)

    def _rule(self, when, do, blame, rationale, source):
        return {"rule_id": self.notebook.next_id(), "version": 1, "kind": "trader", "status": "proposed",
                "when": when, "do": do, "rationale": rationale, "proposed_by": f"reviewer@{self.llm_name}",
                "blame_category": blame, "source_trades": source.decision_id.tolist()[-10:], "priority": 1,
                "expires_after_days": LIMITS["default_expiry_days"]}

    def gate(self, state):
        rp, day, rule = state["replay"], state["day"], state["proposal"]
        games = rp.t["games"]
        traded = games[games.game_id.isin(rp.t["markets"].game_id) & (games.date < day)]
        days = sorted(traded.date.unique())[-GATE_DAYS:]
        decided_at = games.loc[games.date == day, "final_at"].max()
        evidence = {"decided_at": decided_at, "backtest_days": [days[0], days[-1]] if days else [],
                    "metric": "mean_clv", "threshold": GATE_THRESHOLD}
        if len(days) < 2:
            return {"gate": {**evidence, "result": "deferred", "reason": "not enough earlier days"},
                    "trace": log("gate", "deferred: not enough earlier days")}
        without = self.backtest(rp, self.notebook.without(rule), days[0], days[-1])
        with_rule = self.backtest(rp, self.notebook.with_candidate(rule), days[0], days[-1])
        before = float(without.clv.mean()) if len(without) else 0.0
        after = float(with_rule.clv.mean()) if len(with_rule) else 0.0
        cases = abs(len(without) - len(with_rule))
        pnl = [float(f.pnl.sum()) if len(f) else 0.0 for f in (without, with_rule)]
        ok = cases >= LIMITS["min_cases"] and after - before >= GATE_THRESHOLD
        reason = (f"mean CLV {before:+.4f} -> {after:+.4f} over {cases} changed trades, "
                  f"P&L {pnl[0]:+.2f} -> {pnl[1]:+.2f}")
        evidence.update(result="accepted" if ok else "rejected", cases=cases, before=before, after=after,
                        pnl_before=pnl[0], pnl_after=pnl[1], reason=reason)
        return {"gate": evidence, "trace": log("gate", ("accepted: " if ok else "rejected: ") + reason)}

    def backtest(self, rp, notebook, start, end) -> pd.DataFrame:
        """Replay earlier days with an offline, non-learning copy of this agent."""
        child = MarketAgent(notebook, self.forecaster, self.risk, self.confirm, use_llm=False, learn=False,
                            stake=self.stake, channel=self.channel)
        _, fills = Replay(rp.t, child.policy, risk=rp.risk, fee=rp.fee).run(start, end)
        return fills

    def save_rule(self, state):
        g = state["gate"]
        old = [r["rule_id"] for r in self.notebook.rules
               if (r["when"], r["do"]) == (state["proposal"]["when"], state["proposal"]["do"])]
        rule = {**state["proposal"], "status": "active", "proposed_at": g["decided_at"], "gate": g,
                "valid_from": g["decided_at"], "supersedes": old[-1] if old else None}
        self.notebook.record(rule)
        return {"trace": log("save_rule", f"{rule['rule_id']} active from {g['decided_at']}")}

    def reject_rule(self, state):
        g = state["gate"]
        if g["result"] == "deferred":
            return {"trace": log("reject_rule", f"not recorded, the reviewer may propose it again: {g['reason']}")}
        rule = {**state["proposal"], "status": "rejected", "proposed_at": g["decided_at"], "gate": g,
                "valid_from": None}
        self.notebook.record(rule)
        return {"trace": log("reject_rule", f"{rule['rule_id']}: {g['reason']}")}

    # ---------------- wiring ----------------

    def _fresh_quote(self, view, ticker, now):
        q = view.quote(ticker)
        return None if q is None or now - q.ts > MAX_QUOTE_AGE else q

    def _faulty(self, state, kind):
        return state.get("plant") == kind and state.get("tries", 0) == 0

    def build(self):
        g = StateGraph(State)
        for name, fn in [("trigger", self.trigger), ("investigate", self.investigate), ("forecast", self.forecast),
                         ("analyse", self.analyse), ("propose", self.propose), ("no_action", self.no_action),
                         ("checks", self.checks), ("risk", self.risk_limits), ("confirm", self.confirm_orders),
                         ("blocked", self.blocked), ("deliver", self.deliver), ("settle", self.settle),
                         ("review", self.review), ("gate", self.gate), ("save_rule", self.save_rule),
                         ("reject_rule", self.reject_rule)]:
            g.add_node(name, fn)

        g.add_conditional_edges(START, lambda s: "settle" if s["phase"] == "review" else "trigger",
                                ["trigger", "settle"])
        g.add_conditional_edges("trigger", lambda s: END if s["status"] == "idle" else "investigate",
                                ["investigate", END])
        g.add_edge("investigate", "forecast")
        g.add_edge("forecast", "analyse")
        g.add_conditional_edges("analyse", lambda s: "propose" if any(r["act"] for r in s["candidates"])
                                else "no_action", ["propose", "no_action"])
        g.add_edge("propose", "checks")
        g.add_edge("no_action", "checks")
        g.add_conditional_edges("checks", self._after_checks, ["risk", "deliver", "investigate", "blocked"])
        g.add_conditional_edges("risk", lambda s: "confirm" if s["orders"] else "blocked", ["confirm", "blocked"])
        g.add_edge("confirm", "deliver")
        g.add_edge("blocked", "deliver")
        g.add_edge("deliver", END)

        g.add_edge("settle", "review")
        g.add_conditional_edges("review", lambda s: "gate" if s.get("proposal") else END, ["gate", END])
        g.add_conditional_edges("gate", lambda s: "save_rule" if s["gate"]["result"] == "accepted"
                                else "reject_rule", ["save_rule", "reject_rule"])
        g.add_edge("save_rule", END)
        g.add_edge("reject_rule", END)
        return g.compile()

    @staticmethod
    def _after_checks(state):
        if state["failures"]:
            return "investigate" if state["tries"] <= MAX_RETRIES else "blocked"
        return "risk" if state["orders"] else "deliver"

    # ---------------- replay plug-ins ----------------

    def decide(self, view, game, now) -> dict:
        out = self.graph.invoke({"phase": "decide", "view": view, "game": game, "now": now, "tries": 0,
                                 "plant": self.plant, "trace": []})
        self.traces.append({"phase": "decide", "game_id": game.game_id, "as_of": str(now),
                            "status": out.get("status"), "brief": out.get("brief", ""), "trace": out["trace"]})
        return out

    def policy(self, view, game, now) -> list:
        return self.decide(view, game, now).get("orders", [])

    def on_day_end(self, day, fills, rp):
        if not self.learn:
            return
        out = self.graph.invoke({"phase": "review", "day": day, "replay": rp, "trace": []})
        self.traces.append({"phase": "review", "day": day, "trace": out["trace"]})


def load_source(source, start, end):
    if source == "synthetic":
        from agents.demo_data import synthetic_replay_tables
        return synthetic_replay_tables(start, end)
    return replay.load_tables()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true", help="print the compiled graph as Mermaid and exit")
    ap.add_argument("--source", choices=["synthetic", "frozen"], default="synthetic")
    ap.add_argument("--start", default="2026-02-01")
    ap.add_argument("--end", default="2026-02-28")
    ap.add_argument("--name", default="agent")
    ap.add_argument("--llm", action="store_true", help="use the chat model when a key is set")
    ap.add_argument("--no-learn", action="store_true", help="the no-learning ablation")
    ap.add_argument("--notebook", type=Path, default=None, help="start from this notebook (default: empty)")
    ap.add_argument("--plant", choices=FAULTS, default=None)
    args = ap.parse_args()

    agent = MarketAgent(Notebook.load(args.notebook) if args.notebook else Notebook(), use_llm=args.llm,
                        learn=not args.no_learn, plant=args.plant)
    if args.draw:
        print(agent.graph.get_graph().draw_mermaid())
        return
    tables = load_source(args.source, args.start, args.end)
    decisions, fills = Replay(tables, agent.policy, on_day_end=agent.on_day_end).run(args.start, args.end)
    out = replay.RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(out / "decisions.parquet", index=False)
    fills.to_parquet(out / "fills.parquet", index=False)
    agent.notebook.save(out / "notebook.json")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(t, default=str) for t in agent.traces))
    print(f"{len(decisions)} decisions, {len(fills)} fills -> {out}")
    print(replay.summary(fills))
    for r in agent.notebook.rules:
        print(f"{r['rule_id']} {r['status']}: when {r['when']} do {r['do']['action']} ({r['gate']['reason']})")


if __name__ == "__main__":
    main()
