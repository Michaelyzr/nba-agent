"""A6: the agent loop as a LangGraph, wired into replay.py.

    python -m agents.graph --draw                                          # Mermaid of the compiled graph
    python -m agents.graph --source synthetic --start 2026-02-01 --end 2026-02-28 --name agent-feb
    python -m agents.graph --source frozen --start 2026-02-01 --end 2026-02-28 --llm

One graph, two phases. "decide" runs at every replay decision time:

    trigger -> news investigator (LLM) -> forecast models -> market analyst (LLM)
            -> propose | no_action -> checks (one retry) -> risk -> confirm | blocked -> deliver

"review" runs after each replay day:

    settle -> reviewer (LLM) -> gate (back-test on earlier days only) -> save_rule | reject_rule

Two opt-in learning upgrades (off by default, so published runs reproduce):
--edge-templates (adaptive_edge=True) lets the reviewer move the market-wide edge threshold up or
down from --base-edge; --model-review (model_review=True) adds a second subloop after the rule
subloop, review_model -> gate_model -> save_model | reject_model, that chooses the forecast blend.

LLM steps fall back to offline rules when no key is set or --llm is off, so the
loop runs with no network. The LLMs choose and explain; code computes every
number, and an LLM can only drop a candidate trade, never add one.

Gate modes: "split" selects the rule on the last SELECT_DAYS market days and
back-tests it on the GATE_DAYS before them, judged on total CLV dollars;
"split-edge" is the same judged on edge over the entry mid; "legacy" is the
published version (21-day selection overlapping a 14-day test, judged on mean
CLV per trade). The command line defaults to split with the $100 kill switch;
MarketAgent() and Replay() keep legacy and no kill switch so the published
evaluation scripts reproduce. See docs/preregistration_gate.md.
"""
import argparse
import hashlib
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
ANCHOR_LEAD = pd.Timedelta(hours=24)  # market price this long before tip is the anchored base rate
GATE_DAYS = 14
GATE_THRESHOLD = 0.005           # legacy: mean closing-line value must improve by this much
GATE_DOLLARS = 2.0               # split gates: total CLV (or edge) dollars must improve by this much
SELECT_DAYS = 7                  # split gates: the reviewer selects on these most recent market days
GATE_MODES = ("split", "split-edge", "legacy")
SIZINGS = ("flat", "kelly")
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_BANKROLL = 1000.0
MAX_ORDER = 50.0
BANNED = re.compile(r"\block\b|guarantee|can'?t lose|sure thing|risk[- ]free", re.I)
PCT = re.compile(r"(\d+)%")
FAULTS = ("future_citation", "lock_wording", "number_mismatch")
REVIEW_WINDOW = pd.Timedelta(days=21)
SKIP = {"action": "skip_market", "params": {}}
REVIEW_TEMPLATES = [   # (condition, action, blame category, how the reviewer describes the slice)
    ({"side_price_max": 0.25}, SKIP, "model_wrong", "the agent bought a long shot (price 25% or less)"),
    ({"side_price_max": 0.35}, SKIP, "model_wrong", "the agent bought an underdog (price 35% or less)"),
    ({"side_price_min": 0.65}, SKIP, "model_wrong", "the agent bought a heavy favourite (price 65% or more)"),
    ({"gap_min": 0.10}, SKIP, "model_wrong", "the model disagreed with the market by 10 points or more"),
    ({"gap_min": 0.15}, SKIP, "model_wrong", "the model disagreed with the market by 15 points or more"),
    ({"market_move_max": -0.02}, SKIP, "market_priced_in", "the price had moved against the trade since the anchor"),
    ({"market_move_min": 0.03}, SKIP, "market_priced_in", "the price had already moved 3 points toward the trade"),
    ({"news_age_minutes_min": STALE_NEWS_MINUTES}, SKIP, "market_priced_in", "there was no fresh news"),
    ({"model_shift_max": 0.005}, SKIP, "market_priced_in", "the news barely moved the model"),
    ({"hours_to_tip_min": 0.75}, SKIP, "news_misread", "the trade was placed before the inactive list"),
    ({}, {"action": "min_edge", "params": {"edge": DEFAULT_MIN_EDGE + 0.03}}, "model_wrong",
     "the edge after fees was under 7 points"),
]
# Adaptive edge (MarketAgent(adaptive_edge=True), --edge-templates): the latest accepted market-wide
# min_edge rule sets the threshold, up or down from the base edge; slice min_edge rules only raise it.
EDGE_LEVELS = (0.02, 0.03, 0.05, 0.07, 0.10)
EDGE_TEMPLATES = [({}, {"action": "min_edge", "params": {"edge": e}}, "model_wrong",
                   f"the market-wide edge threshold were {e * 100:.0f} points") for e in EDGE_LEVELS] + [
    ({"side_price_max": 0.35}, {"action": "min_edge", "params": {"edge": 0.08}}, "model_wrong",
     "the agent bought an underdog (35% or less) with under 8 points of edge"),
    ({"side_price_max": 0.25}, {"action": "min_edge", "params": {"edge": 0.10}}, "model_wrong",
     "the agent bought a long shot (25% or less) with under 10 points of edge"),
    ({"news_age_minutes_min": STALE_NEWS_MINUTES}, {"action": "min_edge", "params": {"edge": 0.07}},
     "market_priced_in", "there was no fresh news and the edge was under 7 points"),
]
ADAPTIVE_TEMPLATES = [t for t in REVIEW_TEMPLATES if t[1]["action"] != "min_edge"] + EDGE_TEMPLATES
# Model review (MarketAgent(model_review=True), --model-review): every MODEL_REVIEW_EVERY market days the
# reviewer picks a forecast blend on the last SELECT_DAYS market days and the gate back-tests it on the
# GATE_DAYS before them. Its windows start at MODEL_REVIEW_FROM, the first day no forecast model trained on.
MODEL_REVIEW_FROM = "2026-02-01"
MODEL_REVIEW_EVERY = 7
FORECAST_OVERRIDES = {"minutes_share", "minutes_cap", "p_play_adjust"}


def is_global_edge(rule: dict) -> bool:
    """A min_edge rule with no condition beyond the market kind: it sets the market-wide threshold."""
    return rule["do"]["action"] == "min_edge" and set(rule.get("when", {})) <= {"market_kind"}


def _recency(rule: dict) -> tuple:
    # A gate trial (valid_from None) counts as the newest rule, so it overrides the live threshold.
    start = rule.get("valid_from")
    return (start is None, pd.Timestamp(start) if start is not None else pd.Timestamp(0, tz="UTC"),
            rule.get("rule_id", ""))


def effective_edge(base: float, rules: list, adaptive: bool = False) -> float:
    """Threshold for one decision from the base edge and the matching active trader rules.

    Default: the highest of the base and every min_edge rule (rules can only raise it).
    Adaptive: the newest market-wide min_edge rule replaces the base (raise or lower), then
    slice rules can only raise it.
    """
    edges = [r for r in rules if r["do"]["action"] == "min_edge"]
    if not adaptive:
        return max([base] + [r["do"]["params"]["edge"] for r in edges])
    glob = [r for r in edges if is_global_edge(r)]
    if glob:
        base = max(glob, key=_recency)["do"]["params"]["edge"]
    return max([base] + [r["do"]["params"]["edge"] for r in edges if not is_global_edge(r)])


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
    components: dict
    model_proposal: Optional[dict]
    model_gate: dict
    trace: Annotated[list, operator.add]


def log(step, detail):
    return [{"step": step, "detail": detail}]


def fee_per_contract(price: float) -> float:
    return 0.07 * price * (1 - price)


def clip(p: float) -> float:
    return float(min(max(p, 0.02), 0.98))


def minutes_between(a, b) -> float:
    return pd.Timedelta(b - a).total_seconds() / 60


def placebo_bucket(ticker: str, now) -> float:
    """Stable pseudo-random number in [0, 1) per decision, for placebo rules in the gate audit."""
    digest = hashlib.md5(f"{ticker}|{pd.Timestamp(now).value}".encode()).hexdigest()
    return int(digest[:12], 16) / 16 ** 12


def kelly_fraction(p_side: float, price: float) -> float:
    """Full-Kelly share of bankroll for buying one side at `price` (fee included), 0 if no edge.

    A contract costs k = price + fee and pays 1, so the log-optimal share is (p - k) / (1 - k).
    """
    k = price + fee_per_contract(price)
    return max(p_side - k, 0.0) / (1 - k) if k < 1 else 0.0


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


# ---------------- gate windows and metrics ----------------

GATE_METRICS = {"legacy": ("mean_clv", GATE_THRESHOLD), "split": ("clv_dollars", GATE_DOLLARS),
                "split-edge": ("edge_dollars", GATE_DOLLARS)}


def market_days(rp, through: str) -> list:
    """Dates up to and including `through` that had games with markets."""
    games = rp.t["games"]
    traded = games[games.game_id.isin(rp.t["markets"].game_id) & (games.date <= through)]
    return sorted(traded.date.unique())


def gate_windows(rp, day: str, mode: str) -> tuple:
    """(reviewer selection days, gate test days) for the review after `day`.

    legacy: selection is a 21-day time window (not a day list, so None) and the test is the
    last GATE_DAYS market days before `day`, which overlap it. split modes: the selection is the
    last SELECT_DAYS market days up to `day` and the test is the GATE_DAYS market days before them.
    """
    days = market_days(rp, day)
    if mode == "legacy":
        return None, [d for d in days if d < day][-GATE_DAYS:]
    return days[-SELECT_DAYS:], days[-(SELECT_DAYS + GATE_DAYS):-SELECT_DAYS]


def gate_totals(fills: pd.DataFrame) -> dict:
    if fills is None or fills.empty:
        return {"trades": 0, "mean_clv": 0.0, "clv_dollars": 0.0, "edge_dollars": 0.0, "pnl": 0.0}
    contracts = fills["contracts"] if "contracts" in fills else pd.Series(np.nan, index=fills.index)
    edge = fills["edge"] if "edge" in fills else pd.Series(np.nan, index=fills.index)
    return {"trades": len(fills), "mean_clv": float(fills.clv.mean()),
            "clv_dollars": float((fills.clv * contracts).sum()), "edge_dollars": float((edge * contracts).sum()),
            "pnl": float(fills.pnl.sum())}


def judge(without: pd.DataFrame, with_rule: pd.DataFrame, mode: str) -> dict:
    """Gate verdict: enough changed trades and the mode's metric improves by its threshold."""
    b, a = gate_totals(without), gate_totals(with_rule)
    metric, threshold = GATE_METRICS[mode]
    cases = abs(b["trades"] - a["trades"])
    ok = cases >= LIMITS["min_cases"] and a[metric] - b[metric] >= threshold
    reason = (f"mean CLV {b['mean_clv']:+.4f} -> {a['mean_clv']:+.4f}, CLV $ {b['clv_dollars']:+.2f} -> "
              f"{a['clv_dollars']:+.2f}, edge $ {b['edge_dollars']:+.2f} -> {a['edge_dollars']:+.2f} over {cases} "
              f"changed trades, P&L {b['pnl']:+.2f} -> {a['pnl']:+.2f}; judged on {metric}")
    return {"result": "accepted" if ok else "rejected", "cases": cases, "before": b["mean_clv"],
            "after": a["mean_clv"], "clv_dollars_before": b["clv_dollars"], "clv_dollars_after": a["clv_dollars"],
            "edge_dollars_before": b["edge_dollars"], "edge_dollars_after": a["edge_dollars"],
            "pnl_before": b["pnl"], "pnl_after": a["pnl"], "reason": reason}


# ---------------- risk and confirmation defaults ----------------

def pretrade_risk(order: Order, ctx: dict, max_order=MAX_ORDER):
    """Limits checked inside the graph; replay.py enforces the caps and the kill switch again."""
    if ctx.get("kill_switch_tripped"):
        return False, "kill_switch"
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
    """Replay simplification: confirms orders for the platform's own paper account.

    Retail orders are never auto-confirmed; a retail channel needs an explicit
    confirm callable that stands for the user (the live demo waits on a click).
    """
    return [o for o in orders if o.channel != "retail"]


class MarketAgent:
    def __init__(self, notebook: Notebook | None = None, forecaster: Callable = record_forecaster,
                 risk: Callable = pretrade_risk, confirm: Callable = auto_confirm, use_llm: bool = False,
                 learn: bool = True, stake: float = DEFAULT_STAKE, channel: str = "platform", plant=None,
                 anchor: bool = True, impact=None, min_edge: float = DEFAULT_MIN_EDGE, gate: str = "legacy",
                 sizing: str = "flat", kelly_fraction: float = DEFAULT_KELLY_FRACTION,
                 bankroll: float = DEFAULT_BANKROLL, forecast_cache: dict | None = None,
                 adaptive_edge: bool = False, model_review: bool = False, components: Callable | None = None):
        if gate not in GATE_MODES or sizing not in SIZINGS:
            raise ValueError(f"gate must be one of {GATE_MODES} and sizing one of {SIZINGS}")
        if model_review and components is None:
            raise ValueError("model_review needs a components source (forecast.blend.Components)")
        self.notebook = notebook or Notebook()
        self.adaptive_edge = adaptive_edge
        self.model_review = model_review
        # forecast.blend.Components: MLP shift and M6 move per decision; with it the agent logs every
        # decision point's components (home terms) in component_log, keyed (game_id, as_of).
        self.components = components
        self.component_log = {}
        self.candidates_seen = {}       # adaptive only: (market_ticker, as_of) -> candidate, traded or not
        self.anchor = anchor
        self.gate_mode = gate
        self.sizing, self.kelly_fraction, self.bankroll = sizing, kelly_fraction, bankroll
        # Forecasts per (game, decision time, overrides). Each entry was computed from the as-of view at
        # that time, so sharing it with back-test copies of this agent saves work without leaking anything.
        self.forecast_cache = {} if forecast_cache is None else forecast_cache
        # M6 (forecast/impact.py): p(home) = current home mid + predicted move to tip, instead of anchor + news shift.
        self.impact = impact
        self.min_edge = min_edge
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
        rules = {r["rule_id"]: r for s in situations for r in self.notebook.matching(s, now, "forecast")
                 if r["do"]["action"] in FORECAST_OVERRIDES}
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
        before = self._forecast(view, game, markets, {"out": []})
        after = self._forecast(view, game, markets, overrides)
        forecasts = {t: {"before": before[t], "after": after[t]} for t in after}
        shown = ", ".join(f"{t.rsplit('-', 1)[-1]} {f['before']:.2f}->{f['after']:.2f}" for t, f in forecasts.items())
        out = {"forecasts": forecasts, "trace": log("forecast", shown or "no market the models cover")}
        if self.components is not None:
            key = ("components", game.game_id, pd.Timestamp(view.now).value, json.dumps(overrides, sort_keys=True,
                                                                                      default=str))
            if key not in self.forecast_cache:
                self.forecast_cache[key] = self.components(view, game, overrides.get("out", []), view.now)
            out["components"] = dict(self.forecast_cache[key])
        return out

    def active_blend(self, now) -> dict:
        """Params of the newest active forecast_blend rule, or the default anchor + M4 shift."""
        from forecast.blend import DEFAULT_BLEND
        rules = [r for r in self.notebook.active(now, "forecast") if r["do"]["action"] == "forecast_blend"]
        return max(rules, key=_recency)["do"]["params"] if rules else DEFAULT_BLEND

    @staticmethod
    def _is_default_blend(params) -> bool:
        return (params.get("name") != "stack" and params.get("base") == "anchor" and params.get("w_m4") == 1.0
                and params.get("w_mlp") == 0.0 and params.get("w_m6") == 0.0)

    def _log_components(self, view, game, now, state):
        """Home-terms component row for this decision point (logged), or None without a fresh home quote."""
        markets = state["markets"]
        home = markets[(markets.team == game.home_team) & (markets.kind == "game")].market_ticker
        if home.empty or home.iloc[0] not in state["forecasts"]:
            return None
        ticker = home.iloc[0]
        q, anchor = self._fresh_quote(view, ticker, now), self._anchor_mid(view, ticker, game)
        if q is None or anchor is None:
            return None
        f, comp = state["forecasts"][ticker], state["components"]
        row = {"anchor": anchor, "mid": float((q.bid + q.ask) / 2), "m4": f["after"] - f["before"],
               "mlp": comp.get("mlp"), "m6": comp.get("m6")}
        blend = self.active_blend(now) if self.model_review else None
        if blend is None or self._is_default_blend(blend):
            used, name = clip(anchor + row["m4"]), "m4"
        else:
            used, name = self._blend_p(blend, True, anchor, row["mid"], row["m4"], comp, row), blend["name"]
        self.component_log[(game.game_id, now)] = {"game_id": game.game_id, "as_of": now, "tip_time": game.tip_time,
                                                   "date": game.date, **row, "p_used": used, "blend": name}
        return row

    def _blend_p(self, params, home, anchor, mid, m4_shift, comp, home_row):
        """P(yes) for one ticker under a blend; None if a component the blend needs is missing."""
        from forecast.blend import blend_home
        if params["name"] == "stack":
            return None if home_row is None else (lambda p: p if home else 1 - p)(
                float(blend_home(params, pd.DataFrame([home_row]))[0]))
        sign = 1.0 if home else -1.0
        mlp, m6 = comp.get("mlp"), comp.get("m6")
        if (params["w_m6"] and m6 is None) or (params["w_mlp"] and mlp is None) or anchor is None:
            return None
        base = anchor if params["base"] == "anchor" else mid
        return clip(base + params["w_m4"] * m4_shift + params["w_mlp"] * sign * (mlp or 0.0)
                    + params["w_m6"] * sign * (m6 or 0.0))

    def _forecast(self, view, game, markets, overrides):
        key = (game.game_id, pd.Timestamp(view.now).value, tuple(markets.market_ticker),
               json.dumps(overrides, sort_keys=True, default=str))
        if key not in self.forecast_cache:
            self.forecast_cache[key] = self.forecaster(view, game, markets, overrides)
        return dict(self.forecast_cache[key])

    def _stake(self, p_side: float, price: float, scale: float) -> float:
        """Flat stake, or fractional Kelly on a fixed bankroll capped at the order cap (stake = contract notional)."""
        if self.sizing == "flat":
            return round(self.stake * scale, 2)
        outlay = self.kelly_fraction * kelly_fraction(p_side, price) * self.bankroll
        notional = outlay * price / (price + fee_per_contract(price))
        return round(min(notional, MAX_ORDER) * scale, 2)

    def analyse(self, state):
        view, game, now, news = state["view"], state["game"], state["now"], state["news"]
        markets = state["markets"].set_index("market_ticker")
        age = minutes_between(news.published_at.max(), now) if len(news) else NO_NEWS_AGE
        rows = []
        impact = self.impact.predict(view, game, now) if self.impact is not None else None
        comp = state.get("components")
        blend = self.active_blend(now) if self.model_review else None
        if blend is not None and self._is_default_blend(blend):
            blend = None
        home_row = self._log_components(view, game, now, state) if comp is not None else None
        for ticker, f in state["forecasts"].items():
            m, q = markets.loc[ticker], self._fresh_quote(view, ticker, now)
            if q is None or (self.impact is not None and impact is None):
                continue
            shift = f["after"] - f["before"]
            anchor = self._anchor_mid(view, ticker, game) if self.anchor or impact else None
            home = m.team == game.home_team
            if impact is not None:
                p_home = clip(impact["mid"] + impact["move"])
                p, base = (p_home, impact["mid"]) if home else (1 - p_home, 1 - impact["mid"])
            elif blend is not None:
                p = self._blend_p(blend, home, anchor, float((q.bid + q.ask) / 2), shift, comp, home_row)
                if p is None:
                    continue
                base = anchor
            # Anchored: the market is the base rate and only the model's news shift is traded.
            else:
                p = clip(anchor + shift) if anchor is not None else f["after"]
                base = anchor if anchor is not None else f["before"]
            yes_gap = p - q.ask - fee_per_contract(q.ask)
            no_gap = (1 - p) - (1 - q.bid) - fee_per_contract(1 - q.bid)
            side, gap = ("yes", yes_gap) if yes_gap >= no_gap else ("no", no_gap)
            moved = 0.0 if anchor is None else (q.bid + q.ask) / 2 - anchor
            situation = {"team": m.team, "opponent": game.away_team if m.team == game.home_team else game.home_team,
                         "market_kind": m.kind, "hours_to_tip": minutes_between(now, game.tip_time) / 60,
                         "news_age_minutes": float(age), "side_price": float(q.ask if side == "yes" else 1 - q.bid),
                         "gap": float(gap), "market_move": float(moved if side == "yes" else -moved),
                         "model_shift": float(abs(shift)), "placebo_bucket": placebo_bucket(ticker, now)}
            self.situations[(ticker, now)] = situation
            rules = self.notebook.matching(situation, now, "trader")
            min_edge = effective_edge(self.min_edge, rules, self.adaptive_edge)
            scale = min([1.0] + [r["do"]["params"]["scale"] for r in rules if r["do"]["action"] == "stake_scale"])
            skip = [r["rule_id"] for r in rules if r["do"]["action"] == "skip_market"]
            price = situation["side_price"]
            stake = self._stake(p if side == "yes" else 1 - p, price, scale)
            rows.append({"ticker": ticker, "team": m.team, "kind": m.kind, "p": p, "stake": stake,
                         "before": base, "shift": shift, "anchored": anchor is not None and impact is None,
                         "impact": None if impact is None else impact["move"] * (1 if home else -1),
                         "bid": float(q.bid), "ask": float(q.ask), "side": side, "gap": float(gap),
                         "min_edge": min_edge, "scale": scale, "rules": [r["rule_id"] for r in rules],
                         "act": gap > min_edge and not skip, "why_not": "rule " + ", ".join(skip) if skip else
                         ("" if gap > min_edge else "gap below threshold")})
        # Both teams' markets price the same game: hold at most one position per game.
        best = max((r for r in rows if r["act"]), key=lambda r: r["gap"], default=None)
        for r in rows:
            if r["act"] and (game.game_id in self.held or r is not best):
                r.update(act=False, why_not="one position per game")
        if self.adaptive_edge:
            for r in rows:
                self.candidates_seen[(r["ticker"], now)] = {
                    "ticker": r["ticker"], "game_id": game.game_id, "as_of": now, "tip_time": game.tip_time, "side": r["side"],
                    "price": r["ask"] if r["side"] == "yes" else 1 - r["bid"], "stake": r["stake"], "gap": r["gap"],
                    "min_edge": r["min_edge"], "skipped": r["why_not"].startswith("rule ")}
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
        if r.get("impact") is not None:
            basis = (f"estimate {r['p']:.0%} (market mid {r['before']:.0%} now, M6 expects "
                     f"{r['impact'] * 100:+.1f} points by tip-off)")
        elif r.get("anchored"):
            basis = (f"estimate {r['p']:.0%} (market {r['before']:.0%} a day earlier, news shift "
                     f"{r['shift'] * 100:+.1f} points)")
        else:
            moved = f" ({r['before']:.0%} before the news)" if f"{r['before']:.0%}" != f"{r['p']:.0%}" else ""
            basis = f"model {r['p']:.0%}{moved}"
        return (f"{r['team']} win: {basis}, market "
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
            orders.append(Order(r["ticker"], r["side"], r["p"], r["stake"], reason,
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
        analysed = {r["ticker"]: r["p"] for r in state["candidates"]}
        for o in state["orders"]:
            if o.market_ticker not in analysed or not math.isclose(o.p_model, analysed[o.market_ticker], abs_tol=1e-9):
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
        view = state["view"]
        ctx = {"now": state["now"], "tip_time": state["game"].tip_time,
               "realised_day": getattr(view, "realised_day", 0.0),
               "kill_switch_tripped": getattr(view, "kill_switch_tripped", False)}
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
        return {"orders": orders, "status": "sent" if orders else "unconfirmed",
                "trace": log("confirm", f"{len(orders)} of {len(state['orders'])} confirmed")}

    def blocked(self, state):
        why = ("checks failed twice" if state.get("failures") else
               "daily loss limit reached, no new orders today" if getattr(state.get("view"), "kill_switch_tripped", False)
               else "risk limits")
        brief = state.get("brief", "") + f"\nOrder blocked ({why}); reason logged."
        return {"orders": [], "status": "blocked", "brief": brief, "trace": log("blocked", why)}

    def deliver(self, state):
        status = state.get("status") if state.get("status") in ("sent", "blocked", "unconfirmed") else "brief_only"
        return {"status": status, "trace": log("deliver", status)}

    # ---------------- review phase ----------------

    def settle(self, state):
        rp, day = state["replay"], state["day"]
        today = pd.DataFrame(rp.fills)
        today = today[today.game_id.isin(rp.t["games"].loc[rp.t["games"].date == day, "game_id"])] if len(today) else today
        s = replay.summary(today)
        return {"trace": log("settle", f"{day}: {json.dumps({k: round(v, 3) for k, v in s.items()})}")}

    def review(self, state):
        rp, day = state["replay"], state["day"]
        fills = pd.DataFrame(rp.fills)
        if fills.empty:
            return {"proposal": None, "trace": log("review", "no settled trades yet")}
        fills["news_age"] = [self.situations.get((t, a), {}).get("news_age_minutes", np.nan)
                             for t, a in zip(fills.market_ticker, fills.as_of)]
        recent = self.selection(rp, fills, day)
        if recent.empty:
            return {"proposal": None, "trace": log("review", "no settled trades in the selection window")}
        if self.adaptive_edge:
            proposal, how = self._review_adaptive(recent, rp, day), "offline rules, adaptive edge"
        else:
            proposal, how = self._review_offline(recent), "offline rules"
        if self.use_llm:
            try:
                # legacy showed the LLM every losing trade so far, which overlaps the gate's test days
                proposal, how = self._review_llm(fills if self.gate_mode == "legacy" else recent), self.llm_name
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

    def selection(self, rp, fills: pd.DataFrame, day: str, mode=None) -> pd.DataFrame:
        """Trades the reviewer may learn from: legacy, the last REVIEW_WINDOW; split, the last SELECT_DAYS market days."""
        mode = mode or self.gate_mode
        if mode == "legacy":
            return fills[pd.to_datetime(fills.as_of) >= pd.to_datetime(fills.as_of).max() - REVIEW_WINDOW]
        chosen, _ = gate_windows(rp, day, mode)
        games = rp.t["games"]
        return fills[fills.game_id.isin(games.loc[games.date.isin(chosen), "game_id"])]

    def situation_frame(self, fills: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self.situations.get((t, a), {}) for t, a in zip(fills.market_ticker, fills.as_of)],
                            index=fills.index)

    @staticmethod
    def template_mask(sit: pd.DataFrame, when: dict, do: dict, current: float | None = None) -> pd.Series:
        """Decisions a template's condition covers.

        min_edge with current=None (the published reviewer): trades whose gap was below the new edge,
        ignoring the condition. With the threshold in force (`current`, adaptive edge) the condition
        applies too: raising covers trades with current < gap <= new edge (they stop trading);
        lowering covers candidates with new edge < gap <= current (the current threshold blocked them,
        the new one lets them trade), so it must be applied to blocked candidates, not fills.
        """
        if do["action"] == "min_edge" and current is None:
            return sit.get("gap", pd.Series(np.nan, index=sit.index)) < do["params"]["edge"]
        mask = pd.Series(True, index=sit.index)
        for key, want in when.items():
            if key in ("market_kind",):
                continue
            name, end = key.rsplit("_", 1)
            have = sit.get(name, pd.Series(np.nan, index=sit.index))
            mask &= (have >= want) if end == "min" else (have <= want)
        if do["action"] == "min_edge":
            gap, edge = sit.get("gap", pd.Series(np.nan, index=sit.index)), do["params"]["edge"]
            mask &= (gap <= edge) & (gap > current) if edge >= current else (gap > edge) & (gap <= current)
        return mask

    def current_global_edge(self, now) -> float:
        """Market-wide threshold in force at `now`: the newest active market-wide min_edge rule, or the base."""
        glob = [r for r in self.notebook.active(now, "trader") if is_global_edge(r)]
        return max(glob, key=_recency)["do"]["params"]["edge"] if glob else self.min_edge

    def blocked_candidates(self, rp, game_ids, traded_games) -> pd.DataFrame:
        """Per untraded game, decisions the threshold (not a skip rule) blocked, with counterfactual CLV.

        Keeps the best-gap market per decision time (one position per game). CLV is the side's mid at
        tip minus the price that would have been paid, the same as replay settles a fill.
        """
        rows = [c for c in self.candidates_seen.values()
                if c["game_id"] in game_ids and c["game_id"] not in traded_games and not c["skipped"]]
        if not rows:
            return pd.DataFrame(columns=["game_id", "as_of", "gap", "clv", "contracts"])
        frame = pd.DataFrame(rows)
        frame = frame.sort_values("gap", ascending=False).drop_duplicates(["game_id", "as_of"])
        close = {}
        for t, tip in set(zip(frame.ticker, frame.tip_time)):
            q = replay.AsOf(rp.t, tip, rp.price_index).quote(t)
            close[t] = float((q.bid + q.ask) / 2) if q is not None else np.nan
        mid = frame.ticker.map(close)
        frame["clv"] = np.where(frame.side == "yes", mid, 1 - mid) - frame.price
        frame["contracts"] = np.floor(frame.stake / frame.price + 1e-9)
        return frame.sort_values("as_of").reset_index(drop=True)

    def _review_adaptive(self, recent, rp, day):
        """The published reviewer's templates plus edge templates, scored by CLV dollars gained on recent days.

        Skip and raise-edge templates gain the CLV dollars of the trades they would remove. A lower
        market-wide edge gains the counterfactual CLV dollars of the first decision per untraded game
        it would let through. Ignores later re-entries and the one-position interaction; the gate's
        back-test is exact.
        """
        now = pd.Timestamp(recent.as_of.max())
        current = self.current_global_edge(now)
        sit = self.situation_frame(recent)
        dollars = recent.clv * recent.contracts
        games = rp.t["games"]
        chosen, _ = gate_windows(rp, day, self.gate_mode)
        window_games = set(games.loc[games.date.isin(chosen), "game_id"]) if chosen else set(recent.game_id)
        blocked = None
        best = None
        for when, do, blame, says in ADAPTIVE_TEMPLATES:
            glob = do["action"] == "min_edge" and not when
            if glob and math.isclose(do["params"]["edge"], current):
                continue
            if glob and do["params"]["edge"] < current:
                if blocked is None:
                    blocked = self.blocked_candidates(rp, window_games, set(recent.game_id))
                hit = blocked[self.template_mask(blocked, when, do, current)].drop_duplicates("game_id")
                gain = float((hit.clv * hit.contracts).sum())
                if len(hit) < 2 * LIMITS["min_cases"] or hit.clv.mean() <= 0:
                    continue
                text = (f"{len(hit)} recent decisions blocked by the {current * 100:.0f}-point threshold would have "
                        f"averaged {hit.clv.mean():+.3f} closing-line value ({gain:+.2f} dollars) if {says}.")
                source = recent.iloc[:0]
            else:
                mask = self.template_mask(sit, when, do, current if do["action"] == "min_edge" else None)
                hit = recent[mask]
                gain = -float(dollars[mask].sum())
                if len(hit) < 2 * LIMITS["min_cases"] or hit.clv.mean() >= 0:
                    continue
                text = (f"{len(hit)} recent trades where {says} averaged {hit.clv.mean():+.3f} closing-line "
                        f"value ({-gain:+.2f} dollars).")
                source = hit
            rule = self._rule({"market_kind": "game", **when}, do, blame, text, source)
            if self.notebook.seen(rule, now):
                continue
            if best is None or gain > best[0]:
                best = (gain, rule)
        return None if best is None else best[1]

    def _review_offline(self, recent):
        """Blame the slice of recent trades that lost the most closing-line value and propose a rule for it.

        Each template names a situation the reviewer can describe; the gate, not the
        reviewer, decides whether skipping that situation helps on earlier days.
        """
        sit = self.situation_frame(recent)
        dollars = recent.clv * recent.contracts
        best = None
        for when, do, blame, says in REVIEW_TEMPLATES:
            mask = self.template_mask(sit, when, do)
            hit = recent[mask]
            if len(hit) < 2 * LIMITS["min_cases"] or hit.clv.mean() >= 0:
                continue
            rule = self._rule({"market_kind": "game", **when}, do, blame,
                              f"{len(hit)} recent trades where {says} averaged {hit.clv.mean():+.3f} closing-line "
                              f"value ({dollars[mask].sum():+.2f} dollars).", hit)
            if self.notebook.seen(rule, pd.Timestamp(recent.as_of.max())):
                continue
            if best is None or dollars[mask].sum() < best[0]:
                best = (dollars[mask].sum(), rule)
        return None if best is None else best[1]

    def _review_llm(self, fills):
        bad = fills[fills.clv < 0]
        sample = bad[["market_ticker", "side", "p_model", "price", "close_price", "clv", "pnl", "news_age"]].round(3)
        prompt = ("You review losing sports-market trades. Pick one blame_category from news_misread, minutes_wrong, "
                  "model_wrong, market_priced_in, and propose at most one rule. Return JSON {\"rule\": null or "
                  "{\"when\": {...}, \"do\": {\"action\": ..., \"params\": {...}}}, \"blame_category\": ..., "
                  "\"rationale\": one sentence}. Allowed when fields: market_kind, team, opponent, and _min/_max of "
                  "hours_to_tip, news_age_minutes, side_price (price paid), gap (edge after fees), market_move "
                  "(price move toward the trade since a day earlier), model_shift. Allowed actions: skip_market (no "
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
        evidence = self.gate_rule(state["replay"], state["day"], state["proposal"])
        return {"gate": evidence, "trace": log("gate", f"{evidence['result']}: {evidence['reason']}")}

    def gate_rule(self, rp, day, rule, mode=None, without=None) -> dict:
        """Back-test the notebook with and without `rule` on the gate's test days and judge it.

        `mode` overrides the agent's gate mode (the placebo audit judges one rule several ways);
        `without` reuses a back-test of the notebook without the rule on the same days.
        """
        mode = mode or self.gate_mode
        games = rp.t["games"]
        _, days = gate_windows(rp, day, mode)
        metric, threshold = GATE_METRICS[mode]
        evidence = {"decided_at": games.loc[games.date == day, "final_at"].max(), "mode": mode,
                    "backtest_days": [days[0], days[-1]] if days else [], "metric": metric, "threshold": threshold}
        if len(days) < 2:
            return {**evidence, "result": "deferred", "reason": "not enough earlier days"}
        if without is None:
            without = self.backtest(rp, self.notebook.without(rule), days[0], days[-1])
        with_rule = self.backtest(rp, self.notebook.with_candidate(rule), days[0], days[-1])
        return {**evidence, **judge(without, with_rule, mode)}

    def backtest(self, rp, notebook, start, end, return_agent=False):
        """Replay earlier days with an offline, non-learning copy of this agent."""
        child = MarketAgent(notebook, self.forecaster, self.risk, self.confirm, use_llm=False, learn=False,
                            stake=self.stake, channel=self.channel, anchor=self.anchor, impact=self.impact,
                            min_edge=self.min_edge, gate=self.gate_mode, sizing=self.sizing,
                            kelly_fraction=self.kelly_fraction, bankroll=self.bankroll,
                            forecast_cache=self.forecast_cache, adaptive_edge=self.adaptive_edge,
                            model_review=self.model_review, components=self.components)
        _, fills = Replay(rp.t, child.policy, risk=rp.risk, fee=rp.fee, kill_switch=getattr(rp, "kill_switch", None),
                          price_index=getattr(rp, "price_index", None)).run(start, end)
        return (fills, child) if return_agent else fills

    def save_rule(self, state):
        g = state["gate"]
        old = [r["rule_id"] for r in self.notebook.rules
               if (r["when"], r["do"]) == (state["proposal"]["when"], state["proposal"]["do"])]
        rule = {**state["proposal"], "status": "active", "proposed_at": g["decided_at"], "gate": g,
                "valid_from": g["decided_at"], "supersedes": old[-1] if old else None}
        if self.adaptive_edge and is_global_edge(rule):
            replaced = [r for r in self.notebook.active(g["decided_at"], "trader") if is_global_edge(r)]
            for r in replaced:
                r["valid_until"] = g["decided_at"]
            rule["supersedes"] = replaced[-1]["rule_id"] if replaced else rule["supersedes"]
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

    # ---------------- model review: a second subloop that chooses the forecast blend ----------------

    @staticmethod
    def model_windows(rp, day) -> tuple:
        """(selection days, gate days) for a model review after `day`, or None when none is due.

        Only market days from MODEL_REVIEW_FROM count, so no window holds a game a model trained on.
        Due after every MODEL_REVIEW_EVERY market days once SELECT_DAYS + GATE_DAYS are available.
        """
        days = [d for d in market_days(rp, day) if d >= MODEL_REVIEW_FROM]
        need = SELECT_DAYS + GATE_DAYS
        if not days or days[-1] != day or len(days) < need or (len(days) - need) % MODEL_REVIEW_EVERY:
            return None
        return days[-SELECT_DAYS:], days[-need:-SELECT_DAYS]

    def component_frame(self, rp, days, log: dict | None = None) -> pd.DataFrame:
        """Logged decision points on `days` with the home team's result (all final by the review)."""
        rows = [r for r in (self.component_log if log is None else log).values() if r["date"] in set(days)]
        if not rows:
            return pd.DataFrame()
        games = rp.t["games"].set_index("game_id")
        frame = pd.DataFrame(rows)
        frame["home_win"] = (games.loc[frame.game_id, "home_pts"].to_numpy()
                             > games.loc[frame.game_id, "away_pts"].to_numpy()).astype(float)
        return frame

    def review_model(self, state):
        """Pick the candidate blend with the lowest Brier on the selection days (the stack is fit there)."""
        from forecast.blend import CANDIDATE_BLENDS, blend_home, brier, fit_stack
        rp, day = state["replay"], state["day"]
        windows = self.model_windows(rp, day)
        if windows is None:
            return {"model_proposal": None, "trace": log("review_model", "not due")}
        chosen, _ = windows
        frame = self.component_frame(rp, chosen)
        if len(frame) < 20:
            return {"model_proposal": None, "trace": log("review_model", f"only {len(frame)} decision points")}
        games = rp.t["games"]
        now = games.loc[games.date == day, "final_at"].max()
        scores, params = {}, {}
        for cand in CANDIDATE_BLENDS:
            p = fit_stack(frame) if cand["name"] == "stack" else cand
            if p is None:
                continue
            need = frame.m6.notna() if p.get("w_m6") or p["name"] == "stack" else pd.Series(True, index=frame.index)
            need &= frame.mlp.notna() if p.get("w_mlp") or p["name"] == "stack" else True
            if need.mean() < 0.8:
                continue
            scores[p["name"]] = brier(blend_home(p, frame[need]), frame.home_win[need])
            params[p["name"]] = p
        current = self.active_blend(now)
        used = frame.p_used.notna()
        cur_score = brier(frame.p_used[used].astype(float), frame.home_win[used])
        if not scores:
            return {"model_proposal": None, "trace": log("review_model", "no candidate had enough components")}
        best = min(scores, key=scores.get)
        table = ", ".join(f"{k} {v:.4f}" for k, v in sorted(scores.items(), key=lambda kv: kv[1]))
        if best == current["name"] or scores[best] >= cur_score:
            return {"model_proposal": None, "trace": log("review_model", f"keep {current['name']}: Brier {table}")}
        rule = {"rule_id": self.notebook.next_id("f"), "version": 1, "kind": "forecast", "status": "proposed",
                "when": {"market_kind": "game"}, "do": {"action": "forecast_blend", "params": params[best]},
                "rationale": (f"On {len(frame)} decision points ({chosen[0]}..{chosen[-1]}) blend {best} had Brier "
                              f"{scores[best]:.4f} vs {cur_score:.4f} for the forecasts used ({current['name']}). "
                              f"All: {table}."),
                "proposed_by": "model_reviewer@offline", "blame_category": "model_wrong", "source_trades": [],
                "priority": 1, "expires_after_days": LIMITS["default_expiry_days"],
                "selection": {"days": [chosen[0], chosen[-1]], "rows": int(len(frame)), "brier": scores}}
        if validate(rule) or self.notebook.seen(rule, now):
            return {"model_proposal": None, "trace": log("review_model", f"dropped {best}: invalid or seen")}
        return {"model_proposal": rule, "trace": log("review_model", f"propose {best}: Brier {table}")}

    def gate_model(self, state):
        """Back-test with vs without the blend on the gate days: CLV $ must rise >= GATE_DOLLARS and Brier not worsen."""
        from forecast.blend import blend_home, brier
        rp, day, rule = state["replay"], state["day"], state["model_proposal"]
        _, days = self.model_windows(rp, day)
        games = rp.t["games"]
        evidence = {"decided_at": games.loc[games.date == day, "final_at"].max(), "mode": "model",
                    "backtest_days": [days[0], days[-1]], "metric": "clv_dollars+brier", "threshold": GATE_DOLLARS}
        without, child = self.backtest(rp, self.notebook.without(rule), days[0], days[-1], return_agent=True)
        with_rule = self.backtest(rp, self.notebook.with_candidate(rule), days[0], days[-1])
        verdict = judge(without, with_rule, "split")
        frame = self.component_frame(rp, days, child.component_log)
        p = rule["do"]["params"]
        ok_rows = (frame.m6.notna() & frame.mlp.notna() & frame.p_used.notna()) if len(frame) else pd.Series(dtype=bool)
        b_old = brier(frame.p_used[ok_rows].astype(float), frame.home_win[ok_rows]) if ok_rows.any() else np.nan
        b_new = brier(blend_home(p, frame[ok_rows]), frame.home_win[ok_rows]) if ok_rows.any() else np.nan
        accept = verdict["result"] == "accepted" and b_new <= b_old
        reason = verdict["reason"] + f"; Brier {b_old:.4f} -> {b_new:.4f} on {int(ok_rows.sum())} decision points"
        g = {**evidence, **verdict, "brier_before": b_old, "brier_after": b_new, "brier_rows": int(ok_rows.sum()),
             "result": "accepted" if accept else "rejected", "reason": reason}
        return {"model_gate": g, "trace": log("gate_model", f"{g['result']}: {reason}")}

    def save_model(self, state):
        g, proposal = state["model_gate"], state["model_proposal"]
        rule = {**proposal, "status": "active", "proposed_at": g["decided_at"], "gate": g,
                "valid_from": g["decided_at"]}
        replaced = [r for r in self.notebook.active(g["decided_at"], "forecast") if r["do"]["action"] == "forecast_blend"]
        for r in replaced:
            r["valid_until"] = g["decided_at"]
        rule["supersedes"] = replaced[-1]["rule_id"] if replaced else None
        self.notebook.record(rule)
        return {"trace": log("save_model", f"{rule['rule_id']} blend {proposal['do']['params']['name']} active "
                                           f"from {g['decided_at']}")}

    def reject_model(self, state):
        g = state["model_gate"]
        rule = {**state["model_proposal"], "status": "rejected", "proposed_at": g["decided_at"], "gate": g,
                "valid_from": None}
        self.notebook.record(rule)
        return {"trace": log("reject_model", f"{rule['rule_id']}: {g['reason']}")}

    # ---------------- wiring ----------------

    def _fresh_quote(self, view, ticker, now):
        q = view.quote(ticker)
        return None if q is None or now - q.ts > MAX_QUOTE_AGE else q

    @staticmethod
    def _anchor_mid(view, ticker, game):
        """Mid price ANCHOR_LEAD before tip-off (or the earliest quote), before most of the day's news."""
        p = view.prices(ticker)
        if p.empty:
            return None
        early = p[p.ts <= game.tip_time - ANCHOR_LEAD]
        row = early.iloc[-1] if len(early) else p.iloc[0]
        return float((row.bid + row.ask) / 2)

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
        # With model review, the rule subloop hands over to the model subloop instead of ending. They run one
        # after the other (not as parallel branches) because the forecasters' history caches are not thread-safe.
        done = "review_model" if self.model_review else END
        g.add_conditional_edges("review", lambda s: "gate" if s.get("proposal") else done, ["gate", done])
        g.add_conditional_edges("gate", lambda s: "save_rule" if s["gate"]["result"] == "accepted"
                                else "reject_rule", ["save_rule", "reject_rule"])
        g.add_edge("save_rule", done)
        g.add_edge("reject_rule", done)
        if self.model_review:
            for name, fn in [("review_model", self.review_model), ("gate_model", self.gate_model),
                             ("save_model", self.save_model), ("reject_model", self.reject_model)]:
                g.add_node(name, fn)
            g.add_conditional_edges("review_model", lambda s: "gate_model" if s.get("model_proposal") else END,
                                    ["gate_model", END])
            g.add_conditional_edges("gate_model", lambda s: "save_model" if s["model_gate"]["result"] == "accepted"
                                    else "reject_model", ["save_model", "reject_model"])
            g.add_edge("save_model", END)
            g.add_edge("reject_model", END)
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
    if source == "sample":
        from data_sources.sample import SAMPLE
        return replay.load_tables(SAMPLE)
    return replay.load_tables()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true", help="print the compiled graph as Mermaid and exit")
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--start", default="2026-02-01")
    ap.add_argument("--end", default="2026-02-28")
    ap.add_argument("--name", default="agent")
    ap.add_argument("--llm", action="store_true", help="use the chat model when a key is set")
    ap.add_argument("--no-learn", action="store_true", help="the no-learning ablation")
    ap.add_argument("--notebook", type=Path, default=None, help="start from this notebook (default: empty)")
    ap.add_argument("--plant", choices=FAULTS, default=None)
    ap.add_argument("--forecaster", choices=["models", "record"], default="models",
                    help="trained models via forecast/api.py, or the win-rate placeholder")
    ap.add_argument("--signal", choices=["anchor", "impact"], default="anchor",
                    help="anchor + M4 news shift (default), or M6: current mid + predicted move to tip")
    ap.add_argument("--impact-model", type=Path, default=None, help="M6 pickle (default models/m6/impact.pkl)")
    ap.add_argument("--gate", choices=GATE_MODES, default="split",
                    help="split (default): disjoint selection and test days, CLV dollars; split-edge: edge over "
                         "the mid; legacy: the published runs (overlapping windows, mean CLV)")
    ap.add_argument("--kill-switch", type=float, default=100.0,
                    help="stop new orders for the day once realised losses exceed this (0 = off, as published)")
    ap.add_argument("--sizing", choices=SIZINGS, default="flat")
    ap.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION)
    ap.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    ap.add_argument("--base-edge", type=float, default=DEFAULT_MIN_EDGE,
                    help="starting edge threshold after fees (default 0.04)")
    ap.add_argument("--edge-templates", action="store_true",
                    help="adaptive edge: the reviewer may propose market-wide thresholds (raise or lower) and "
                         "slice-specific edges; the gate decides")
    ap.add_argument("--model-review", action="store_true",
                    help="second subloop: weekly choose a forecast blend (M4, MLP, M6, stack) and gate it")
    args = ap.parse_args()

    forecaster = record_forecaster
    if args.forecaster == "models":
        from forecast.api import Forecaster
        forecaster = Forecaster.load()
    impact = None
    if args.signal == "impact":
        from forecast.impact import MODEL_PATH, load_impact
        impact = load_impact(args.impact_model or MODEL_PATH)
    components = None
    if args.model_review:
        from forecast.blend import Components, load_mlp
        from forecast.impact import MODEL_PATH as M6_PATH, load_impact as load_m6
        components = Components(forecaster, load_mlp(), load_m6(args.impact_model or M6_PATH))
    agent = MarketAgent(Notebook.load(args.notebook) if args.notebook else Notebook(), forecaster=forecaster,
                        use_llm=args.llm, learn=not args.no_learn, plant=args.plant, impact=impact, gate=args.gate,
                        sizing=args.sizing, kelly_fraction=args.kelly_fraction, bankroll=args.bankroll,
                        min_edge=args.base_edge, adaptive_edge=args.edge_templates,
                        model_review=args.model_review, components=components)
    if args.draw:
        print(agent.graph.get_graph().draw_mermaid())
        return
    tables = load_source(args.source, args.start, args.end)
    rp = Replay(tables, agent.policy, on_day_end=agent.on_day_end, kill_switch=args.kill_switch or None)
    decisions, fills = rp.run(args.start, args.end)
    out = save_run(args.name, decisions, fills, agent, rp.kill_trips)
    print(f"{len(decisions)} decisions, {len(fills)} fills, {len(rp.kill_trips)} kill-switch trips -> {out}")
    print(replay.summary(fills))
    for r in agent.notebook.rules:
        print(f"{r['rule_id']} {r['status']}: when {r['when']} do {r['do']['action']} ({r['gate']['reason']})")


def save_run(name, decisions, fills, agent=None, kill_trips=None) -> Path:
    out = replay.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(out / "decisions.parquet", index=False)
    fills.to_parquet(out / "fills.parquet", index=False)
    if kill_trips is not None:
        (out / "kill_switch.json").write_text(json.dumps(kill_trips, indent=1, default=str))
    if agent is not None:
        agent.notebook.save(out / "notebook.json")
        (out / "trace.jsonl").write_text("\n".join(json.dumps(t, default=str) for t in agent.traces))
    return out


if __name__ == "__main__":
    main()
