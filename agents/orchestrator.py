"""E (thin): one game night as a multi-agent system, routed by a LangGraph supervisor.

    python -m agents.orchestrator --date 2026-03-10
    python -m agents.orchestrator --date 2026-03-10 --games 2 --lead-minutes 90
    python -m agents.orchestrator --date 2026-02-01 --source synthetic --forecaster record

    supervisor -> pregame -> supervisor -> trader -> supervisor -> coach -> supervisor -> briefs -> supervisor -> END

Each worker is an existing team component, called as it is:

    pregame  agents/pregame.py PregameAgent polls the news table up to the decision time and writes factors
             (out, doubtful, questionable weights) and fair odds
    trader   agents/graph.py MarketAgent decides at the decision time; with --trader-input pregame the pregame
             factors become its investigation (expected lost minutes share per player), else official statuses
    coach    agents/coach.py explains the game and the trader's decision to a retail user
    briefs   agents/briefs.py writes the four channel briefs, quoting the trader's order if there is one

All workers see the same as-of moment (tip minus --lead-minutes) and nothing later. State is shared through
typed fields and a message log of handoffs. A failing component becomes an error message; the supervisor moves
on and later agents degrade (the trader falls back to official statuses, the coach explains without the agent
panel). The supervisor routes by rules, not an LLM: the order is fixed, the failures are not.
"""
import argparse
import json
import operator
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.graph import MarketAgent, log
from replay import LEAD, RUNS, AsOf

ORDER = ("pregame", "trader", "coach", "briefs")
PREGAME_WINDOW = pd.Timedelta(hours=6)
PREGAME_POLL = pd.Timedelta(minutes=60)


class Message(TypedDict):
    sender: str
    receiver: str
    kind: str                    # route | handoff | error | degraded | done
    game_id: Optional[str]
    as_of: Optional[str]
    content: Any


class NightState(TypedDict, total=False):
    date: str
    games: list                  # [{"game_id", "matchup", "tip_time", "as_of"}]
    pregame: dict                # game_id -> pregame snapshot summary
    trader: dict                 # game_id -> trader decision summary
    coach: dict                  # game_id -> explanation, lessons, agent panel
    briefs: dict                 # game_id -> channel briefs
    done: Annotated[list, operator.add]
    failed: Annotated[list, operator.add]
    messages: Annotated[list, operator.add]
    next: str


def msg(sender, receiver, kind, content, game_id=None, as_of=None) -> Message:
    return {"sender": sender, "receiver": receiver, "kind": kind, "game_id": game_id,
            "as_of": None if as_of is None else str(as_of), "content": content}


# ---------------- pregame -> trader adapter ----------------

class AvailabilityForecaster:
    """MarketAgent forecaster that applies pregame expected lost shares (overrides["availability"]) via M4."""

    def __init__(self, base):
        self.base = base
        self.name = getattr(base, "name", "forecaster")

    def __call__(self, view, game, markets, overrides):
        loss = overrides.get("availability")
        if not loss or not hasattr(self.base, "win"):
            return self.base(view, game, markets, overrides)
        p_home = self.base.win(view, game, [], availability=loss)["p_home"]
        return {m.market_ticker: p_home if m.team == game.home_team else 1 - p_home
                for m in markets.itertuples() if m.kind == "game"}

    def win(self, *args, **kwargs):
        return self.base.win(*args, **kwargs)


class PregameInformedTrader(MarketAgent):
    """The unchanged MarketAgent, with the pregame snapshot's lost shares added to its investigation."""

    def __init__(self, *args, pregame_loss: dict | None = None, **kwargs):
        self.pregame_loss = pregame_loss or {}
        super().__init__(*args, **kwargs)

    def investigate(self, state):
        out = super().investigate(state)
        loss = self.pregame_loss.get(state["game"].game_id)
        if loss:
            out["investigation"]["overrides"]["availability"] = dict(loss)
            out["trace"] = out["trace"] + log("pregame_handoff", f"expected lost share {loss}")
        return out


# ---------------- orchestrator ----------------

class NightOrchestrator:
    def __init__(self, tables: dict, forecaster, players: pd.DataFrame | None = None, lead=LEAD,
                 trader_input: str = "pregame", out_dir: Path | None = None, components: dict | None = None,
                 max_games: int | None = None):
        if trader_input not in ("pregame", "status"):
            raise ValueError("trader_input must be 'pregame' or 'status'")
        self.tables, self.forecaster, self.lead = tables, forecaster, pd.Timedelta(lead)
        self.players = players if players is not None else pd.DataFrame(columns=["player_id", "player_name"])
        self.names = dict(zip(self.players.player_id, self.players.player_name))
        self.trader_input, self.max_games = trader_input, max_games
        self.out_dir = Path(out_dir) if out_dir else RUNS / "orchestrator" / pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        prices = tables["prices"].sort_values("ts")
        self.price_index = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}
        self.components: dict[str, Callable] = {"pregame": self.run_pregame, "trader": self.run_trader,
                                                "coach": self.run_coach, "briefs": self.run_briefs}
        self.components.update(components or {})
        self.graph = self.build()

    # ---------- helpers ----------

    def view(self, now) -> AsOf:
        return AsOf(self.tables, now, self.price_index)

    def slate(self, date: str) -> list:
        g = self.tables["games"].dropna(subset=["tip_time"])
        g = g[(g.date == date) & g.game_id.isin(self.tables["markets"].game_id)].sort_values("tip_time")
        rows = list(g.itertuples(index=False))
        return rows[: self.max_games] if self.max_games else rows

    def _game(self, game_id):
        return next(self.tables["games"][self.tables["games"].game_id == game_id].itertuples(index=False))

    # ---------- components (each called as it is) ----------

    def run_pregame(self, game, now, night: dict) -> dict:
        from agents.pregame import PregameAgent
        from data_sources.live_news import TableNews

        tables = {"games": self.tables["games"], "player_games": self.tables["player_games"],
                  "news": self.tables["news"]}
        agent = PregameAgent(tables, self.players, game, self.forecaster, TableNews(self.tables["news"]),
                             self.out_dir / "pregame" / str(game.game_id))
        polls = list(pd.date_range(game.tip_time - PREGAME_WINDOW, now, freq=PREGAME_POLL))
        if not polls or polls[-1] < now:
            polls.append(now)
        snapshot = None
        for t in polls:
            out = agent.poll(t)
            snapshot = out.get("snapshot", snapshot)
        if snapshot is None:
            raise RuntimeError("pregame agent produced no snapshot before the decision time")
        return {"as_of": snapshot["as_of"], "p_home": snapshot["p_home"],
                "p_home_baseline": snapshot["baseline"]["p_home"],
                "delta_from_baseline": snapshot["delta_home_from_baseline"],
                "expected_lost_share": snapshot["expected_lost_share"],
                "factors": [{"player": self.names.get(f["player_id"], str(f["player_id"])),
                             "player_id": f["player_id"], "status": f["status"], "news_id": f["news_id"],
                             "published_at": f["published_at"], "source": f.get("source")}
                            for f in snapshot["factors"]],
                "news_health": snapshot["news_health"], "polls": len(polls)}

    def run_trader(self, game, now, night: dict) -> dict:
        snap = night.get("pregame", {}).get(game.game_id)
        use_pregame = self.trader_input == "pregame" and snap is not None
        loss = ({int(k): float(v) for k, v in snap["expected_lost_share"].items() if float(v) > 0}
                if use_pregame else {})
        agent = PregameInformedTrader(forecaster=AvailabilityForecaster(self.forecaster), learn=False,
                                      pregame_loss={game.game_id: loss} if loss else {})
        out = agent.decide(self.view(now), game, now)
        return {"input": "pregame factors" if use_pregame else "official statuses",
                "status": out.get("status"), "brief": out.get("brief", ""),
                "orders": [{"ticker": o.market_ticker, "side": o.side, "stake": o.stake, "p_model": o.p_model,
                            "reason": o.reason, "citations": list(o.citations)} for o in out.get("orders", [])],
                "candidates": [{k: r[k] for k in ("ticker", "team", "p", "bid", "ask", "side", "gap", "act",
                                                  "why_not")} for r in out.get("candidates", [])],
                "citations": list(out.get("investigation", {}).get("citations", [])),
                "steps": [s["step"] for s in out["trace"]], "trace": out["trace"]}

    def run_coach(self, game, now, night: dict) -> dict:
        from agents import coach

        s = coach.snapshot(self.forecaster, self.view(now), game, self.names)
        lines = coach.explain(s)
        t = night.get("trader", {}).get(game.game_id)
        if t is None:
            panel = ["The trading agent's decision is not available for this game, so this explanation uses the "
                     "model and market only."]
        elif t["orders"]:
            o = t["orders"][0]
            panel = [f"What the agent did: it proposed buying {o['side']} on {o['ticker'].rsplit('-', 1)[-1]} "
                     f"with a ${o['stake']:.0f} paper stake, then passed its checks and risk limits."]
        else:
            why = next((c["why_not"] for c in t["candidates"] if c.get("why_not")), "no gap after fees")
            panel = [f"What the agent did: it passed ({why}). Passing is a decision, not a failure."]
        cards = coach.lessons(s)
        text = lines + panel + [c["body"] for c in cards]
        return {"explanation": lines, "agent_panel": panel, "lessons": [c["title"] for c in cards],
                "coach_pick": s["pick"], "banned_words": coach.banned_words(text), "notice": coach.NOTICE}

    def run_briefs(self, game, now, night: dict) -> dict:
        from agents.briefs import channel_briefs, outlook

        view = self.view(now)
        news = view.news(game.game_id)
        latest = news.sort_values("published_at").groupby("player_id").tail(1)
        out = [p for p, s in zip(latest.player_id, latest.status) if str(s).lower() in ("out", "doubtful")]
        o = outlook(self.forecaster, view, game, out)
        market = None
        for m in view.markets(game.game_id).itertuples():
            q = view.quote(m.market_ticker)
            if m.team == game.home_team and q is not None:
                market = {"bid": float(q.bid), "ask": float(q.ask)}
        t = night.get("trader", {}).get(game.game_id)
        order_text = t["orders"][0]["reason"] if t and t["orders"] else None
        b = channel_briefs(game, o, news if "text" in news else news.assign(text=""), self.names, market, order_text)
        return {k: b[k] for k in ("platform", "media", "team", "retail")}

    # ---------- graph nodes ----------

    def supervisor(self, state):
        finished = set(state.get("done", []))
        if not state.get("games"):
            return {"next": "end", "messages": [msg("supervisor", "user", "done", f"no games with markets on "
                                                                                  f"{state['date']}")]}
        nxt = next((c for c in ORDER if c not in finished), None)
        if nxt is None:
            failed = sorted(set(state.get("failed", [])))
            return {"next": "end", "messages": [msg("supervisor", "user", "done",
                                                    "night complete" + (f"; degraded: {failed}" if failed else ""))]}
        note = {"pregame": "poll news and set fair odds for each game",
                "trader": "decide at the decision time using the pregame factors"
                if self.trader_input == "pregame" else "decide at the decision time from official statuses",
                "coach": "explain each game and the trader's decision to the user",
                "briefs": "write the channel briefs"}[nxt]
        if nxt == "trader" and self.trader_input == "pregame" and not state.get("pregame"):
            note += " (pregame unavailable: falling back to official statuses)"
        if nxt == "coach" and not state.get("trader"):
            note += " (trader unavailable: explain without the agent panel)"
        return {"next": nxt, "messages": [msg("supervisor", nxt, "route", note)]}

    def _worker(self, name: str, receiver: str):
        def node(state):
            results, messages, failures = {}, [], []
            night = {k: state.get(k, {}) for k in ORDER}
            for g in state["games"]:
                game, now = self._game(g["game_id"]), pd.Timestamp(g["as_of"])
                t0 = time.time()
                try:
                    r = self.components[name](game, now, night)
                    leak = _look_ahead(r, now, self.tables["news"])
                    if leak:
                        raise ValueError(f"look-ahead: {leak}")
                    results[game.game_id] = r
                    messages.append(msg(name, receiver, "handoff", {**_summary(name, r),
                                                                    "seconds": round(time.time() - t0, 2)},
                                        game.game_id, now))
                except Exception as exc:
                    failures.append(game.game_id)
                    messages.append(msg(name, receiver, "error", f"{type(exc).__name__}: {exc}", game.game_id, now))
            out = {name: results, "done": [name], "messages": messages}
            if failures:
                out["failed"] = [name]
            return out
        return node

    def build(self):
        g = StateGraph(NightState)
        g.add_node("supervisor", self.supervisor)
        receivers = dict(zip(ORDER, list(ORDER[1:]) + ["user"]))
        for name in ORDER:
            g.add_node(name, self._worker(name, receivers[name]))
            g.add_edge(name, "supervisor")
        g.add_edge(START, "supervisor")
        g.add_conditional_edges("supervisor", lambda s: END if s["next"] == "end" else s["next"], [*ORDER, END])
        return g.compile()

    def run(self, date: str) -> dict:
        games = [{"game_id": g.game_id, "matchup": f"{g.away_team}@{g.home_team}", "tip_time": str(g.tip_time),
                  "as_of": str(g.tip_time - self.lead)} for g in self.slate(date)]
        start = msg("user", "supervisor", "route", f"run the night of {date}: {len(games)} games, decisions "
                                                    f"{self.lead.total_seconds() / 60:.0f} minutes before tip")
        return self.graph.invoke({"date": date, "games": games, "pregame": {}, "trader": {}, "coach": {},
                                  "briefs": {}, "done": [], "failed": [], "messages": [start]},
                                 {"recursion_limit": 4 * len(ORDER) + 4})


# ---------------- leakage guard and summaries ----------------

def _look_ahead(result: dict, now: pd.Timestamp, news: pd.DataFrame) -> str | None:
    """Any timestamp a component returns, and any news it cites, must be at or before the decision time."""
    stamps = [result.get("as_of")] + [f.get("published_at") for f in result.get("factors", [])]
    for s in stamps:
        if s is not None and pd.Timestamp(s) > now:
            return f"{s} is after {now}"
    cited = set(result.get("citations", [])) | {c for o in result.get("orders", []) for c in o.get("citations", [])}
    published = dict(zip(news.news_id, news.published_at))
    for c in sorted(cited):
        if c not in published or published[c] > now:
            return f"cited {c} was not public at {now}"
    return None


def _summary(name: str, r: dict) -> dict:
    if name == "pregame":
        return {"p_home": round(r.get("p_home", 0) or 0, 3),
                "baseline": round(r.get("p_home_baseline", 0) or 0, 3),
                "factors": [f"{f.get('player', '?')} {f.get('status', '?')}" for f in r.get("factors", [])],
                "news_health": r.get("news_health")}
    if name == "trader":
        orders = r.get("orders") or []
        cands = r.get("candidates") or []
        return {"input": r.get("input", "unknown"), "status": r.get("status"),
                "orders": [f"{o.get('side')} {str(o.get('ticker', '')).rsplit('-', 1)[-1]} "
                           f"${float(o.get('stake', 0)):.0f}" for o in orders],
                "best_gap": max((round(c.get("gap", 0) or 0, 3) for c in cands), default=None)}
    if name == "coach":
        panel = r.get("agent_panel") or ["(no panel)"]
        return {"coach_pick": r.get("coach_pick"), "lessons": r.get("lessons", []),
                "agent_panel": panel[0] if panel else "(no panel)"}
    retail = r.get("retail") or ""
    return {"retail": retail[:160]}


def format_trace(out: dict) -> str:
    lines = []
    for m in out["messages"]:
        where = f" {m['game_id']}" if m["game_id"] else ""
        content = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], default=str)
        lines.append(f"[{m['sender']} -> {m['receiver']}] {m['kind']}{where}: {content}")
    return "\n".join(lines)


def save_trace(out: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: out.get(k) for k in ("date", "games", "done", "failed", "messages", "pregame", "trader", "coach",
                                    "briefs")}
    path.write_text(json.dumps(keep, indent=2, default=str))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="game date (YYYY-MM-DD, the schedule's date column)")
    ap.add_argument("--source", choices=["frozen", "sample", "synthetic"], default="frozen")
    ap.add_argument("--forecaster", choices=["models", "record"], default="models")
    ap.add_argument("--lead-minutes", type=float, default=LEAD.total_seconds() / 60)
    ap.add_argument("--trader-input", choices=["pregame", "status"], default="pregame")
    ap.add_argument("--games", type=int, default=None, help="only the first N games of the night")
    ap.add_argument("--out", type=Path, default=None, help="JSON trace (default runs/orchestrator/<date>.json)")
    args = ap.parse_args()

    from agents.graph import load_source, record_forecaster
    from data_sources import FROZEN, ROOT, read_table
    tables = load_source(args.source, args.date, args.date)
    try:
        players = read_table("players", ROOT / "data" / "sample" if args.source == "sample" else FROZEN)
    except FileNotFoundError:
        players = None
    if args.forecaster == "models":
        from forecast.api import Forecaster
        forecaster = Forecaster.load()
    else:
        forecaster = record_forecaster
    run_dir = RUNS / "orchestrator" / f"{args.date}-{pd.Timestamp.now():%H%M%S}"
    orch = NightOrchestrator(tables, forecaster, players, pd.Timedelta(minutes=args.lead_minutes),
                             args.trader_input, run_dir, max_games=args.games)
    out = orch.run(args.date)
    print(format_trace(out))
    path = save_trace(out, args.out or RUNS / "orchestrator" / f"{args.date}.json")
    print(f"\ntrace -> {path}")


if __name__ == "__main__":
    main()
