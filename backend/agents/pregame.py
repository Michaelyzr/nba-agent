"""Pregame-only agent: historical baseline -> poll news -> factors -> fair odds.

    python -m agents.pregame --mode replay --source sample --game-id <id> --forecaster record
    python -m agents.pregame --mode live --source frozen --game-id <id> --poll-seconds 300

The runner stops at tip-off. It does not submit orders or settle games. Model
history is fixed at baseline time so subsequent movement is attributable to
news, rather than another finished game's score or a market-price anchor.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from data_sources import FROZEN, ROOT, read_table
from data_sources.live_news import LiveNews, TableNews
from data_sources.news_registry import NewsRegistry, DEFAULT_REGISTRY
from forecast.history import History
from replay import AsOf, RUNS

# Scenario assumptions, not calibrated participation probabilities.
STATUS_LOSS = {"out": 1.0, "doubtful": 0.75, "questionable": 0.5, "probable": 0.25,
               "available": 0.0, "active": 0.0}


def utc(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return t.tz_convert("UTC")


def odds(p_home):
    p = float(p_home)
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("win probability must be finite and between zero and one")
    return {"p_home": p, "p_away": 1 - p, "home_decimal_odds": 1 / p,
            "away_decimal_odds": 1 / (1 - p)}


class PollState(TypedDict, total=False):
    now: Any
    batch: Any
    new_rows: list
    errors: list
    snapshot: dict
    stopped: bool
    items: list
    coverage: list
    new_evidence_count: int


class PregameAgent:
    def __init__(self, tables, players, game, model, provider, output: Path,
                 lookback_hours=48, clock=None, market_provider=None):
        if not math.isfinite(lookback_hours) or lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        self.tables, self.players, self.game = tables, players, game
        self.model, self.provider, self.output = model, provider, Path(output)
        self.registry = getattr(provider, "registry", None) or NewsRegistry.load()
        self.lookback = pd.Timedelta(hours=lookback_hours)
        self.clock = clock  # live: observation time is AFTER the network request
        self.market_provider = market_provider
        self.baseline, self.factors, self.seen, self.last_as_of = None, {}, set(), None
        self.latest = None
        self._history_index = None
        self.factor_evidence, self.conflicts, self.seen_items = {}, [], set()
        self.state_path = self.output / "state.json"
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            if saved["game_id"] != game.game_id or saved["tip_time"] != utc(game.tip_time).isoformat():
                raise ValueError("output belongs to another game or tip time; choose a new --name")
            if saved["model"] != self.model_name or saved["lookback_hours"] != lookback_hours:
                raise ValueError("run configuration changed; choose a new --name")
            self.baseline, self.factors = saved["baseline"], saved["factors"]
            self.seen, self.last_as_of = set(saved["seen"]), utc(saved["last_as_of"])
            self.latest = saved.get("latest")
            self.factor_evidence = saved.get("factor_evidence", {
                key: {row["source"]: row} for key, row in self.factors.items()})
            self.seen_items = set(saved.get("seen_items", []))
            if hasattr(self.provider, "load_state"):
                self.provider.load_state(saved.get("provider", {}))
        g = StateGraph(PollState)
        g.add_node("baseline", self._baseline)
        g.add_node("retrieve_news", self._retrieve)
        g.add_node("extract_factors", self._extract)
        g.add_node("reforecast", self._reforecast)
        g.add_node("record", self._record)
        g.add_edge(START, "baseline")
        g.add_edge("baseline", "retrieve_news")
        g.add_conditional_edges("retrieve_news", lambda s: END if s.get("stopped") else "extract_factors",
                                [END, "extract_factors"])
        g.add_edge("extract_factors", "reforecast")
        g.add_edge("reforecast", "record")
        g.add_edge("record", END)
        self.graph = g.compile()

    @property
    def model_name(self):
        return getattr(self.model, "name", getattr(self.model, "__name__", type(self.model).__name__))

    def _view(self):
        return AsOf(self.tables, utc(self.baseline["as_of"]), {})

    def _history(self):
        if self._history_index is None:
            view = self._view()
            self._history_index = (self.model.history(view) if hasattr(self.model, "history")
                                   else History(view.player_games(), view.games()))
        return self._history_index

    def _probability(self, loss):
        view = self._view()
        if hasattr(self.model, "win"):
            return self.model.win(view, self.game, availability=loss)["p_home"]
        # Explicit record ablation: same usual-minutes adjustment as agents.graph.
        from agents.graph import clip, log5_home, win_rate, OUT_MINUTE_VALUE
        done = view.games().dropna(subset=["home_pts"])
        p = log5_home(win_rate(done, self.game.home_team_id), win_rate(done, self.game.away_team_id))
        h = self._history()
        for pid, share in loss.items():
            team, _ = h.last_played(pid, self.game.tip_time)
            sign = -1 if team == self.game.home_team_id else 1 if team == self.game.away_team_id else 0
            p += sign * OUT_MINUTE_VALUE * h.usual(pid, self.game.tip_time)[0] * share
        return clip(p)

    def _append(self, filename, value):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / filename).open("a") as f:
            f.write(json.dumps(value, default=str, allow_nan=False) + "\n")

    def _save_state(self, now):
        saved = {"game_id": self.game.game_id, "tip_time": utc(self.game.tip_time).isoformat(),
                 "model": self.model_name, "lookback_hours": self.lookback.total_seconds() / 3600,
                 "baseline": self.baseline, "factors": self.factors, "seen": sorted(self.seen),
                 "last_as_of": now.isoformat(), "latest": self.latest,
                 "factor_evidence": self.factor_evidence, "seen_items": sorted(self.seen_items),
                 "provider": self.provider.dump_state() if hasattr(self.provider, "dump_state") else {},
                 "registry_reviewed_at": self.registry.data["reviewed_at"]}
        self.output.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(saved, default=str, allow_nan=False, indent=2))
        tmp.replace(self.state_path)
        self.last_as_of = now

    def _baseline(self, state):
        if self.baseline is None:
            self.baseline = {"as_of": state["now"].isoformat()}
            self.baseline.update(odds(self._probability({})))
            self._append("snapshots.jsonl", {"phase": "baseline", "game_id": self.game.game_id,
                                             "model": self.model_name, **self.baseline})
            self._save_state(state["now"])
        return {}

    def _retrieve(self, state):
        h = self._history()
        roster = set(h.roster(self.game.home_team_id, self.game.tip_time)) | set(h.roster(self.game.away_team_id, self.game.tip_time))
        self.roster = roster
        since = utc(self.baseline["as_of"]) - self.lookback
        batch = self.provider.fetch(self.game, self.players, roster, since, state["now"])
        now = utc(self.clock()) if self.clock else state["now"]
        if now >= utc(self.game.tip_time):
            return {"now": now, "stopped": True}
        if self.clock:
            for row in batch.rows + batch.items:
                row["observed_at"] = now
        return {"batch": batch, "now": now, "errors": list(batch.errors), "stopped": False,
                "items": batch.items, "coverage": batch.coverage}

    def _resolve_factors(self):
        self.conflicts = []
        for key, candidates in self.factor_evidence.items():
            if not candidates:
                continue
            chosen = max(candidates.values(), key=lambda r: (
                self.registry.priority(r["source"]), r["published_at"], r["observed_at"], r["news_id"]))
            self.factors[key] = {**chosen, "source_role": self.registry.role(chosen["source"]),
                                 "confirmation": "primary" if self.registry.priority(chosen["source"]) >= 30 else "secondary_only"}
            value = lambda r: (str(r["status"]).lower(), r.get("minutes_limit") if key.endswith(":minutes") else None)
            alternatives = [r for r in candidates.values() if value(r) != value(chosen)]
            if alternatives:
                self.conflicts.append({"factor": key, "selected_news_id": chosen["news_id"],
                                       "selected_source": chosen["source"], "selected_status": chosen["status"],
                                       "reason": "official/authoritative source first; newest within the same tier",
                                       "alternatives": alternatives,
                                       "newer_secondary_pending": any(r["published_at"] > chosen["published_at"]
                                                                      for r in alternatives)})

    def _extract(self, state):
        now, rows, errors = state["now"], [], list(state["errors"])
        since = utc(self.baseline["as_of"]) - self.lookback
        evidence_count = 0
        for raw in state.get("items", []):
            try:
                item = dict(raw)
                published = utc(item["published_at"])
                observed = utc(item.get("observed_at", now))
                if (item["game_id"] != self.game.game_id or not since <= published <= now
                        or observed > now or published >= utc(self.game.tip_time)):
                    continue
                if item["item_id"] in self.seen_items:
                    continue
                item.update(published_at=published.isoformat(), observed_at=observed.isoformat())
                self._append("evidence.jsonl", item)
                self.seen_items.add(item["item_id"])
                evidence_count += 1
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"source": "evidence_validation", "error": str(exc)})
        for raw in state["batch"].rows:
            try:
                row = {k: None if pd.api.types.is_scalar(v) and pd.isna(v) else v for k, v in raw.items()}
                published = utc(row["published_at"])
                observed = utc(row.get("observed_at", now))
                pid = int(row["player_id"])
                if (row["game_id"] != self.game.game_id or pid not in self.roster
                        or not since <= published <= now or observed > now or published >= utc(self.game.tip_time)):
                    continue
                if not isinstance(row.get("news_id"), str) or not row["news_id"] or not isinstance(row.get("source"), str) or not row["source"]:
                    raise ValueError("news_id and source are required")
                if row["news_id"] in self.seen:
                    continue
                row.update(player_id=pid, published_at=published.isoformat(), observed_at=observed.isoformat())
                rows.append(row)
                self.seen.add(row["news_id"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"source": "validation", "error": str(exc)})
        rows.sort(key=lambda r: (r["published_at"], r["observed_at"], r["news_id"]))
        for row in rows:
            status = str(row.get("status", "")).lower()
            kind = "status" if status in STATUS_LOSS else "minutes" if status == "minutes_limit" else None
            row["factor_result"] = "review_only"
            if kind:
                key = f"{row['player_id']}:{kind}"
                candidates = self.factor_evidence.setdefault(key, {})
                previous = candidates.get(row["source"])
                rank = lambda r: (r["published_at"], r["observed_at"], r["news_id"])
                if previous is not None and rank(row) <= rank(previous):
                    row["factor_result"] = "superseded"
                else:
                    if kind == "minutes":
                        try:
                            limit = float(row["minutes_limit"])
                            if not math.isfinite(limit) or not 0 <= limit <= 48:
                                raise ValueError("minutes_limit must be between zero and 48")
                            row["minutes_limit"] = limit
                        except (KeyError, TypeError, ValueError) as exc:
                            errors.append({"source": row["news_id"], "error": str(exc)})
                            self._append("news.jsonl", row)
                            continue
                    candidates[row["source"]] = row.copy()
                    self._resolve_factors()
                    row["factor_result"] = "applied" if self.factors[key]["news_id"] == row["news_id"] else "conflict_pending"
            self._append("news.jsonl", row)
        self._resolve_factors()
        return {"new_rows": rows, "errors": errors, "new_evidence_count": evidence_count}

    def _reforecast(self, state):
        status_loss, minutes_loss = {}, {}
        h = self._history()
        for factor in self.factors.values():
            pid = factor["player_id"]
            if factor["status"].lower() in STATUS_LOSS:
                share = STATUS_LOSS[factor["status"].lower()]
                status_loss[pid] = share
            else:
                usual = h.usual(pid, self.game.tip_time)[0]
                share = max(0.0, 1 - factor["minutes_limit"] / usual) if usual > 0 else 0.0
                minutes_loss[pid] = share
        loss = {pid: 1 - (1 - status_loss.get(pid, 0.0)) * (1 - minutes_loss.get(pid, 0.0))
                for pid in status_loss.keys() | minutes_loss.keys()}
        current = odds(self._probability(loss))
        previous = self.latest or self.baseline
        snapshot = {"phase": "update", "game_id": self.game.game_id, "as_of": state["now"].isoformat(),
                    "tip_time": utc(self.game.tip_time).isoformat(), "model": self.model_name,
                    "baseline": self.baseline, **current,
                    "delta_home": current["p_home"] - previous["p_home"],
                    "delta_home_from_baseline": current["p_home"] - self.baseline["p_home"],
                    "expected_lost_share": {str(k): v for k, v in loss.items()},
                    "factors": list(self.factors.values()), "new_news_ids": [r["news_id"] for r in state["new_rows"]],
                    "conflicts": self.conflicts, "source_coverage": state.get("coverage", []),
                    "new_evidence_count": state.get("new_evidence_count", 0),
                    "errors": state["errors"], "news_health": "degraded" if state["errors"] else "ok",
                    "odds_basis": "fair decimal odds, no bookmaker margin",
                    "factor_assumptions": STATUS_LOSS}
        if self.market_provider is not None:
            try:
                snapshot["polymarket"] = self.market_provider(self.game, state["now"])
            except Exception as exc:
                snapshot["polymarket"] = {
                    "source": "polymarket", "is_live": True, "status": "UNAVAILABLE",
                    "source_state": "SOURCE DATA UNAVAILABLE", "fetched_at": state["now"].isoformat(),
                    "age_seconds": 0.0, "usable_for_future_analysis": False,
                    "reasons": [f"Polymarket provider failed: {exc}"], "markets": [],
                }
        return {"snapshot": snapshot}

    def _record(self, state):
        snapshot = state["snapshot"]
        self._append("snapshots.jsonl", snapshot)
        self.latest = snapshot
        self._save_state(state["now"])
        return {}

    def poll(self, now):
        now = utc(now)
        if now >= utc(self.game.tip_time):
            return {"stopped": True, "now": now}
        if self.last_as_of is not None and now < self.last_as_of:
            raise ValueError("cannot move a persisted pregame run backward in time; choose a new --name")
        return self.graph.invoke({"now": now})


def run_loop(agent, interval_seconds=300, start=None, until=None, replay_mode=False,
             once=False, clock=lambda: pd.Timestamp.now(tz="UTC"), sleeper=time.sleep):
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("poll interval must be positive and finite")
    tip = utc(agent.game.tip_time)
    end = min(tip, utc(until)) if until is not None else tip
    now = utc(start) if replay_mode else utc(clock())
    if replay_mode and now >= end:
        raise ValueError("replay start must be before the end time")
    next_poll = now
    while now < end:
        if now >= next_poll:
            out = agent.poll(now)
            if out.get("stopped"):
                break
            s = out["snapshot"]
            print(f"{s['as_of']} {agent.game.away_team}@{agent.game.home_team} "
                  f"home {s['p_home']:.2%} / {s['home_decimal_odds']:.3f}, "
                  f"away {s['p_away']:.2%} / {s['away_decimal_odds']:.3f}; "
                  f"change {s['delta_home']:+.2%}; news {len(s['new_news_ids'])}; {s['news_health']}", flush=True)
            if once:
                return
            # Fixed cadence; skip slots missed during a slow HTTP request.
            after = utc(clock()) if not replay_mode else now
            steps = max(1, int((after - next_poll).total_seconds() // interval_seconds) + 1)
            next_poll += pd.Timedelta(seconds=steps * interval_seconds)
        if replay_mode:
            now = next_poll
        else:
            remaining = (min(next_poll, end) - utc(clock())).total_seconds()
            if remaining > 0:
                sleeper(min(remaining, 30))
            now = utc(clock())
    print(f"Stopped: {'tip-off' if end == tip else 'end of requested window'}; {agent.output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--source", choices=["sample", "frozen"], default="sample")
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--forecaster", choices=["models", "record"], default="models")
    ap.add_argument("--poll-seconds", type=float, default=60)
    ap.add_argument("--window-hours", type=float, default=6)
    ap.add_argument("--lookback-hours", type=float, default=48)
    ap.add_argument("--start-time", help="replay only, timezone-aware ISO timestamp")
    ap.add_argument("--until", help="optional end time, always capped at tip-off")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--rss-url", action="append", help="replace default media RSS feeds; repeat for multiple feeds")
    ap.add_argument("--no-official", action="store_true")
    ap.add_argument("--no-x", action="store_true", help="disable X requests (otherwise missing credentials are logged)")
    ap.add_argument("--news-registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--name", default="pregame")
    args = ap.parse_args()
    if args.mode == "live" and args.start_time:
        ap.error("--start-time is replay-only; live uses the current UTC clock")
    if not math.isfinite(args.window_hours) or args.window_hours <= 0:
        ap.error("--window-hours must be positive and finite")
    folder = ROOT / "data" / "sample" if args.source == "sample" else FROZEN
    tables = {n: read_table(n, folder) for n in ("games", "player_games")}
    if args.mode == "replay":
        tables["news"] = read_table("news", folder)
    else:
        # Live must not consume the inactive-list fallback stamped retrospectively.
        tables["news"] = pd.DataFrame(columns=["news_id", "game_id", "published_at", "player_id", "status"])
    players = read_table("players", folder)
    found = tables["games"][tables["games"].game_id.astype(str) == args.game_id]
    if len(found) != 1:
        ap.error(f"game id {args.game_id!r} must match exactly one schedule row")
    game = next(found.itertuples(index=False))
    if args.mode == "live" and utc(game.tip_time) <= pd.Timestamp.now(tz="UTC"):
        ap.error("live mode requires a future game in the local schedule")
    if args.forecaster == "models":
        from forecast.api import Forecaster
        model = Forecaster.load()
    else:
        from agents.graph import record_forecaster
        model = record_forecaster
    live_clock = lambda: pd.Timestamp.now(tz="UTC")
    provider = (LiveNews(args.rss_url, official=not args.no_official, registry=NewsRegistry.load(args.news_registry),
                         x_enabled=not args.no_x) if args.mode == "live" else TableNews(tables["news"]))
    market_provider = None
    if args.mode == "live":
        from data_sources.polymarket_live import PolymarketLiveProvider
        polymarket = PolymarketLiveProvider()
        market_provider = polymarket.for_game
    agent = PregameAgent(tables, players, game, model, provider, RUNS / args.name,
                         args.lookback_hours, clock=live_clock if args.mode == "live" else None,
                         market_provider=market_provider)
    start = utc(args.start_time) if args.start_time else utc(game.tip_time) - pd.Timedelta(hours=args.window_hours)
    if args.mode == "replay" and agent.last_as_of is not None and not args.start_time:
        start = agent.last_as_of + pd.Timedelta(seconds=args.poll_seconds)
    try:
        run_loop(agent, args.poll_seconds, start, args.until, args.mode == "replay", args.once)
    except KeyboardInterrupt:
        print(f"Stopped by user; state saved in {agent.output}")


if __name__ == "__main__":
    main()
