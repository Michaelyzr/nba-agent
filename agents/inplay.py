"""Independent live-game agent: scoreboard + events -> probabilities/fair odds.

    python -m agents.inplay --mode replay --source sample --game-id 401811041
    python -m agents.inplay --mode live --source frozen --game-id <id>

All output is under runs/inplay/. No pregame state, news cache or runtime imports.
"""
import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import pandas as pd

from data_sources import ROOT, FROZEN, read_table
from data_sources.inplay import LiveInPlay, TableInPlay
from data_sources.inplay_news import LOSSES, TERMINAL
from data_sources.inplay_types import utc, validate_score
from data_sources.news_registry import NewsRegistry, DEFAULT_REGISTRY
from forecast.history import History
from forecast.inplay import InPlayWinModel, features, remaining_seconds
from replay import AsOf

INPLAY_RUNS = ROOT / "runs" / "inplay"


def fair_odds(p):
    return {"p_home": p, "p_away": 1 - p, "home_decimal_odds": 1 / p if p else None,
            "away_decimal_odds": 1 / (1 - p) if p < 1 else None}


class InPlayAgent:
    def __init__(self, tables, players, game, provider, output, prior_model=None,
                 initial_p_home=None, win_model=None, max_score_age=30, point_value=0.12,
                 clock=None):
        if not math.isfinite(max_score_age) or max_score_age <= 0:
            raise ValueError("score age limit must be positive and finite")
        if not math.isfinite(point_value) or point_value < 0:
            raise ValueError("replacement-adjusted point value must be nonnegative and finite")
        if initial_p_home is not None and (not math.isfinite(initial_p_home) or not 0 < initial_p_home < 1):
            raise ValueError("initial probability must be strictly between zero and one")
        self.tables, self.players, self.game, self.provider = tables, players, game, provider
        self.output, self.clock = Path(output), clock
        self.registry = provider.registry
        self.win_model = win_model or InPlayWinModel()
        self.prior_model, self.initial_p_home = prior_model, initial_p_home
        self.max_age, self.point_value = max_score_age, point_value
        self.prior, self.score, self.last_as_of, self.latest = None, None, None, None
        self.candidates, self.factors, self.conflicts = {}, {}, []
        self.episodes = {}
        self.seen, self.seen_items, self.history = set(), set(), None
        self.stopped = False
        model_config = {"sigma": self.win_model.sigma, "model": self.win_model.name,
                        "coefs": self.win_model.fitted.coef_.tolist() if hasattr(self.win_model.fitted, "coef_") else None,
                        "intercept": self.win_model.fitted.intercept_.tolist() if hasattr(self.win_model.fitted, "intercept_") else None}
        if self.win_model.fitted is not None and hasattr(self.win_model.fitted, "kind"):
            model_config["artifact_hash"] = hashlib.sha256(pickle.dumps(self.win_model.fitted)).hexdigest()
        self.config = {"pipeline": "inplay-v1", "game_id": str(game.game_id),
                       "tip_time": utc(game.tip_time).isoformat(), "initial_p_home": initial_p_home,
                       "max_score_age": max_score_age, "point_value": point_value, **model_config}
        self.state_path = self.output / "inplay_state.json"
        # Refuse to mix both systems in the same directory even via a custom API caller.
        if (self.output / "state.json").exists() or (self.output / "snapshots.jsonl").exists():
            raise ValueError("in-play output cannot share a pregame run directory")
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            if saved["config"] != self.config:
                raise ValueError("in-play run configuration changed; choose a new --name")
            self.prior, self.score = saved["prior"], saved["score"]
            self.last_as_of = utc(saved["last_as_of"])
            self.latest = saved.get("latest")
            self.candidates = saved["candidates"]
            self.episodes = saved.get("episodes", {})
            self.seen, self.seen_items = set(saved["seen"]), set(saved["seen_items"])
            self.stopped = saved.get("stopped", False)
            self.provider.load_state(saved.get("provider", {}))

    def _append(self, name, row):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / name).open("a") as file:
            file.write(json.dumps(row, default=str, allow_nan=False) + "\n")

    def _save(self, now, snapshot):
        saved = {"config": self.config, "prior": self.prior, "score": self.score,
                 "last_as_of": now.isoformat(), "latest": snapshot, "candidates": self.candidates,
                 "episodes": self.episodes,
                 "seen": sorted(self.seen), "seen_items": sorted(self.seen_items),
                 "stopped": self.stopped, "provider": self.provider.dump_state()}
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, default=str, allow_nan=False, indent=2))
        temporary.replace(self.state_path)
        self.last_as_of, self.latest = now, snapshot

    def _history(self):
        if self.history is None:
            view = AsOf(self.tables, utc(self.game.tip_time), {})
            self.history = History(view.player_games(), view.games())
        return self.history

    def _initialize(self):
        if self.prior is not None:
            return
        if self.initial_p_home is not None:
            p, source = self.initial_p_home, "explicit_initial_probability"
        else:
            if self.prior_model is None:
                from forecast.api import Forecaster
                self.prior_model = Forecaster.load()
            p = self.prior_model.win(AsOf(self.tables, utc(self.game.tip_time), {}), self.game)["p_home"]
            source = self.prior_model.name
        if not math.isfinite(p) or not 0 < p < 1:
            raise ValueError("historical initial probability is invalid")
        self.prior = {"p_home": p, "as_of": utc(self.game.tip_time).isoformat(), "model": source}

    def _priority(self, source):
        if source == "nba_live":
            return 60
        if source == "espn_live" or self.registry.role(source) == "authority_media":
            return 50
        return self.registry.priority(source)

    def _extract(self, batch, roster, now, errors):
        for raw in batch.evidence:
            try:
                item = dict(raw)
                published, observed = utc(item["published_at"]), utc(item["observed_at"])
                if (str(item["game_id"]) != str(self.game.game_id) or not utc(self.game.tip_time) <= published <= now
                        or observed > now or item["item_id"] in self.seen_items):
                    continue
                item.update(published_at=published.isoformat(), observed_at=observed.isoformat())
                self._append("inplay_evidence.jsonl", item)
                self.seen_items.add(item["item_id"])
            except (KeyError, ValueError, TypeError):
                errors.append({"source": "evidence", "error": "invalid in-game evidence"})
        new_ids = []
        def event_time(raw):
            try:
                return utc(raw["published_at"])
            except (KeyError, ValueError, TypeError):
                return utc(self.game.tip_time)
        for raw in sorted(batch.events, key=event_time):
            try:
                row = dict(raw)
                published, observed = utc(row["published_at"]), utc(row["observed_at"])
                pid = int(row["player_id"])
                if (str(row["game_id"]) != str(self.game.game_id) or pid not in roster
                        or not utc(self.game.tip_time) <= published <= now or observed > now
                        or row["event_id"] in self.seen):
                    continue
                if not isinstance(row.get("source"), str) or not row["source"]:
                    raise ValueError("missing event source")
                row.update(player_id=pid, published_at=published.isoformat(), observed_at=observed.isoformat(),
                           publication_to_observation_seconds=max(0.0, (observed - published).total_seconds()))
                self.seen.add(row["event_id"])
                new_ids.append(row["event_id"])
                if row["status"] in LOSSES:
                    key = str(pid)
                    by_source = self.candidates.setdefault(key, {})
                    prior = max(by_source.values(), key=lambda r: (self._priority(r["source"]), r["published_at"])) if by_source else None
                    if (prior and prior["status"] in {"returned", "rescinded"} and LOSSES[row["status"]] > 0
                            and row["published_at"] > prior["published_at"]):
                        # A new injury after recovery is a new episode, not a dispute
                        # with an earlier report that the player returned to the court.
                        self.episodes[key] = row["published_at"]
                        by_source = self.candidates[key] = {}
                    previous = by_source.get(row["source"])
                    rank = lambda r: (r["published_at"], r["observed_at"], r["event_id"])
                    in_episode = key not in self.episodes or row["published_at"] >= self.episodes[key]
                    if in_episode and (previous is None or rank(row) > rank(previous)):
                        # Returning to the court does not reverse an ejection/foul-out.
                        if not (previous and previous["status"] in TERMINAL and row["status"] not in TERMINAL | {"rescinded"}):
                            by_source[row["source"]] = row
                self._append("inplay_events.jsonl", row)
            except (KeyError, ValueError, TypeError):
                errors.append({"source": "events", "error": "invalid in-game event"})
        self.factors, self.conflicts = {}, []
        for pid, candidates in self.candidates.items():
            chosen = max(candidates.values(), key=lambda r: (self._priority(r["source"]), r["published_at"], r["observed_at"]))
            if chosen["status"] == "left_injured":
                prognosis = [r for r in candidates.values() if r["status"] in {"injury_out", "questionable_return", "doubtful_return"}
                             and r["published_at"] >= chosen["published_at"]]
                if prognosis:
                    # Departure alone says nothing about a subsequent return prognosis.
                    chosen = max(prognosis, key=lambda r: (self._priority(r["source"]), r["published_at"]))
            self.factors[pid] = {**chosen, "confirmation": "primary" if self._priority(chosen["source"]) >= 30 else "secondary_only"}
            alternatives = [r for r in candidates.values() if LOSSES[r["status"]] != LOSSES[chosen["status"]]
                            and not (r["status"] == "left_injured" and r["published_at"] <= chosen["published_at"]
                                     and chosen["status"] in {"injury_out", "questionable_return", "doubtful_return", "returned"})]
            if alternatives:
                self.conflicts.append({"player_id": int(pid), "selected": self.factors[pid], "alternatives": alternatives})
        return new_ids

    def _news_margin(self, score, factors=None):
        h = self._history()
        remaining = remaining_seconds(score) / 60
        total, effects = 0.0, []
        for factor in self.factors.values() if factors is None else factors:
            pid, lost = factor["player_id"], LOSSES[factor["status"]]
            team = score.get("player_teams", {}).get(str(pid)) or h.last_played(pid, self.game.tip_time)[0]
            usual, usual_points = h.usual(pid, self.game.tip_time)
            played = score.get("player_minutes", {}).get(str(pid))
            if score["period"] > 4 or played is None:
                minutes = min(remaining, usual * remaining / 48)
            else:
                minutes = min(remaining, max(0.0, usual - float(played)))
            sign = -1 if team == self.game.home_team_id else 1 if team == self.game.away_team_id else 0
            impact = sign * self.point_value * minutes * lost
            total += impact
            effects.append({"player_id": pid, "status": factor["status"], "expected_remaining_minutes": minutes,
                            "player": dict(zip(self.players.player_id, self.players.player_name)).get(pid, str(pid)),
                            "expected_lost_share": lost, "home_margin_adjustment": impact,
                            "historical_usual_minutes": usual, "historical_usual_points": usual_points,
                            "observed_played_minutes": played, "point_value_per_lost_minute": self.point_value,
                            "team_id": int(team) if team is not None else None,
                            "source": factor["source"], "impact_basis": "uniform_replacement_adjusted_assumption"})
        return total, effects

    def poll(self, now):
        now = utc(now)
        if self.last_as_of and now < self.last_as_of:
            raise ValueError("cannot move a persisted in-play run backward")
        if self.stopped:
            return {"stopped": True, "snapshot": self.latest}
        errors, new_ids = [], []
        coverage, batch = [], None
        accepted_score = False
        if now < utc(self.game.tip_time):
            quote_state = "waiting_for_tip"
        else:
            h = self._history()
            roster = set(h.roster(self.game.home_team_id, self.game.tip_time)) | set(h.roster(self.game.away_team_id, self.game.tip_time))
            batch = self.provider.fetch(self.game, self.players, roster, now)
            now = utc(self.clock()) if self.clock else now
            errors, coverage = list(batch.errors), batch.coverage
            if batch.score is not None:
                try:
                    candidate = validate_score(batch.score, self.game, now)
                    if self.score and candidate["period"] < self.score["period"]:
                        raise ValueError("game period moved backward")
                    if self.score and self.score["phase"] == "live" and candidate["phase"] == "scheduled":
                        raise ValueError("started game moved back to scheduled")
                    if (self.score and self.score.get("updated_at") and candidate.get("updated_at")
                            and utc(candidate["updated_at"]) < utc(self.score["updated_at"])):
                        raise ValueError("source timestamp moved backward")
                    self.score = candidate
                    accepted_score = True
                except (KeyError, ValueError, TypeError):
                    errors.append({"source": "score_validation", "error": "invalid or regressed live state"})
            if self.score is not None:
                roster |= set(map(int, self.score.get("roster", [])))
            new_ids = self._extract(batch, roster, now, errors)
            quote_state = "score_unavailable" if self.score is None else self.score["phase"]
        snapshot = {"pipeline": "inplay", "game_id": str(self.game.game_id), "as_of": now.isoformat(),
                    "model": self.win_model.name, "quote_state": quote_state, "score": self.score,
                    "p_home": None, "p_away": None, "home_decimal_odds": None, "away_decimal_odds": None,
                    "new_event_ids": new_ids, "factors": list(self.factors.values()), "conflicts": self.conflicts,
                    "source_coverage": coverage, "errors": errors, "news_health": "degraded" if errors else "ok",
                    "model_trained": self.win_model.fitted is not None,
                    "heldout_validation": self.win_model.validation,
                    "calibration_status": "calibrated_heldout" if (self.win_model.validation or {}).get("calibrated") else "heldout_evaluated" if self.win_model.validation else "validation_required",
                    "model_parameters": {"sigma": self.win_model.sigma},
                    "assumptions": {"point_value_per_lost_minute": self.point_value, "uncertain_status_loss": LOSSES},
                    "odds_basis": "fair decimal odds, no bookmaker margin"}
        if self.score:
            # Observation age and source age are separate; polling cannot make old content fresh.
            observed_age = max(0.0, (now - utc(self.score["observed_at"])).total_seconds())
            source_age = max(0.0, (now - utc(self.score["updated_at"])).total_seconds()) if self.score.get("updated_at") else None
            snapshot.update(score_observation_age_seconds=observed_age, score_source_age_seconds=source_age)
            if batch is not None and not accepted_score:
                # Even a recent cached score must not price new injuries after a feed failure.
                snapshot["quote_state"] = "score_unavailable"
            elif snapshot["quote_state"] == "final":
                if self.score["home_score"] == self.score["away_score"]:
                    snapshot["quote_state"] = "invalid_final_tie"
                else:
                    snapshot.update(fair_odds(float(self.score["home_score"] > self.score["away_score"])))
                    self.stopped = True
            elif observed_age > self.max_age or (source_age is not None and source_age > self.max_age):
                snapshot["quote_state"] = "stale_score"
            elif snapshot["quote_state"] == "live":
                self._initialize()
                news_margin, effects = self._news_margin(self.score)
                p = self.win_model.predict(self.score, self.prior["p_home"], news_margin)
                old_margin, _ = self._news_margin(self.score, (self.latest or {}).get("factors", []))
                without_news = self.win_model.predict(self.score, self.prior["p_home"], 0)
                before_new_events = self.win_model.predict(self.score, self.prior["p_home"], old_margin)
                snapshot.update(fair_odds(p), prior=self.prior, news_margin=news_margin, player_effects=effects,
                                p_home_without_news=without_news, p_home_before_new_events=before_new_events,
                                news_effect_pp=(p - without_news) * 100,
                                new_event_effect_pp=(p - before_new_events) * 100,
                                training_features=features(self.score, self.prior["p_home"], news_margin),
                                remaining_seconds=remaining_seconds(self.score),
                                quote_quality="provisional" if source_age is None or errors or self.win_model.fitted is None else "observed")
                if source_age is None:
                    snapshot["freshness"] = "source_timestamp_unverified"
                else:
                    snapshot["freshness"] = "within_age_limit"
        from agents.news_report import make_report, persist_report
        snapshot["report"] = make_report(snapshot, self.game, self.players, self.latest, batch.events if batch else [])
        from agents.forecast_audit import audit_snapshot
        snapshot["forecast_audit"] = audit_snapshot(snapshot, self.latest)
        persist_report(self.output, snapshot["report"])
        self._append("inplay_snapshots.jsonl", snapshot)
        self._save(now, snapshot)
        return {"stopped": self.stopped, "snapshot": snapshot}


def run_loop(agent, interval_seconds=5, replay_times=None, max_runtime_seconds=21600,
             clock=lambda: pd.Timestamp.now(tz="UTC"), sleeper=time.sleep):
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("in-play polling interval must be positive and finite")
    if not math.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0:
        raise ValueError("runtime limit must be positive and finite")
    start = utc(clock())
    try:
        times = replay_times if replay_times is not None else iter(lambda: utc(clock()), None)
        for now in times:
            if replay_times is None and (now - start).total_seconds() >= max_runtime_seconds:
                break
            poll_started = time.monotonic()
            result = agent.poll(now)
            s = result["snapshot"]
            p = "unavailable" if s["p_home"] is None else f"{s['p_home']:.2%} / {s['home_decimal_odds']}"
            print(f"{s['as_of']} {agent.game.away_team}@{agent.game.home_team} {s['quote_state']}: "
                  f"home {p}; news {len(s['new_event_ids'])}; {s['news_health']}", flush=True)
            if result["stopped"]:
                break
            if replay_times is None:
                # New streamed evidence wakes the loop; score polling remains a fallback.
                wait_seconds = min(30, max(0, interval_seconds - (time.monotonic() - poll_started)),
                                   max(0, max_runtime_seconds - (utc(clock()) - start).total_seconds()))
                wake = getattr(agent.provider, "wake", None)
                if wake is not None:
                    wake.wait(wait_seconds)
                    wake.clear()
                    sleeper(min(0.25, interval_seconds))
                else:
                    sleeper(wait_seconds)
    finally:
        agent.provider.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["live", "replay"], default="replay")
    ap.add_argument("--source", choices=["sample", "frozen"], default="sample")
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--nba-game-id")
    ap.add_argument("--initial-p-home", type=float)
    ap.add_argument("--model-file", type=Path, help="trusted locally trained in-play model")
    ap.add_argument("--scores", type=Path, help="replay score snapshots parquet")
    ap.add_argument("--events", type=Path, help="replay in-game events parquet")
    ap.add_argument("--poll-seconds", type=float, default=5)
    ap.add_argument("--max-runtime-minutes", type=float, default=360)
    ap.add_argument("--max-score-age", type=float, default=30)
    ap.add_argument("--no-x", action="store_true")
    ap.add_argument("--no-media", action="store_true")
    ap.add_argument("--no-espn-fallback", action="store_true")
    ap.add_argument("--news-registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--name", default="demo")
    args = ap.parse_args()
    if not re_safe_name(args.name):
        ap.error("--name must be a single directory name using letters, digits, dot, underscore or dash")
    folder = ROOT / "data" / "sample" if args.source == "sample" else FROZEN
    tables = {name: read_table(name, folder) for name in ("games", "player_games")}
    tables["news"] = pd.DataFrame()  # Independent: the pregame news table is never read.
    players = read_table("players", folder)
    matches = tables["games"][tables["games"].game_id.astype(str) == args.game_id]
    if len(matches) != 1:
        ap.error("game id must match exactly one local schedule entry")
    game = next(matches.itertuples(index=False))
    registry = NewsRegistry.load(args.news_registry)
    times = None
    if args.mode == "replay":
        if args.scores:
            scores = pd.read_parquet(args.scores)
            events = pd.read_parquet(args.events) if args.events else None
        elif args.events:
            ap.error("--events requires --scores")
        else:
            from data_sources.inplay_demo import synthetic_game
            scores, events = synthetic_game(tables, players, game)
            print("SYNTHETIC in-game score/news scenarios over historical rosters; not historical play-by-play.")
        provider = TableInPlay(scores, events, registry)
        times = sorted(set(pd.to_datetime(scores.observed_at, utc=True))
                       | (set(pd.to_datetime(events.observed_at, utc=True)) if events is not None else set()))
    else:
        if args.scores or args.events:
            ap.error("live mode cannot consume replay score/event files")
        if pd.Timestamp.now(tz="UTC") - utc(game.tip_time) > pd.Timedelta(hours=8):
            ap.error("live mode requires a current game, not an old historical sample")
        provider = LiveInPlay(registry, args.nba_game_id, not args.no_x, not args.no_media,
                              not args.no_espn_fallback)
    model = pickle.loads(args.model_file.read_bytes()) if args.model_file else InPlayWinModel()
    if not isinstance(model, InPlayWinModel):
        ap.error("--model-file is not an independent InPlayWinModel")
    agent = InPlayAgent(tables, players, game, provider, INPLAY_RUNS / args.name,
                        initial_p_home=args.initial_p_home, win_model=model, max_score_age=args.max_score_age,
                        clock=(lambda: pd.Timestamp.now(tz="UTC")) if args.mode == "live" else None)
    if agent.last_as_of and times is not None:
        times = [t for t in times if t > agent.last_as_of]
    try:
        run_loop(agent, args.poll_seconds, times, args.max_runtime_minutes * 60)
    except KeyboardInterrupt:
        print("In-play stopped; its independent state has been saved.")


def re_safe_name(name):
    import re
    return name not in {".", ".."} and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None


if __name__ == "__main__":
    main()
