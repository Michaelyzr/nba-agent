"""Uncached NBA liveData with an explicitly labelled ESPN fallback.

NBA and ESPN player IDs differ. Map by unique full names in the local roster,
never by treating a provider's personId as a local historical player ID.
"""
import re
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

from data_sources import name_key
from data_sources.inplay_types import InPlayBatch, clock_seconds, event_id, utc
from data_sources.inplay_news import classify
from data_sources.news_registry import TEAM_ALIASES

NBA_BASE = "https://cdn.nba.com/static/json/liveData/"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def team_code(value):
    # ESPN uses UTAH; keep this adapter's aliases out of the pregame registry.
    return "UTA" if value == "UTAH" else TEAM_ALIASES.get(value, value)


def mapped_players(players, roster):
    names = {}
    for pid, name in zip(players.player_id, players.player_name):
        # Current box-score membership is stronger than last season's roster:
        # traded players can be present locally but absent from that old roster.
        names.setdefault(name_key(name), []).append(int(pid))
    return {k: v[0] for k, v in names.items() if len(v) == 1}


def source_time(response, payload):
    modified = response.headers.get("Last-Modified")
    if modified:
        try:
            return utc(parsedate_to_datetime(modified)).isoformat(), "last_modified"
        except (ValueError, TypeError):
            pass
    raw = payload.get("meta", {}).get("lastUpdatedAt") or payload.get("meta", {}).get("time")
    if raw:
        try:
            stamp = pd.Timestamp(raw)
            # Do not guess a timezone for an undocumented naive server timestamp.
            if stamp.tzinfo is not None:
                return utc(stamp).isoformat(), "source_meta"
        except (ValueError, TypeError):
            pass
    return None, "observation_only"


def event_row(game, pid, status, source, published, now, text, key, url, **extra):
    return {"event_id": event_id(source, game.game_id, key, pid, status, text),
            "item_id": event_id(source, game.game_id, key, text), "game_id": str(game.game_id),
            "player_id": pid, "status": status, "source": source,
            "published_at": utc(published).isoformat(), "observed_at": utc(now).isoformat(),
            "text": text, "url": url, **extra}


def nba_batch(box, pbp, game, players, roster, now, updated=None, timestamp_basis="observation_only"):
    data = box["game"]
    home, away = data["homeTeam"], data["awayTeam"]
    for side, expected in ((home, game.home_team), (away, game.away_team)):
        if team_code(side["teamTricode"]) != team_code(expected):
            raise ValueError("NBA live game does not match selected teams")
    if utc(data["gameTimeUTC"]).tz_convert("America/New_York").date() != pd.Timestamp(game.date).date():
        raise ValueError("NBA live game does not match selected date")
    nba_id = str(data["gameId"])
    if str(pbp["game"]["gameId"]) != nba_id:
        raise ValueError("score and play-by-play belong to different games")
    phase = {1: "scheduled", 2: "live", 3: "final"}.get(data["gameStatus"], "suspended")
    score = {"game_id": str(game.game_id), "provider_game_id": nba_id, "source": "nba_live",
             "home_score": home["score"], "away_score": away["score"], "period": data["period"],
             "clock_seconds": clock_seconds(data["gameClock"] or "PT12M00S"), "phase": phase,
             "observed_at": now.isoformat(), "updated_at": updated, "timestamp_basis": timestamp_basis,
             "player_minutes": {}, "player_teams": {}, "roster": []}
    batch = InPlayBatch(score=score)
    lookup = mapped_players(players, roster)
    provider_map = {}
    url = NBA_BASE + f"boxscore/boxscore_{nba_id}.json"
    for team, local_team in ((home, game.home_team_id), (away, game.away_team_id)):
        for player in team.get("players", []):
            pid = lookup.get(name_key(player.get("name", "")))
            if pid is None:
                continue
            provider_map[int(player["personId"])] = pid
            score["roster"].append(pid)
            score["player_teams"][str(pid)] = int(local_team)
            minutes = clock_seconds(player.get("statistics", {}).get("minutes", "PT00M00S")) / 60
            score["player_minutes"][str(pid)] = minutes
            if phase == "scheduled" or minutes <= 0:
                continue
            stats = player.get("statistics", {})
            reason = str(player.get("notPlayingReason", "")) + " " + str(player.get("notPlayingDescription", ""))
            status = ("fouled_out" if stats.get("foulsPersonal", 0) >= 6 else
                      "ejected" if re.search(r"\b(?:eject\w*|disqualif\w*)\b", reason, re.I) else
                      "injury_out" if player.get("status") == "INACTIVE" and re.search(r"\binjur\w*\b", reason, re.I) else None)
            if status:
                batch.events.append(event_row(game, pid, status, "nba_live", updated or now, now,
                                               player["name"] + ": " + (reason.strip() or "six personal fouls"),
                                               f"box-{pid}-{status}", url))
    for action in pbp["game"].get("actions", []):
        pid = provider_map.get(int(action.get("personId") or 0))
        if pid is None or not action.get("timeActual"):
            continue
        published = utc(action["timeActual"])
        if not utc(game.tip_time) <= published <= now:
            continue
        text = str(action.get("description", ""))
        classified = classify(text)
        status = "rescinded" if classified == "rescinded" else "ejected" if action.get("actionType") == "ejection" else classified
        if status == "review":
            continue
        # Edits of an action carry a new evidence ID; the same action is replaced.
        batch.events.append(event_row(game, pid, status, "nba_live", published, now, text,
                                       action["actionNumber"], NBA_BASE + f"playbyplay/playbyplay_{nba_id}.json",
                                       action_key=f"nba-{action['actionNumber']}", revision_at=action.get("edited")))
    return batch


def espn_batch(payload, game, players, roster, now, updated=None, timestamp_basis=None):
    header = payload["header"]
    if str(header["id"]) != str(game.game_id):
        raise ValueError("ESPN event id differs from local game id")
    competition = header["competitions"][0]
    sides = {c["homeAway"]: c for c in competition["competitors"]}
    for key, expected in (("home", game.home_team), ("away", game.away_team)):
        abbreviation = team_code(sides[key]["team"]["abbreviation"])
        if abbreviation != team_code(expected):
            raise ValueError("ESPN event does not match selected teams")
    if utc(competition["date"]).tz_convert("America/New_York").date() != pd.Timestamp(game.date).date():
        raise ValueError("ESPN event does not match selected date")
    status = competition["status"]
    phase = {"pre": "scheduled", "in": "live", "post": "final"}.get(status["type"]["state"], "suspended")
    if status["type"].get("name") in {"STATUS_SUSPENDED", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_DELAYED"}:
        phase = "suspended"
    if phase == "live" and ("clock" not in status or "period" not in status):
        raise ValueError("ESPN live state has no verified period/clock")
    score = {"game_id": str(game.game_id), "provider_game_id": str(game.game_id), "source": "espn_live",
             "phase": phase, "home_score": sides["home"].get("score", 0), "away_score": sides["away"].get("score", 0),
             "period": status.get("period", 4 if phase == "final" else 0),
             "clock_seconds": 0.0 if phase == "final" else float(status.get("clock", 0)),
             "observed_at": now.isoformat(), "updated_at": updated,
             "timestamp_basis": timestamp_basis or ("source_update" if updated else "observation_only"),
             "player_minutes": {}, "player_teams": {}, "roster": [],
             "possession_sign": 1 if sides["home"].get("possession") else -1 if sides["away"].get("possession") else 0}
    batch = InPlayBatch(score=score)
    lookup = mapped_players(players, roster)
    provider_map = {}
    url = f"https://www.espn.com/nba/game/_/gameId/{game.game_id}"
    for team in payload.get("boxscore", {}).get("players", []):
        code = team_code(team["team"]["abbreviation"])
        if code not in {team_code(game.home_team), team_code(game.away_team)}:
            continue
        local_team = game.home_team_id if code == team_code(game.home_team) else game.away_team_id
        for block in team.get("statistics", []):
            for entry in block.get("athletes", []):
                athlete = entry.get("athlete", {})
                pid = lookup.get(name_key(athlete.get("displayName", "")))
                if pid is None:
                    continue
                provider_map[str(athlete["id"])] = pid
                score["roster"].append(pid)
                score["player_teams"][str(pid)] = int(local_team)
                stats = dict(zip(block.get("keys", []), entry.get("stats", [])))
                raw_minutes = stats.get("minutes", "0")
                try:
                    minutes = clock_seconds(raw_minutes) / 60 if ":" in str(raw_minutes) else float(raw_minutes)
                except (ValueError, TypeError):
                    minutes = 0.0
                score["player_minutes"][str(pid)] = minutes
                try:
                    fouls = float(stats.get("fouls", stats.get("personalFouls", "0")) or 0)
                except (ValueError, TypeError):
                    fouls = 0
                if fouls >= 6 and minutes > 0:
                    batch.events.append(event_row(game, pid, "fouled_out", "espn_live", updated or now, now,
                                                   athlete["displayName"] + " fouled out (six personal fouls)",
                                                   f"box-{pid}-fouled-out", url))
    for play in payload.get("plays", []):
        involved = {provider_map.get(str(p.get("athlete", {}).get("id"))) for p in play.get("participants", [])}
        involved.discard(None)
        text = play.get("text", "")
        status = classify(text)
        if status == "review" or len(involved) != 1 or not play.get("wallclock"):
            continue
        published = utc(play["wallclock"])
        if utc(game.tip_time) <= published <= now:
            pid = next(iter(involved))
            batch.events.append(event_row(game, pid, status, "espn_live", published, now, text, play["id"], url,
                                           action_key=f"espn-{play['id']}"))
    return batch


class LiveScore:
    def __init__(self, nba_game_id=None, espn_fallback=True, timeout=4, session=None,
                 clock=lambda: pd.Timestamp.now(tz="UTC")):
        if nba_game_id is not None and not re.fullmatch(r"\d{10}", nba_game_id):
            raise ValueError("NBA game id must contain 10 digits")
        self.nba_id, self.fallback, self.timeout = nba_game_id, espn_fallback, timeout
        self.session, self.clock = session or requests.Session(), clock
        self.retry_nba_at = None

    def dump_state(self):
        return {"nba_id": self.nba_id, "retry_nba_at": self.retry_nba_at}

    def load_state(self, state):
        if self.nba_id and state.get("nba_id") and self.nba_id != state["nba_id"]:
            raise ValueError("NBA game id changed during resume")
        self.nba_id = self.nba_id or state.get("nba_id")
        self.retry_nba_at = state.get("retry_nba_at")

    def _get(self, url, **kwargs):
        response = self.session.get(url, timeout=self.timeout, headers={"Cache-Control": "no-cache"}, **kwargs)
        response.raise_for_status()
        return response, response.json()

    def fetch(self, game, players, roster, now):
        errors, coverage = [], []
        cooling_down = self.retry_nba_at and now < utc(self.retry_nba_at)
        try:
            if cooling_down:
                raise ValueError("NBA endpoint is in retry cooldown")
            if not self.nba_id:
                _, board = self._get(NBA_BASE + "scoreboard/todaysScoreboard_00.json")
                matches = [g for g in board["scoreboard"]["games"]
                           if g["homeTeam"]["teamTricode"] == TEAM_ALIASES.get(game.home_team, game.home_team)
                           and g["awayTeam"]["teamTricode"] == TEAM_ALIASES.get(game.away_team, game.away_team)
                           and utc(g["gameTimeUTC"]).tz_convert("America/New_York").date() == pd.Timestamp(game.date).date()]
                if len(matches) != 1:
                    raise ValueError("no unique NBA live match; supply --nba-game-id")
                self.nba_id = str(matches[0]["gameId"])
            response, box = self._get(NBA_BASE + f"boxscore/boxscore_{self.nba_id}.json")
            score_observed = utc(self.clock())
            _, pbp = self._get(NBA_BASE + f"playbyplay/playbyplay_{self.nba_id}.json")
            updated, basis = source_time(response, box)
            batch = nba_batch(box, pbp, game, players, roster, utc(self.clock()), updated, basis)
            batch.score["observed_at"] = score_observed.isoformat()
            self.retry_nba_at = None
            batch.coverage.append({"source": "nba_live", "state": "ok"})
            return batch
        except (requests.RequestException, ValueError, KeyError, TypeError):
            errors.append({"source": "nba_live", "error": "NBA live request or game validation failed"})
            coverage.append({"source": "nba_live", "state": "failed"})
            if not cooling_down:
                self.retry_nba_at = (now + pd.Timedelta(seconds=60)).isoformat()
        batch = InPlayBatch(errors=errors, coverage=coverage)
        if self.fallback:
            try:
                response, payload = self._get(ESPN_BASE + "/summary", params={"event": str(game.game_id)})
                competition = payload["header"]["competitions"][0]
                if competition["status"]["type"]["state"] == "in":
                    _, board = self._get(ESPN_BASE + "/scoreboard", params={"dates": pd.Timestamp(game.date).strftime("%Y%m%d"), "limit": 100})
                    event = next(e for e in board["events"] if str(e["id"]) == str(game.game_id))
                    live_competition = event["competitions"][0]
                    competition["status"] = live_competition["status"]
                    competition["competitors"] = live_competition["competitors"]
                updated, basis = source_time(response, payload)
                alternate = espn_batch(payload, game, players, roster, utc(self.clock()), updated, basis)
                batch.score, batch.events = alternate.score, alternate.events
                batch.coverage.append({"source": "espn_live", "state": "ok", "fallback": True})
            except (requests.RequestException, ValueError, KeyError, TypeError, StopIteration):
                batch.errors.append({"source": "espn_live", "error": "ESPN live request or game validation failed"})
                batch.coverage.append({"source": "espn_live", "state": "failed"})
        return batch
