"""Dedicated X filtered stream plus recent-search recovery for in-play news.

Uses INPLAY_X_BEARER_TOKEN, a separate session, tagged rules and independent
checkpoints. It only adds its own missing rule; existing rules are not deleted.
No posting, following, or changes to the pregame recent-search client.
"""
import json
import os
import threading
from collections import deque

import pandas as pd
import requests
from dotenv import load_dotenv

from data_sources import ROOT, name_key
from data_sources.inplay_news import text_events
from data_sources.inplay_types import InPlayBatch, event_id, utc

BASE = "https://api.x.com/2/tweets/search/"
FIELDS = {"tweet.fields": "created_at,author_id,attachments,referenced_tweets,note_tweet",
          "expansions": "author_id,attachments.media_keys", "user.fields": "username,name",
          "media.fields": "url,preview_image_url,type"}


def iso(value):
    return utc(value).floor("s").strftime("%Y-%m-%dT%H:%M:%SZ")


class InPlayX:
    def __init__(self, registry, token=None, timeout=3, clock=lambda: pd.Timestamp.now(tz="UTC"),
                 rest_session=None, stream_session=None, queue_limit=5000):
        if token is None:
            load_dotenv(ROOT / "backend" / ".env", override=False)
            load_dotenv(ROOT / ".env", override=False)
        self.token = token if token is not None else os.getenv("INPLAY_X_BEARER_TOKEN", "")
        self.registry, self.timeout, self.clock = registry, timeout, clock
        self.rest = rest_session or requests.Session()
        self.stream = stream_session or requests.Session()
        self.lock, self.stop_event, self.wake = threading.Lock(), threading.Event(), threading.Event()
        self.rest_lock = threading.Lock()
        self.buffer, self.queue_limit = deque(), queue_limit
        self.thread, self.response = None, None
        self.stream_state, self.last_heartbeat, self.gap_until = "unconfigured" if not self.token else "idle", None, None
        self.cursor, self.pending, self.retry_at, self.next_recovery = None, None, None, None
        self.dropped, self.malformed = 0, 0

    @property
    def headers(self):
        return {"Authorization": "Bearer " + self.token}

    def dump_state(self):
        return {"cursor": self.cursor, "pending": self.pending, "retry_at": self.retry_at,
                "gap_until": self.gap_until, "dropped": self.dropped}

    def load_state(self, state):
        for key in ("cursor", "pending", "retry_at", "gap_until", "dropped"):
            if key in state:
                setattr(self, key, state[key])

    def _plan(self, game):
        plan = self.registry.for_game(game)
        if plan["missing_teams"]:
            raise ValueError("selected game is absent from the NBA account catalog")
        accounts = plan["x_accounts"]
        query = "(" + " OR ".join("from:" + a["handle"] for a in accounts) + ") -is:retweet"
        return accounts, query

    def ensure_rule(self, game):
        with self.rest_lock:
            return self._ensure_rule(game)

    def _ensure_rule(self, game):
        _, query = self._plan(game)
        tag = "nba-agent-inplay:" + event_id(game.game_id, query)
        response = self.rest.get(BASE + "stream/rules", headers=self.headers, timeout=self.timeout)
        if response.status_code != 200:
            raise ValueError(f"X rules HTTP {response.status_code}")
        payload = response.json()
        if payload.get("errors"):
            raise ValueError("X rules returned an incomplete response")
        if not any(r.get("tag") == tag and r.get("value") == query for r in payload.get("data", [])):
            response = self.rest.post(BASE + "stream/rules", headers=self.headers,
                                      json={"add": [{"value": query, "tag": tag}]}, timeout=self.timeout)
            if response.status_code not in {200, 201} or response.json().get("errors"):
                raise ValueError("X stream rule could not be added")
        return tag

    def accept(self, payload, observed=None):
        """Capture first observation in the reader thread, not at the next poll."""
        with self.lock:
            if len(self.buffer) >= self.queue_limit:
                self.buffer.popleft()
                self.dropped += 1
            self.buffer.append((payload, utc(observed or self.clock()).isoformat()))
        self.wake.set()

    def start(self, game):
        if not self.token or self.stop_event.is_set() or (self.thread and self.thread.is_alive()):
            return
        self.thread = threading.Thread(target=self._run, args=(game,), daemon=True, name="nba-inplay-x")
        self.thread.start()

    def _run(self, game):
        delay = 1
        while not self.stop_event.is_set():
            try:
                self.stream_state = "connecting"
                self.ensure_rule(game)
                if self.stop_event.is_set():
                    return
                response = self.stream.get(BASE + "stream", params=FIELDS, headers=self.headers,
                                           stream=True, timeout=(self.timeout, 25))
                self.response = response
                if response.status_code != 200:
                    self.stream_state = f"http_{response.status_code}"
                    raise ValueError("stream refused")
                self.gap_until = utc(self.clock()).isoformat()
                self.stream_state, delay = "connected", 1
                for line in response.iter_lines(chunk_size=1):
                    if self.stop_event.is_set():
                        return
                    self.last_heartbeat = utc(self.clock()).isoformat()
                    if line:
                        try:
                            payload = json.loads(line)
                            if payload.get("errors") or "data" not in payload:
                                self.malformed += 1
                            else:
                                self.accept(payload)
                        except (ValueError, TypeError):
                            self.malformed += 1
                self.stream_state = "disconnected"
            except (requests.RequestException, ValueError, TypeError, KeyError):
                if self.stream_state in {"connected", "connecting"}:
                    self.stream_state = "disconnected"
            finally:
                if self.response is not None:
                    self.response.close()
                    self.response = None
            self.wake.set()
            # Event wait is interruptible on shutdown; never busy-poll an outage.
            self.stop_event.wait(delay)
            delay = min(delay * 2, 60)
        self.stream_state = "stopped"

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.response is not None:
            self.response.close()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1)

    def recover(self, game, now):
        """Overlap recovery resumes capped pagination instead of skipping a gap."""
        errors = []
        if not self.token or (self.retry_at and now < utc(self.retry_at)):
            return errors
        if self.next_recovery and now < utc(self.next_recovery) and not self.pending:
            return errors
        _, query = self._plan(game)
        if not self.pending:
            since = max(utc(game.tip_time), now - pd.Timedelta(days=7) + pd.Timedelta(minutes=1))
            if self.cursor:
                since = max(since, utc(self.cursor) - pd.Timedelta(minutes=2))
            end = now - pd.Timedelta(seconds=30)
            if end <= since:
                return errors
            self.pending = {"query": query, "start_time": iso(since), "end_time": iso(end)}
        try:
            for _ in range(2):
                with self.rest_lock:
                    response = self.rest.get(BASE + "recent", params={**self.pending, **FIELDS, "max_results": 100},
                                             headers=self.headers, timeout=self.timeout)
                if response.status_code == 429:
                    self.retry_at = (now + pd.Timedelta(seconds=60)).isoformat()
                    raise ValueError("X recovery rate limited")
                if response.status_code != 200:
                    raise ValueError(f"X recovery HTTP {response.status_code}")
                payload = response.json()
                if payload.get("errors"):
                    raise ValueError("X recovery incomplete response")
                includes = payload.get("includes", {})
                observed = utc(self.clock())
                accounts, _ = self._plan(game)
                allowed = {a["handle"].lower() for a in accounts}
                authors = {str(u["id"]): u["username"].lower() for u in includes.get("users", [])}
                if any(authors.get(str(p.get("author_id"))) not in allowed for p in payload.get("data", [])):
                    raise ValueError("X recovery contains an unverified author")
                for post in payload.get("data", []):
                    self.accept({"data": post, "includes": includes}, observed)
                next_token = payload.get("meta", {}).get("next_token")
                if not next_token:
                    self.cursor = self.pending["end_time"]
                    self.pending = None
                    self.next_recovery = (now + pd.Timedelta(seconds=30)).isoformat()
                    if self.gap_until and utc(self.cursor) >= utc(self.gap_until):
                        self.gap_until = None
                    return errors
                self.pending["next_token"] = next_token
            errors.append({"source": "inplay_x_recovery", "error": "backfill pages remain; next poll resumes"})
        except (requests.RequestException, ValueError, TypeError, KeyError):
            errors.append({"source": "inplay_x_recovery", "error": "backfill failed; pending window retained"})
            self.next_recovery = (now + pd.Timedelta(seconds=30)).isoformat()
            if not self.retry_at or utc(self.retry_at) <= now:
                self.retry_at = self.next_recovery
        return errors

    def fetch(self, game, players, roster, now):
        batch = InPlayBatch()
        accounts, _ = self._plan(game)
        if not self.token:
            batch.errors.append({"source": "inplay_x", "error": "INPLAY_X_BEARER_TOKEN is not configured"})
        else:
            self.start(game)
            batch.errors.extend(self.recover(game, now))
            # Recovery requests take time. Do not drop newly observed messages
            # merely because their receipt followed the start of this fetch.
            now = max(now, utc(self.clock()))
        state = self.stream_state
        if state == "connected" and self.last_heartbeat and (now - utc(self.last_heartbeat)).total_seconds() > 30:
            state = "stale_connection"
        if self.token and state != "connected":
            batch.errors.append({"source": "inplay_x", "error": "stream is " + state})
        if self.gap_until or self.dropped or self.malformed:
            batch.errors.append({"source": "inplay_x", "error": "news gap or queue loss; completeness is unverified"})
        batch.coverage = [{"source": a["source"], "state": state, "transport": "filtered_stream",
                           "last_heartbeat": self.last_heartbeat, "gap_until": self.gap_until,
                           "backfill_cursor": self.cursor, "dropped": self.dropped} for a in accounts]
        with self.lock:
            queued = list(self.buffer)
            self.buffer.clear()
            self.wake.clear()
        allowed = {a["handle"].lower(): a for a in accounts}
        names = [name_key(n) for p, n in zip(players.player_id, players.player_name) if int(p) in roster]
        for payload, observed in queued:
            try:
                post = payload["data"]
                author = next((u["username"].lower() for u in payload.get("includes", {}).get("users", [])
                               if str(u["id"]) == str(post["author_id"])), None)
                if author not in allowed:
                    raise ValueError("X post author is not in this game's allowlist")
                published = utc(post["created_at"])
                if not utc(game.tip_time) <= published <= now or utc(observed) > now:
                    continue
                if any(r.get("type") in {"retweeted", "reposted"} for r in post.get("referenced_tweets", post.get("referenced_posts", []))):
                    continue
                text = post.get("note_tweet", {}).get("text") or post.get("note_post", {}).get("text") or post["text"]
                account = allowed[author]
                if account["team"] is None and not any(n and n in name_key(text) for n in names):
                    continue
                item_id = "inplay-x-" + str(post["id"])
                url = f"https://x.com/{account['handle']}/status/{post['id']}"
                rows = text_events(text, game, players, roster, account["source"], published, observed, url, item_id)
                batch.events.extend(rows)
                batch.evidence.append({"item_id": item_id, "game_id": str(game.game_id), "source": account["source"],
                                       "url": url, "text": text, "published_at": published.isoformat(),
                                       "observed_at": observed, "attachments": post.get("attachments", {}),
                                       "media": payload.get("includes", {}).get("media", []),
                                       "review_required": bool(post.get("attachments")) or not rows or any(r["status"] == "review" for r in rows)})
            except (ValueError, KeyError, TypeError):
                batch.errors.append({"source": "inplay_x", "error": "post rejected: invalid author or timestamps"})
        return batch
