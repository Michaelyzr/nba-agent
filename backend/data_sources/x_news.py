"""Read only selected X authors via the official recent-search API.

Docs: https://docs.x.com/x-api/posts/search/quickstart/recent-search
No login scraping, fake RSS mirrors, posting or account-follow mutations.
"""
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from data_sources import ET as EASTERN, ROOT
from data_sources.live_news import NewsBatch, post_rows, relevant_text

SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
INDEX_LAG = pd.Timedelta(seconds=30)
OVERLAP = pd.Timedelta(minutes=5)


def _iso(t):
    return pd.Timestamp(t).tz_convert("UTC").floor("s").strftime("%Y-%m-%dT%H:%M:%SZ")


class XNews:
    def __init__(self, registry, bearer_token=None, timeout=10, max_pages=5,
                 clock=lambda: pd.Timestamp.now(tz="UTC"), session=None):
        self.registry, self.timeout, self.max_pages, self.clock = registry, timeout, max_pages, clock
        if max_pages < 1:
            raise ValueError("X max_pages must be positive")
        if bearer_token is None:
            load_dotenv(ROOT / "backend" / ".env", override=False)
            load_dotenv(ROOT / ".env", override=False)
        self.token = bearer_token if bearer_token is not None else os.getenv("X_BEARER_TOKEN", "")
        self.session = session or requests.Session()
        self.cursor, self.pending, self.retry_at = {}, {}, None

    def dump_state(self):
        # Tokens and Authorization headers are deliberately never serialized.
        return {"cursor": self.cursor, "pending": self.pending, "retry_at": self.retry_at}

    def load_state(self, saved):
        self.cursor = dict(saved.get("cursor", {}))
        self.pending = dict(saved.get("pending", {}))
        self.retry_at = saved.get("retry_at")

    def fetch(self, game, players, roster, since, now):
        batch = NewsBatch()
        accounts = self.registry.for_game(game)["x_accounts"]
        allowed = {a["handle"].lower(): a for a in accounts}
        def coverage(state, **extra):
            batch.coverage = [{"source": a["source"], "url": a["url"], "role": a["role"],
                               "team": a["team"], "state": state, "checked_at": now.isoformat(), **extra}
                              for a in accounts]
        if not self.token:
            batch.errors.append({"source": "x_api", "error": "X_BEARER_TOKEN is not configured; X sources were not queried"})
            coverage("unconfigured")
            return batch
        if self.retry_at and now < pd.Timestamp(self.retry_at):
            batch.errors.append({"source": "x_api", "error": "rate-limit cooldown", "retry_at": self.retry_at})
            coverage("rate_limited", retry_at=self.retry_at)
            return batch
        query = "(" + " OR ".join(f"from:{a['handle']}" for a in accounts) + ") -is:retweet"
        key = str(game.game_id)
        window = self.pending.get(key)
        if window and window["query"] != query:
            window = None  # Rebuild the overlap window if the registry changed.
        if not window:
            start = max(since, now - pd.Timedelta(days=7) + pd.Timedelta(minutes=1))
            if key in self.cursor:
                start = max(start, pd.Timestamp(self.cursor[key]) - OVERLAP)
            end = now - INDEX_LAG
            if end <= start:
                coverage("indexing_delay", indexing_lag_seconds=30)
                return batch
            window = {"query": query, "start_time": _iso(start), "end_time": _iso(end)}
        coverage("ok", window_end=window["end_time"], indexing_lag_seconds=30)
        names = [n for p, n in zip(players.player_id, players.player_name) if int(p) in roster]
        counts = {a["source"]: 0 for a in accounts}
        for _ in range(self.max_pages):
            if self.clock() >= game.tip_time:
                coverage("stopped_at_tip")
                return batch
            params = {**window, "max_results": 100, "sort_order": "recency",
                      "tweet.fields": "created_at,author_id,attachments,referenced_tweets,note_tweet",
                      "expansions": "author_id,attachments.media_keys", "user.fields": "username,name",
                      "media.fields": "url,preview_image_url,type"}
            try:
                response = self.session.get(SEARCH_URL, params=params,
                                            headers={"Authorization": f"Bearer {self.token}"}, timeout=self.timeout)
                if response.status_code == 429:
                    try:
                        reset = pd.Timestamp(float(response.headers["x-rate-limit-reset"]), unit="s", tz="UTC")
                    except (KeyError, ValueError):
                        reset = now + pd.Timedelta(minutes=1)
                    self.retry_at = max(reset, now + pd.Timedelta(seconds=1)).isoformat()
                    batch.errors.append({"source": "x_api", "error": "HTTP 429 rate limit", "retry_at": self.retry_at})
                    self.pending[key] = window
                    coverage("rate_limited", retry_at=self.retry_at)
                    return batch
                if response.status_code != 200:
                    # Never include a response body or HTTP exception containing credentials.
                    raise ValueError(f"X API HTTP {response.status_code}; check access permissions")
                payload = response.json()
                if payload.get("errors"):
                    raise ValueError("X returned a partial/error response; window will be retried")
                authors = {str(u["id"]): u["username"].lower() for u in payload.get("includes", {}).get("users", [])}
                media = {m["media_key"]: m for m in payload.get("includes", {}).get("media", [])}
                if any(authors.get(str(p.get("author_id"))) not in allowed for p in payload.get("data", [])):
                    raise ValueError("post author absent from selected allowlist/expansion; window will be retried")
                for post in payload.get("data", []):
                    handle = authors.get(str(post.get("author_id")))
                    published = pd.Timestamp(post["created_at"])
                    if pd.isna(published) or published.tzinfo is None or not since <= published <= now:
                        continue
                    published = published.tz_convert("UTC")
                    if published >= game.tip_time:
                        continue
                    account = allowed[handle]
                    references = post.get("referenced_tweets", post.get("referenced_posts", []))
                    if any(r.get("type") in {"retweeted", "reposted"} for r in references):
                        continue
                    text = (post.get("note_tweet", {}).get("text") or post.get("note_post", {}).get("text")
                            or post["text"])
                    # Team authors' game-day posts also survive without a player name
                    # (e.g. image-only lineups); Shams must mention a team/player.
                    team_day = account["team"] is not None and published.tz_convert(EASTERN).date() == pd.Timestamp(game.date).date()
                    if not team_day and not relevant_text(text, game, names, self.registry):
                        continue
                    counts[account["source"]] += 1
                    post_id = f"x-{post['id']}"
                    url = f"https://x.com/{account['handle']}/status/{post['id']}"
                    attachments = [media[k] for k in post.get("attachments", {}).get("media_keys", []) if k in media]
                    rows = post_rows(text, game, players, roster, account["source"], published, now, url, post_id)
                    batch.rows.extend(rows)
                    batch.items.append({"item_id": post_id, "game_id": game.game_id, "published_at": published,
                                        "observed_at": now, "source": account["source"], "url": url,
                                        "author": account["handle"], "text": text, "attachments": attachments,
                                        "review_required": bool(attachments) or not rows or any(r["status"] == "review" for r in rows)})
                token = payload.get("meta", {}).get("next_token")
                if not token:
                    self.cursor[key] = window["end_time"]
                    self.pending.pop(key, None)
                    for c in batch.coverage:
                        c["matching_items"] = counts[c["source"]]
                    return batch
                window = {**window, "next_token": token}
                self.pending[key] = window
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                self.pending[key] = window
                error = "network request failed" if isinstance(exc, requests.RequestException) else str(exc)
                batch.errors.append({"source": "x_api", "error": error})
                coverage("failed")
                return batch
        batch.errors.append({"source": "x_api", "error": "pagination budget reached; remaining pages resume next poll"})
        coverage("backlog", window_end=window["end_time"])
        return batch
