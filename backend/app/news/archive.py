"""Project-local append-only evidence and model archive; no PostgreSQL migration.

SQLite transactions also coordinate collectors in multiple API workers. Source
versions and review times are retained so approval cannot leak into past runs.
"""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BACKEND_DIR, settings


def utcnow():
    return datetime.now(timezone.utc)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class NewsArchive:
    def __init__(self, path=None):
        self.path = Path(path) if path else BACKEND_DIR / "data" / "news" / "archive.sqlite3"

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, url TEXT NOT NULL, provider TEXT NOT NULL,
                    published_at TEXT NOT NULL, observed_at TEXT NOT NULL,
                    title TEXT NOT NULL, body TEXT NOT NULL, extracted INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, data TEXT NOT NULL,
                    review TEXT NOT NULL, reviewed_at TEXT, observed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY, game_id TEXT NOT NULL, as_of TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshots_game ON snapshots(game_id, as_of);
            """)
            yield conn

    def get(self, key, default=None):
        with self.connection() as c:
            row = c.execute("SELECT data FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.connection() as c:
            c.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def preferences(self):
        return self.get("preferences", {"auto_collect": settings.news_auto_collect,
                                         "auto_extract": settings.news_auto_extract})

    def lease(self, owner):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT data FROM state WHERE key='lease'").fetchone()
            current = json.loads(row[0]) if row else {}
            if current.get("expires", 0) > time.time():
                return False
            c.execute("INSERT OR REPLACE INTO state VALUES ('lease',?)",
                      (json.dumps({"owner": owner, "expires": time.time() + 300}),))
        return True

    def release(self, owner):
        with self.connection() as c:
            row = c.execute("SELECT data FROM state WHERE key='lease'").fetchone()
            if row and json.loads(row[0]).get("owner") == owner:
                c.execute("DELETE FROM state WHERE key='lease'")

    def document(self, *, url, provider, published_at, title, body, observed_at=None):
        # A changed body creates a new version even when the URL is unchanged.
        identity = fingerprint([provider, url, published_at, body])
        with self.connection() as c:
            previous = c.execute("SELECT id FROM documents WHERE url=? AND provider=? "
                "AND published_at=? AND body=? LIMIT 1", (url, provider, published_at, body)).fetchone()
            if previous:
                return previous[0], False
            cursor = c.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,0)",
                               (identity, url, provider, published_at,
                                observed_at or utcnow().isoformat(), title[:500], body))
        return identity, cursor.rowcount == 1

    def unextracted(self, limit):
        with self.connection() as c:
            return [dict(r) for r in c.execute("SELECT * FROM documents WHERE provider='espn_rss' "
                    "AND extracted=0 GROUP BY url,provider,published_at,body ORDER BY published_at DESC LIMIT ?", (limit,))]

    def mark_extracted(self, identity):
        with self.connection() as c:
            c.execute("UPDATE documents SET extracted=1 WHERE (url,provider,published_at,body)="
                      "(SELECT url,provider,published_at,body FROM documents WHERE id=?)", (identity,))

    def add_event(self, document_id, data, review="pending"):
        # Stable per fact/provider: an unrelated player changing in a bulk report
        # must not duplicate every unchanged injury or reset its evidence time.
        matching_fields = {"team_id", "player_id", "game_id", "match_error"}
        fact = {k: v for k, v in data.items() if k not in matching_fields}
        identity = fingerprint(fact)
        now = utcnow().isoformat()
        with self.connection() as c:
            # Preserve existing IDs and prevent duplication after roster rematch,
            # including archives produced before the stable fact-key policy.
            candidates = c.execute("SELECT id,data FROM events WHERE json_extract(data,'$.provider')=? "
                "AND json_extract(data,'$.published_at')=?", (data.get("provider"), data.get("published_at"))).fetchall()
            for row in candidates:
                previous = {k: v for k, v in json.loads(row["data"]).items() if k not in matching_fields}
                if fingerprint(previous) == identity:
                    return row["id"], False
            cur = c.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)",
                            (identity, document_id, json.dumps(data, ensure_ascii=False), review,
                             now if review == "approved" else None, now))
        return identity, cur.rowcount == 1

    def events(self, limit=None):
        with self.connection() as c:
            sql = "SELECT e.*, d.url,d.provider,d.title,d.published_at FROM events e " \
                  "JOIN documents d ON d.id=e.document_id ORDER BY d.published_at DESC, e.id"
            rows = c.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def review(self, identity, decision):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM events WHERE id=?", (identity,)).fetchone()
            if not row:
                raise ValueError("事件不存在")
            data = json.loads(row["data"])
            if decision == "approved" and (not data.get("team_id") or
                    (data.get("player_name") and not data.get("player_id"))):
                raise ValueError("球员或球队尚未准确匹配，请先同步球员和阵容，再重新采集")
            # First review only: later corrections must be a new source version.
            if row["review"] != "pending":
                raise ValueError("事件已审核；状态更正须由新事件记录")
            c.execute("UPDATE events SET review=?,reviewed_at=? WHERE id=?",
                      (decision, utcnow().isoformat(), identity))

    def rematch(self, mapper):
        with self.connection() as c:
            rows = c.execute("SELECT * FROM events WHERE review='pending'").fetchall()
            for row in rows:
                data = json.loads(row["data"])
                mapped = mapper(data)
                c.execute("UPDATE events SET data=? WHERE id=?",
                          (json.dumps(mapped, ensure_ascii=False), row["id"]))

    def snapshot(self, game_id, as_of, data):
        with self.connection() as c:
            c.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?)",
                      (fingerprint([game_id, as_of]), game_id, as_of, json.dumps(data)))

    def snapshots(self):
        with self.connection() as c:
            return [{**dict(r), "data": json.loads(r["data"])}
                    for r in c.execute("SELECT * FROM snapshots ORDER BY as_of")]

    def status(self):
        with self.connection() as c:
            counts = {name: c.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                      for name in ("documents", "events", "snapshots")}
            counts["pending"] = c.execute("SELECT COUNT(*) FROM events WHERE review='pending'").fetchone()[0]
        return {**counts, "preferences": self.preferences(), "last_collection": self.get("last_collection"),
                "model": self.get("injury_model"), "last_training": self.get("last_training"),
                "poll_seconds": settings.news_poll_seconds, "extract_limit": settings.news_extract_limit,
                "training_min_games": settings.injury_training_min_games,
                "injury_status_max_hours": settings.injury_status_max_hours,
                "openai_configured": bool(settings.openai_api_key)}


archive = NewsArchive()
