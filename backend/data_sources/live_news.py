"""Timestamped pregame news adapters. Live sources never backfill a replay.

ESPN RSS is documented at https://www.espn.com/espn/news/story?page=rssinfo.
Official reports reuse data_sources.news's PDF parser. HTTP errors are logged
and retried at the next poll; an outage never means a player is available.
"""
import hashlib
import html
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from data_sources import ET as EASTERN, name_key, player_lookup
from data_sources.news import BASE, parse_pdf, report_names
from data_sources.news_registry import NewsRegistry, TEAM_ALIASES

ESPN_RSS = "https://www.espn.com/espn/rss/nba/news"
NON_GAME_CONTEXT = r"\b(tomorrow|yesterday|last night|last game|next game|next week|next season|rest of the game|for trade|out of)\b"


@dataclass
class NewsBatch:
    rows: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    items: list = field(default_factory=list)
    coverage: list = field(default_factory=list)


def event_id(source, game_id, player_id, published_at, text):
    value = f"{source}|{game_id}|{player_id}|{published_at}|{text}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


class TableNews:
    """Offline replay adapter: publication AND first observation must be as-of."""
    def __init__(self, frame):
        self.frame = frame

    def fetch(self, game, players, roster, since, now):
        n = self.frame
        mask = ((n.game_id == game.game_id) & (n.published_at >= since) & (n.published_at <= now))
        if "observed_at" in n:
            mask &= n.observed_at <= now
        return NewsBatch(n[mask].to_dict("records"))


def rss_rows(content, game, players, roster, since, now, source="espn_rss", items=None, domains=None, registry=None):
    """Keep matched player sentences; ambiguous prose is retained for review.

    Full names and the as-of roster are required. No team-wide point adjustment
    is guessed from a headline. Multi-player sentences are review-only.
    """
    root = ET.fromstring(content)
    if root.tag.rsplit("}", 1)[-1].lower() != "rss":
        raise ValueError("news endpoint returned non-RSS content")
    names = {int(p): n for p, n in zip(players.player_id, players.player_name) if int(p) in roster}
    rows = []
    for item in root.findall(".//item"):
        try:
            published = pd.Timestamp(item.findtext("pubDate"))
            if pd.isna(published) or published.tzinfo is None:
                continue
            published = published.tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        url = item.findtext("link", "")
        if not since <= published <= now or not url.startswith("https://"):
            continue
        host = (urlparse(url).hostname or "").lower()
        if domains and not any(host == d or host.endswith("." + d) for d in domains):
            continue
        headline = html.unescape(item.findtext("title", ""))
        summary = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description", "")))
        text = headline + ". " + summary
        if items is not None and relevant_text(text, game, names.values(), registry):
            items.append({"item_id": event_id(source, game.game_id, "article", published, url + text),
                          "game_id": game.game_id, "published_at": published, "observed_at": now,
                          "source": source, "url": url, "title": headline, "text": text,
                          "review_required": True})
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            matches = [p for p, n in names.items() if name_key(n) in name_key(sentence)]
            for pid in matches:
                status, minutes = classify_sentence(sentence, names[pid]) if len(matches) == 1 else ("review", None)
                # A current feed can contain tomorrow's lineup or an old recap.
                same_day = published.tz_convert(EASTERN).date() == pd.Timestamp(game.date).date()
                if not same_day or not re.search(r"\b(today|tonight)\b", sentence, re.I):
                    status, minutes = "review", None
                rows.append({"news_id": event_id(source, game.game_id, pid, published, sentence),
                             "game_id": game.game_id, "player_id": pid, "published_at": published,
                             "observed_at": now, "status": status, "minutes_limit": minutes,
                             "source": source, "url": url, "text": sentence})
    return rows


def relevant_text(text, game, names, registry=None):
    registry = registry or NewsRegistry.load()
    keys = [name_key(n) for n in names]
    if any(k and k in name_key(text) for k in keys):
        return True
    for team in (game.home_team, game.away_team):
        team = TEAM_ALIASES.get(team, team)
        entry = registry.data["teams"].get(team, {})
        if any(re.search(r"\b" + re.escape(a) + r"\b", text, re.I) for a in entry.get("aliases", [])):
            return True
    return False


def post_rows(text, game, players, roster, source, published, now, url, post_id):
    """Explicit X statuses, including injury lists and unique roster surnames.

    Full post evidence is stored separately even when no factor is supported.
    Each clause has its own subject; list headings apply only to their list.
    """
    names = {int(p): n for p, n in zip(players.player_id, players.player_name) if int(p) in roster}
    aliases = {}
    for pid, name in names.items():
        parts = name.split()
        surname = parts[-2] if parts[-1].rstrip(".").lower() in {"jr", "sr", "ii", "iii", "iv"} and len(parts) > 1 else parts[-1]
        for alias in (name, surname):
            aliases.setdefault(alias, set()).add(pid)
    rows = []
    same_day = published.tz_convert(EASTERN).date() == pd.Timestamp(game.date).date()
    actionable = same_day and not re.search(NON_GAME_CONTEXT, text, re.I)
    for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", text):
        matched = {}
        for alias, ids in aliases.items():
            if len(ids) == 1 and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", sentence, re.I):
                pid = next(iter(ids))
                if pid not in matched or len(alias) > len(matched[pid]):
                    matched[pid] = alias
        heading = re.match(r"\s*(out|doubtful|questionable|probable|available)\s*:\s*(.+)", sentence, re.I)
        for pid, alias in matched.items():
            status, minutes = "review", None
            if actionable:
                if heading:
                    if not re.search(r"\b(could|might|may|not|if)\b|\b(out|doubtful|questionable|probable|available)\s*:", heading[2], re.I):
                        status = heading[1].lower()
                elif len(matched) == 1:
                    status, minutes = classify_sentence(sentence, alias)
                    # Official short labels: "Brown (ankle): OUT", "Tatum - Available".
                    short = re.search(re.escape(alias) + r"(?:\s*\([^)]*\))?\s*[-:]\s*(out|doubtful|questionable|probable|available)\b", sentence, re.I)
                    if short and not re.search(r"\b(could|might|may|not|if)\b", sentence, re.I):
                        status = short[1].lower()
            rows.append({"news_id": event_id(source, game.game_id, pid, published, post_id + sentence),
                         "item_id": post_id, "game_id": game.game_id, "player_id": pid,
                         "published_at": published, "observed_at": now, "status": status,
                         "minutes_limit": minutes, "source": source, "url": url, "text": sentence})
    return rows


def classify_sentence(text, player_name):
    """Only explicit subject/status clauses are actionable; never use an LLM number."""
    # A recap, rumour, conditional or negation cannot alter today's forecast.
    guard_text = re.sub(r"\bwill not play\b", "will miss", text, flags=re.I)
    if (re.search(NON_GAME_CONTEXT, text, re.I)
            or re.search(r"\b(could|might|may|if|not|isn't|won't be ruled out)\b", guard_text, re.I)):
        return "review", None
    name = r"\s+".join(re.escape(p) for p in player_name.split())
    match = re.search(name + r"(?:\s*\([^)]*\))?\s*[:,]?\s*(.*)", text, re.I)
    if not match:
        return "review", None
    clause = match[1]
    patterns = [
        ("out", r"^(?:(?:has been|is|was|remains|listed as)\s+)?(?:ruled out|out\b(?!\s+of\b)|will miss|will sit out|will not play|won't play)\b"),
        ("available", r"^(?:(?:has been|is)\s+)?(?:cleared to play|available|will play|will return)\b"),
        ("doubtful", r"^(?:(?:is|remains|listed as)\s+)?doubtful\b"),
        ("questionable", r"^(?:(?:is|remains|listed as)\s+)?questionable\b"),
        ("probable", r"^(?:(?:is|remains|listed as)\s+)?probable\b"),
    ]
    for status, pattern in patterns:
        if re.search(pattern, clause, re.I):
            return status, None
    limit = re.search(r"^(?:will (?:have|be on)|is on|has) (?:a )?(\d{1,2})[ -]minute (?:limit|restriction|cap)\b", clause, re.I)
    return ("minutes_limit", float(limit[1])) if limit else ("review", None)


class LiveNews:
    def __init__(self, rss_urls=None, official=True, timeout=10, report_slots=4,
                 clock=lambda: pd.Timestamp.now(tz="UTC"), registry=None, x_enabled=True,
                 x_client=None):
        self.registry = registry or NewsRegistry.load()
        self.rss_urls = [m["url"] for m in self.registry.media.values()] if rss_urls is None else rss_urls
        self.official, self.timeout, self.report_slots = official, timeout, report_slots
        self.clock = clock
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "nba-agent/0.1"})
        self.x_enabled = x_enabled
        if x_enabled:
            from data_sources.x_news import XNews
            self.x_client = x_client or XNews(self.registry, timeout=timeout, clock=clock)

    def dump_state(self):
        return self.x_client.dump_state() if self.x_enabled else {}

    def load_state(self, saved):
        if self.x_enabled:
            self.x_client.load_state(saved)

    def _get(self, url, game):
        if self.clock() >= game.tip_time:
            raise ValueError("tip-off reached; no further news requests")
        return self.session.get(url, timeout=self.timeout)

    def fetch(self, game, players, roster, since, now):
        batch = NewsBatch()
        plan = self.registry.for_game(game)
        for team in plan["missing_teams"] if self.x_enabled else []:
            batch.errors.append({"source": "source_plan", "error": f"no team account plan for {team}"})
        for url in self.rss_urls:
            media = next((m for m in self.registry.media.values() if m["url"] == url), None)
            source = media["source"] if media else f"rss:{urlparse(url).netloc}"
            try:
                response = self._get(url, game)
                response.raise_for_status()
                rows = rss_rows(response.content, game, players, roster, since, now, source,
                                batch.items, media.get("domains") if media else None, self.registry)
                batch.rows.extend(rows)
                batch.coverage.append({"source": source, "url": url, "state": "ok", "checked_at": now.isoformat(),
                                       "matching_factors": len(rows)})
            except (requests.RequestException, ET.ParseError, ValueError) as exc:
                batch.errors.append({"source": url, "error": f"{type(exc).__name__}: {exc}"})
                batch.coverage.append({"source": source, "url": url, "state": "failed", "checked_at": now.isoformat()})
        if self.official:
            try:
                official_rows = self._official(game, players, since, now)
                batch.rows.extend(official_rows)
                batch.items.extend({**r, "item_id": r["news_id"], "review_required": False} for r in official_rows)
                batch.coverage.append({"source": "nba_injury_report", "state": "ok", "checked_at": now.isoformat()})
            except Exception as exc:
                batch.errors.append({"source": "nba_injury_report", "error": f"{type(exc).__name__}: {exc}"})
                batch.coverage.append({"source": "nba_injury_report", "state": "failed", "checked_at": now.isoformat()})
        if self.x_enabled:
            x_batch = self.x_client.fetch(game, players, roster, since, now)
            batch.rows.extend(x_batch.rows)
            batch.items.extend(x_batch.items)
            batch.errors.extend(x_batch.errors)
            batch.coverage.extend(x_batch.coverage)
        else:
            batch.coverage.extend({"source": a["source"], "state": "disabled"} for a in plan["x_accounts"])
        return batch

    def _official(self, game, players, since, now):
        lookup = player_lookup(players)
        latest = now.tz_convert(EASTERN).floor("15min", ambiguous=False, nonexistent="shift_forward")
        for offset in range(self.report_slots):
            slot = latest - pd.Timedelta(minutes=15 * offset)
            for filename in report_names(slot.date(), slot.hour, slot.minute):
                url = BASE + filename
                r = self._get(url, game)
                if r.status_code == 404:
                    continue  # Missing live report is retried next poll, never cached permanently.
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("injury endpoint did not return a PDF")
                with tempfile.TemporaryDirectory(prefix="nba-news-") as folder:
                    path = Path(folder) / filename
                    path.write_bytes(r.content)
                    raw = parse_pdf(path)
                rows = []
                for row in raw:
                    if (row["Matchup"] != f"{game.away_team}@{game.home_team}"
                            or pd.Timestamp(row["GameDate"]).date() != pd.Timestamp(game.date).date()):
                        continue
                    pid = lookup.get(name_key(row["PlayerName"]))
                    if pid is None:
                        continue
                    text = f"{row['PlayerName']}: {row['CurrentStatus']}; {row['Reason']}"
                    published = slot.tz_convert("UTC")
                    if published < since:
                        continue
                    rows.append({"news_id": event_id("nba_injury_report", game.game_id, pid, published, text),
                                 "game_id": game.game_id, "player_id": pid, "published_at": published,
                                 "observed_at": now, "status": row["CurrentStatus"].lower(),
                                 "source": "nba_injury_report", "url": url, "text": text})
                if rows:
                    return rows
        raise ValueError("no matching injury report found in recent report slots")
