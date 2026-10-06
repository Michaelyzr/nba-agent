"""In-game text extraction and media RSS. Independent of pregame adapters."""
import html
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from data_sources import name_key
from data_sources.inplay_types import InPlayBatch, event_id, utc

LOSSES = {"ejected": 1.0, "fouled_out": 1.0, "injury_out": 1.0,
          "left_injured": 0.35, "questionable_return": 0.5, "doubtful_return": 0.75,
          "returned": 0.0, "rescinded": 0.0}
TERMINAL = {"ejected", "fouled_out"}


def classify(text):
    if re.search(r"\b(yesterday|last night|last game|tomorrow|next game|earlier|previously|pregame|warmup|before tip|could|might|may|if)\b|\?", text, re.I):
        return "review"
    if re.search(r"\b(?:not|isn't|wasn't|hasn't been|has not been)\s+(?:ejected|injured|ruled out)\b|\b(?:won't|will not) be ejected\b", text, re.I):
        return "review"
    if re.search(r"\b(?:ejection|disqualification|sixth foul)\b.{0,35}\b(?:rescinded|overturned|withdrawn)\b", text, re.I):
        return "rescinded"
    if re.search(r"\b(?:ejected|disqualified)\b", text, re.I):
        return "ejected"
    if re.search(r"\b(?:fouled out|fouls out)\b", text, re.I):
        return "fouled_out"
    if re.search(r"\b(?:will not|won't|cannot|can't) return\b|\b(?:ruled )?out for (?:the )?(?:rest|remainder) of (?:the )?game\b", text, re.I):
        return "injury_out"
    if re.search(r"\bdoubtful (?:to|for) return\b", text, re.I):
        return "doubtful_return"
    if re.search(r"\bquestionable (?:to|for) return\b", text, re.I):
        return "questionable_return"
    if re.search(r"\b(?:will|expected to|plans to) return\b", text, re.I):
        return "review"
    if re.search(r"\b(?:has returned|returns? to (?:the )?game|returned to (?:the )?game|back in (?:the )?game|checks? back in)\b", text, re.I):
        return "returned"
    if (re.search(r"\b(?:left|leaves|exits?|exited|heads?|headed|went|taken)\b", text, re.I)
            and re.search(r"\b(?:injur\w*|locker room|ankle|knee|hamstring|concussion|wrist|shoulder|back pain)\b", text, re.I)):
        return "left_injured"
    return "review"


def text_events(text, game, players, roster, source, published, observed, url, item_id):
    published, observed = utc(published), utc(observed)
    if published < utc(game.tip_time):
        return []
    names = {int(p): str(n) for p, n in zip(players.player_id, players.player_name) if int(p) in roster}
    surnames = {}
    for pid, name in names.items():
        parts = name.split()
        surname = parts[-2] if len(parts) > 1 and parts[-1].rstrip(".").lower() in {"jr", "sr", "ii", "iii", "iv"} else parts[-1]
        surnames.setdefault(surname.lower(), set()).add(pid)
    rows = []
    for clause in re.split(r"[\n;]+|(?<=[.!?])\s+", text):
        matches = {}
        for pid, name in names.items():
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause, re.I):
                matches[pid] = name
        if not matches:
            for surname, ids in surnames.items():
                if len(ids) == 1 and re.search(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", clause, re.I):
                    matches[next(iter(ids))] = surname
        for pid, alias in matches.items():
            # A multi-player sentence has no reliable subject/status association.
            subject = re.search(re.escape(alias) + r"(.*)", clause, re.I)
            status = classify(subject[1]) if len(matches) == 1 else "review"
            rows.append({"event_id": event_id(item_id, pid, clause), "item_id": item_id,
                         "game_id": str(game.game_id), "player_id": pid, "status": status,
                         "published_at": published.isoformat(), "observed_at": observed.isoformat(),
                         "source": source, "url": url, "text": clause})
    return rows


class InPlayMedia:
    def __init__(self, registry, interval_seconds=60, timeout=3, session=None, clock=None):
        self.registry, self.interval, self.timeout = registry, interval_seconds, timeout
        self.session = session or requests.Session()
        self.last_check = None
        self.last_errors, self.last_coverage, self.clock = [], [], clock

    def fetch(self, game, players, roster, now):
        batch = InPlayBatch()
        if self.last_check and (now - utc(self.last_check)).total_seconds() < self.interval:
            batch.coverage = [{**c, "next_check_at": (utc(self.last_check) + pd.Timedelta(seconds=self.interval)).isoformat()}
                              for c in self.last_coverage]
            batch.errors = list(self.last_errors)
            return batch
        self.last_check = now.isoformat()
        player_names = [name_key(n) for p, n in zip(players.player_id, players.player_name) if int(p) in roster]
        team_names = [a for team in self.registry.for_game(game)["teams"]
                      for a in self.registry.data["teams"].get(team, {}).get("aliases", [])]
        for source, media in self.registry.media.items():
            try:
                response = self.session.get(media["url"], timeout=self.timeout)
                response.raise_for_status()
                observed = utc(self.clock()) if self.clock else now
                root = ET.fromstring(response.content)
                if root.tag.rsplit("}", 1)[-1].lower() != "rss":
                    raise ValueError("not an RSS response")
                for item in root.findall(".//item"):
                    from urllib.parse import urlparse
                    try:
                        published = utc(item.findtext("pubDate"))
                    except (ValueError, TypeError):
                        continue
                    url = item.findtext("link", "")
                    host = urlparse(url).hostname or ""
                    if not url.startswith("https://") or not any(host == d or host.endswith("." + d) for d in media["domains"]):
                        continue
                    if not utc(game.tip_time) <= published <= observed:
                        continue
                    text = html.unescape(item.findtext("title", "") + ". " + re.sub(r"<[^>]*>", " ", item.findtext("description", "")))
                    if not (any(n and n in name_key(text) for n in player_names)
                            or any(re.search(r"\b" + re.escape(n) + r"\b", text, re.I) for n in team_names)):
                        continue
                    item_id = event_id(source, url, published, text)
                    rows = text_events(text, game, players, roster, source, published, observed, url, item_id)
                    batch.events.extend(rows)
                    batch.evidence.append({"item_id": item_id, "game_id": str(game.game_id), "source": source,
                                           "published_at": published.isoformat(), "observed_at": observed.isoformat(),
                                           "url": url, "text": text, "review_required": not rows or any(r["status"] == "review" for r in rows)})
                batch.coverage.append({"source": source, "state": "ok", "checked_at": now.isoformat()})
            except (requests.RequestException, ET.ParseError, ValueError, TypeError):
                batch.errors.append({"source": source, "error": "media request or feed validation failed"})
                batch.coverage.append({"source": source, "state": "failed", "checked_at": now.isoformat()})
        self.last_errors, self.last_coverage = list(batch.errors), list(batch.coverage)
        return batch
