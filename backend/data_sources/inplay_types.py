"""Contracts for the independent in-play pipeline. No pregame runtime imports."""
import hashlib
import math
import re
from dataclasses import dataclass, field

import pandas as pd


def utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("in-play timestamps must include a timezone")
    return value.tz_convert("UTC")


def clock_seconds(value):
    match = re.fullmatch(r"PT(?:(\d+)M)?(\d+(?:\.\d+)?)S", str(value))
    if match:
        return float(match[1] or 0) * 60 + float(match[2])
    match = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", str(value))
    if match:
        return int(match[1]) * 60 + float(match[2])
    raise ValueError(f"unsupported game clock: {value!r}")


def validate_score(state, game, now):
    s = dict(state)
    if str(s["game_id"]) != str(game.game_id):
        raise ValueError("live state belongs to another game")
    if s["phase"] not in {"scheduled", "live", "final", "suspended"}:
        raise ValueError("unsupported game phase")
    for key in ("home_score", "away_score", "period"):
        value = float(s[key])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"invalid {key}")
        s[key] = int(value)
    seconds = float(s["clock_seconds"])
    length = 720 if s["period"] <= 4 else 300
    if not math.isfinite(seconds) or not 0 <= seconds <= length:
        raise ValueError("invalid live game clock")
    if s["phase"] == "live" and s["period"] < 1:
        raise ValueError("live game must have a period")
    s["clock_seconds"] = seconds
    s["observed_at"] = utc(s["observed_at"]).isoformat()
    if utc(s["observed_at"]) > now:
        raise ValueError("future score observation")
    if pd.isna(s.get("updated_at")):
        s["updated_at"] = None
    if s.get("updated_at") is not None:
        s["updated_at"] = utc(s["updated_at"]).isoformat()
        if utc(s["updated_at"]) > now + pd.Timedelta(seconds=5):
            raise ValueError("future source timestamp")
    for key in ("player_minutes", "player_teams"):
        if not isinstance(s.get(key), dict):
            s[key] = {}
    if hasattr(s.get("roster"), "tolist"):
        s["roster"] = s["roster"].tolist()
    if not isinstance(s.get("roster"), (list, tuple)):
        s["roster"] = []
    s["roster"] = list(map(int, s["roster"]))
    for pid, minutes in s["player_minutes"].items():
        if not math.isfinite(float(minutes)) or float(minutes) < 0:
            raise ValueError("invalid played minutes")
    return s


def event_id(*values):
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:24]


@dataclass
class InPlayBatch:
    score: dict | None = None
    events: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    errors: list = field(default_factory=list)
