"""Shared paths, table schemas and HTTP helper for the data-and-replay subgroup.

Every frozen table lives in data/frozen/<name>.parquet and must have at least the
columns in SCHEMAS (README section 5). Times are timezone-aware UTC.
"""
import re
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
RAW = ROOT / "data" / "raw"
ET = ZoneInfo("America/New_York")

SCHEMAS = {
    "games": ["game_id", "date", "tip_time", "final_at", "home_team_id", "away_team_id",
              "home_team", "away_team", "home_pts", "away_pts"],
    "player_games": ["game_id", "player_id", "team_id", "min", "pts", "fga", "fta", "usage", "started"],
    "players": ["player_id", "player_name"],
    "news": ["news_id", "published_at", "game_id", "player_id", "status", "source", "url", "text"],
    "markets": ["venue", "market_ticker", "kind", "game_id", "team", "player_id", "line", "title"],
    "prices": ["venue", "market_ticker", "ts", "bid", "ask", "volume"],
    "settlements": ["market_ticker", "settled_at", "outcome"],
}

# Finished box scores are treated as public this long after tip-off.
GAME_LENGTH = pd.Timedelta(hours=3)


def write_table(name: str, frame: pd.DataFrame, folder: Path = FROZEN) -> Path:
    missing = [c for c in SCHEMAS[name] if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path


def read_table(name: str, folder: Path = FROZEN) -> pd.DataFrame:
    path = folder / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it with the matching data_sources module")
    return pd.read_parquet(path)


def get_json(url: str, params: dict | None = None, tries: int = 4, timeout: int = 30) -> dict:
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "nba-agent/0.1"})
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{url} kept returning 429")


def name_key(name: str) -> str:
    """'D'Angelo Russell', 'Russell,D'Angelo' and 'Russell, D'Angelo' all map to 'dangelorussell'."""
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


def player_lookup(players: pd.DataFrame) -> dict:
    return {name_key(n): int(p) for p, n in zip(players.player_id, players.player_name)}


def et_to_utc(day: str, clock: str) -> pd.Timestamp:
    """day '2026-02-07', clock '07:30 PM' (Eastern) -> UTC timestamp."""
    local = pd.Timestamp(f"{day} {clock}").tz_localize(ET)
    return local.tz_convert("UTC")
