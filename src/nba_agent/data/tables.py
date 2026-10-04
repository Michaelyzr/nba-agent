"""Paths and cached access to the frozen season files."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data"
INFO = DATA / "info"
NEWS = DATA / "news"
EVAL = DATA / "eval"
MODELS = PROJECT_ROOT / "models"
SKILLS = PACKAGE_ROOT / "skills"

SEASON = "2025-26"
SEASON_DIR = DATA / "seasons" / SEASON
MAX_TRIES = 3

FILES = {
    "players": INFO / "players.parquet",
    "teams": INFO / "teams.parquet",
    "trades": INFO / "trades.parquet",
    "contracts": INFO / "contracts.parquet",
    "paragraphs": NEWS / "paragraphs.parquet",
    "player_regular": SEASON_DIR / "player_regular.parquet",
    "player_playoffs": SEASON_DIR / "player_playoffs.parquet",
    "team_regular": SEASON_DIR / "team_regular.parquet",
    "team_playoffs": SEASON_DIR / "team_playoffs.parquet",
    "advanced_regular": SEASON_DIR / "advanced_regular.parquet",
    "advanced_playoffs": SEASON_DIR / "advanced_playoffs.parquet",
    "recaps": SEASON_DIR / "recaps.parquet",
}


@lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    path = FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python make_sample_data.py` or `python download_season.py` first."
        )
    return pd.read_parquet(path)


def all_tables() -> dict:
    return {name: load(name) for name in FILES}


def player_name(player_id: int) -> str:
    p = load("players")
    return p.loc[p.PLAYER_ID == player_id, "PLAYER_NAME"].iloc[0]


def team_name(team_id: int) -> str:
    t = load("teams")
    return t.loc[t.TEAM_ID == team_id, "TEAM_NAME"].iloc[0]
