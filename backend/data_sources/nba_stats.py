"""D1: freeze NBA box scores for several seasons into data/frozen/.

    python -m data_sources.nba_stats --seasons 2025-26 --limit 5      # smoke test
    python -m data_sources.nba_stats                                  # 2022-23 to 2025-26
    python -m data_sources.nba_stats --advanced                       # + usage and starters (1 call per game)

Writes games, player_games and players. stats.nba.com throttles and sometimes
hangs; every call is cached under data/raw/nba/, so rerun to resume.
Tip-off times come from the league schedule; if that call fails, they are
filled from tip_times.parquet written by data_sources.news.
"""
import argparse
import time

import pandas as pd

from data_sources import FROZEN, GAME_LENGTH, RAW, read_table, write_table

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
CACHE = RAW / "nba"
PAUSE = 0.7


def _cached(path, fetch):
    if path.exists():
        return pd.read_parquet(path)
    for attempt in range(3):
        try:
            frame = fetch()
            break
        except Exception as exc:
            print(f"  {path.name} attempt {attempt + 1} failed: {exc.__class__.__name__}")
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"could not fetch {path.name}; stats.nba.com may be blocking this network")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    time.sleep(PAUSE)
    return frame


def game_logs(season: str, season_type: str):
    from nba_api.stats.endpoints import playergamelogs, teamgamelogs

    tag = season_type.split()[0].lower()
    players = _cached(CACHE / season / f"players_{tag}.parquet", lambda: playergamelogs.PlayerGameLogs(
        season_nullable=season, season_type_nullable=season_type, timeout=60).get_data_frames()[0])
    teams = _cached(CACHE / season / f"teams_{tag}.parquet", lambda: teamgamelogs.TeamGameLogs(
        season_nullable=season, season_type_nullable=season_type, timeout=60).get_data_frames()[0])
    return players, teams


def schedule(season: str) -> pd.DataFrame:
    """game_id and tip_time (UTC) from the league schedule; empty if the endpoint fails."""
    from nba_api.stats.endpoints import scheduleleaguev2

    try:
        raw = _cached(CACHE / season / "schedule.parquet", lambda: scheduleleaguev2.ScheduleLeagueV2(
            season=season, timeout=60).get_data_frames()[0])
    except RuntimeError as exc:
        print(f"  schedule {season}: {exc}")
        return pd.DataFrame({"game_id": pd.Series(dtype=str), "tip_time": pd.Series(dtype="datetime64[ns, UTC]")})
    col = next(c for c in ("gameDateTimeUTC", "gameDateUTC", "gameDateTimeEst") if c in raw.columns)
    tip = pd.to_datetime(raw[col], utc=col.endswith("UTC"))
    if not col.endswith("UTC"):
        tip = tip.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return pd.DataFrame({"game_id": raw.gameId.astype(str), "tip_time": tip})


def build_games(team_logs: pd.DataFrame, tips: pd.DataFrame) -> pd.DataFrame:
    """One row per game from two team rows; 'vs.' in MATCHUP marks the home team."""
    t = team_logs.assign(GAME_ID=team_logs.GAME_ID.astype(str), home=team_logs.MATCHUP.str.contains("vs."))
    home = t[t.home].rename(columns={"TEAM_ID": "home_team_id", "TEAM_ABBREVIATION": "home_team", "PTS": "home_pts"})
    away = t[~t.home].rename(columns={"TEAM_ID": "away_team_id", "TEAM_ABBREVIATION": "away_team", "PTS": "away_pts"})
    games = home[["GAME_ID", "GAME_DATE", "home_team_id", "home_team", "home_pts"]].merge(
        away[["GAME_ID", "away_team_id", "away_team", "away_pts"]], on="GAME_ID")
    games = games.rename(columns={"GAME_ID": "game_id"})
    games["date"] = pd.to_datetime(games.GAME_DATE).dt.date.astype(str)
    games = games.drop(columns="GAME_DATE").merge(tips, on="game_id", how="left")
    games["final_at"] = games.tip_time + GAME_LENGTH
    return games.sort_values(["tip_time", "game_id"]).reset_index(drop=True)


def advanced_extra(game_ids) -> pd.DataFrame:
    """Usage and starter flag from advanced box scores. Games that keep failing are skipped; rerun to fill them."""
    from nba_api.stats.endpoints import boxscoreadvancedv3

    parts = []
    for i, gid in enumerate(game_ids):
        try:
            parts.append(_cached(CACHE / "advanced" / f"{gid}.parquet",
                                 lambda: boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid, timeout=30).get_data_frames()[0]))
        except RuntimeError as exc:
            print(f"  skipped: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  advanced box scores: {i + 1}/{len(game_ids)}")
    if not parts:
        return pd.DataFrame(columns=["game_id", "player_id", "usage", "started"])
    adv = pd.concat(parts, ignore_index=True)
    started = adv["position"].fillna("").astype(str).ne("") if "position" in adv else pd.NA
    return pd.DataFrame({"game_id": adv.gameId.astype(str), "player_id": adv.personId.astype(int),
                         "usage": adv.usagePercentage, "started": started})


def build_player_games(player_logs: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    p = player_logs.rename(columns={"GAME_ID": "game_id", "PLAYER_ID": "player_id", "TEAM_ID": "team_id",
                                    "MIN": "min", "PTS": "pts", "FGA": "fga", "FTA": "fta"})
    p["game_id"] = p.game_id.astype(str)
    p["min"] = pd.to_numeric(p["min"], errors="coerce")
    p = p[["game_id", "player_id", "team_id", "min", "pts", "fga", "fta"]]
    if extra is not None and len(extra):
        return p.merge(extra, on=["game_id", "player_id"], how="left")
    return p.assign(usage=float("nan"), started=pd.NA)


def fill_tips_from_news(games: pd.DataFrame) -> pd.DataFrame:
    path = FROZEN / "tip_times.parquet"
    if games.tip_time.notna().all() or not path.exists():
        return games
    tips = read_table("tip_times").rename(columns={"tip_time": "news_tip"})
    games = games.merge(tips, on=["date", "home_team", "away_team"], how="left")
    games["tip_time"] = games.tip_time.fillna(games.pop("news_tip"))
    games["final_at"] = games.tip_time + GAME_LENGTH
    return games


def freeze(seasons=SEASONS, limit=None, with_advanced=False):
    games, player_games, players = [], [], []
    for season in seasons:
        tips = schedule(season)
        for season_type in SEASON_TYPES:
            print(f"{season} {season_type}")
            p_logs, t_logs = game_logs(season, season_type)
            if t_logs.empty:
                continue
            ids = sorted(t_logs.GAME_ID.astype(str).unique())[:limit]
            p_logs = p_logs[p_logs.GAME_ID.astype(str).isin(ids)]
            t_logs = t_logs[t_logs.GAME_ID.astype(str).isin(ids)]
            games.append(build_games(t_logs, tips))
            player_games.append(build_player_games(p_logs, advanced_extra(ids) if with_advanced else None))
            players.append(p_logs[["PLAYER_ID", "PLAYER_NAME"]])
    games = fill_tips_from_news(pd.concat(games, ignore_index=True))
    missing = games.tip_time.isna().sum()
    if missing:
        print(f"warning: {missing} games have no tip_time; run data_sources.news then rerun this")
    write_table("games", games)
    write_table("player_games", pd.concat(player_games, ignore_index=True))
    write_table("players", pd.concat(players).drop_duplicates("PLAYER_ID").rename(
        columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"}))
    print(f"wrote {len(games)} games to {FROZEN}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--limit", type=int, default=None, help="games per season type (smoke test)")
    ap.add_argument("--advanced", action="store_true", help="also fetch usage and starters, one call per game")
    args = ap.parse_args()
    freeze(args.seasons, args.limit, args.advanced)


if __name__ == "__main__":
    main()
