"""One pull of the real 2025-26 season from stats.nba.com via nba_api. Not called during the demo.

    python download_season.py              # full season (about 1,300 box-score calls; resumable)
    python download_season.py --limit 5    # smoke test on the first five games

Writes the same files as make_sample_data.py. Contracts and news paragraphs are not in nba_api:
put hand-made CSVs at data/info/contracts.csv and data/news/paragraphs.csv and they are converted.
NBA.com terms: private, non-commercial use with attribution; do not publish the dump.
"""
import argparse
import json
import time
import urllib.request

import pandas as pd

from nba_agent.data.tables import DATA, EVAL, FILES, INFO, NEWS, SEASON, SEASON_DIR

CACHE = DATA / "cache" / "advanced_v3"
MOVEMENT_URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"}
PAUSE = 0.7


def game_logs(season_type):
    from nba_api.stats.endpoints import playergamelogs, teamgamelogs

    players = playergamelogs.PlayerGameLogs(season_nullable=SEASON, season_type_nullable=season_type).get_data_frames()[0]
    time.sleep(PAUSE)
    teams = teamgamelogs.TeamGameLogs(season_nullable=SEASON, season_type_nullable=season_type).get_data_frames()[0]
    time.sleep(PAUSE)
    for f in (players, teams):
        f["GAME_DATE"] = pd.to_datetime(f["GAME_DATE"]).dt.normalize()
        f["SEASON_TYPE"] = season_type
    players["MIN"] = pd.to_numeric(players["MIN"], errors="coerce")
    opp = teams[["GAME_ID", "TEAM_ID", "PTS"]].rename(columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"})
    teams = teams.merge(opp, on="GAME_ID")
    teams = teams[teams.TEAM_ID != teams.OPP_TEAM_ID]
    return players, teams


def advanced(game_ids):
    from nba_api.stats.endpoints import boxscoreadvancedv3

    CACHE.mkdir(parents=True, exist_ok=True)
    player_parts, team_parts = [], []
    for i, gid in enumerate(game_ids):
        p_path, t_path = CACHE / f"{gid}_players.parquet", CACHE / f"{gid}_teams.parquet"
        if not p_path.exists():
            for attempt in range(3):
                try:
                    frames = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gid, timeout=30).get_data_frames()
                    frames[0].to_parquet(p_path)
                    frames[1].to_parquet(t_path)
                    break
                except Exception as exc:
                    print(f"  {gid} attempt {attempt + 1} failed: {exc.__class__.__name__}")
                    time.sleep(5 * (attempt + 1))
            time.sleep(PAUSE)
        if p_path.exists():
            player_parts.append(pd.read_parquet(p_path))
            team_parts.append(pd.read_parquet(t_path))
        if (i + 1) % 50 == 0:
            print(f"  advanced box scores: {i + 1}/{len(game_ids)}")
    players = pd.concat(player_parts, ignore_index=True).rename(columns={
        "gameId": "GAME_ID", "personId": "PLAYER_ID", "teamId": "TEAM_ID", "trueShootingPercentage": "TS_PCT",
        "possessions": "POSS", "usagePercentage": "USG_PCT"})
    teams = pd.concat(team_parts, ignore_index=True).rename(columns={
        "gameId": "GAME_ID", "teamId": "TEAM_ID", "possessions": "POSS"})
    return players[["GAME_ID", "PLAYER_ID", "TEAM_ID", "TS_PCT", "POSS", "USG_PCT"]], teams[["GAME_ID", "TEAM_ID", "POSS"]]


def trades_from_movement(player_logs):
    req = urllib.request.Request(MOVEMENT_URL, headers=HEADERS)
    rows = json.loads(urllib.request.urlopen(req, timeout=30).read())["NBA_Player_Movement"]["rows"]
    moves = pd.DataFrame(rows)
    moves = moves[moves.Transaction_Type == "Trade"].copy()
    moves["TRADE_DATE"] = pd.to_datetime(moves.TRANSACTION_DATE).dt.normalize()
    moves = moves[(moves.TRADE_DATE >= "2025-07-01") & (moves.PLAYER_ID > 0)]
    out = []
    for m in moves.drop_duplicates(["PLAYER_ID", "TRADE_DATE"]).itertuples():
        games = player_logs[player_logs.PLAYER_ID == m.PLAYER_ID].sort_values("GAME_DATE")
        before, after = games[games.GAME_DATE < m.TRADE_DATE], games[games.GAME_DATE >= m.TRADE_DATE]
        if len(before) and len(after) and before.TEAM_ID.iloc[-1] != after.TEAM_ID.iloc[0]:
            out.append({"PLAYER_ID": int(m.PLAYER_ID), "TRADE_DATE": m.TRADE_DATE,
                        "FROM_TEAM_ID": int(before.TEAM_ID.iloc[-1]), "TO_TEAM_ID": int(after.TEAM_ID.iloc[0])})
    return pd.DataFrame(out, columns=["PLAYER_ID", "TRADE_DATE", "FROM_TEAM_ID", "TO_TEAM_ID"])


def recaps(player_logs, teams):
    names = dict(zip(teams.TEAM_ID, teams.TEAM_NAME))
    rows = []
    for i, r in enumerate(player_logs.itertuples()):
        rows.append({"RECAP_ID": f"r{i:06d}", "GAME_ID": r.GAME_ID, "GAME_DATE": r.GAME_DATE, "PLAYER_ID": r.PLAYER_ID,
                     "TEAM_ID": r.TEAM_ID,
                     "TEXT": f"{r.PLAYER_NAME} scored {r.PTS} points on {r.FGM}-of-{r.FGA} shooting with {r.REB} "
                             f"rebounds and {r.AST} assists in {r.MIN:.0f} minutes for the {names.get(r.TEAM_ID, r.TEAM_ID)} "
                             f"({r.MATCHUP}) on {r.GAME_DATE.strftime('%-d %B %Y')}."})
    return pd.DataFrame(rows)


def main():
    from nba_api.stats.endpoints import leaguedashplayerstats
    from nba_api.stats.static import teams as static_teams

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only this many games per season type (smoke test)")
    args = ap.parse_args()
    for d in (INFO, NEWS, EVAL, SEASON_DIR):
        d.mkdir(parents=True, exist_ok=True)

    teams = pd.DataFrame(static_teams.get_teams()).rename(columns={"id": "TEAM_ID", "abbreviation": "TEAM_ABBREVIATION",
                                                                   "full_name": "TEAM_NAME"})[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]]
    dash = leaguedashplayerstats.LeagueDashPlayerStats(season=SEASON).get_data_frames()[0]
    players = dash[["PLAYER_ID", "PLAYER_NAME", "AGE"]].drop_duplicates("PLAYER_ID")
    teams.to_parquet(FILES["teams"], index=False)
    players.to_parquet(FILES["players"], index=False)

    for season_type, suffix in (("Regular Season", "regular"), ("Playoffs", "playoffs")):
        print(f"{season_type}: game logs")
        p_logs, t_logs = game_logs(season_type)
        game_ids = sorted(t_logs.GAME_ID.unique())[: args.limit]
        p_logs, t_logs = p_logs[p_logs.GAME_ID.isin(game_ids)], t_logs[t_logs.GAME_ID.isin(game_ids)]
        if not game_ids:
            print(f"  no {season_type} games yet; writing empty files")
        adv_p, adv_t = advanced(game_ids) if game_ids else (pd.DataFrame(columns=["GAME_ID", "PLAYER_ID", "TEAM_ID", "TS_PCT", "POSS", "USG_PCT"]),
                                                            pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "POSS"]))
        t_logs = t_logs.merge(adv_t, on=["GAME_ID", "TEAM_ID"], how="left")
        t_logs["OFF_RATING"] = t_logs.PTS / t_logs.POSS * 100
        t_logs["DEF_RATING"] = t_logs.OPP_PTS / t_logs.POSS * 100
        t_logs["NET_RATING"] = t_logs.OFF_RATING - t_logs.DEF_RATING
        p_logs.to_parquet(FILES[f"player_{suffix}"], index=False)
        t_logs.to_parquet(FILES[f"team_{suffix}"], index=False)
        adv_p.to_parquet(FILES[f"advanced_{suffix}"], index=False)
        if suffix == "regular":
            recaps(p_logs, teams).to_parquet(FILES["recaps"], index=False)
            trades_from_movement(p_logs).to_parquet(FILES["trades"], index=False)

    for name, csv in (("contracts", INFO / "contracts.csv"), ("paragraphs", NEWS / "paragraphs.csv")):
        if csv.exists():
            frame = pd.read_csv(csv, parse_dates=["DATE"] if name == "paragraphs" else None)
            frame.to_parquet(FILES[name], index=False)
            print(f"converted {csv.name}")
        else:
            print(f"missing {csv}: {name} must be built by hand (see README)")


if __name__ == "__main__":
    main()
