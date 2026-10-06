"""Synthetic 2025-26 season in the frozen-table format, for developing models and the agent loop before D1 lands.

Team names and IDs are real; every player, minute and score is generated here.
Never report these numbers as findings. Built in memory with fixed seeds, so it
needs no downloads and gives the same season on every machine.
"""
import random

import numpy as np
import pandas as pd

from data_sources.nba_stats import build_games, build_player_games

TEAMS = [
    ("ATL", "Atlanta Hawks"), ("BOS", "Boston Celtics"), ("BKN", "Brooklyn Nets"),
    ("CHA", "Charlotte Hornets"), ("CHI", "Chicago Bulls"), ("CLE", "Cleveland Cavaliers"),
    ("DAL", "Dallas Mavericks"), ("DEN", "Denver Nuggets"), ("DET", "Detroit Pistons"),
    ("GSW", "Golden State Warriors"), ("HOU", "Houston Rockets"), ("IND", "Indiana Pacers"),
    ("LAC", "LA Clippers"), ("LAL", "Los Angeles Lakers"), ("MEM", "Memphis Grizzlies"),
    ("MIA", "Miami Heat"), ("MIL", "Milwaukee Bucks"), ("MIN", "Minnesota Timberwolves"),
    ("NOP", "New Orleans Pelicans"), ("NYK", "New York Knicks"), ("OKC", "Oklahoma City Thunder"),
    ("ORL", "Orlando Magic"), ("PHI", "Philadelphia 76ers"), ("PHX", "Phoenix Suns"),
    ("POR", "Portland Trail Blazers"), ("SAC", "Sacramento Kings"), ("SAS", "San Antonio Spurs"),
    ("TOR", "Toronto Raptors"), ("UTA", "Utah Jazz"), ("WAS", "Washington Wizards"),
]
FIRST = ["Marcus", "Dante", "Elijah", "Theo", "Jalen", "Andre", "Cole", "Isaiah", "Malik", "Owen",
         "Rafael", "Quincy", "Tobias", "Victor", "Wes", "Xavier", "Yusuf", "Zion", "Bryce", "Caleb"]
LAST = ["Hale", "Okafor", "Brandt", "Castillo", "Duvall", "Ferreira", "Garrow", "Holloway", "Ingram",
        "Jessup", "Kowalski", "Lindqvist", "Mbeki", "Navarro", "Osei", "Pruitt", "Quarles", "Rourke"]
PER_TEAM = 8
SEED = 7606
REG_START, REG_DAYS, N_REG = pd.Timestamp("2025-10-21"), 174, 1230
PO_START, PO_DAYS, N_PO = pd.Timestamp("2026-04-18"), 58, 60
TIP_ET = "19:30"


def _players(teams, rng, pick):
    names = pick.sample([f"{f} {l}" for f in FIRST for l in LAST], len(teams) * PER_TEAM)
    return pd.DataFrame([{
        "PLAYER_ID": 1_700_000 + i, "PLAYER_NAME": name, "AGE": int(rng.integers(20, 36)),
        "START_TEAM_ID": int(teams.TEAM_ID.iloc[i // PER_TEAM]), "MIN_WEIGHT": float(rng.uniform(0.5, 1.6)),
        "USAGE": float(rng.uniform(0.55, 1.05)), "BASE_TS": float(rng.normal(0.575, 0.035)),
    } for i, name in enumerate(names)])


def _trades(players, teams, rng):
    rows = []
    for pid in players.sample(12, random_state=7).PLAYER_ID:
        start = int(players.loc[players.PLAYER_ID == pid, "START_TEAM_ID"].iloc[0])
        to = int(rng.choice([t for t in teams.TEAM_ID if t != start]))
        date = pd.Timestamp("2025-12-15") + pd.Timedelta(days=int(rng.integers(0, 52)))
        rows.append({"PLAYER_ID": pid, "TRADE_DATE": date, "TO_TEAM_ID": to})
    return {r["PLAYER_ID"]: r for r in rows}


def _schedule(n, start, days, prefix, team_ids, rng):
    out = []
    for g in range(n):
        home, away = rng.choice(team_ids, 2, replace=False)
        out.append({"GAME_ID": f"{prefix}{g + 1:05d}", "GAME_DATE": start + pd.Timedelta(days=int(g * days / n)),
                    "HOME": int(home), "AWAY": int(away)})
    return out


def _box_line(p, minutes, ts_shift, rng):
    fga = max(1, int(round(minutes * p.USAGE * 0.42)))
    fta = int(round(fga * 0.25))
    fg3a = int(round(fga * 0.38))
    ftm = int(round(fta * rng.uniform(0.65, 0.9)))
    fg3m = min(fg3a, int(round(fg3a * rng.uniform(0.25, 0.45))))
    ts = float(np.clip(rng.normal(p.BASE_TS + ts_shift, 0.09), 0.25, 0.9))
    fgm = int(np.clip(round((ts * 2 * (fga + 0.44 * fta) - ftm - fg3m) / 2), fg3m, fga))
    rng.poisson(minutes * 0.18), rng.poisson(minutes * 0.11)       # rebounds and assists keep the seed sequence
    return {"MIN": round(minutes, 1), "FGA": fga, "FTA": fta, "PTS": 2 * fgm + fg3m + ftm}


def _play(schedule, players, abbr, trades, rng):
    player_rows, team_rows = [], []
    for g in schedule:
        rng.normal(99, 4)                                               # possessions keep the seed sequence
        points = {}
        for team, opp, at in ((g["HOME"], g["AWAY"], "vs."), (g["AWAY"], g["HOME"], "@")):
            roster = [p for p in players.itertuples()
                      if (trades[p.PLAYER_ID]["TO_TEAM_ID"] if p.PLAYER_ID in trades
                          and g["GAME_DATE"] >= trades[p.PLAYER_ID]["TRADE_DATE"] else p.START_TEAM_ID) == team]
            weights = np.array([p.MIN_WEIGHT for p in roster]) * rng.uniform(0.8, 1.2, len(roster))
            minutes = np.minimum(weights / weights.sum() * 240, 42)
            points[team] = 0
            for p, m in zip(roster, minutes):
                t = trades.get(p.PLAYER_ID)
                shift = 0.0 if t is None else (0.035 if g["GAME_DATE"] >= t["TRADE_DATE"] else -0.01)
                line = _box_line(p, float(m), shift, rng)
                points[team] += line["PTS"]
                player_rows.append({"GAME_ID": g["GAME_ID"], "PLAYER_ID": p.PLAYER_ID, "TEAM_ID": team, **line})
            team_rows.append({"GAME_ID": g["GAME_ID"], "GAME_DATE": g["GAME_DATE"], "TEAM_ID": team,
                              "TEAM_ABBREVIATION": abbr[team], "MATCHUP": f"{abbr[team]} {at} {abbr[opp]}"})
        for row in team_rows[-2:]:
            row["PTS"] = points[row["TEAM_ID"]]
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)


def synthetic_tables():
    """(player_games, games) for the synthetic regular season, tipping off at 19:30 Eastern."""
    rng, pick = np.random.default_rng(SEED), random.Random(SEED)
    teams = pd.DataFrame([{"TEAM_ID": 1610612737 + i, "TEAM_ABBREVIATION": a} for i, (a, _) in enumerate(TEAMS)])
    players = _players(teams, rng, pick)
    trades = _trades(players, teams, rng)
    regular = _schedule(N_REG, REG_START, REG_DAYS, "00225", teams.TEAM_ID.to_numpy(), rng)
    _schedule(N_PO, PO_START, PO_DAYS, "00425", teams.TEAM_ID.to_numpy()[:16], rng)  # keeps the seed sequence
    abbr = dict(zip(teams.TEAM_ID, teams.TEAM_ABBREVIATION))
    player_logs, team_logs = _play(regular, players, abbr, trades, rng)

    days = pd.to_datetime(team_logs.GAME_DATE).dt.strftime("%Y-%m-%d")
    tips = pd.DataFrame({"game_id": team_logs.GAME_ID.astype(str),
                         "tip_time": pd.to_datetime(days + " " + TIP_ET).dt.tz_localize("America/New_York")
                         .dt.tz_convert("UTC")}).drop_duplicates("game_id")
    return build_player_games(player_logs), build_games(team_logs, tips)
