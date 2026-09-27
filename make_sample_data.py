"""Build a SYNTHETIC 2025-26 season with the same file layout as download_season.py.

Every player name, stat line, contract, and news paragraph here is made up so the
prototype can run offline before the real download. Team names are real; nothing
else is. Replace these files with `python download_season.py` for the real build.
"""
import json
import random

import numpy as np
import pandas as pd

from tables import EVAL, FILES, INFO, NEWS, SEASON_DIR

RNG = np.random.default_rng(7606)
random.seed(7606)

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
REG_START, REG_DAYS, N_REG = pd.Timestamp("2025-10-21"), 174, 1230
PO_START, PO_DAYS, N_PO = pd.Timestamp("2026-04-18"), 58, 60


def build_teams():
    return pd.DataFrame(
        [{"TEAM_ID": 1610612737 + i, "TEAM_ABBREVIATION": a, "TEAM_NAME": n} for i, (a, n) in enumerate(TEAMS)]
    )


def build_players(teams):
    names = random.sample([f"{f} {l}" for f in FIRST for l in LAST], len(teams) * PER_TEAM)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "PLAYER_ID": 1_700_000 + i,
            "PLAYER_NAME": name,
            "AGE": int(RNG.integers(20, 36)),
            "START_TEAM_ID": int(teams.TEAM_ID.iloc[i // PER_TEAM]),
            "MIN_WEIGHT": float(RNG.uniform(0.5, 1.6)),
            "USAGE": float(RNG.uniform(0.55, 1.05)),
            "BASE_TS": float(RNG.normal(0.575, 0.035)),
        })
    return pd.DataFrame(rows)


def build_trades(players, teams):
    picked = players.sample(12, random_state=7).PLAYER_ID.tolist()
    rows = []
    for pid in picked:
        start = int(players.loc[players.PLAYER_ID == pid, "START_TEAM_ID"].iloc[0])
        to = int(RNG.choice([t for t in teams.TEAM_ID if t != start]))
        date = pd.Timestamp("2025-12-15") + pd.Timedelta(days=int(RNG.integers(0, 52)))
        rows.append({"PLAYER_ID": pid, "TRADE_DATE": date, "FROM_TEAM_ID": start, "TO_TEAM_ID": to})
    return pd.DataFrame(rows)


def team_of(pid, date, start_team, trades_by_player):
    t = trades_by_player.get(pid)
    if t is not None and date >= t["TRADE_DATE"]:
        return t["TO_TEAM_ID"]
    return start_team


def box_line(p, minutes, ts_shift):
    fga = max(1, int(round(minutes * p.USAGE * 0.42)))
    fta = int(round(fga * 0.25))
    fg3a = int(round(fga * 0.38))
    ftm = int(round(fta * RNG.uniform(0.65, 0.9)))
    fg3m = min(fg3a, int(round(fg3a * RNG.uniform(0.25, 0.45))))
    ts = float(np.clip(RNG.normal(p.BASE_TS + ts_shift, 0.09), 0.25, 0.9))
    target = ts * 2 * (fga + 0.44 * fta)
    fgm = int(np.clip(round((target - ftm - fg3m) / 2), fg3m, fga))
    return {
        "MIN": round(minutes, 1), "FGM": fgm, "FGA": fga, "FG3M": fg3m, "FG3A": fg3a,
        "FTM": ftm, "FTA": fta, "PTS": 2 * fgm + fg3m + ftm,
        "REB": int(RNG.poisson(minutes * 0.18)), "AST": int(RNG.poisson(minutes * 0.11)),
    }


def build_games(n, start, days, prefix, team_pool):
    games = []
    for g in range(n):
        home, away = RNG.choice(team_pool, 2, replace=False)
        games.append({
            "GAME_ID": f"{prefix}{g + 1:05d}",
            "GAME_DATE": start + pd.Timedelta(days=int(g * days / n)),
            "HOME": int(home), "AWAY": int(away),
        })
    return games


def play_games(games, season_type, players, teams, trades):
    trades_by_player = {r.PLAYER_ID: r._asdict() for r in trades.itertuples(index=False)}
    abbr = dict(zip(teams.TEAM_ID, teams.TEAM_ABBREVIATION))
    player_rows, team_rows, adv_rows = [], [], []
    for g in games:
        poss = float(RNG.normal(99, 4))
        sides = {}
        for team, opp, at in ((g["HOME"], g["AWAY"], "vs."), (g["AWAY"], g["HOME"], "@")):
            roster = [p for p in players.itertuples()
                      if team_of(p.PLAYER_ID, g["GAME_DATE"], p.START_TEAM_ID, trades_by_player) == team]
            weights = np.array([p.MIN_WEIGHT for p in roster]) * RNG.uniform(0.8, 1.2, len(roster))
            minutes = np.minimum(weights / weights.sum() * 240, 42)
            lines = []
            for p, m in zip(roster, minutes):
                t = trades_by_player.get(p.PLAYER_ID)
                shift = 0.0
                if t is not None:
                    shift = 0.035 if g["GAME_DATE"] >= t["TRADE_DATE"] else -0.01
                line = box_line(p, float(m), shift)
                lines.append((p, line))
                player_rows.append({
                    "GAME_ID": g["GAME_ID"], "GAME_DATE": g["GAME_DATE"], "SEASON_TYPE": season_type,
                    "PLAYER_ID": p.PLAYER_ID, "PLAYER_NAME": p.PLAYER_NAME, "TEAM_ID": team,
                    "TEAM_ABBREVIATION": abbr[team], "MATCHUP": f"{abbr[team]} {at} {abbr[opp]}", **line,
                })
            plays = sum(l["FGA"] + 0.44 * l["FTA"] for _, l in lines)
            for p, l in lines:
                own = l["FGA"] + 0.44 * l["FTA"]
                adv_rows.append({
                    "GAME_ID": g["GAME_ID"], "PLAYER_ID": p.PLAYER_ID, "TEAM_ID": team,
                    "TS_PCT": l["PTS"] / (2 * own) if own else 0.0,
                    "POSS": round(poss * l["MIN"] / 48, 2),
                    "USG_PCT": (own / plays) * (240 / max(l["MIN"], 1)) / 5 if plays else 0.0,
                })
            sides[team] = sum(l["PTS"] for _, l in lines)
        for team, opp in ((g["HOME"], g["AWAY"]), (g["AWAY"], g["HOME"])):
            pts, opp_pts = sides[team], sides[opp]
            team_rows.append({
                "GAME_ID": g["GAME_ID"], "GAME_DATE": g["GAME_DATE"], "SEASON_TYPE": season_type,
                "TEAM_ID": team, "TEAM_ABBREVIATION": abbr[team], "OPP_TEAM_ID": opp,
                "WL": "W" if pts > opp_pts else "L", "PTS": pts, "OPP_PTS": opp_pts, "POSS": round(poss, 2),
                "OFF_RATING": pts / poss * 100, "DEF_RATING": opp_pts / poss * 100,
                "NET_RATING": (pts - opp_pts) / poss * 100,
            })
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows), pd.DataFrame(adv_rows)


def build_recaps(player_reg, trades, teams):
    names = dict(zip(teams.TEAM_ID, teams.TEAM_NAME))
    first_after = {}
    for t in trades.itertuples():
        after = player_reg[(player_reg.PLAYER_ID == t.PLAYER_ID) & (player_reg.GAME_DATE >= t.TRADE_DATE)]
        if len(after):
            first_after[after.sort_values("GAME_DATE").GAME_ID.iloc[0], t.PLAYER_ID] = t.FROM_TEAM_ID
    rows = []
    for i, r in enumerate(player_reg.itertuples()):
        date = r.GAME_DATE.strftime("%-d %B %Y")
        text = (f"{r.PLAYER_NAME} scored {r.PTS} points on {r.FGM}-of-{r.FGA} shooting with {r.REB} rebounds "
                f"and {r.AST} assists in {r.MIN:.0f} minutes for the {names[r.TEAM_ID]} ({r.MATCHUP}) on {date}.")
        old = first_after.get((r.GAME_ID, r.PLAYER_ID))
        if old is not None:
            text += f" It was his first game for the {names[r.TEAM_ID]} since the trade from the {names[old]}."
        rows.append({"RECAP_ID": f"r{i:06d}", "GAME_ID": r.GAME_ID, "GAME_DATE": r.GAME_DATE,
                     "PLAYER_ID": r.PLAYER_ID, "TEAM_ID": r.TEAM_ID, "TEXT": text})
    return pd.DataFrame(rows)


def build_paragraphs(players, trades, teams):
    names = dict(zip(teams.TEAM_ID, teams.TEAM_NAME))
    rows = []

    def add(date, pid, kind, text):
        n = len(rows)
        rows.append({"PARA_ID": f"p{n:04d}", "DATE": pd.Timestamp(date), "PLAYER_ID": pid, "KIND": kind,
                     "SOURCE": "Sample source (synthetic)", "URL": f"https://example.com/news/p{n:04d}", "TEXT": text})

    for t in trades.itertuples():
        name = players.loc[players.PLAYER_ID == t.PLAYER_ID, "PLAYER_NAME"].iloc[0]
        add(t.TRADE_DATE, t.PLAYER_ID, "trade",
            f"The {names[t.FROM_TEAM_ID]} traded {name} to the {names[t.TO_TEAM_ID]} for two reserves. "
            f"League sources said the {names[t.TO_TEAM_ID]} wanted more spacing on the second unit.")
        add(t.TRADE_DATE + pd.Timedelta(days=9), t.PLAYER_ID, "role",
            f"{name} said the new role suits him. \"I'm getting cleaner looks here,\" he said. "
            f"His coach said he expects {name.split()[0]} to play about 32 minutes a night.")
    for p in players.sample(40, random_state=11).itertuples():
        date = pd.Timestamp("2025-11-01") + pd.Timedelta(days=int(RNG.integers(0, 150)))
        part = random.choice(["left ankle soreness", "right hamstring tightness", "illness", "low back spasms"])
        status = random.choice(["Questionable", "Out", "Probable", "Doubtful"])
        add(date, p.PLAYER_ID, "injury", f"Injury report: {p.PLAYER_NAME} - {status} ({part}).")
    for p in players.sample(20, random_state=13).itertuples():
        date = pd.Timestamp("2025-11-15") + pd.Timedelta(days=int(RNG.integers(0, 120)))
        add(date, p.PLAYER_ID, "role",
            f"{p.PLAYER_NAME} has moved into the starting lineup. Teammates praised his defense, "
            f"and one scout called him a 20-point scorer in waiting.")
    return pd.DataFrame(rows)


def build_contracts(players, player_reg, adv_reg):
    agg = player_reg.groupby("PLAYER_ID").agg(GP=("GAME_ID", "nunique"), MPG=("MIN", "mean"),
                                              PPG=("PTS", "mean"), FGA=("FGA", "sum"), FTA=("FTA", "sum"),
                                              PTS=("PTS", "sum"))
    agg["TS"] = agg.PTS / (2 * (agg.FGA + 0.44 * agg.FTA))
    agg["USG"] = adv_reg.groupby("PLAYER_ID").USG_PCT.mean()
    rows = []
    for p in players.itertuples():
        a = agg.loc[p.PLAYER_ID]
        prior = float(RNG.uniform(1.5e6, 35e6)) if RNG.random() < 0.7 else np.nan
        pay = (1.2e6 + 0.9e6 * a.PPG + 0.25e6 * a.MPG + 40e6 * (a.TS - 0.55) + 12e6 * (a.USG - 0.2)
               - 0.12e6 * (p.AGE - 27) ** 2 + (0.3 * prior if prior == prior else 0) + RNG.normal(0, 2.5e6))
        rows.append({"PLAYER_ID": p.PLAYER_ID, "SEASON": "2026-27", "ANNUAL_PAY": round(max(1.2e6, pay), -3),
                     "YEARS": int(RNG.integers(1, 6)), "PRIOR_PAY": None if prior != prior else round(prior, -3)})
    return pd.DataFrame(rows)


def ts(frame):
    return float(frame.PTS.sum() / (2 * (frame.FGA.sum() + 0.44 * frame.FTA.sum())))


def build_eval(players, teams, trades, player_reg, adv_reg, team_reg, paragraphs, recaps, contracts):
    pname = dict(zip(players.PLAYER_ID, players.PLAYER_NAME))
    q = []

    def add(kind, text, expect):
        q.append({"id": f"q{len(q) + 1:02d}", "type": kind, "question": text, "expect": expect})

    for t in trades.head(5).itertuples():
        rows = player_reg[player_reg.PLAYER_ID == t.PLAYER_ID]
        before, after = rows[rows.GAME_DATE < t.TRADE_DATE], rows[rows.GAME_DATE >= t.TRADE_DATE]
        add("trade_split", f"Did {pname[t.PLAYER_ID]}'s true shooting change after he was traded this season?",
            {"metric": "trade_split", "player_id": int(t.PLAYER_ID),
             "values": {"before_games": len(before), "after_games": len(after),
                        "before_ts": round(ts(before), 4), "after_ts": round(ts(after), 4)}})
    for p in players.sample(5, random_state=21).itertuples():
        rows = player_reg[(player_reg.PLAYER_ID == p.PLAYER_ID) & (player_reg.GAME_DATE.dt.month == 12)]
        poss = adv_reg[adv_reg.GAME_ID.isin(rows.GAME_ID) & (adv_reg.PLAYER_ID == p.PLAYER_ID)].POSS.sum()
        add("rate", f"What were {p.PLAYER_NAME}'s points per game and per 100 possessions in December this season?",
            {"metric": "rate", "player_id": int(p.PLAYER_ID),
             "values": {"games": len(rows), "pts_per_game": round(rows.PTS.mean(), 4),
                        "pts_per_100": round(rows.PTS.sum() / poss * 100, 4)}})
    for tm in teams.sample(5, random_state=31).itertuples():
        rows = team_reg[team_reg.TEAM_ID == tm.TEAM_ID]
        cut = pd.Timestamp("2026-01-15")
        b, a = rows[rows.GAME_DATE < cut], rows[rows.GAME_DATE >= cut]
        net = lambda f: (f.PTS.sum() - f.OPP_PTS.sum()) / f.POSS.sum() * 100
        add("team_window", f"What was the {tm.TEAM_NAME}' net rating before and after 15 January 2026 this season?".replace("' ", "'s "),
            {"metric": "team_window", "team_id": int(tm.TEAM_ID),
             "values": {"before_games": len(b), "after_games": len(a),
                        "before_net": round(net(b), 4), "after_net": round(net(a), 4)}})
    inj = paragraphs[paragraphs.KIND == "injury"].head(3)
    for r in inj.itertuples():
        wk = r.DATE - pd.Timedelta(days=r.DATE.weekday())
        add("context", f"What did the injury report say about {pname[r.PLAYER_ID]} in the week of "
                       f"{wk.strftime('%-d %B %Y')} this season?",
            {"metric": "context", "player_id": int(r.PLAYER_ID), "relevant_para_ids": [r.PARA_ID]})
    for t in trades.iloc[5:7].itertuples():
        ids = paragraphs[(paragraphs.PLAYER_ID == t.PLAYER_ID) & (paragraphs.KIND.isin(["trade", "role"]))].PARA_ID
        add("context", f"What did saved reports say about {pname[t.PLAYER_ID]}'s trade and new role this season?",
            {"metric": "context", "player_id": int(t.PLAYER_ID), "relevant_para_ids": ids.tolist()})
    held = json.loads((EVAL / "contracts_heldout.json").read_text())
    for h in held[:4]:
        add("signing", f"What annual salary would a team have to offer to sign {pname[h['PLAYER_ID']]} "
                       f"after this season?",
            {"metric": "signing", "player_id": h["PLAYER_ID"], "actual_annual_pay": h["ANNUAL_PAY"]})
    some = players.iloc[3].PLAYER_NAME
    add("followup", "How did he shoot after the trade?", {"followup_field": "who"})
    add("followup", f"What was {some}'s scoring rate per 100 possessions?", {"followup_field": "season"})
    add("followup", f"Will {some} win MVP this season?", {"followup_field": "metric"})
    t = trades.iloc[7]
    add("planted", f"Did {pname[t.PLAYER_ID]}'s true shooting change after he was traded this season?",
        {"metric": "trade_split", "plant": "playoffs", "expect_check": "playoff_mix"})
    t = trades.iloc[8]
    add("planted", f"Did {pname[t.PLAYER_ID]}'s true shooting change after he was traded this season?",
        {"metric": "trade_split", "plant": "wrong_team", "expect_check": "wrong_team"})
    r = inj.iloc[0]
    add("planted", f"What did the injury report say about {pname[r.PLAYER_ID]} this season?",
        {"metric": "context", "plant": "invented_quote", "expect_check": "quote_not_saved"})
    (EVAL / "questions.json").write_text(json.dumps(q, indent=2))

    rank = []
    eval_players = {e["expect"].get("player_id") for e in q}
    for t in trades.itertuples():
        split = "test" if t.PLAYER_ID in eval_players else "train"
        mine = recaps[recaps.PLAYER_ID == t.PLAYER_ID]
        for query in (f"Did {pname[t.PLAYER_ID]} play better after the trade?",
                      f"{pname[t.PLAYER_ID]} first game with new team after trade"):
            for r in mine.itertuples():
                label = int("since the trade" in r.TEXT)
                if label or RNG.random() < 0.08:
                    rank.append({"split": split, "query": query, "doc_id": r.RECAP_ID, "text": r.TEXT, "label": label})
    for pid, group in paragraphs.groupby("PLAYER_ID"):
        split = "test" if pid in eval_players else "train"
        for kind, query in (("injury", f"What did the injury report say about {pname[pid]}?"),
                            ("trade", f"What did reports say about {pname[pid]}'s trade?"),
                            ("role", f"What did reports say about {pname[pid]}'s new role?")):
            for r in group.itertuples():
                rank.append({"split": split, "query": query, "doc_id": r.PARA_ID, "text": r.TEXT,
                             "label": int(r.KIND == kind)})
            for r in recaps[recaps.PLAYER_ID == pid].sample(3, random_state=1).itertuples():
                rank.append({"split": split, "query": query, "doc_id": r.RECAP_ID, "text": r.TEXT, "label": 0})
    pd.DataFrame(rank).to_csv(EVAL / "rank_labels.csv", index=False)

    from checks import strip_for_numbers
    from models import evidence_text

    templates = [
        ("{name}'s true shooting was {a:.3f} over {g} games before the trade [W:before].", "ratio"),
        ("Over {g} regular-season games, {name} scored {a:.1f} points per game [W:window].", "ppg"),
        ("Over the same games he scored {a:.1f} points per 100 possessions [W:window].", "p100"),
        ("The team had a net rating of {a:+.1f} over {g} games before the date [W:before].", "net"),
        ("Predicted annual pay to sign {name}: ${m:.2f}M [M:signing].", "pay"),
        ("This is a model prediction, with a mean absolute error of ${m:.2f}M on held-out contracts [M:signing].", "mae"),
    ]
    ranges = {"ratio": (0.45, 0.7), "ppg": (3, 32), "p100": (8, 45), "net": (-12, 12), "pay": (1.2e6, 45e6), "mae": (1e6, 5e6)}
    support = []
    for i in range(800):
        text, kind = templates[i % len(templates)]
        lo, hi = ranges[kind]
        true = float(RNG.uniform(lo, hi))
        games = int(RNG.integers(5, 80))
        values = [true, games] + ([true / 1e6] if kind in ("pay", "mae") else [])
        values += [float(RNG.uniform(lo, hi)) for _ in range(2)]
        good = i % 2 == 0
        shown = true if good else true * float(RNG.choice([0.7, 0.8, 1.25, 1.4])) + (0.0 if kind in ("pay", "mae") else 1.0)
        sentence = text.format(name=random.choice(players.PLAYER_NAME.tolist()), a=shown, g=games, m=shown / 1e6)
        support.append({"split": "test" if i < 200 else "train", "sentence": strip_for_numbers(sentence),
                        "evidence": evidence_text(values), "label": int(good)})
    pd.DataFrame(support).to_csv(EVAL / "support_labels.csv", index=False)


def main():
    for d in (INFO, NEWS, EVAL, SEASON_DIR):
        d.mkdir(parents=True, exist_ok=True)
    teams = build_teams()
    players = build_players(teams)
    trades = build_trades(players, teams)
    reg_games = build_games(N_REG, REG_START, REG_DAYS, "00225", teams.TEAM_ID.to_numpy())
    po_games = build_games(N_PO, PO_START, PO_DAYS, "00425", teams.TEAM_ID.to_numpy()[:16])
    player_reg, team_reg, adv_reg = play_games(reg_games, "Regular Season", players, teams, trades)
    player_po, team_po, adv_po = play_games(po_games, "Playoffs", players, teams, trades)
    recaps = build_recaps(player_reg, trades, teams)
    paragraphs = build_paragraphs(players, trades, teams)
    contracts = build_contracts(players, player_reg, adv_reg)

    held_ids = set(contracts.sample(frac=0.2, random_state=51).PLAYER_ID)
    contracts["SPLIT"] = np.where(contracts.PLAYER_ID.isin(held_ids), "heldout", "train")
    held = contracts[contracts.SPLIT == "heldout"][["PLAYER_ID", "ANNUAL_PAY"]]
    (EVAL / "contracts_heldout.json").write_text(json.dumps(held.to_dict("records"), indent=2))

    out = {
        "teams": teams, "players": players[["PLAYER_ID", "PLAYER_NAME", "AGE"]], "trades": trades,
        "contracts": contracts, "paragraphs": paragraphs, "recaps": recaps,
        "player_regular": player_reg, "player_playoffs": player_po, "team_regular": team_reg,
        "team_playoffs": team_po, "advanced_regular": adv_reg, "advanced_playoffs": adv_po,
    }
    for name, frame in out.items():
        frame.to_parquet(FILES[name], index=False)
    build_eval(players, teams, trades, player_reg, adv_reg, team_reg, paragraphs, recaps, contracts)
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
