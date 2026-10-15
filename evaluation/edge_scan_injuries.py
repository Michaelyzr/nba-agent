"""Part 3a: do official NBA injury-report status changes leave a tradable Kalshi move after report_time + 5 min?

Reads runs/edge_scan/injury_rows.parquet (evaluation/edge_scan_parse_injuries.py). For each status change between
consecutive reports for a rotation player (usual minutes >= MIN_MINUTES) on a game in the next 36 h, the implied
direction is: worse status (-> Out/Doubtful) = buy NO on that team, better (-> Available/Probable) = buy YES.
Moves are the team's Kalshi mid in that direction relative to the mid at report_time.
    python -m evaluation.edge_scan_injuries   -> runs/edge_scan/injury_changes.parquet, injury_moves.csv, PNG
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from evaluation.edge_scan_common import RESULTS, RUNS, Quotes, day_boot, fee, frozen

MIN_MINUTES = 20.0
HORIZONS = (0, 5, 15, 30, 60)
RANK = {"Available": 0, "Probable": 1, "Questionable": 2, "Doubtful": 3, "Out": 4}
TEAMS = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
         "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
         "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
         "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
         "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
         "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
         "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
         "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"}


def norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name)
    return re.sub(r"[^a-z]", "", name)


def report_name(raw: str) -> str:
    raw = re.sub(r"-\s+", "-", raw)
    last, _, first = raw.partition(",")
    return norm(f"{first} {last}")


def team_abbrev(full: str, abbrevs: set) -> str | None:
    full = " ".join(full.split())
    a = TEAMS.get(full)
    alias = {"BKN": "BRK", "GSW": "GS", "NOP": "NO", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHX": "PHO",
             "NYK": "NY"}
    if a in abbrevs:
        return a
    for k, v in alias.items():
        if a == k and v in abbrevs:
            return v
    return None


def changes(rows: pd.DataFrame, q: Quotes) -> pd.DataFrame:
    games = q.games[(q.games.date >= "2026-01-30") & (q.games.date <= "2026-04-14")]
    abbrevs = set(games.home_team) | set(games.away_team)
    rows = rows.assign(abbr=[team_abbrev(t or "", abbrevs) for t in rows.team], key=rows.player.map(report_name))
    rows = rows.dropna(subset=["abbr"])
    sched = pd.concat([games.reset_index()[["game_id", "tip_time", "home_team"]].rename(columns={"home_team": "abbr"}),
                       games.reset_index()[["game_id", "tip_time", "away_team"]].rename(columns={"away_team": "abbr"})])
    sched = sched.sort_values("tip_time")
    rows = pd.merge_asof(rows.sort_values("report_time"), sched.rename(columns={"tip_time": "tip"}),
                         left_on="report_time", right_on="tip", by="abbr", direction="forward")
    rows = rows[(rows.tip - rows.report_time) <= pd.Timedelta(hours=36)]
    files = sorted(rows.report_time.unique())
    nxt = dict(zip(files[:-1], files[1:]))
    rows = rows.sort_values("report_time").drop_duplicates(["game_id", "abbr", "key", "report_time"])
    out = []
    for (gid, abbr, key), g in rows.groupby(["game_id", "abbr", "key"], sort=False):
        st, ts, pl, tip = g.status.to_numpy(), g.report_time.tolist(), g.player.iloc[0], g.tip.iloc[0]
        for i in range(1, len(st)):
            if st[i] != st[i - 1]:
                out.append({"game_id": gid, "team": abbr, "key": key, "player": pl, "from_status": st[i - 1],
                            "to": st[i], "report_time": ts[i], "tip": tip})
        # dropping off the list before tip = cleared to play
        n = nxt.get(ts[-1])
        if n is not None and pd.Timestamp(n) < tip and st[-1] != "Available":
            out.append({"game_id": gid, "team": abbr, "key": key, "player": pl, "from_status": st[-1],
                        "to": "Removed", "report_time": pd.Timestamp(n), "tip": tip})
    return pd.DataFrame(out)


def usual_minutes(ch: pd.DataFrame) -> pd.Series:
    pg = frozen("player_games")
    pl = frozen("players")
    pl = pl.assign(key=pl.player_name.map(norm))
    g = frozen("games")
    pg = pg.merge(g[["game_id", "date", "season"]], on="game_id")
    pg = pg[pg.season == "2025-26"]
    mins = pg.groupby("player_id")["min"].mean()
    played = pg[pg["min"] > 0].groupby("player_id")["min"].mean()
    ids = pl.drop_duplicates("key").set_index("key").player_id
    return ch.key.map(ids).map(played.combine_first(mins))


def measure(ch: pd.DataFrame, q: Quotes, news: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in ch.itertuples():
        t = q.ticker.get((str(r.game_id), r.team))
        if t is None:
            continue
        side = "no" if r.to in ("Out", "Doubtful") else "yes"
        sign = -1 if side == "no" else 1
        base = q.mid(t, r.report_time)
        close = q.mid(t, r.tip)
        if base is None or close is None:
            continue
        rec = {"ticker": t, "side": side, "base_mid": base, "close_move": sign * (close - base)}
        pre = q.mid(t, r.report_time - pd.Timedelta(minutes=60))
        rec["pre_move_60"] = sign * (base - pre) if pre is not None else np.nan
        for h in HORIZONS[1:]:
            when = r.report_time + pd.Timedelta(minutes=h)
            m = q.mid(t, min(when, r.tip))
            rec[f"move_{h}"] = sign * (m - base) if m is not None else np.nan
        entry_t = r.report_time + pd.Timedelta(minutes=5)
        if entry_t < r.tip:
            qq = q.quote(t, entry_t)
            if qq is not None:
                price = qq[1] if side == "yes" else 1 - qq[0]
                rec["entry_price"] = price
                rec["half_spread"] = (qq[1] - qq[0]) / 2
                rec["fee"] = fee(price)
                rec["clv_5"] = (close if side == "yes" else 1 - close) - price - fee(price)
        rec["lead_vs_espn_min"] = (r.tip - pd.Timedelta(minutes=30) - r.report_time).total_seconds() / 60
        recs.append({**r._asdict(), **rec})
    return pd.DataFrame(recs)


def chart(df: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [0, 5, 15, 30, 60, 90]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, sub, c in (("-> Out/Doubtful", df[df.side == "no"], "tab:red"),
                          ("-> Out/Doubtful, usual min >= 28", df[(df.side == "no") & (df.usual_min >= 28)], "darkred"),
                          ("-> Available/Probable", df[(df.side == "yes") & (df.to != "Removed")], "tab:green")):
        ys = [0] + [sub[f"move_{h}"].mean() * 100 for h in HORIZONS[1:]] + [sub.close_move.mean() * 100]
        ax.plot(xs, ys, marker="o", color=c, label=f"{label} (n={len(sub)})")
    band = (df.fee + df.half_spread).mean() * 100
    ax.axhspan(-band, band, color="grey", alpha=0.2, label=f"fee + half spread ({band:.1f}c)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xs, ["report", "+5", "+15", "+30", "+60", "tip"])
    ax.set_ylabel("mid move in implied direction (cents)")
    ax.set_title("Kalshi mid after official injury-report status changes (rotation players)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    rows = pd.read_parquet(RUNS / "injury_rows.parquet")
    rows["report_time"] = pd.to_datetime(rows.report_time, utc=True)
    q = Quotes()
    news = frozen("news")
    news = news.assign(game_id=news.game_id.astype(str))
    ch = changes(rows, q)
    ch["usual_min"] = usual_minutes(ch)
    print(f"raw status changes: {len(ch)}; matched to a usual-minutes figure: {ch.usual_min.notna().sum()}")
    ch = ch[ch.usual_min >= MIN_MINUTES]
    ch = ch[~((ch["from_status"].isin(["Out", "Doubtful"])) & (ch.to.isin(["Out", "Doubtful"])))]
    ch = ch[~((ch["from_status"] == "Probable") & (ch.to.isin(["Available", "Removed"])))]
    ch = ch[~((ch["from_status"] == "Available") & (ch.to == "Removed"))]
    df = measure(ch, q, news)
    df["date"] = df.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    df.to_parquet(RUNS / "injury_changes.parquet", index=False)
    out = []
    for label, sub in (("all", df), ("-> Out/Doubtful", df[df.side == "no"]),
                       ("-> Available/Probable (explicit)", df[(df.side == "yes") & (df.to != "Removed")]),
                       ("dropped off the list (noisy)", df[df.to == "Removed"]),
                       ("usual min >= 28", df[df.usual_min >= 28]),
                       ("-> Out/Doubtful, usual min >= 28", df[(df.side == "no") & (df.usual_min >= 28)]),
                       ("Questionable -> Out, usual min >= 28",
                        df[(df["from_status"] == "Questionable") & (df.to == "Out") & (df.usual_min >= 28)]),
                       ("Questionable -> Available, usual min >= 28",
                        df[(df["from_status"] == "Questionable") & (df.to == "Available") & (df.usual_min >= 28)]),
                       ("report < 3h before tip", df[(df.tip - df.report_time) < pd.Timedelta(hours=3)])):
        row = {"part": "3a", "analysis": "injury_status_change", "bucket": label, "split": "test",
               **day_boot(sub.clv_5, sub.date)}
        for h in HORIZONS[1:]:
            row[f"move_{h}"] = sub[f"move_{h}"].mean()
        row["close_move"] = sub.close_move.mean()
        row["pre_move_60"] = sub.pre_move_60.mean()
        b = day_boot(sub.close_move - sub.move_5, sub.date)
        row["after5_lo"], row["after5_hi"] = b["lo"], b["hi"]
        row["move_after_5_to_close"] = (sub.close_move - sub.move_5).mean()
        row["cost_fee_half_spread"] = (sub.fee + sub.half_spread).mean()
        row["lead_vs_espn_median_min"] = sub.lead_vs_espn_min.median()
        row["n_pos_clv"] = int((sub.clv_5 > 0).sum())
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(RUNS / "injury_moves.csv", index=False)
    chart(df.dropna(subset=["clv_5"]), RESULTS / "edge_scan_injuries.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"changes: {len(ch)} rotation-player status changes, {len(df)} with quotes")
        print(res.round(4).to_string())


if __name__ == "__main__":
    main()
