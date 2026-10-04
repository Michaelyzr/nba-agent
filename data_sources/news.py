"""D2: late-news table from the NBA's official injury-report PDFs.

    python -m data_sources.news --start 2026-02-01 --end 2026-02-07          # download + parse
    python -m data_sources.news --start 2026-02-01 --end 2026-02-07 --every 15

Needs data/frozen/games.parquet and players.parquet from data_sources.nba_stats
to attach game_id and player_id. Writes news.parquet (one row per status
change) and tip_times.parquet (tip-off times printed in the reports).

Reports are published several times a day; a row's published_at is the report's
nominal time plus REPORT_LAG, so the replay never sees a report early.
"""
import argparse
from datetime import date, timedelta

import pandas as pd
import requests

from data_sources import ET, FROZEN, RAW, et_to_utc, name_key, player_lookup, read_table, write_table

BASE = "https://ak-static.cms.nba.com/referee/injury/"
CACHE = RAW / "injury"
REPORT_LAG = pd.Timedelta(minutes=15)
COLUMNS = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]
CARRY = ["GameDate", "GameTime", "Matchup", "Team"]
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}


def report_names(day: date, hour: int, minute: int) -> list[str]:
    """Filenames for one report time; the season uses both 15-minute and hourly names."""
    h12, ampm = (hour % 12) or 12, "AM" if hour < 12 else "PM"
    names = [f"Injury-Report_{day:%Y-%m-%d}_{h12:02d}_{minute:02d}{ampm}.pdf"]
    if minute == 0:
        names.append(f"Injury-Report_{day:%Y-%m-%d}_{h12:02d}{ampm}.pdf")
    return names


def report_times(day: date, first="11:00", last="22:00", every=30):
    start = pd.Timestamp(f"{day} {first}")
    end = pd.Timestamp(f"{day} {last}")
    return list(pd.date_range(start, end, freq=f"{every}min"))


def download(day: date, when: pd.Timestamp):
    """Cached path of the report at local time `when`, or None if the league did not publish one."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in report_names(day, when.hour, when.minute):
        path, marker = CACHE / name, CACHE / f"{name}.missing"
        if path.exists():
            return path
        if marker.exists():
            continue
        r = requests.get(BASE + name, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            path.write_bytes(r.content)
            return path
        marker.touch()
    return None


def rows_from_words(pages_words) -> list[dict]:
    """Rebuild report rows from pdfplumber words, one list of words per page.

    Columns are assigned by x position under the header, which only the first
    page prints. Game, time, matchup and team are blank on continuation lines
    and carry forward, across pages too. A wrapped reason is printed centred on
    its player line, so a fragment just above a player line belongs to it.
    """
    rows, state, starts = [], {c: "" for c in CARRY}, None
    for words in pages_words:
        header = {w["text"]: w for w in words if w["text"] in COLUMNS}
        if len(header) == len(COLUMNS):
            starts = sorted((header[c]["x0"] - 3, c) for c in COLUMNS)
            skip_above = header["GameDate"]["top"]
        else:
            skip_above = max((w["top"] for w in words if w["text"] == "Report:"), default=-1)
        if starts is None:
            continue
        lines = {}
        for w in words:
            if w["top"] <= skip_above + 2 or w["text"].startswith("Page"):
                continue
            col = [c for x, c in starts if w["x0"] >= x]
            if col:
                lines.setdefault(round(w["top"]), {}).setdefault(col[-1], []).append(w["text"])
        pending = []
        for top in sorted(lines):
            cells = {c: " ".join(v) for c, v in lines[top].items()}
            if set(cells) == {"Reason"}:
                pending.append((top, cells["Reason"]))
                continue
            for c in CARRY:
                if cells.get(c):
                    state[c] = cells[c]
            if cells.get("PlayerName") and cells.get("CurrentStatus") in STATUSES:
                before = [t for y, t in pending if top - y < 12]
                if rows:
                    rows[-1]["Reason"] = " ".join([rows[-1]["Reason"]] + [t for y, t in pending if top - y >= 12]).strip()
                rows.append({**state, "PlayerName": cells["PlayerName"], "CurrentStatus": cells["CurrentStatus"],
                             "Reason": " ".join(before + [cells.get("Reason", "")]).strip()})
                pending = []
        if rows and pending:
            rows[-1]["Reason"] = " ".join([rows[-1]["Reason"]] + [t for _, t in pending]).strip()
    return rows


def parse_pdf(path) -> list[dict]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return rows_from_words([p.extract_words() for p in pdf.pages])


def to_frames(raw: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame):
    """Attach ids, keep one row per status change, and collect printed tip-off times."""
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw.GameDate, format="%m/%d/%Y").dt.date.astype(str)
    raw[["away_team", "home_team"]] = raw.Matchup.str.split("@", expand=True)
    # Report times omit AM/PM; NBA tip-offs are noon or later Eastern.
    clock = raw.GameTime.str.replace("(ET)", "", regex=False).str.strip()
    raw["tip_time"] = [et_to_utc(d, f"{c} PM") for d, c in zip(raw.date, clock)]

    tips = raw.drop_duplicates(["date", "home_team", "away_team"])[["date", "home_team", "away_team", "tip_time"]]
    raw = raw.merge(games[["game_id", "date", "home_team", "away_team"]], on=["date", "home_team", "away_team"],
                    how="left")
    lookup = player_lookup(players)
    raw["player_id"] = raw.PlayerName.map(lambda n: lookup.get(name_key(n)))
    unmatched = raw[raw.player_id.isna() | raw.game_id.isna()]
    if len(unmatched):
        print(f"  {len(unmatched)} rows without a game or player match, e.g. "
              f"{unmatched[['PlayerName', 'Matchup', 'date']].drop_duplicates().head(5).values.tolist()}")
    raw = raw.dropna(subset=["player_id", "game_id"]).astype({"player_id": int})

    raw = raw.sort_values("published_at")
    changed = raw.CurrentStatus != raw.groupby(["game_id", "player_id"]).CurrentStatus.shift()
    news = raw[changed].copy()
    news["status"] = news.CurrentStatus.str.lower()
    news["source"] = "nba_injury_report"
    news["text"] = news.PlayerName + " (" + news.Team + ") listed " + news.CurrentStatus + ": " + news.Reason
    news["news_id"] = ("inj-" + news.game_id + "-" + news.player_id.astype(str) + "-"
                       + news.published_at.dt.strftime("%Y%m%d%H%M"))
    return news[["news_id", "published_at", "game_id", "player_id", "status", "source", "url", "text"]], tips


def inactive_news(inactive: pd.DataFrame, games: pd.DataFrame, lead=pd.Timedelta(minutes=30)) -> pd.DataFrame:
    """Fallback for games with no report: each inactive player becomes 'out' news 30 minutes before tip."""
    rows = inactive.merge(games[["game_id", "tip_time"]], on="game_id")
    return pd.DataFrame({
        "news_id": "inactive-" + rows.game_id + "-" + rows.player_id.astype(str),
        "published_at": rows.tip_time - lead, "game_id": rows.game_id, "player_id": rows.player_id,
        "status": "out", "source": "inactive_list", "url": "", "text": "listed inactive"})


def build(start: date, end: date, every=30, first="11:00", last="22:00"):
    games, players = read_table("games"), read_table("players")
    raw = []
    day = start
    while day <= end:
        for when in report_times(day, first, last, every):
            path = download(day, when)
            if path is None:
                continue
            published = when.tz_localize(ET).tz_convert("UTC") + REPORT_LAG
            raw += [{**r, "published_at": published, "url": BASE + path.name} for r in parse_pdf(path)]
        print(f"  {day}: {len(raw)} report rows so far")
        day += timedelta(days=1)
    if not raw:
        raise SystemExit("no reports found for that range")
    news, tips = to_frames(pd.DataFrame(raw), games, players)
    write_table("news", news)
    tips.to_parquet(FROZEN / "tip_times.parquet", index=False)
    print(f"wrote {len(news)} news rows and {len(tips)} tip-off times")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--every", type=int, default=30, help="minutes between reports to fetch (15, 30 or 60)")
    ap.add_argument("--first", default="11:00", help="first report time, Eastern")
    ap.add_argument("--last", default="22:00", help="last report time, Eastern")
    args = ap.parse_args()
    build(args.start, args.end, args.every, args.first, args.last)


if __name__ == "__main__":
    main()
