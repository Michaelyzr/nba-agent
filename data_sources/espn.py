"""D1/D2 fallback: freeze games, box scores and inactive lists from ESPN's public JSON.

    python -m data_sources.espn --seasons 2025-26 --limit-days 3      # smoke test
    python -m data_sources.espn                                       # 2023-24 to 2025-26

Use this when stats.nba.com and the NBA injury PDFs are blocked (they block
cloud and VPN addresses; ESPN does not). Writes games, player_games, players and
news in the section 5 formats. game_id is ESPN's event id; team_id is the NBA
team id; player_id is ESPN's athlete id. Every response is cached under
data/raw/espn/, so rerun to resume.

News is the README fallback: each game's inactive list (players who did not
play for a reason other than a coach's decision), stamped NEWS_LEAD before
tip-off, the latest time teams must submit it. Real injury reports come out
earlier, so this understates how early news is public.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from data_sources import GAME_LENGTH, RAW, get_json, write_table

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE = RAW / "espn"
SEASONS = {"2023-24": ("2023-10-24", "2024-06-20"), "2024-25": ("2024-10-22", "2025-06-25"),
           "2025-26": ("2025-10-21", "2026-06-30")}
GAME_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NEWS_LEAD = pd.Timedelta(minutes=30)
NOT_NEWS = ("COACH", "DNP-CD", "NOT WITH TEAM")      # reasons that are not injury or rest news


def _cached_json(path, url, params=None) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    data = get_json(url, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def nba_team_ids() -> dict:
    from nba_api.stats.static import teams
    return {t["abbreviation"]: t["id"] for t in teams.get_teams()}


def tricode(espn_abbr: str) -> str:
    return ESPN_TO_NBA.get(espn_abbr, espn_abbr)


def scoreboard(day: date) -> list[dict]:
    key = day.strftime("%Y%m%d")
    data = _cached_json(CACHE / "scoreboard" / f"{key}.json", f"{BASE}/scoreboard", {"dates": key, "limit": 100})
    return data.get("events", [])


def summary(event_id: str) -> dict:
    return _cached_json(CACHE / "summary" / f"{event_id}.json", f"{BASE}/summary", {"event": event_id})


def game_row(event: dict, team_ids: dict, season: str) -> dict | None:
    comp = event["competitions"][0]
    if event["season"]["type"] not in GAME_TYPES or comp["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    side = {c["homeAway"]: c for c in comp["competitors"]}
    home, away = tricode(side["home"]["team"]["abbreviation"]), tricode(side["away"]["team"]["abbreviation"])
    if home not in team_ids or away not in team_ids:
        return None                                   # All-Star and exhibition games
    tip = pd.Timestamp(event["date"])
    return {"game_id": str(event["id"]), "date": str(tip.tz_convert("America/New_York").date()),
            "tip_time": tip, "final_at": tip + GAME_LENGTH, "season": season,
            "season_type": GAME_TYPES[event["season"]["type"]],
            "home_team_id": team_ids[home], "away_team_id": team_ids[away], "home_team": home, "away_team": away,
            "home_pts": int(side["home"]["score"]), "away_pts": int(side["away"]["score"])}


def _made_attempted(value: str) -> int:
    try:
        return int(value.split("-")[1])
    except (IndexError, ValueError):
        return 0


def box_rows(game: dict, data: dict, team_ids: dict):
    """(player_games rows, players rows, news rows) for one finished game."""
    played, people, news = [], [], []
    for team in data.get("boxscore", {}).get("players", []):
        team_id = team_ids[tricode(team["team"]["abbreviation"])]
        block = team["statistics"][0]
        keys = block.get("keys") or []
        for a in block.get("athletes", []):
            if "id" not in a.get("athlete", {}):
                continue
            pid, name = int(a["athlete"]["id"]), a["athlete"]["displayName"]
            people.append({"player_id": pid, "player_name": name})
            stats = dict(zip(keys, a.get("stats") or []))
            if a.get("didNotPlay") or not stats:
                reason = (a.get("reason") or "").strip().upper()
                if reason and not any(r in reason for r in NOT_NEWS):
                    news.append({"news_id": f"espn-{game['game_id']}-{pid}",
                                 "published_at": game["tip_time"] - NEWS_LEAD, "game_id": game["game_id"],
                                 "player_id": pid, "team_id": team_id, "status": "out", "source": "espn_inactive",
                                 "url": f"https://www.espn.com/nba/boxscore/_/gameId/{game['game_id']}",
                                 "text": f"{name} ({tricode(team['team']['abbreviation'])}) out: {reason.title()}"})
                continue
            played.append({"game_id": game["game_id"], "player_id": pid, "team_id": team_id,
                           "min": pd.to_numeric(stats.get("minutes"), errors="coerce"),
                           "pts": pd.to_numeric(stats.get("points"), errors="coerce"),
                           "fga": _made_attempted(stats.get("fieldGoalsMade-fieldGoalsAttempted", "")),
                           "fta": _made_attempted(stats.get("freeThrowsMade-freeThrowsAttempted", "")),
                           "usage": float("nan"), "started": bool(a.get("starter"))})
    return played, people, news


def freeze(seasons=tuple(SEASONS), limit_days=None, workers=8):
    team_ids = nba_team_ids()
    games = []
    for season in seasons:
        start, end = (date.fromisoformat(d) for d in SEASONS[season])
        days = [start + timedelta(days=i) for i in range((min(end, date.today()) - start).days + 1)][:limit_days]
        with ThreadPoolExecutor(workers) as pool:
            boards = list(pool.map(scoreboard, days))
        found = [g for events in boards for e in events if (g := game_row(e, team_ids, season))]
        print(f"{season}: {len(found)} games over {len(days)} days")
        games += found
    games = pd.DataFrame(games).drop_duplicates("game_id").sort_values(["tip_time", "game_id"]).reset_index(drop=True)

    with ThreadPoolExecutor(workers) as pool:
        summaries = list(pool.map(summary, games.game_id))
    played, people, news = [], [], []
    for game, data in zip(games.to_dict("records"), summaries):
        p, n, w = box_rows(game, data, team_ids)
        played += p
        people += n
        news += w
    player_games = pd.DataFrame(played)
    news = pd.DataFrame(news)
    write_table("games", games)
    write_table("player_games", player_games)
    write_table("players", pd.DataFrame(people).drop_duplicates("player_id"))
    write_table("news", news)
    print(f"wrote {len(games)} games, {len(player_games)} player lines, {len(news)} inactive-list items")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS), choices=list(SEASONS))
    ap.add_argument("--limit-days", type=int, default=None, help="only the first N days of each season")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    freeze(args.seasons, args.limit_days, args.workers)


if __name__ == "__main__":
    main()
