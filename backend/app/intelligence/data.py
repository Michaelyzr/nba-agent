"""Historical NBA results, with date-only timestamps explicitly identified."""
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
from sqlalchemy import select

from app.models.game import Game
from app.models.team import Team


def normalize_results(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["GAME_ID"]).zfill(10)].append(row)
    games = []
    for game_id, pair in groups.items():
        if not game_id.startswith("002") or len(pair) != 2:
            continue
        home = next((r for r in pair if "vs." in r.get("MATCHUP", "")), None)
        away = next((r for r in pair if "@" in r.get("MATCHUP", "")), None)
        if home is None or away is None or int(home["TEAM_ID"]) == int(away["TEAM_ID"]):
            continue
        if home.get("PTS") is None or away.get("PTS") is None:
            continue
        hs, aws = int(home["PTS"]), int(away["PTS"])
        if hs < 0 or aws < 0 or hs == aws:
            continue
        raw_date = str(home["GAME_DATE"])[:10]
        # The endpoint lacks exact tipoff time. Next local midnight is a
        # conservative upper bound for ordinary games on this NBA calendar date.
        local = datetime.strptime(raw_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/New_York"))
        tipoff_proxy = (local + timedelta(days=1)).astimezone(timezone.utc)
        games.append({"nba_game_id": game_id, "home_nba_id": int(home["TEAM_ID"]),
                      "away_nba_id": int(away["TEAM_ID"]), "tipoff_proxy": tipoff_proxy,
                      "home_score": hs, "away_score": aws})
    return games


async def sync_results(db, season):
    def fetch():
        endpoint = LeagueGameLog(player_or_team_abbreviation="T", season=season,
                                season_type_all_star="Regular Season", league_id="00", timeout=30)
        return endpoint.league_game_log.get_data_frame().to_dict(orient="records")
    rows = await asyncio.to_thread(fetch)
    normalized = normalize_results(rows)
    teams = {t.nba_team_id: t for t in (await db.scalars(select(Team))).all()}
    if not teams:
        raise ValueError("请先同步球队目录")
    existing = {g.nba_game_id: g for g in (await db.scalars(select(Game).where(Game.season == season))).all()}
    created, updated, skipped = 0, 0, 0
    now = datetime.now(timezone.utc)
    for data in normalized:
        home, away = teams.get(data["home_nba_id"]), teams.get(data["away_nba_id"])
        if not home or not away or data["tipoff_proxy"] > now:
            skipped += 1
            continue
        game = existing.get(data["nba_game_id"])
        if game is None:
            game = Game(nba_game_id=data["nba_game_id"], season=season,
                        season_type="Regular Season", home_team_id=home.id, away_team_id=away.id,
                        tipoff_time=data["tipoff_proxy"], tipoff_time_estimated=True, status="final")
            db.add(game)
            existing[game.nba_game_id] = game
            created += 1
        else:
            if game.home_team_id != home.id or game.away_team_id != away.id:
                skipped += 1
                continue
            game.status = "final"
            updated += 1
        game.home_score = data["home_score"]
        game.away_score = data["away_score"]
    await db.commit()
    return {"season": season, "source": "NBA Stats LeagueGameLog", "created": created,
            "updated": updated, "skipped": skipped, "rows_received": len(rows),
            "time_policy": "New rows use next midnight America/New_York as estimated tipoff; schedule sync replaces it with actual tipoff."}
