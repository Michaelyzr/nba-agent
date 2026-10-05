from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

from app.nba.service import nba_data_service
from app.nba.games_service import nba_games_service
from app.nba.stats_service import nba_stats_service


router = APIRouter()


# ============================================================
# Sync Teams
# ============================================================

@router.post("/sync/teams")
async def sync_nba_teams(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    同步 NBA 30 支球队。

    NBA API
        ↓
    NBADataService
        ↓
    PostgreSQL teams
    """

    try:
        result = await nba_data_service.sync_teams(
            session=db,
        )

        return {
            "status": "ok",
            "message": (
                "NBA teams synchronized successfully."
            ),
            "data": result,
        }

    except Exception as exc:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Sync Players
# ============================================================

@router.post("/sync/players")
async def sync_nba_players(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    同步 NBA active players。

    NBA API
        ↓
    NBADataService
        ↓
    PostgreSQL players
    """

    try:
        result = await nba_data_service.sync_players(
            session=db,
        )

        return {
            "status": "ok",
            "message": (
                "NBA players synchronized successfully."
            ),
            "data": result,
        }

    except Exception as exc:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Sync Rosters
# ============================================================

@router.post("/sync/rosters")
async def sync_nba_rosters(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    同步 NBA 当前球员阵容关系。

    更新：
    - Player.current_team_id
    - Player.position
    """

    try:
        result = await nba_data_service.sync_rosters(
            session=db,
        )

        return {
            "status": "ok",
            "message": (
                "NBA rosters synchronized successfully."
            ),
            "data": result,
        }

    except Exception as exc:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Sync Games
# ============================================================

@router.post("/sync/games")
async def sync_nba_games(
    season: str = Query(
        default="2026-27",
        pattern=r"^\d{4}-\d{2}$",
        description=(
            "NBA season. "
            "Example: 2025-26 or 2026-27"
        ),
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    同步指定 NBA 赛季的比赛。

    例如：

    season = 2025-26
        -> 获取 2025-26 赛季历史比赛

    season = 2026-27
        -> 获取当前赛季比赛
    """

    try:
        result = await nba_games_service.sync_games(
            session=db,
            season=season,
        )

        return {
            "status": "ok",
            "message": (
                "NBA games synchronized successfully."
            ),
            "data": result,
        }

    except Exception as exc:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Sync Player Game Stats
# ============================================================

@router.post("/sync/player-game-stats")
async def sync_player_game_stats(
    season: str = Query(
        default="2026-27",
        pattern=r"^\d{4}-\d{2}$",
        description=(
            "NBA season. "
            "Example: 2025-26 or 2026-27"
        ),
    ),

    season_type: str = Query(
        default="Pre Season",
        pattern=(
            r"^(Pre Season|Regular Season|Playoffs)$"
        ),
        description=(
            "Pre Season, Regular Season, "
            "or Playoffs"
        ),
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    同步指定赛季的球员逐场比赛数据。

    示例：

    历史常规赛：
    season = 2025-26
    season_type = Regular Season

    当前季前赛：
    season = 2026-27
    season_type = Pre Season
    """

    try:
        result = (
            await nba_stats_service
            .sync_player_game_stats(
                session=db,
                season=season,
                season_type=season_type,
            )
        )

        return {
            "status": "ok",
            "message": (
                "NBA player game stats "
                "synchronized successfully."
            ),
            "data": result,
        }

    except Exception as exc:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Get Teams
# ============================================================

@router.get("/teams")
async def get_nba_teams(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    获取数据库中的 NBA 球队。
    """

    try:
        teams = await nba_data_service.get_teams(
            session=db,
        )

        return {
            "status": "ok",
            "count": len(teams),

            "data": [
                {
                    "id": str(team.id),

                    "nba_team_id": (
                        team.nba_team_id
                    ),

                    "abbreviation": (
                        team.abbreviation
                    ),

                    "city": team.city,

                    "name": team.name,

                    "full_name": (
                        team.full_name
                    ),

                    "conference": (
                        team.conference
                    ),

                    "division": (
                        team.division
                    ),

                    "active": (
                        team.active
                    ),
                }

                for team in teams
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Get Players
# ============================================================

@router.get("/players")
async def get_nba_players(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

    active_only: bool = Query(
        default=True,
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    获取数据库中的 NBA 球员。
    """

    try:
        players = (
            await nba_data_service.get_players(
                session=db,
                limit=limit,
                active_only=active_only,
            )
        )

        return {
            "status": "ok",
            "count": len(players),

            "data": [
                {
                    "id": str(
                        player.id
                    ),

                    "nba_player_id": (
                        player.nba_player_id
                    ),

                    "full_name": (
                        player.full_name
                    ),

                    "first_name": (
                        player.first_name
                    ),

                    "last_name": (
                        player.last_name
                    ),

                    "position": (
                        player.position
                    ),

                    "current_team_id": (
                        str(
                            player.current_team_id
                        )
                        if player.current_team_id
                        else None
                    ),

                    "active": (
                        player.active
                    ),
                }

                for player in players
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Get Games
# ============================================================

@router.get("/games")
async def get_nba_games(
    limit: int = Query(
        default=100,
        ge=1,
        le=2000,
    ),

    season: str | None = Query(
        default=None,
        pattern=r"^\d{4}-\d{2}$",
        description=(
            "Optional season filter. "
            "Example: 2025-26"
        ),
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    获取数据库中的 NBA 比赛。

    可以按 season 筛选。
    """

    try:
        games = await nba_games_service.get_games(
            session=db,
            limit=limit,
            season=season,
        )

        return {
            "status": "ok",
            "count": len(games),

            "data": [
                {
                    "id": str(
                        game.id
                    ),

                    "nba_game_id": (
                        game.nba_game_id
                    ),

                    "season": (
                        game.season
                    ),

                    "season_type": (
                        game.season_type
                    ),

                    "home_team_id": str(
                        game.home_team_id
                    ),

                    "away_team_id": str(
                        game.away_team_id
                    ),

                    "tipoff_time": (
                        game.tipoff_time.isoformat()
                    ),

                    "status": (
                        game.status
                    ),

                    "home_score": (
                        game.home_score
                    ),

                    "away_score": (
                        game.away_score
                    ),
                }

                for game in games
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# Get Player Game Stats
# ============================================================

@router.get("/player-game-stats")
async def get_player_game_stats(
    limit: int = Query(
        default=100,
        ge=1,
        le=2000,
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    获取数据库中的球员逐场比赛统计。
    """

    try:
        stats = (
            await nba_stats_service
            .get_player_game_stats(
                session=db,
                limit=limit,
            )
        )

        return {
            "status": "ok",
            "count": len(stats),

            "data": [
                {
                    "id": str(
                        stat.id
                    ),

                    "game_id": str(
                        stat.game_id
                    ),

                    "player_id": str(
                        stat.player_id
                    ),

                    "team_id": str(
                        stat.team_id
                    ),

                    "starter": (
                        stat.starter
                    ),

                    # =========================================
                    # Basic
                    # =========================================

                    "minutes": (
                        stat.minutes
                    ),

                    "points": (
                        stat.points
                    ),

                    "rebounds": (
                        stat.rebounds
                    ),

                    "assists": (
                        stat.assists
                    ),

                    "steals": (
                        stat.steals
                    ),

                    "blocks": (
                        stat.blocks
                    ),

                    "turnovers": (
                        stat.turnovers
                    ),

                    # =========================================
                    # Shooting
                    # =========================================

                    "field_goals_made": (
                        stat.field_goals_made
                    ),

                    "field_goals_attempted": (
                        stat.field_goals_attempted
                    ),

                    "three_points_made": (
                        stat.three_points_made
                    ),

                    "three_points_attempted": (
                        stat.three_points_attempted
                    ),

                    "free_throws_made": (
                        stat.free_throws_made
                    ),

                    "free_throws_attempted": (
                        stat.free_throws_attempted
                    ),

                    # =========================================
                    # Advanced
                    # =========================================

                    "usage_pct": (
                        stat.usage_pct
                    ),

                    "true_shooting_pct": (
                        stat.true_shooting_pct
                    ),

                    "offensive_rating": (
                        stat.offensive_rating
                    ),

                    "defensive_rating": (
                        stat.defensive_rating
                    ),

                    "pace": (
                        stat.pace
                    ),

                    "plus_minus": (
                        stat.plus_minus
                    ),
                }

                for stat in stats
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc