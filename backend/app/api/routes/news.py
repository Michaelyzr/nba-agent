from sqlalchemy import select

from app.models.team import Team
from app.news.web_fetcher import (
    web_news_fetcher,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.nba.news_service import (
    nba_news_service,
)
from app.schemas.news import (
    NewsEventCreate,
)


router = APIRouter()


# ============================================================
# Create News Event
# ============================================================

@router.post("/events")
async def create_news_event(
    payload: NewsEventCreate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    手动创建结构化 NBA 新闻事件。

    当前用于验证整个 News Pipeline。

    后续真实新闻 Provider
    也会通过同一套 Service 写入数据库。
    """

    try:
        event = (
            await nba_news_service
            .create_event(
                session=db,
                payload=payload,
            )
        )

        return {
            "status": "ok",

            "data": {
                "id": str(event.id),

                "game_id": (
                    str(event.game_id)
                    if event.game_id
                    else None
                ),

                "player_id": (
                    str(event.player_id)
                    if event.player_id
                    else None
                ),

                "team_id": (
                    str(event.team_id)
                    if event.team_id
                    else None
                ),

                "event_type": (
                    event.event_type
                ),

                "player_status": (
                    event.player_status
                ),

                "title": event.title,

                "body": event.body,

                "source": event.source,

                "source_url": (
                    event.source_url
                ),

                "published_at": (
                    event
                    .published_at
                    .isoformat()
                ),

                "ingested_at": (
                    event
                    .ingested_at
                    .isoformat()
                ),
            },
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
# Get Recent Events
# ============================================================

@router.get("/events")
async def get_news_events(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    db: AsyncSession = Depends(get_db),
) -> dict:

    events = (
        await nba_news_service
        .get_recent_events(
            session=db,
            limit=limit,
        )
    )

    return {
        "status": "ok",
        "count": len(events),

        "data": [
            {
                "id": str(event.id),

                "event_type": (
                    event.event_type
                ),

                "player_status": (
                    event.player_status
                ),

                "title": event.title,

                "body": event.body,

                "source": event.source,

                "published_at": (
                    event
                    .published_at
                    .isoformat()
                ),
            }

            for event in events
        ],
    }
# ============================================================
# Search Player News From Web
# ============================================================

@router.post(
    "/search/player/{nba_player_id}"
)
async def search_player_news(
    nba_player_id: int,

    lookback_hours: int = Query(
        default=72,
        ge=1,
        le=168,
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    自动联网搜索某 NBA 球员的最新新闻。

    当前阶段：

    Internet
        ↓
    OpenAI Web Search
        ↓
    Raw News

    暂时不会写 news_events。

    下一阶段加入：
    News Extraction Agent
        ↓
    news_events
    """

    try:
        # ====================================================
        # Player
        # ====================================================

        player = (
            await nba_news_service
            .get_player_by_nba_id(
                session=db,
                nba_player_id=(
                    nba_player_id
                ),
            )
        )

        # ====================================================
        # Team
        # ====================================================

        team_name = None

        if (
            player.current_team_id
            is not None
        ):
            team_result = (
                await db.execute(
                    select(Team).where(
                        Team.id
                        == player.current_team_id
                    )
                )
            )

            team = (
                team_result
                .scalar_one_or_none()
            )

            if team is not None:
                team_name = (
                    team.full_name
                )

        # ====================================================
        # Web Search
        # ====================================================

        result = (
            await web_news_fetcher
            .fetch_player_news(
                player_name=(
                    player.full_name
                ),

                team_name=(
                    team_name
                ),

                lookback_hours=(
                    lookback_hours
                ),
            )
        )

        return {
            "status": "ok",

            "data": {
                "nba_player_id": (
                    player.nba_player_id
                ),

                **result,
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc