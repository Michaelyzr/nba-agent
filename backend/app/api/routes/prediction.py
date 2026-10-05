from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.prediction.service import (
    prediction_service,
)


router = APIRouter()


# ============================================================
# Create Player Forecast
# ============================================================

@router.post(
    "/players/{nba_player_id}/forecast"
)
async def create_player_forecast(
    nba_player_id: int,

    history_games: int = Query(
        default=10,
        ge=3,
        le=30,
    ),

    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    为指定球员下一场比赛生成 baseline forecast。

    当前预测：
    - Minutes
    - Points
    """

    try:
        result = (
            await prediction_service
            .create_player_forecast(
                session=db,
                nba_player_id=(
                    nba_player_id
                ),
                history_games=(
                    history_games
                ),
            )
        )

        return {
            "status": "ok",
            "data": result,
        }

    except ValueError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc),
            },
        ) from exc