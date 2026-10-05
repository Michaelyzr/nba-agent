from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.forecaster import (
    forecaster_agent,
)
from app.db.session import get_db
from app.schemas.forecaster import (
    ForecasterRunRequest,
)


router = APIRouter()


@router.post(
    "/forecaster/players/{nba_player_id}"
)
async def run_forecaster_agent(
    nba_player_id: int,
    request: ForecasterRunRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    运行 Forecaster Agent。

    Baseline
        ↓
    Context
        ↓
    GPT Adjustment
        ↓
    Final Forecast
    """

    try:
        result = await forecaster_agent.run(
            session=db,
            nba_player_id=nba_player_id,
            context=request.context,
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