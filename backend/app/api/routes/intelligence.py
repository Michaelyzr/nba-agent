import logging
import requests
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.intelligence import service
from app.intelligence.data import sync_results
from app.intelligence.market import MarketDataError
from app.models.game import Game
from app.models.intelligence import GamePrediction, PaperPosition
from app.models.market import MarketSnapshot
from app.models.news import NewsEvent
from app.models.team import Team
from app.schemas.intelligence import MarketRequest, PaperRequest

router = APIRouter()
logger = logging.getLogger(__name__)


async def result(operation):
    try:
        return {"status": "ok", "data": await operation}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except (httpx.HTTPError, requests.RequestException, MarketDataError) as exc:
        raise HTTPException(502, detail="外部数据请求失败，请检查数据标识与网络后重试") from exc
    except SQLAlchemyError as exc:
        logger.warning("Intelligence database operation failed: %s", type(exc).__name__)
        raise HTTPException(503, detail="数据库不可用或迁移尚未执行，请检查 PostgreSQL 并运行 Alembic") from exc


@router.get("/config")
async def config():
    return {"status": "ok", "data": {
        "model": "Elo v1", "calibrated": False, "live_trading": False,
        "openai_configured": bool(settings.openai_api_key), "season": settings.nba_season,
        "min_team_history": settings.min_team_history,
        "signal_min_edge": settings.signal_min_edge,
        "market_max_age_seconds": settings.market_max_age_seconds,
        "market_min_liquidity": settings.market_min_liquidity,
        "market_max_spread": settings.market_max_spread,
        "paper_cost_buffer": settings.paper_cost_buffer,
        "paper_max_stake": settings.paper_max_stake, "paper_max_exposure": settings.paper_max_exposure,
        "news_recent_hours": settings.news_recent_hours,
        "news_review_after_hours": settings.news_review_after_hours,
    }}


@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)):
    async def operation():
        counts = {}
        for model in (Team, Game, NewsEvent, GamePrediction, MarketSnapshot, PaperPosition):
            counts[model.__tablename__] = await db.scalar(select(func.count()).select_from(model))
        return counts
    return await result(operation())


@router.get("/games")
async def games(season: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
                limit: int = Query(100, ge=1, le=500), db: AsyncSession = Depends(get_db)):
    return await result(service.list_games(db, season, limit))


@router.post("/games/{game_id}/run")
async def run(game_id: UUID, db: AsyncSession = Depends(get_db)):
    return await result(service.run_game(db, game_id))


@router.post("/games/{game_id}/markets")
async def market(game_id: UUID, request: MarketRequest, db: AsyncSession = Depends(get_db)):
    return await result(service.save_market(db, game_id, request))


@router.get("/backtest")
async def backtest(season: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
                   db: AsyncSession = Depends(get_db)):
    return await result(service.backtest(db, season))


@router.post("/sync/results")
async def historical_results(season: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
                              db: AsyncSession = Depends(get_db)):
    return await result(sync_results(db, season))


@router.get("/paper")
async def paper(db: AsyncSession = Depends(get_db)):
    async def operation():
        positions = (await db.scalars(select(PaperPosition).order_by(PaperPosition.created_at.desc()))).all()
        return {"positions": [service.position_data(p) for p in positions],
                "open_exposure": sum(p.stake for p in positions if p.status == "open"),
                "realized_pnl": sum(p.pnl or 0 for p in positions if p.status == "settled")}
    return await result(operation())


@router.post("/paper")
async def open_paper(request: PaperRequest, db: AsyncSession = Depends(get_db)):
    return await result(service.open_paper(db, request))


@router.post("/paper/settle")
async def settle(db: AsyncSession = Depends(get_db)):
    return await result(service.settle_paper(db))
