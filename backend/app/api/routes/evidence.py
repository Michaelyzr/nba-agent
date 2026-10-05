import asyncio
import logging
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.intelligence.injury import train
from app.news.archive import archive
from app.news.collector import collect, extract_task, import_report

router = APIRouter()
logger = logging.getLogger(__name__)


class Preferences(BaseModel):
    auto_collect: bool
    auto_extract: bool


class Review(BaseModel):
    decision: Literal["approved", "rejected"]


class Report(BaseModel):
    url: str = Field(max_length=500)


async def respond(operation):
    try:
        return {"status": "ok", "data": await operation}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, detail="来源请求失败，请检查网络；未读取到报告不代表没有伤病") from exc
    except Exception as exc:
        logger.warning("Evidence operation failed: %s", type(exc).__name__)
        raise HTTPException(503, detail="采集或本地档案不可用，请检查后端日志和数据连接") from exc


@router.get("/status")
async def status():
    return await respond(asyncio.to_thread(archive.status))


@router.get("/events")
async def events(limit: int = Query(100, ge=1, le=500)):
    return await respond(asyncio.to_thread(archive.events, limit))


@router.get("/documents/{identity}")
async def document(identity: str):
    def read():
        with archive.connection() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (identity,)).fetchone()
        if not row:
            raise ValueError("来源文档不存在")
        return dict(row)
    return await respond(asyncio.to_thread(read))


@router.put("/preferences")
async def preferences(request: Preferences):
    def save():
        archive.put("preferences", request.model_dump())
        return archive.preferences()
    return await respond(asyncio.to_thread(save))


@router.post("/collect")
async def run_collect(season: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
                      db: AsyncSession = Depends(get_db)):
    return await respond(collect(db, season=season, store=getattr(db, "news_archive", None)))


@router.post("/extract")
async def extract(db: AsyncSession = Depends(get_db)):
    # Explicit manual request, separately from free source collection.
    return await respond(extract_task(db, store=getattr(db, "news_archive", None)))


@router.post("/reports")
async def report(request: Report, db: AsyncSession = Depends(get_db)):
    return await respond(import_report(db, request.url, store=getattr(db, "news_archive", None)))


@router.post("/events/{identity}/review")
async def review(identity: str, request: Review):
    async def operation():
        await asyncio.to_thread(archive.review, identity, request.decision)
        return {"id": identity, "review": request.decision}
    return await respond(operation())


@router.post("/train")
async def training(db: AsyncSession = Depends(get_db)):
    return await respond(train(db, store=getattr(db, "news_archive", None)))
