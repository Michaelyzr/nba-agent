from contextlib import asynccontextmanager
import asyncio
import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage a cancellable source collector and dispose the database engine."""

    logging.getLogger(__name__).info("Starting %s (%s)", settings.app_name, settings.app_env)

    from app.news.collector import poll_news
    # Isolated browser fixtures must never collect real network data.
    worker = asyncio.create_task(poll_news()) if os.environ.get("NBA_QA_FIXTURE") != "1" else None
    try:
        yield
    finally:
        if worker:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    from app.db.session import engine
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for NBA Game-Impact Intelligence."
    ),
    version="1.0.0",
    debug=settings.app_debug,
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API
# ============================================================

app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)


# ============================================================
# Root
# ============================================================

@app.get("/")
async def root() -> dict:

    return {
        "service": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }
