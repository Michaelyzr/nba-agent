from fastapi import APIRouter

from app.api.routes import (
    analysis,
    evidence,
    agents,
    ai,
    health,
    intelligence,
    nba,
    news,
    prediction,
)


api_router = APIRouter()
api_router.include_router(analysis.router, prefix="/analysis", tags=["NBA Analysis"])
api_router.include_router(evidence.router, prefix="/evidence", tags=["News Evidence"])

api_router.include_router(intelligence.router, prefix="/intelligence", tags=["Game Intelligence"])


# ============================================================
# System
# ============================================================

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["System"],
)


# ============================================================
# AI
# ============================================================

api_router.include_router(
    ai.router,
    prefix="/ai",
    tags=["AI"],
)


# ============================================================
# NBA Data
# ============================================================

api_router.include_router(
    nba.router,
    prefix="/nba",
    tags=["NBA Data"],
)


# ============================================================
# Prediction
# ============================================================

api_router.include_router(
    prediction.router,
    prefix="/prediction",
    tags=["Prediction"],
)

# ============================================================
# Agents
# ============================================================

api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["Agents"],
)

# ============================================================
# News
# ============================================================

api_router.include_router(
    news.router,
    prefix="/news",
    tags=["News"],
)
