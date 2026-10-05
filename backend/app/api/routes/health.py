from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.core.config import settings
from app.db.session import AsyncSessionLocal


router = APIRouter()


@router.get("")
async def health_check() -> dict:
    """
    FastAPI 服务健康检查。
    """

    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/db")
async def database_health_check() -> dict:
    """
    PostgreSQL 数据库连接检查。

    通过执行 SELECT 1 验证：
    FastAPI -> SQLAlchemy -> asyncpg -> PostgreSQL
    整条数据库链路是否正常。
    """

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("SELECT 1")
            )

            value = result.scalar_one()

        return {
            "status": "ok",
            "database": "postgresql",
            "connected": value == 1,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "database": "postgresql",
                "connected": False,
                "message": str(exc),
            },
        ) from exc