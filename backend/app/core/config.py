from functools import lru_cache
from pathlib import Path
from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Project Paths
# ============================================================

# config.py:
# backend/app/core/config.py
#
# parents[0] = core
# parents[1] = app
# parents[2] = backend
BACKEND_DIR = Path(__file__).resolve().parents[2]

# 固定读取：
# backend/.env
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """
    NBA Game-Impact Intelligence 后端全局配置。
    """

    # ========================================================
    # Application
    # ========================================================

    app_name: str = "NBA Game-Impact Intelligence API"
    app_env: str = "development"
    app_debug: bool = True

    api_v1_prefix: str = "/api/v1"

    # ========================================================
    # OpenAI
    # ========================================================

    openai_api_key: str = ""
    openai_model: str = "gpt-5"

    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 3

    # ========================================================
    # Database
    # ========================================================

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/nba_agent"
    )

    # ========================================================
    # NBA
    # ========================================================

    nba_season: str = "2026-27"

    # ========================================================
    # Frontend
    # ========================================================

    frontend_origin: str = "http://localhost:3000"

    news_recent_hours: int = Field(168, ge=1, le=8760)
    news_review_after_hours: int = Field(168, ge=1, le=8760)
    news_poll_seconds: int = Field(900, ge=60, le=86400)
    news_auto_collect: bool = True
    news_auto_extract: bool = False
    news_extract_limit: int = Field(3, ge=1, le=10)
    injury_status_max_hours: int = Field(24, ge=1, le=168)
    injury_training_min_games: int = Field(200, ge=100, le=10000)

    # Paper research only. No wallet credentials or live execution.
    elo_k: float = Field(20.0, gt=0, le=100, allow_inf_nan=False)
    elo_home_advantage: float = Field(65.0, ge=0, le=200, allow_inf_nan=False)
    elo_season_retention: float = Field(0.75, ge=0, le=1, allow_inf_nan=False)
    min_team_history: int = Field(10, ge=1)
    market_max_age_seconds: int = Field(900, ge=1, le=86400)
    market_min_liquidity: float = Field(1000.0, gt=0, allow_inf_nan=False)
    market_max_spread: float = Field(0.05, gt=0, lt=1, allow_inf_nan=False)
    signal_min_edge: float = Field(0.05, gt=0, lt=1, allow_inf_nan=False)
    paper_cost_buffer: float = Field(0.01, ge=0, lt=1, allow_inf_nan=False)
    paper_max_stake: float = Field(100.0, gt=0, allow_inf_nan=False)
    paper_max_exposure: float = Field(500.0, gt=0, allow_inf_nan=False)

    # ========================================================
    # Pydantic Settings
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
