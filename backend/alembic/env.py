from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# ============================================================
# Import application configuration
# ============================================================

from app.core.config import settings
from app.db.base import Base

# 这一行非常重要：
# 导入所有 SQLAlchemy Models，
# 让 Base.metadata 能够发现所有数据库表。
import app.models  # noqa: F401


# ============================================================
# Alembic Config
# ============================================================

config = context.config


# ============================================================
# Logging
# ============================================================

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ============================================================
# Database URL
# ============================================================

# 不从 alembic.ini 写死数据库密码。
# 直接读取 backend/.env 中的 DATABASE_URL。
config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.replace("%", "%%"),
)


# ============================================================
# SQLAlchemy Metadata
# ============================================================

# Alembic --autogenerate 会比较：
#
# Base.metadata
#       VS
# PostgreSQL 当前数据库结构
#
# 然后自动生成 migration。
target_metadata = Base.metadata


# ============================================================
# Offline Migration
# ============================================================

def run_migrations_offline() -> None:
    """
    Offline migration。

    不真正建立数据库连接，
    主要用于生成 SQL script。
    """

    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================
# Async Online Migration
# ============================================================

def do_run_migrations(connection) -> None:
    """
    Alembic migration 核心执行函数。
    """

    context.configure(
        connection=connection,
        target_metadata=target_metadata,

        # Model column type 改变时，
        # Alembic 能检测出来。
        compare_type=True,

        # Server default 改变也尝试检测。
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    使用 SQLAlchemy Async Engine 连接 PostgreSQL。
    """

    connectable = async_engine_from_config(
        config.get_section(
            config.config_ini_section,
            {}
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(
            do_run_migrations
        )

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Online migration 入口。
    """

    import asyncio

    asyncio.run(
        run_async_migrations()
    )


# ============================================================
# Alembic Entry Point
# ============================================================

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
