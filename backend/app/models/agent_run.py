from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class AgentRun(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    每一次 Agent 执行记录。

    用于：
    - 可审计性
    - Debug
    - Token / latency 统计
    - 模型版本追踪
    """

    __tablename__ = "agent_runs"

    game_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "games.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # forecaster / trader / reviewer
    agent_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    # OpenAI
    model_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # gpt-5 ...
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="running",
        nullable=False,
        index=True,
    )

    input_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    output_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )