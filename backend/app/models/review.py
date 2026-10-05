from uuid import UUID

from sqlalchemy import (
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


class Review(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    比赛结算后 Reviewer Agent 的复盘结果。
    """

    __tablename__ = "reviews"

    trade_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "trades.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    agent_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "agent_runs.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # news_misread
    # minutes_wrong
    # model_wrong
    # market_already_priced
    # correct
    cause: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evidence: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )