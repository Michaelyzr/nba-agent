from uuid import UUID

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
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


class Rule(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Reviewer Agent 提出的经验规则。

    只有通过 Rule Validation Gate 后，
    status 才能变成 active。
    """

    __tablename__ = "rules"

    review_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "reviews.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # 条件
    #
    # {
    #   "ruled_out_position": "C",
    #   "hours_to_tip_max": 2
    # }
    conditions: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    # 行为
    #
    # {
    #   "backup_minutes_share": 0.70
    # }
    actions: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    # proposed / active / rejected / retired
    status: Mapped[str] = mapped_column(
        String(30),
        default="proposed",
        nullable=False,
        index=True,
    )

    approved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    evidence_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class RuleBacktest(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Rule Validation Gate 的历史回测结果。

    Agent 不能批准自己的 Rule。
    Gate 通过代码评估后才能批准。
    """

    __tablename__ = "rule_backtests"

    rule_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "rules.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    cases_tested: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    cases_improved: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    calibration_before: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    calibration_after: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    clv_before: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    clv_after: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    details: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )