from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
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


class TradeProposal(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Market Grader / Trader Agent 提出的交易建议。

    注意：
    Proposal != Trade

    Agent 只能提出交易。
    Risk Engine 批准后才能真正执行。
    """

    __tablename__ = "trade_proposals"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "games.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    forecast_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "forecasts.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    market_snapshot_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "market_snapshots.id",
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

    # buy_yes / buy_no / over / under
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    model_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    market_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    probability_gap: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    proposed_size: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


class RiskCheck(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Risk Engine 对 TradeProposal 的检查结果。

    这是纯代码逻辑产生的，
    不允许 LLM override。
    """

    __tablename__ = "risk_checks"

    trade_proposal_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "trade_proposals.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    # 每个风险检查详细结果
    #
    # {
    #   "trade_size": true,
    #   "daily_exposure": true,
    #   "before_tipoff": true,
    #   "liquidity": true,
    #   "calibration": false
    # }
    checks: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    rejection_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class Trade(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    实际执行或模拟执行的交易。
    """

    __tablename__ = "trades"

    trade_proposal_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "trade_proposals.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    risk_check_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "risk_checks.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    # replay / paper / live
    mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    size: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    entry_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    fees: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    pnl: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    closing_price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    closing_line_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )