from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class MarketSnapshot(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Prediction Market 某个时间点的价格快照。

    数据源可能包括：
    - Kalshi
    - Polymarket
    - Historical replay dataset
    """

    __tablename__ = "market_snapshots"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "games.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    # game_winner / player_points / player_minutes ...
    market_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    # Exchange 自己的 contract ID
    contract_id: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    # 例如：
    # Lakers YES
    # Reaves OVER 22.5
    selection: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    line: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # 0 ~ 1
    market_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    bid_price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    ask_price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    liquidity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    raw_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )