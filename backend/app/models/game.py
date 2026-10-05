from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class Game(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    NBA 比赛。
    """

    __tablename__ = "games"

    # NBA 官方 game ID
    # 通常是字符串形式
    nba_game_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
    )

    # 例如 2025-26
    season: Mapped[str] = mapped_column(
        String(20),
        index=True,
        nullable=False,
    )

    # Regular Season / Playoffs
    season_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    home_team_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    away_team_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    # 正式开赛时间
    tipoff_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )

    # scheduled / live / final / postponed
    status: Mapped[str] = mapped_column(
        String(30),
        default="scheduled",
        nullable=False,
        index=True,
    )

    # Historical game logs supply calendar dates, not exact tipoff timestamps.
    tipoff_time_estimated: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)

    home_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    away_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
