from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class Player(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    NBA 球员基本信息。
    """

    __tablename__ = "players"

    # NBA 官方球员 ID
    nba_player_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        index=True,
        nullable=False,
    )

    full_name: Mapped[str] = mapped_column(
        String(200),
        index=True,
        nullable=False,
    )

    first_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    last_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # PG / SG / SF / PF / C
    position: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    # 当前球队
    # 自由球员时允许为空
    current_team_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "teams.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

