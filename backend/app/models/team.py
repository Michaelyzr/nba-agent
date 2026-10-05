from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class Team(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    NBA 球队。

    使用内部 UUID 作为主键，
    同时保存 NBA 官方 team_id。
    """

    __tablename__ = "teams"

    # NBA 官方 ID
    nba_team_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        index=True,
        nullable=False,
    )

    # 例如：
    # LAL
    # GSW
    abbreviation: Mapped[str] = mapped_column(
        String(10),
        unique=True,
        index=True,
        nullable=False,
    )

    # Los Angeles
    city: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # Lakers
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # Los Angeles Lakers
    full_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    # East / West
    conference: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Pacific / Atlantic ...
    division: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    