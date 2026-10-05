from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class NewsEvent(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    NBA 临场新闻事件。

    例如：
    - injury
    - rest
    - ruled_out
    - questionable
    - starting_lineup
    - minutes_restriction
    - role_change
    - trade
    """

    __tablename__ = "news_events"

    # 新闻可能对应具体比赛
    game_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "games.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # 可能对应具体球员
    player_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "players.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # 可能对应球队
    team_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "teams.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # injury / rest / lineup / role_change ...
    event_type: Mapped[str] = mapped_column(
        String(50),
        index=True,
        nullable=False,
    )

    # out / questionable / probable / available ...
    player_status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    body: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # NBA.com / ESPN / Team Report ...
    source: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    source_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # 新闻原始发布时间
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )

    # 我们系统什么时候抓到新闻
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )