from uuid import UUID

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class PlayerGameStat(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    球员单场比赛统计数据。

    后续：
    - rolling average
    - gradient boosting
    - GRU
    - Forecaster Agent

    都会从这里读取数据。
    """

    __tablename__ = "player_game_stats"

    # 一个球员在一场比赛原则上只能有一条记录
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "player_id",
            name="uq_player_game_stats_game_player",
        ),
    )

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "games.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    player_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "players.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    team_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    starter: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # ========================================================
    # Basic Box Score
    # ========================================================

    minutes: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    points: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rebounds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    assists: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    steals: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    blocks: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    turnovers: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # Shooting
    # ========================================================

    field_goals_made: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    field_goals_attempted: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    three_points_made: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    three_points_attempted: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    free_throws_made: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    free_throws_attempted: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # Advanced
    # ========================================================

    usage_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    true_shooting_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    offensive_rating: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    defensive_rating: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    pace: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    plus_minus: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )