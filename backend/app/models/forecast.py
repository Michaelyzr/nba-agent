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


class Forecast(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Agent / Prediction Service 生成的预测结果。

    可以表示：
    - 球员是否出场概率
    - minutes distribution
    - points distribution
    - team win probability
    """

    __tablename__ = "forecasts"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "games.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # 球员级预测时填写
    # 比赛级 win probability 可以为空
    player_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "players.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # play_probability / minutes / points / win_probability
    forecast_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    # 最主要的概率或预测值
    predicted_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # 例如：
    #
    # {
    #   "mean": 26.5,
    #   "std": 5.2,
    #   "p10": 19.0,
    #   "p50": 26.0,
    #   "p90": 34.0
    # }
    distribution: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # 模型名称
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # 预测时使用了哪些 feature
    features: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    explanation: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )