from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class GamePrediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "game_predictions"
    game_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("games.id"), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    home_probability: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    features: Mapped[dict] = mapped_column(JSONB)
    output: Mapped[dict] = mapped_column(JSONB)


class GameSignal(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "game_signals"
    prediction_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("game_predictions.id"), index=True)
    snapshot_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("market_snapshots.id"))
    selection: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(30))
    decision: Mapped[dict] = mapped_column(JSONB)


class PaperPosition(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "paper_positions"
    __table_args__ = (UniqueConstraint("game_id", "selection", name="uq_paper_game_selection"),)
    game_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("games.id"), index=True)
    signal_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("game_signals.id"))
    selection: Mapped[str] = mapped_column(String(10))
    stake: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    shares: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="open")
    pnl: Mapped[float | None] = mapped_column(Float)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
