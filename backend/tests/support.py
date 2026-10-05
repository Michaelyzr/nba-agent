"""SQLite adapter for offline workflow tests, not a production DB implementation.

PostgreSQL lock statements are bypassed here. Lock/concurrency semantics require
a separate PostgreSQL integration environment and are not covered by this adapter.
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.game import Game
from app.models.team import Team
import app.models  # noqa: F401


@compiles(JSONB, "sqlite")
def json_for_test_sqlite(type_, compiler, **kwargs):
    return "JSON"


def aware(obj):
    if obj is not None and hasattr(obj, "__table__"):
        for col in obj.__table__.columns:
            value = getattr(obj, col.name)
            if isinstance(value, datetime) and value.tzinfo is None:
                setattr(obj, col.name, value.replace(tzinfo=timezone.utc))
    return obj


class Rows:
    def __init__(self, rows): self.rows = rows
    def all(self): return self.rows


class TestDB:
    __test__ = False

    def __init__(self):
        from tempfile import TemporaryDirectory
        from app.core.config import BACKEND_DIR
        from app.news.archive import NewsArchive
        root = BACKEND_DIR.parent / ".cache" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = TemporaryDirectory(dir=root)
        self.news_archive = NewsArchive(root / self.directory.name / "news.sqlite3")
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine, expire_on_commit=False)

    def add(self, obj): self.session.add(obj)
    async def get(self, model, pk): return aware(self.session.get(model, pk))
    async def scalar(self, statement): return aware(self.session.scalar(statement))
    async def scalars(self, statement): return Rows([aware(r) for r in self.session.scalars(statement).all()])
    async def execute(self, statement):
        if "pg_advisory_xact_lock" in str(statement): return Rows([])
        return Rows([tuple(aware(obj) for obj in row) for row in self.session.execute(statement).all()])
    async def flush(self): self.session.flush()
    async def commit(self): self.session.commit()
    async def rollback(self): self.session.rollback()
    def close(self):
        self.session.close(); self.engine.dispose(); self.directory.cleanup()


def seed(db):
    now = datetime.now(timezone.utc)
    teams = [Team(id=uuid4(), nba_team_id=i, abbreviation=f"T{i}", city="Test",
                  name=f"Team {i}", full_name=f"Test Team {i}", active=True) for i in (1, 2)]
    db.session.add_all(teams)
    db.session.flush()
    for i in range(40):
        db.add(Game(id=uuid4(), nba_game_id=f"002250{i:04d}", season="2025-26", season_type="Regular Season",
                    home_team_id=teams[0].id, away_team_id=teams[1].id,
                    tipoff_time=now - timedelta(days=90-i*2), status="final", home_score=110, away_score=100))
    upcoming = Game(id=uuid4(), nba_game_id="0022600001", season="2026-27", season_type="Regular Season",
                    home_team_id=teams[0].id, away_team_id=teams[1].id,
                    tipoff_time=now + timedelta(days=1), status="scheduled")
    db.add(upcoming)
    db.session.commit()
    return upcoming
