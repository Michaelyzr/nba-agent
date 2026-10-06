"""Providers for the independent in-play loop; never load pregame state/news."""
import pandas as pd

from data_sources.inplay_news import InPlayMedia
from data_sources.inplay_score import LiveScore
from data_sources.inplay_types import InPlayBatch, utc
from data_sources.inplay_x import InPlayX
from data_sources.news_registry import NewsRegistry


class TableInPlay:
    """Replay requires publication AND observation times. No today's HTTP data."""
    def __init__(self, scores, events=None, registry=None):
        self.scores = scores.copy()
        self.events = events.copy() if events is not None else pd.DataFrame()
        self.registry = registry or NewsRegistry.load()

    def fetch(self, game, players, roster, now):
        scores = self.scores[(self.scores.game_id.astype(str) == str(game.game_id))
                             & (pd.to_datetime(self.scores.observed_at, utc=True) <= now)]
        score = None if scores.empty else scores.loc[pd.to_datetime(scores.observed_at, utc=True).idxmax()].to_dict()
        rows = []
        if not self.events.empty:
            mask = ((self.events.game_id.astype(str) == str(game.game_id))
                    & (pd.to_datetime(self.events.published_at, utc=True) >= utc(game.tip_time))
                    & (pd.to_datetime(self.events.published_at, utc=True) <= now)
                    & (pd.to_datetime(self.events.observed_at, utc=True) <= now))
            rows = self.events[mask].to_dict("records")
        return InPlayBatch(score=score, events=rows,
                           coverage=[{"source": "table_inplay", "state": "replay"}])

    def dump_state(self):
        return {}

    def load_state(self, state):
        pass

    def close(self):
        pass


class LiveInPlay:
    def __init__(self, registry=None, nba_game_id=None, x_enabled=True, media_enabled=True,
                 espn_fallback=True, clock=lambda: pd.Timestamp.now(tz="UTC"), score_client=None,
                 x_client=None, media_client=None):
        self.registry = registry or NewsRegistry.load()
        self.clock = clock
        self.score = score_client or LiveScore(nba_game_id, espn_fallback, clock=clock)
        self.x = (x_client or InPlayX(self.registry, clock=clock)) if x_enabled else None
        self.media = (media_client or InPlayMedia(self.registry, clock=clock)) if media_enabled else None

    @property
    def wake(self):
        return self.x.wake if self.x else None

    def dump_state(self):
        return {"score": self.score.dump_state(), "x": self.x.dump_state() if self.x else {},
                "media_last_check": self.media.last_check if self.media else None,
                "media_errors": self.media.last_errors if self.media else [],
                "media_coverage": self.media.last_coverage if self.media else []}

    def load_state(self, state):
        self.score.load_state(state.get("score", {}))
        if self.x:
            self.x.load_state(state.get("x", {}))
        if self.media:
            self.media.last_check = state.get("media_last_check")
            self.media.last_errors = state.get("media_errors", [])
            self.media.last_coverage = state.get("media_coverage", [])

    def fetch(self, game, players, roster, now):
        batch = self.score.fetch(game, players, roster, now)
        if not batch.score or batch.score["phase"] != "live":
            return batch
        current_roster = set(batch.score.get("roster", [])) | roster
        for client in (self.x, self.media):
            if client:
                extra = client.fetch(game, players, current_roster, utc(self.clock()))
                for field in ("events", "evidence", "coverage", "errors"):
                    getattr(batch, field).extend(getattr(extra, field))
        batch.evidence.extend({**e, "review_required": False} for e in batch.events
                              if e["source"] in {"nba_live", "espn_live"})
        return batch

    def close(self):
        if self.x:
            self.x.close()
