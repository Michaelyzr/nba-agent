from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4
import unittest

from app.news.context import load_evidence, select_evidence
from app.models.news import NewsEvent
from app.services.context_service import context_service
from tests.support import TestDB, seed


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 5, tzinfo=timezone.utc)
        self.team, self.player, self.game = uuid4(), uuid4(), uuid4()

    def event(self, kind, days=0, **kwargs):
        when = self.now - timedelta(days=days)
        data = dict(id=uuid4(), event_type=kind, player_id=self.player,
                    team_id=self.team, game_id=None, player_status=None,
                    published_at=when, ingested_at=when)
        data.update(kwargs)
        return SimpleNamespace(**data)

    def select(self, events, **kwargs):
        return select_evidence(events, as_of=self.now, team_ids=[self.team],
                               game_id=self.game, **kwargs)

    def test_old_injury_retained_but_old_ordinary_news_expires(self):
        injury, old_news, recent = self.event("injury", 30), self.event("news", 8), self.event("news", 5)
        rows = self.select([injury, old_news, recent])
        self.assertEqual([r.event.id for r in rows], [injury.id, recent.id])
        self.assertTrue(rows[0].needs_review)
        self.assertFalse(rows[1].persistent)

    def test_return_clears_injury_and_is_visible_as_recent_news(self):
        recovery = self.event("return_from_injury", 1)
        rows = self.select([self.event("injury", 30), recovery])
        self.assertEqual([r.event.id for r in rows], [recovery.id])
        self.assertFalse(rows[0].persistent)

    def test_recovery_of_another_player_does_not_clear_injury(self):
        injury = self.event("injury", 30)
        rows = self.select([injury, self.event("available", 1, player_id=uuid4())])
        self.assertIn(injury.id, [r.event.id for r in rows])

    def test_injury_recovery_does_not_clear_minutes_restriction(self):
        restriction = self.event("minutes_restriction", 14)
        rows = self.select([restriction, self.event("injury", 15), self.event("return_from_injury", 1)])
        self.assertEqual([r.event.id for r in rows if r.persistent], [restriction.id])

    def test_no_future_publication_or_late_ingestion_leakage(self):
        injury = self.event("injury", 30)
        rows = self.select([injury, self.event("available", -1),
                            self.event("return_from_injury", 1, ingested_at=self.now + timedelta(hours=1))])
        self.assertEqual([r.event.id for r in rows], [injury.id])

    def test_later_trade_to_another_team_supersedes_old_trade(self):
        rows = self.select([self.event("trade", 30), self.event("trade", 1, team_id=uuid4())])
        self.assertEqual(rows, [])

    def test_game_scoped_news_is_not_reused_for_another_game(self):
        rows = self.select([self.event("ruled_out", 1, game_id=uuid4())])
        self.assertEqual(rows, [])

    def test_one_game_availability_does_not_clear_global_injury(self):
        injury = self.event("injury", 30)
        rows = self.select([injury, self.event("available", 1, game_id=self.game)])
        self.assertIn(injury.id, [r.event.id for r in rows if r.persistent])

    def test_unnamed_players_are_not_merged_or_resolved_as_one_player(self):
        a, b = self.event("injury", 2, player_id=None), self.event("injury", 1, player_id=None)
        rows = self.select([a, b, self.event("available", player_id=None)])
        self.assertEqual({r.event.id for r in rows if r.persistent}, {a.id, b.id})
        self.assertTrue(all(r.needs_review for r in rows if r.persistent))

    def test_ordinary_news_window_is_configurable(self):
        old = self.event("news", 8)
        self.assertEqual(self.select([old]), [])
        self.assertEqual(self.select([old], lookback_hours=240)[0].event.id, old.id)

    def test_recent_limit_does_not_drop_old_long_term_reports(self):
        old = self.event("suspension", 60)
        rows = self.select([old, *[self.event("news", i / 24) for i in range(40)]])
        self.assertEqual(len(rows), 31)
        self.assertEqual(rows[0].event.id, old.id)

    def test_latest_injury_report_replaces_earlier_duplicate(self):
        latest = self.event("injury", 1)
        rows = self.select([self.event("injury", 30), latest])
        self.assertEqual([r.event.id for r in rows], [latest.id])
        self.assertFalse(rows[0].needs_review)

    def test_suspension_and_restriction_have_independent_resolution(self):
        rows = self.select([self.event("suspension", 30), self.event("minutes_restriction", 14),
                            self.event("suspension_ended", 10), self.event("minutes_restriction_lifted", 10)])
        self.assertEqual(rows, [])


class EvidenceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = TestDB()
        self.game = seed(self.db)
        self.now = datetime.now(timezone.utc)
        from app.models.player import Player
        self.player = Player(id=uuid4(), nba_player_id=12345, full_name="Test Player",
                             current_team_id=self.game.home_team_id, active=True)
        self.db.add(self.player)
        await self.db.commit()

    async def asyncTearDown(self):
        self.db.close()

    async def event(self, kind, days, **kwargs):
        when = self.now - timedelta(days=days)
        values = dict(player_id=self.player.id, team_id=self.game.home_team_id,
                      event_type=kind, title=f"Test {kind}", source="Offline fixture",
                      published_at=when, ingested_at=when)
        values.update(kwargs)
        event = NewsEvent(**values)
        self.db.add(event)
        await self.db.commit()
        return event

    async def evidence(self):
        return await load_evidence(self.db, as_of=self.now, game_id=self.game.id,
                                   team_ids=[self.game.home_team_id, self.game.away_team_id])

    async def test_followup_without_original_team_is_used_to_resolve(self):
        injury = await self.event("injury", 30)
        self.assertEqual((await self.evidence())[0].event.id, injury.id)
        await self.event(" AVAILABLE ", 10, team_id=None)
        self.assertEqual(await self.evidence(), [])

    async def test_prediction_exposes_long_term_status_and_policy(self):
        await self.event("injury", 30)
        from app.intelligence.service import run_game
        output = (await run_game(self.db, self.game.id))["prediction"]
        self.assertEqual(output["news_policy"]["recent_hours"], 168)
        self.assertEqual(output["news_policy"]["persistent_events"], 1)
        self.assertTrue(output["events"][0]["needs_review"])
        self.assertEqual(output["pipeline"][1]["status"], "context_only")

    async def test_player_context_uses_same_long_term_policy(self):
        await self.event("injury", 30)
        context = await context_service.build_player_context(self.db, player=self.player)
        self.assertIn("Test injury", context)
        self.assertIn("needs_review=True", context)
