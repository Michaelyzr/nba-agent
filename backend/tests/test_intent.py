import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.analysis.intent import QuestionIntent, Target, explicit_scope, resolve_scope
from app.models.player import Player
from app.models.team import Team
from app.schemas.analysis import AnalysisRequest
from tests.support import TestDB, seed


class IntentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db=TestDB();self.game=seed(self.db)
        self.thunder=Team(nba_team_id=10,abbreviation='OKC',city='Oklahoma City',name='Thunder',full_name='Oklahoma City Thunder',active=True)
        self.warriors=Team(nba_team_id=11,abbreviation='GSW',city='Golden State',name='Warriors',full_name='Golden State Warriors',active=True)
        self.db.add(self.thunder);self.db.add(self.warriors);self.db.session.flush()
        self.player=Player(nba_player_id=1628983,full_name='Shai Gilgeous-Alexander',current_team_id=self.thunder.id,active=True)
        self.curry=Player(nba_player_id=201939,full_name='Stephen Curry',current_team_id=self.warriors.id,active=True)
        self.db.add(self.player);self.db.add(self.curry);self.db.session.commit()

    def tearDown(self):self.db.close()

    def payload(self,text,**extra):
        return AnalysisRequest(team_id=self.game.home_team_id,season='2025-26',messages=[dict(role='user',content=text)],**extra)

    async def test_screenshot_question_overrides_wrong_team_and_season_without_model(self):
        with patch('app.analysis.intent.extract_intent',AsyncMock(side_effect=AssertionError('No model needed'))):
            resolved,routing=await resolve_scope(self.db,self.payload('Shai Gilgeous-Alexander在26-27赛季的表现'))
        self.assertEqual(resolved.player_id,self.player.id)
        self.assertEqual(resolved.team_id,self.thunder.id)
        self.assertEqual(resolved.season,'2026-27')
        self.assertTrue(routing['changed'])

    async def test_chinese_alias_and_abbreviation_map_to_same_player_and_playoffs(self):
        for text in ('分析亚历山大25-26赛季季后赛表现','SGA 2025/26 playoffs report'):
            resolved,_=await resolve_scope(self.db,self.payload(text))
            self.assertEqual(resolved.player_id,self.player.id)
            self.assertEqual(resolved.season,'2025-26')
            self.assertEqual(resolved.season_type,'Playoffs')

    async def test_explicit_team_replaces_previous_player(self):
        resolved,_=await resolve_scope(self.db,self.payload('勇士24-25常规赛表现',player_id=self.player.id))
        self.assertEqual(resolved.team_id,self.warriors.id)
        self.assertIsNone(resolved.player_id)
        self.assertEqual(resolved.season,'2024-25')

    async def test_model_alias_is_grounded_and_matched_to_local_catalog(self):
        intent=QuestionIntent(targets=[Target(kind='player',mention='鸭梨',name='Shai Gilgeous-Alexander')],season=None,season_type=None,clarification=None)
        with patch('app.analysis.intent.extract_intent',AsyncMock(return_value=intent)):
            resolved,routing=await resolve_scope(self.db,self.payload('评价一下鸭梨的表现'))
        self.assertEqual(resolved.player_id,self.player.id)
        self.assertEqual(routing['method'],'model_intent')

    async def test_pronoun_followup_retains_resolved_scope(self):
        intent=QuestionIntent(targets=[],season=None,season_type='Playoffs',clarification=None)
        payload=self.payload('那他季后赛呢？',player_id=self.player.id);payload.team_id=self.thunder.id
        with patch('app.analysis.intent.extract_intent',AsyncMock(return_value=intent)):
            resolved,_=await resolve_scope(self.db,payload)
        self.assertEqual(resolved.player_id,self.player.id)
        self.assertEqual(resolved.team_id,self.thunder.id)
        self.assertEqual(resolved.season_type,'Playoffs')

    async def test_negated_name_does_not_become_analysis_target(self):
        intent=QuestionIntent(targets=[Target(kind='player',mention='SGA',name='Shai Gilgeous-Alexander')],season=None,season_type=None,clarification=None)
        with patch('app.analysis.intent.extract_intent',AsyncMock(return_value=intent)):
            resolved,_=await resolve_scope(self.db,self.payload('不要分析库里，而是分析SGA'))
        self.assertEqual(resolved.player_id,self.player.id)

    async def test_ambiguous_name_requests_clarification_instead_of_default_team(self):
        intent=QuestionIntent(targets=[],season=None,season_type=None,clarification='请确认是杰伦还是杰林·威廉姆斯？')
        with patch('app.analysis.intent.extract_intent',AsyncMock(return_value=intent)):
            with self.assertRaises(HTTPException) as ctx:await resolve_scope(self.db,self.payload('分析威廉姆斯'))
        self.assertEqual(ctx.exception.status_code,422)
        self.assertIn('请确认',ctx.exception.detail)

    async def test_invented_mention_and_unknown_catalog_entity_do_not_fall_back(self):
        for mention,name in [('不在问题中','Shai Gilgeous-Alexander'),('#','Shai Gilgeous-Alexander'),('新球员','Unknown Player')]:
            intent=QuestionIntent(targets=[Target(kind='player',mention=mention,name=name)],season=None,season_type=None,clarification=None)
            with patch('app.analysis.intent.extract_intent',AsyncMock(return_value=intent)):
                with self.assertRaises(HTTPException):await resolve_scope(self.db,self.payload('分析新球员'))

    async def test_model_failure_never_answers_about_default_team(self):
        with patch('app.analysis.intent.extract_intent',AsyncMock(side_effect=RuntimeError('synthetic failure'))):
            with self.assertRaises(HTTPException) as ctx:await resolve_scope(self.db,self.payload('分析陌生绰号'))
        self.assertEqual(ctx.exception.status_code,503)

    def test_multi_seasons_types_and_invalid_years_are_explicitly_rejected(self):
        for text in ('比较25-26和26-27赛季','常规赛和季后赛','25-27赛季'):
            with self.assertRaises(HTTPException):explicit_scope(text)
