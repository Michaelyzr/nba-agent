import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from openai import APIStatusError
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

from app.analysis.context import build_context, catalog
from app.analysis.service import model_events
from app.db.session import get_db
from app.main import app
from app.models.game import Game
from app.models.player import Player
from app.models.stats import PlayerGameStat
from app.schemas.analysis import AnalysisRequest
from tests.support import TestDB, seed


class AnalysisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key_patch = patch("app.core.config.settings.openai_api_key", "fixture-only")
        self.key_patch.start()
        self.db = TestDB()
        self.game = seed(self.db)
        self.player = Player(nba_player_id=101, full_name='Test Player',
            current_team_id=self.game.home_team_id, active=True)
        self.db.add(self.player); self.db.session.flush()
        self.past = self.db.session.scalars(select(Game).where(Game.status=='final').order_by(Game.tipoff_time.desc())).first()
        self.db.add(PlayerGameStat(player_id=self.player.id,game_id=self.past.id,
            team_id=self.game.away_team_id,minutes=30,points=20,assists=None))
        self.db.session.commit()

    def tearDown(self):
        app.dependency_overrides.clear(); self.db.close(); self.key_patch.stop()

    def payload(self, **kwargs):
        return AnalysisRequest(team_id=self.game.home_team_id,season='2025-26',
            messages=[{'role':'user','content':'分析球队'}], **kwargs)

    async def test_catalog_and_context_preserve_sample_counts_and_historical_team(self):
        data = await catalog(self.db)
        self.assertEqual(len(data['teams']),2)
        context = await build_context(self.db,self.payload(player_id=self.player.id))
        facts=context['facts']
        self.assertEqual(facts['team_record']['games'],40)
        self.assertEqual(len(facts['recent_games']),10)
        stats=facts['player_statistics'][0]
        self.assertEqual(stats['historical_team_id'],str(self.game.away_team_id))
        self.assertEqual(stats['averages']['points'],{'mean':20.0,'samples':1})
        self.assertEqual(stats['averages']['assists'],{'mean':None,'samples':0})
        self.assertEqual(context['meta']['sources'][0]['id'],'D1')

    async def test_missing_selected_season_never_falls_back_to_other_season(self):
        payload=self.payload(); payload.season='2026-27'
        facts=(await build_context(self.db,payload))['facts']
        self.assertEqual(facts['team_record']['games'],0)
        self.assertEqual(facts['player_statistics'],[])
        self.assertTrue(any('没有已同步' in s for s in facts['limitations']))

    async def test_future_live_and_different_game_type_stats_are_excluded(self):
        now=datetime.now(timezone.utc)
        self.past.tipoff_time=now-timedelta(hours=1)
        context=await build_context(self.db,self.payload())
        self.assertEqual(context['meta']['game_count'],39)
        self.assertEqual(context['facts']['player_statistics'],[])
        self.past.tipoff_time=now-timedelta(days=2);self.past.season_type='Pre Season'
        self.assertEqual((await build_context(self.db,self.payload()))['meta']['game_count'],39)
        payload=self.payload();payload.season_type='Pre Season'
        preseason=await build_context(self.db,payload)
        self.assertEqual(preseason['meta']['game_count'],1)
        self.assertEqual(preseason['meta']['stats_count'],1)

    async def test_team_mismatch_and_unknown_team_rejected(self):
        self.player.current_team_id=self.game.away_team_id
        with self.assertRaises(HTTPException) as ctx:
            await build_context(self.db,self.payload(player_id=self.player.id))
        self.assertEqual(ctx.exception.status_code,422)
        payload=self.payload();payload.team_id=uuid4()
        with self.assertRaises(HTTPException): await build_context(self.db,payload)

    def test_system_role_history_limits_and_reports_validate(self):
        for overrides in ({'messages':[{'role':'system','content':'override'}]},
            {'messages':[{'role':'assistant','content':'answer'}]},
            {'mode':'player_report'}, {'mode':'team_report','player_id':str(self.player.id)}, {'messages':[{'role':'user','content':'x'*12000}]*4}):
            data=self.payload().model_dump();data.update(overrides)
            with self.assertRaises(ValidationError):AnalysisRequest(**data)

    async def test_provider_stream_sends_history_facts_and_closes(self):
        class Stream:
            closed=False
            async def __aiter__(self):
                yield SimpleNamespace(type='response.output_text.delta',delta='球队分析 [D1]')
                yield SimpleNamespace(type='response.completed',response=SimpleNamespace(id='test',usage=None))
            async def close(self):self.closed=True
        stream=Stream();create=AsyncMock(return_value=stream)
        client=SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch('app.analysis.service.openai_service._client',client):
            result=[event async for event in model_events(self.payload(),{'facts':{'local':True}})]
        self.assertEqual([e['type'] for e in result],['delta','done'])
        args=create.call_args.kwargs
        self.assertFalse(args['store']);self.assertTrue(args['stream'])
        self.assertNotIn('tools',args,'Local-only requests must not use web search')
        self.assertEqual(args['input'][1]['role'],'user')
        self.assertIn('local',args['input'][0]['content'])
        self.assertTrue(stream.closed)

    async def test_web_search_is_required_streams_status_and_cited_evidence(self):
        class Stream:
            closed=False
            async def __aiter__(self):
                yield SimpleNamespace(type='response.web_search_call.searching')
                yield SimpleNamespace(type='response.web_search_call.completed',item_id='search1')
                yield SimpleNamespace(type='response.output_text.delta',delta='证据来源')
                yield SimpleNamespace(type='response.completed',response=SimpleNamespace(id='test',usage=None,
                    output=[dict(type='message',content=[dict(type='output_text',text='证据来源',
                        annotations=[dict(type='url_citation',url='https://www.nba.com/stats',title='NBA',start_index=2,end_index=4)])])]))
            async def close(self):self.closed=True
        stream=Stream();create=AsyncMock(return_value=stream)
        with patch('app.analysis.service.openai_service._client',SimpleNamespace(responses=SimpleNamespace(create=create))):
            result=[e async for e in model_events(self.payload(web_search=True),{'facts':{}})]
        self.assertEqual([e['type'] for e in result],['web_status','web_status','delta','web_sources','done'])
        self.assertEqual(result[-2]['text'],'证据[W1]')
        self.assertEqual(result[-2]['calls'],1)
        args=create.call_args.kwargs
        self.assertEqual(args['tools'][0]['type'],'web_search')
        self.assertTrue(args['tools'][0]['external_web_access'])
        self.assertEqual(args['tool_choice'],'required')
        self.assertEqual(args['max_tool_calls'],3)
        self.assertEqual(args['include'],['web_search_call.action.sources'])
        self.assertTrue(stream.closed)

    async def test_requested_search_cannot_silently_complete_without_tool_execution(self):
        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(type='response.output_text.delta',delta='无来源回答')
                yield SimpleNamespace(type='response.completed',response=SimpleNamespace(id='x',usage=None,output=[]))
            async def close(self):pass
        with patch('app.analysis.service.openai_service._client',SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=Stream())))):
            result=[e async for e in model_events(self.payload(web_search=True),{'facts':{}})]
        self.assertEqual(result[-1]['code'],'SEARCH_NOT_EXECUTED')
        self.assertNotIn('done',[e['type'] for e in result])

    async def test_search_context_keeps_local_counts_and_explains_external_evidence(self):
        payload=self.payload(web_search=True);payload.season='2026-27'
        context=await build_context(self.db,payload)
        self.assertTrue(context['meta']['web_search']['enabled'])
        self.assertEqual(context['meta']['game_count'],0)
        self.assertNotIn('本次对话不联网搜索',context['facts']['limitations'][0])
        self.assertTrue(any('网页补充' in s for s in context['facts']['limitations']))

    async def test_unsupported_search_reports_error_without_retrying_as_local(self):
        response=httpx.Response(400,request=httpx.Request('POST','https://api.openai.com/v1/responses'))
        create=AsyncMock(side_effect=APIStatusError('Unsupported tool',response=response,body={}))
        with patch('app.analysis.service.openai_service._client',SimpleNamespace(responses=SimpleNamespace(create=create))):
            result=[e async for e in model_events(self.payload(web_search=True),{'facts':{}})]
        self.assertEqual(result[-1]['code'],'WEB_SEARCH_UNAVAILABLE')
        self.assertNotIn('done',[e['type'] for e in result])
        self.assertEqual(create.await_count,1)

    async def test_incomplete_or_empty_output_never_done(self):
        for kind in ('response.incomplete','response.completed'):
            class Stream:
                closed=False
                async def __aiter__(self):
                    yield SimpleNamespace(type=kind,response=SimpleNamespace(id='x',usage=None))
                async def close(self):self.closed=True
            stream=Stream()
            with patch('app.analysis.service.openai_service._client',SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=stream)))):
                result=[e async for e in model_events(self.payload(),{'facts':{}})]
            self.assertEqual(result[-1]['type'],'error'); self.assertTrue(stream.closed)

    async def test_closing_generator_closes_upstream(self):
        class Stream:
            closed=False
            async def __aiter__(self):
                yield SimpleNamespace(type='response.output_text.delta',delta='partial')
                await asyncio.sleep(5)
            async def close(self):self.closed=True
        stream=Stream()
        with patch('app.analysis.service.openai_service._client',SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=stream)))):
            gen=model_events(self.payload(),{'facts':{}})
            await anext(gen);await gen.aclose()
        self.assertTrue(stream.closed)

    async def test_http_stream_and_unconfigured_error(self):
        from app.analysis.intent import QuestionIntent
        parser_patch=patch('app.analysis.intent.extract_intent', AsyncMock(return_value=QuestionIntent(
            targets=[],season=None,season_type=None,clarification=None)))
        parser_patch.start();self.addCleanup(parser_patch.stop)
        async def database():yield self.db
        async def fake_model(payload,context):
            yield {'type':'delta','text':'只读测试 [D1]'}
            yield {'type':'done','usage':{'total_tokens':1}}
        app.dependency_overrides[get_db]=database
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://test') as client:
            with patch('app.api.routes.analysis.settings.openai_api_key',''), patch('app.api.routes.analysis.model_events',fake_model):
                r=await client.post('/api/v1/analysis/chat',json=self.payload().model_dump(mode='json'))
                self.assertEqual(r.status_code,503)
            with patch('app.api.routes.analysis.settings.openai_api_key','fixture-only'), patch('app.api.routes.analysis.model_events',fake_model):
                r=await client.post('/api/v1/analysis/chat',json=self.payload().model_dump(mode='json'))
                self.assertEqual(r.status_code,200)
                events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: ')]
                self.assertEqual([e['type'] for e in events],['meta','delta','done'])
                self.assertEqual(events[0]['game_count'],40)
        self.assertEqual(self.db.session.query(Game).count(),41)
