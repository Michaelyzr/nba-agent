import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from sqlalchemy import select

from app.intelligence.injury import active_events, apply_model, exposures, fit_model, train
from app.models.game import Game
from app.models.player import Player
from app.models.stats import PlayerGameStat
from app.models.team import Team
from app.news.archive import utcnow
from app.news.collector import collect
from app.news.context import archived_evidence
from app.news.extraction import EntityMapper, ExtractedEvent, grounded
from app.news.sources import parse_espn_injuries, parse_report, parse_rss, report_time
from tests.support import TestDB, seed


class SourceTests(unittest.TestCase):
    def test_report_timestamp_is_eastern_with_dst(self):
        self.assertEqual(report_time('https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-02-05_10_30AM.pdf').hour, 15)
        self.assertEqual(report_time('https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-04-23_08_30PM.pdf').hour, 0)

    def test_pdf_url_cannot_target_arbitrary_host_or_path(self):
        for url in ('http://localhost/private', 'https://ak-static.cms.nba.com/../../file.pdf',
                    'https://evil.com/Injury-Report_2026-02-05_10_30AM.pdf'):
            with self.assertRaises(ValueError): report_time(url)

    def test_report_carries_game_date_team_across_pages_and_marks_submission(self):
        teams = [SimpleNamespace(full_name='Brooklyn Nets'), SimpleNamespace(full_name='Orlando Magic')]
        text = '''Injury Report: 02/05/26 10:30 AM
Game Date Game Time Matchup Team Player Name Current Status
02/05/2026 07:00(ET) BKN@ORL Brooklyn Nets  Smith, John  Out  Injury/Illness - Knee
Page 1 of 2
Injury Report: 02/05/26 10:30 AM
Orlando Magic  Jones, Tom  Questionable  Injury/Illness - Ankle
'''
        rows = parse_report(text, teams)
        self.assertEqual([r['player_status'] for r in rows[:2]], ['out', 'questionable'])
        self.assertEqual(rows[1]['game_date'], '2026-02-05')
        self.assertEqual(rows[1]['team_name'], 'Orlando Magic')
        self.assertEqual(sum(r['event_type']=='report_coverage' for r in rows), 2)

    def test_unsubmitted_and_no_player_not_assumed_healthy(self):
        teams = [SimpleNamespace(full_name='Brooklyn Nets')]
        rows=parse_report('02/05/2026 BKN@ORL Brooklyn Nets NOT YET SUBMITTED', teams)
        self.assertEqual(len(rows),1); self.assertFalse(rows[0]['submitted'])
        with self.assertRaises(ValueError): parse_report('John Smith Out Knee', teams)

    def test_g_league_not_classified_as_medical_injury(self):
        rows = parse_report('02/05/2026 BKN@ORL Brooklyn Nets  Smith, John  Out  G League - Two-Way',
                            [SimpleNamespace(full_name='Brooklyn Nets')])
        self.assertEqual(rows[0]['event_type'], 'role_change')

    def test_espn_ids_not_used_as_nba_ids_and_unknown_status_skipped(self):
        payload = {'injuries':[{'id':'999', 'displayName':'Brooklyn Nets', 'injuries':[
            {'date':'2026-10-04T12:00Z', 'status':'Out', 'athlete':{'id':'999', 'displayName':'John Smith'}, 'shortComment':'Knee'},
            {'date':'2026-10-04T12:00Z', 'status':'Unexpected', 'athlete':{'displayName':'Other'}}]}]}
        rows = parse_espn_injuries(payload)
        self.assertEqual(len(rows),1); self.assertNotIn('player_id',rows[0])

    def test_rss_preserves_published_time_and_rejects_untrusted_link_entities(self):
        rss = b'<rss><channel><item><title>Return</title><description>John returns.</description><link>https://www.espn.com/nba/story/1</link><pubDate>Sun, 4 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>'
        self.assertEqual(parse_rss(rss)[0]['published_at'], '2026-10-04T12:00:00+00:00')
        self.assertEqual(parse_rss(rss.replace(b'www.espn.com', b'evil.com')), [])
        with self.assertRaises(ValueError): parse_rss(b'<!DOCTYPE a>'+rss)

    def test_training_return_and_exact_quote_guard(self):
        event = ExtractedEvent(player_name='John Smith', team_name='Brooklyn Nets', event_type='training_return',
            player_status='available', minutes_limit=None, game_date=None, quote='John Smith returned to practice.', reason='Practice')
        doc = {'body':event.quote,'published_at':'2026-10-04T12:00:00+00:00'}
        self.assertEqual(grounded([event],doc)[0]['player_status'],'unknown')
        with self.assertRaises(ValueError): grounded([event],{**doc,'body':'Other news'})
        with self.assertRaises(ValueError): grounded([event.model_copy(update={'player_name':'Invented Player'})],doc)

    def test_minute_limit_requires_quoted_number(self):
        event = ExtractedEvent(player_name='John Smith', team_name='Brooklyn Nets', event_type='minutes_restriction',
            player_status='available', minutes_limit=20, game_date=None, quote='John Smith has a minutes restriction.', reason='Limit')
        with self.assertRaises(ValueError): grounded([event], {'body':event.quote,'published_at':'2026-10-04T12:00:00+00:00'})


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = TestDB(); self.game = seed(self.db); self.store = self.db.news_archive
        self.now = utcnow()
        self.player = Player(id=uuid4(), nba_player_id=101, full_name='John Smith', current_team_id=self.game.home_team_id, active=True)
        self.db.add(self.player); await self.db.commit()
        games = (await self.db.scalars(select(Game).where(Game.status=='final').order_by(Game.tipoff_time.desc()).limit(10))).all()
        for game in games:
            self.db.add(PlayerGameStat(game_id=game.id, player_id=self.player.id, team_id=self.game.home_team_id, minutes=30, points=20))
        await self.db.commit()
        self.mapper = EntityMapper((await self.db.scalars(select(Team))).all(), [self.player], [self.game])

    async def asyncTearDown(self): self.db.close()

    def add(self, kind='injury', status='out', hours=1, review='approved', **updates):
        data = {'player_name':self.player.full_name, 'team_name':'Test Team 1','player_id':str(self.player.id),
                'team_id':str(self.game.home_team_id),'game_id':None,'game_date':None,'matchup':None,
                'event_type':kind,'player_status':status,'minutes_limit':None,'quote':'Exact text',
                'reason':'Knee', 'published_at':(self.now-timedelta(hours=hours)).isoformat(), 'provider':'espn_injuries'}
        data.update(updates)
        if data['team_id'] == str(self.game.away_team_id):
            data['team_name']='Test Team 2'
        doc,_ = self.store.document(url='https://www.espn.com/nba/injuries',provider=data['provider'],
            title='Report',body='Exact text',published_at=data['published_at'])
        return self.store.add_event(doc,data,review)[0]

    async def test_repeat_source_does_not_duplicate_or_refresh_old_fact(self):
        a=self.add(hours=30); b=self.add(hours=30)
        self.assertEqual(a,b); self.assertEqual(self.store.status()['events'],1)
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['players'][0]['status'],'unknown')
        self.assertEqual(output['teams']['home']['out_minutes'],0)

    async def test_fact_identity_survives_entity_rematching(self):
        first=self.add(review='pending',player_id=None,match_error='not matched')
        self.store.rematch(lambda data:{**data,'player_id':str(self.player.id),'match_error':None})
        second=self.add()
        self.assertEqual(first,second); self.assertEqual(self.store.status()['events'],1)
        self.assertEqual(self.store.events()[0]['review'],'pending')

    async def test_document_duplicate_policy_preserves_legacy_ids(self):
        doc=dict(url='https://www.espn.com/nba/story/1',provider='espn_rss',published_at=self.now.isoformat(),title='Report',body='Exact report')
        identity,_=self.store.document(**doc)
        with self.store.connection() as c:
            c.execute('UPDATE documents SET id=? WHERE id=?',('legacy-id',identity))
        found,created=self.store.document(**doc)
        self.assertEqual(found,'legacy-id');self.assertFalse(created)

    async def test_identical_archived_news_is_not_extracted_twice(self):
        identity,_=self.store.document(url='https://www.espn.com/nba/story/1',provider='espn_rss',
            published_at=self.now.isoformat(),title='Report',body='Exact report')
        with self.store.connection() as c:
            c.execute("INSERT INTO documents SELECT 'duplicate',url,provider,published_at,observed_at,title,body,0 FROM documents WHERE id=?",(identity,))
        self.assertEqual(len(self.store.unextracted(10)),1)
        self.store.mark_extracted(identity)
        self.assertEqual(self.store.unextracted(10),[])

    async def test_trade_does_not_drop_global_injury_evidence_or_reuse_old_team_out(self):
        self.add(team_id=str(uuid4()))
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['players'][0]['status'],'unknown')
        rows=await archived_evidence(as_of=utcnow(),team_ids=[self.game.home_team_id],game_id=self.game.id,
                                    player_ids=[self.player.id],store=self.store)
        self.assertEqual(len(rows),1)

    async def test_verified_out_has_quantified_minutes_without_fabricated_probability(self):
        self.add()
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['teams']['home']['out_minutes'],30)
        self.assertEqual(output['players'][0]['minutes_range'],[0,0])
        self.assertFalse(output['training_eligible'])
        self.assertEqual(apply_model(.6,output,None,utcnow())[0],.6)

    async def test_uncertain_status_has_range_not_fixed_play_probability(self):
        self.add(status='questionable')
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['players'][0]['minutes_range'],[0,30])
        self.assertNotIn('play_probability',output['players'][0])

    async def test_recovery_preserves_independent_minutes_limit(self):
        self.add(hours=3); self.add(kind='minutes_restriction',status='available',hours=2,minutes_limit=20)
        self.add(kind='available',status='available',hours=1)
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['players'][0]['minutes_range'],[20,20])
        self.assertEqual(output['teams']['home']['limited_minutes'],10)
        self.assertEqual(output['teams']['home']['out_minutes'],0)

    async def test_injury_available_does_not_clear_suspension(self):
        self.add(kind='suspension',hours=2)
        self.add(kind='available',status='available',hours=1)
        self.assertEqual((await exposures(self.db,self.game,store=self.store))['teams']['home']['out_minutes'],30)
        self.add(kind='suspension_ended',status='available',hours=.1)
        self.assertEqual((await exposures(self.db,self.game,store=self.store))['teams']['home']['out_minutes'],0)

    async def test_removal_from_new_report_requires_review_not_old_out_or_recovery(self):
        self.add(provider='nba_official',game_id=str(self.game.id),hours=2)
        self.add(kind='report_coverage',player_name=None,player_id=None,provider='nba_official',game_id=str(self.game.id),hours=1)
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['players'][0]['status'],'unknown')
        self.assertTrue(output['players'][0]['needs_review'])

    async def test_late_snapshot_cannot_train_on_already_known_result(self):
        past=(await self.db.scalars(select(Game).where(Game.status=='final'))).all()[0]
        data={'version':'injury-exposure-v1','training_eligible':True,'features':dict(out_minutes_diff=.1,
              uncertain_minutes_diff=0,limited_minutes_diff=0,out_points_diff=.1),'base_probability':.5}
        self.store.snapshot(str(past.id),utcnow().isoformat(),data)
        self.assertEqual((await train(self.db,self.store))['sample_size'],0)

    async def test_future_box_score_never_changes_baseline_minutes(self):
        self.game.status='final'; self.game.home_score=110; self.game.away_score=100
        self.db.add(PlayerGameStat(game_id=self.game.id,player_id=self.player.id,team_id=self.game.home_team_id,minutes=48,points=100))
        await self.db.commit();self.add()
        self.assertEqual((await exposures(self.db,self.game,store=self.store))['players'][0]['baseline_minutes'],30)

    async def test_pending_future_and_other_game_are_excluded(self):
        self.add(review='pending'); self.add(hours=-1); self.add(game_id=str(uuid4()))
        output=await exposures(self.db,self.game,store=self.store)
        self.assertEqual(output['players'],[])

    async def test_review_time_cannot_leak_backwards(self):
        identity=self.add(review='pending')
        before=utcnow(); self.store.review(identity,'approved')
        self.assertEqual(active_events(self.store.events(),self.game,before),[])
        with self.assertRaises(ValueError): self.store.review(identity,'rejected')

    async def test_same_publication_time_uses_observation_order_not_hash_order(self):
        self.add(); self.add(kind='available',status='available')
        rows=self.store.events()
        for row in rows:
            old=row['data']['event_type']=='injury'
            row['id']='z' if old else 'a'
            row['observed_at']=(self.now-timedelta(minutes=30 if old else 20)).isoformat()
        selected=active_events(rows,self.game,utcnow())
        self.assertEqual(selected[0]['data']['player_status'],'available')

    async def test_archived_long_injury_resolved_by_recovery(self):
        self.add(hours=24*30); self.add(kind='available',status='available')
        rows=await archived_evidence(as_of=utcnow(),team_ids=[self.game.home_team_id],game_id=self.game.id,store=self.store)
        self.assertEqual([r['event_type'] for r in rows],['available'])

    async def test_wrong_team_or_ambiguous_name_is_never_auto_matched(self):
        event={'player_name':'Smith, John','team_name':'Test Team 1','game_date':None}
        self.assertEqual(self.mapper(event)['player_id'],str(self.player.id))
        self.assertIsNone(self.mapper({**event,'team_name':'Test Team 2'})['player_id'])

    async def test_collector_partial_failure_visible_and_lease_released(self):
        async def fake_download(url):
            if 'injuries' in url:
                return json.dumps({'injuries':[]}).encode()
            return b'<rss><channel/></rss>'
        with patch('app.news.collector.download',side_effect=fake_download), patch('app.news.collector.latest_official',new=AsyncMock(side_effect=ValueError('blocked'))):
            result=await collect(self.db,store=self.store)
        self.assertEqual(result['sources']['nba_official']['status'],'error')
        self.assertEqual(result['sources']['espn_rss']['status'],'ok')
        self.assertTrue(self.store.lease('test')); self.store.release('test')

    async def test_no_prospective_samples_training_is_explicitly_insufficient(self):
        result=await train(self.db,self.store)
        self.assertEqual(result['status'],'insufficient_data'); self.assertIsNone(self.store.get('injury_model'))

    async def test_missing_stats_or_fresh_official_coverage_blocks_training(self):
        self.add()
        for team_id in [self.game.home_team_id,self.game.away_team_id]:
            self.add(kind='report_coverage', player_name=None,player_id=None,team_id=str(team_id),
                     game_id=str(self.game.id),provider='nba_official')
        self.assertTrue((await exposures(self.db,self.game,store=self.store))['training_eligible'])
        self.add(kind='minutes_restriction',status='available',hours=.1,minutes_limit=None)
        self.assertFalse((await exposures(self.db,self.game,store=self.store))['training_eligible'])

    async def test_new_unsubmitted_report_supersedes_old_coverage(self):
        for team_id in [self.game.home_team_id,self.game.away_team_id]:
            self.add(kind='report_coverage',player_name=None,player_id=None,team_id=str(team_id),
                     game_id=str(self.game.id),provider='nba_official',submitted=True)
        self.assertTrue((await exposures(self.db,self.game,store=self.store))['training_eligible'])
        self.add(kind='report_coverage',player_name=None,player_id=None,game_id=str(self.game.id),
                 provider='nba_official',hours=.1,submitted=False)
        self.assertFalse((await exposures(self.db,self.game,store=self.store))['training_eligible'])


class TrainingTests(unittest.TestCase):
    def samples(self, reverse=False):
        # Deterministic controlled effect, entirely synthetic and never installed.
        now=datetime(2025,1,1,tzinfo=timezone.utc)
        rows=[]
        for i in range(250):
            effect=-.2 if i%2 else .2
            won=int(effect>0)
            if reverse and i>=200: won=1-won
            rows.append({'game_id':str(i),'tipoff_time':(now+timedelta(days=i)).isoformat(),
                'home_win':won,'base_probability':.5,'features':dict(out_minutes_diff=effect,
                    uncertain_minutes_diff=0,limited_minutes_diff=0,out_points_diff=effect)})
        return rows

    def test_chronological_fit_improves_holdout_and_applies_only_after_training(self):
        model=fit_model(self.samples()); model['trained_at']=utcnow().isoformat()
        self.assertEqual(model['status'],'validated')
        self.assertLess(model['model_metrics']['brier'],model['baseline_metrics']['brier'])
        exposure={'training_eligible':True,'features':self.samples()[0]['features']}
        self.assertGreater(apply_model(.5,exposure,model,utcnow())[0],.5)
        self.assertEqual(apply_model(.5,exposure,model,datetime(2025,1,1,tzinfo=timezone.utc))[0],.5)
        exposure['features']['out_minutes_diff']=10
        self.assertEqual(apply_model(.5,exposure,model,utcnow())[0],.5)

    def test_validation_failure_does_not_activate_coefficients(self):
        self.assertEqual(fit_model(self.samples(reverse=True))['status'],'rejected')

    def test_no_feature_variation_is_not_a_trained_injury_model(self):
        rows=self.samples()
        for row in rows: row['features']={key:0 for key in row['features']}
        self.assertEqual(fit_model(rows)['status'],'insufficient_variation')
