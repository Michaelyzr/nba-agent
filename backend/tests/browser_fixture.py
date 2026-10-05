"""Opt-in synthetic API fixture for browser QA. No external DB writes."""
import os
from datetime import datetime, timedelta, timezone

if os.environ.get("NBA_QA_FIXTURE") != "1":
    raise RuntimeError("This fixture requires NBA_QA_FIXTURE=1; do not use it as the application server")

from app.db.session import get_db
from app.main import app
from app.models.market import MarketSnapshot
from app.models.news import NewsEvent
from app.models.player import Player
from app.models.stats import PlayerGameStat
from app.models.game import Game
from sqlalchemy import select
from tests.support import TestDB, seed

def make_db():
    db = TestDB()
    game = seed(db)
    db.add(MarketSnapshot(game_id=game.id, provider="polymarket", market_type="game_winner",
        contract_id="123", selection="home", market_probability=.5, bid_price=.49,
        ask_price=.51, liquidity=2000, observed_at=datetime.now(timezone.utc),
        raw_data={"synthetic_qa_only": True, "mapping_confirmed": True}))
    now = datetime.now(timezone.utc)
    for player_id, resolved in [(101, False), (102, True)]:
        player = Player(nba_player_id=player_id, full_name=f"QA Player {player_id}",
                        current_team_id=game.home_team_id, active=True)
        db.add(player)
        db.session.flush()
        for past in db.session.scalars(select(Game).where(Game.status == "final").order_by(Game.tipoff_time.desc()).limit(10)):
            db.add(PlayerGameStat(player_id=player.id,game_id=past.id,team_id=game.home_team_id,
                                 minutes=30,points=20))
        data = dict(player_name=player.full_name,team_name="Test Team 1",player_id=str(player.id),
            team_id=str(game.home_team_id),game_id=None,game_date=None,matchup=None,provider="espn_injuries",
            event_type="injury",player_status="out",quote="Synthetic QA: player ruled out.",reason="合成伤病测试",
            minutes_limit=None,published_at=(now-timedelta(hours=2)).isoformat(),match_error=None)
        doc,_ = db.news_archive.document(url="https://www.espn.com/nba/injuries",provider="espn_injuries",
            published_at=data["published_at"],title="合成测试来源",body=data["quote"])
        db.news_archive.add_event(doc,data,"approved")
        if resolved:
            for kind, cap in [("available",None),("minutes_restriction",20)]:
                db.news_archive.add_event(doc,{**data,"event_type":kind,"player_status":"available",
                    "minutes_limit":cap,"published_at":(now-timedelta(hours=1)).isoformat()},"approved")
        else:
            db.news_archive.add_event(doc,{**data,"event_type":"available","player_status":"available",
                "provider":"espn_rss","reason":"合成测试：已获准出场",
                "published_at":(now-timedelta(minutes=30)).isoformat()},"pending")
        when = now - timedelta(days=30)
        db.add(NewsEvent(player_id=player.id, team_id=game.home_team_id,
            event_type="injury", player_status="out",
            title="测试：30 天前长期伤病" if not resolved else "测试：已解除旧伤病（不应显示）",
            source="合成测试事件", published_at=when, ingested_at=when))
        if resolved:
            returned = now - timedelta(days=1)
            db.add(NewsEvent(player_id=player.id, team_id=game.home_team_id,
                event_type="return_from_injury", player_status="available",
                title="测试：另一球员已复出", source="合成测试事件",
                published_at=returned, ingested_at=returned))
    db.session.commit()
    db.news_archive.put("last_collection",{"finished_at":now.isoformat(),"sources":{
        "espn_injuries":{"status":"ok","parsed":2},"espn_rss":{"status":"ok","parsed":1},
        "nba_official":{"status":"error","http_status":403,"detail":"合成测试：来源被拒绝"}}})
    return db


db = make_db()
from app.api.routes import evidence
evidence.archive = db.news_archive


@app.post("/__qa/reset", include_in_schema=False)
async def reset_fixture():
    global db
    db.close()
    db = make_db()
    evidence.archive = db.news_archive
    return {"synthetic_qa_only": True}


async def test_db():
    try:
        yield db
    except Exception:
        await db.rollback()
        raise


app.dependency_overrides[get_db] = test_db

# Only this opt-in fixture substitutes the provider. Production uses OpenAI.
from app.api.routes import analysis
import asyncio

async def synthetic_analysis(payload, context):
    text = payload.messages[-1].content
    if '测试模型失败' in text:
        yield {'type':'error','code':'QA_ERROR','message':'合成测试：模型暂时不可用，请重试。'}
        return
    target = context['facts']['selected_player'] or context['facts']['team']
    if payload.web_search:
        yield {'type':'web_status','status':'searching'}
        await asyncio.sleep(.1)
    reply = f"## {target['name']} 分析报告\n\n**数据事实**：当前阵容有 {context['meta']['roster_count']} 名球员。[D1]\n\n- 所选赛季有 {context['meta']['game_count']} 场比赛记录。[D2]\n- 这是隔离测试的合成回答，不用于实际研究。\n\n## 人员评价\n统计缺失时不推断最新表现。\n"
    if payload.mode == 'team_report':
        reply += '\n## 长报告布局测试\n' + '\n'.join(
            f'- 合成测试段落 {i+1}：用于验证长报告结束后输入框仍可见，可以连续追问，不是实际球队评价。'
            for i in range(45)) + '\n'
    for part in [reply[:30], reply[30:70], reply[70:]]:
        yield {'type':'delta','text':part}
        await asyncio.sleep(3 if '测试停止' in text else .08)
    if payload.web_search:
        yield {'type':'web_sources','status':'completed','calls':1,
            'searched_at':'2026-10-05T00:00:00+00:00',
            'sources':[{'id':'W1','title':'NBA 官方资料（合成测试）','url':'https://www.nba.com/stats','kind':'web'}],
            'consulted_urls':['https://www.nba.com/stats'],
            'text':reply+'\n网页补充：这是合成测试来源。[W1]\n'}
    yield {'type':'done','response_id':'synthetic-only','usage':{'total_tokens':120}}

analysis.model_events = synthetic_analysis

# Language understanding is also synthetic in browser QA; deterministic name
# matching and the real context builder remain enabled.
from app.analysis import intent as intent_module
async def synthetic_intent(text, payload, teams, players):
    return intent_module.QuestionIntent(targets=[],season=None,season_type=None,
        clarification='请确认是杰伦还是杰林·威廉姆斯？' if '威廉姆斯' in text else None)
intent_module.extract_intent = synthetic_intent
