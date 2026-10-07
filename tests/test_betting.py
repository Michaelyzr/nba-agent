"""Own probabilities, correct line semantics, stake outcomes and distinct-game parlays."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.betting import make_board, evaluate, recommend_single, recommend_parlay, parlay_result, simulated_contracts, hedge_bet
from forecast.betting import distribution, historical_distribution, probability
from data_sources.polymarket_books import bet_contract_spec
from data_sources import ROOT, read_table

NOW=pd.Timestamp('2026-04-12T21:00:00Z')
GAME=SimpleNamespace(game_id='401811041', home_team='BOS', away_team='ORL',home_team_id=1610612738,away_team_id=1610612753,tip_time=NOW+pd.Timedelta(hours=1))
BASE={'total_mean':220.,'total_sd':20.,'margin_sd':14.,'history_games':100,'history_cutoff':NOW.isoformat(),'model':'own'}


def snapshot(p=.65):
    return {'game_id':GAME.game_id,'as_of':NOW.isoformat(),'p_home':p,'quote_state':'pregame','baseline':{'p_home':.65}}


def board(game=GAME,p=.65):
    s=snapshot(p);s['game_id']=game.game_id;m=distribution(BASE,s)
    return make_board(game,s,m,simulated_contracts(game,m,s),True)


def test_probabilities_are_own_model_outputs_and_independent_of_reference_prices():
    s=snapshot();m=distribution(BASE,s);contracts=simulated_contracts(GAME,m,s)
    one=make_board(GAME,s,m,contracts,True)
    contracts=deepcopy(contracts)
    for c in contracts:
        for q in c['quotes'].values():q.update(ask=.9,asks=[{'price':.9,'size':500}])
    two=make_board(GAME,s,m,contracts,True)
    assert [r['probability'] for r in one['bets']]==[r['probability'] for r in two['bets']]
    assert [r['model_odds'] for r in one['bets']]==[r['model_odds'] for r in two['bets']]
    assert evaluate(one['bets'][1])['reference_odds'] != evaluate(two['bets'][1])['reference_odds']


def test_both_sides_and_half_point_lines_have_correct_orientation():
    m=distribution(BASE,snapshot())
    for kind,sides,line in [('moneyline',('home','away'),None),('spread',('home','away'),-7.5),('total',('over','under'),221.5)]:
        assert probability(m,kind,sides[0],line)+probability(m,kind,sides[1],line)==pytest.approx(1)
    assert probability(m,'spread','home',-7.5)<probability(m,'moneyline','home')
    assert probability(m,'total','over',240.5)<probability(m,'total','over',200.5)
    assert distribution(BASE,{**snapshot(),'p_home':None}) is None
    assert distribution(BASE,{**snapshot(),'quote_state':'final'}) is None


def test_historical_model_excludes_future_and_same_game_final_scores():
    games=read_table('games',ROOT/'data'/'sample')
    base=historical_distribution(games,GAME,NOW)
    changed=games.copy()
    mask=pd.to_datetime(changed.final_at,utc=True)>=NOW
    changed.loc[mask,['home_pts','away_pts']]=99999
    assert historical_distribution(changed,GAME,NOW)==base
    assert base['history_games']>=30


def test_news_and_score_update_own_spread_total_probabilities():
    s=snapshot();old=distribution(BASE,s)
    s.update(quote_state='live',p_home=.45,score={'period':2,'clock_seconds':360,'phase':'live','home_score':40,'away_score':45},player_effects=[{'home_margin_adjustment':-4}])
    new=distribution(BASE,s)
    assert new['total_mean']>=85 and new['total_sd']<old['total_sd']
    assert probability(new,'spread','home',-5.5)<probability(old,'spread','home',-5.5)
    no_news=distribution(BASE,{**s,'player_effects':[]})
    assert new['total_mean']==pytest.approx(no_news['total_mean']-4)


def test_single_stake_profit_and_stale_reference_not_recommended():
    rows=board()['bets'];r=next(r for r in rows if r['side']=='over')
    e=evaluate(r,20)
    assert e['estimated_return']*r['probability']-e['cost']==pytest.approx(e['expected_profit'])
    assert e['reference_odds'] != r['model_odds']
    stale=deepcopy(rows)
    for row in stale:row['reference_current']=False
    assert recommend_single(stale)['pick'] is None
    assert evaluate(stale[0])['expected_profit'] is None
    with pytest.raises(ValueError):evaluate(r,float('nan'))


def test_parlay_ranking_exhaustive_distinct_games_and_indicative_return():
    rows=[]
    for i,p in enumerate([.65,.55,.75]):
        game=SimpleNamespace(**{**GAME.__dict__,'game_id':str(i),'home_team':'H'+str(i),'away_team':'A'+str(i)})
        rows+=board(game,p)['bets']
    result=recommend_parlay(rows,20,2,'profit')['parlay']
    assert result and len({r['game_id'] for r in result['picks']})==2
    assert not result['executable_quote']
    assert result['estimated_return']==pytest.approx(20*result['combined_odds'])
    assert result['expected_profit']==pytest.approx(result['probability']*result['estimated_return']-20)
    for a in rows:
        for b in rows:
            if a['game_id']==b['game_id'] or not evaluate(a)['eligible'] or not evaluate(b)['eligible']:continue
            candidate=parlay_result([a,b])
            assert result['stress_profit']>=candidate['stress_profit']-1e-8
    with pytest.raises(ValueError):parlay_result([rows[0],rows[1]])
    assert recommend_parlay(board()['bets'])['parlay'] is None


def test_hedge_uses_bet_amount_and_placed_odds_with_depth_cap():
    rows=board()['bets'];home=rows[0]
    result=hedge_bet(rows,home['id'],20,2,1000)
    assert result['balanced']
    assert result['if_original_wins']==pytest.approx(result['if_opposite_wins'])
    assert result['floor']==pytest.approx(40-20-result['amount'])
    partial=hedge_bet(rows,home['id'],20,2,5)
    assert not partial['balanced'] and partial['amount']<=5
    rows[1]['reference_current']=False
    with pytest.raises(ValueError):hedge_bet(rows,home['id'],20,2)


RULES='If postponed, this market remains open until completed. If canceled entirely it will resolve 50-50.'

def market(kind='spreads',line=-7.5):
    return {'sportsMarketType':kind,'line':line,'outcomes':['Magic','Celtics'] if kind=='spreads' else ['Over','Under'],
      'clobTokenIds':['first','second'],'conditionId':'c','gameStartTime':GAME.tip_time.isoformat(),
      'question':'Spread: Magic (-7.5)', 'description':('Magic win the game by 8 or more points. '+RULES if kind=='spreads' else 'Teams combine to score 221 or more points in this game. '+RULES)}


def test_reference_contract_team_token_mapping_full_game_and_threshold_matching():
    found=bet_contract_spec(market(),GAME)
    assert found['line']==7.5 and found['tokens']=={'home':'second','away':'first'}
    found=bet_contract_spec(market('totals',220.5),GAME)
    assert found['tokens']=={'over':'first','under':'second'}
    for bad in [market('first_half_spreads'),market('totals',221),{**market(),'question':'Spread: Celtics (-7.5)'},
                {**market(),'gameStartTime':(GAME.tip_time+pd.Timedelta(days=1)).isoformat()},
                {**market(),'description':'Overtime only. '+RULES}]:
        assert bet_contract_spec(bad,GAME) is None


def test_live_reference_and_model_age_expire_even_when_board_was_cached():
    row=deepcopy(board()['bets'][1]);row['simulation']=False
    now=pd.Timestamp.now(tz='UTC').isoformat()
    row.update(model_as_of=now,reference_current=True)
    row['reference_quote'].update(observed_at=now,updated_at=now)
    assert evaluate(row)['reference_current']
    row['model_as_of']=(pd.Timestamp(now)-pd.Timedelta(seconds=16)).isoformat()
    assert not evaluate(row)['reference_current']
    assert evaluate(row)['expected_profit'] is None
    assert evaluate(row)['estimated_return'] is None
    with pytest.raises(ValueError):hedge_bet([board()['bets'][0],row],board()['bets'][0]['id'],20,2)


def test_public_batch_books_match_by_token_not_response_order():
    from data_sources.polymarket_books import PolymarketReader
    markets=[{**market(),'conditionId':'spread','feesEnabled':False,'active':True,'closed':False,'acceptingOrders':True},
             {**market('totals',220.5),'conditionId':'total','clobTokenIds':['over','under'],'feesEnabled':False,'active':True,'closed':False,'acceptingOrders':True}]
    calls=[]
    class Response:
        def __init__(self,data):self.data=data
        def raise_for_status(self):pass
        def json(self):return self.data
    class Session:
        def post(self,url,json,**kw):
            calls.append((url,json))
            return Response([{'market':c,'asset_id':t,'timestamp':NOW.timestamp()*1000,
                'bids':[{'price':'.4','size':'100'}],'asks':[{'price':'.5','size':'100'}],'min_order_size':'5'}
                for t,c in [('under','total'),('second','spread'),('over','total'),('first','spread')]])
        def get(self,*a,**kw):return Response({})
    reader=PolymarketReader(Session(),lambda:NOW)
    reader._discover_game_event=lambda game:{'slug':'fixture','markets':markets}
    result=reader.fetch_bet_contracts(GAME)
    assert len(calls)==1 and calls[0][0].endswith('/books')
    assert len(calls[0][1])==4
    spread,total=result['contracts']
    assert spread['line']==7.5 and spread['quotes']['home']['token_id']=='second'
    assert total['quotes']['under']['token_id']=='under'
    assert spread['quotes']['home']['minimum_shares']==5
    assert spread['quotes']['home']['fee_verified']
    board_view=make_board(GAME,snapshot(),distribution(BASE,snapshot()),result['contracts'],True)
    assert all(r['reference_current'] for r in board_view['bets'] if r['kind']!='moneyline')


def test_live_board_uses_separate_agents_before_and_after_tip(tmp_path,monkeypatch):
    import agents.dashboard_live as module
    from agents.dashboard_live import LiveDashboard
    now=pd.Timestamp.now(tz='UTC')
    calls=[]
    class Agent:
        def __init__(self,*a,**kw):assert isinstance(a[2].tip_time,pd.Timestamp)
        def poll(self,now):
            calls.append('pregame')
            return {'snapshot':{**snapshot(),'as_of':now.isoformat()}}
    class LiveAgent:
        def poll(self,now):
            calls.append('inplay')
            return {'snapshot':{**snapshot(.60),'as_of':now.isoformat(),'quote_state':'live',
                'score':{'phase':'live','period':1,'clock_seconds':600,'home_score':8,'away_score':6}}}
    class Reader:
        def fetch_bet_contracts(self,game):calls.append('reference');return {'contracts':[],'errors':[]}
    game={**GAME.__dict__,'tip_time':(now+pd.Timedelta(hours=1)).isoformat(),'phase':'pre'}
    monkeypatch.setattr(module,'PregameAgent',Agent)
    controller=LiveDashboard(market_reader=Reader());controller.run_root=tmp_path
    monkeypatch.setattr(controller,'games',lambda:[game])
    first=controller.betting(GAME.game_id);second=controller.betting(GAME.game_id)
    assert first['phase']=='pregame' and len(calls)==2
    assert first['bets']==second['bets'] and first['own_model']['p_home']==.65
    assert first['pre_curve'] and not controller.agents
    game.update(tip_time=(now-pd.Timedelta(minutes=2)).isoformat(),phase='in')
    monkeypatch.setattr(controller,'_agent',lambda game:LiveAgent())
    live=controller.betting(GAME.game_id)
    assert live['phase']=='inplay' and 'inplay' in calls
    assert live['own_model']['p_home']==.60
    assert len(controller.pregame_agents)==1
    assert live['own_model']['total_mean']!=first['own_model']['total_mean']
    assert all(r['reference_quote'] is None for r in live['bets'])
