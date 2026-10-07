"""Run the actual betting interfaces, including replay, stake math and reconnect."""
import json
import re
import subprocess
from pathlib import Path

import pytest
from agents.loop_demo import SAMPLE_OUTPUT
from agents.betting import evaluate, recommend_single, recommend_parlay, hedge_bet


def test_streamlit_betting_choices_parlay_and_full_game():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/'loop_demo_app.py', default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'Courtside'
    assert not any(r.label=='Match mode' for r in app.radio)
    assert not app.get('vega_lite_chart')
    app.button(key='news_loop_sample_play').click().run()
    assert not app.get('file_uploader')
    app.button(key='news_loop_sample_recommend').click().run()
    assert not app.exception
    assert app.metric[2].label == 'Our estimated chance'
    app.number_input(key='news_loop_sample_stake').set_value(10.).run()
    assert not app.exception
    app.radio(key='news_loop_sample_style').set_value('Parlay').run()
    app.button(key='news_loop_sample_recommend').click().run()
    assert not app.exception and len(app.session_state['news_loop_sample_selected']) == 2
    assert app.metric[3].label == 'Illustrative return if all win'
    app.selectbox(key='news_loop_sample_legs').set_value(4).run()
    app.button(key='news_loop_sample_recommend').click().run()
    assert not app.exception and len(app.session_state['news_loop_sample_selected']) == 4
    app.slider[0].set_value(5).run()
    assert not app.exception and app.slider[0].value == 5
    assert app.get('vega_lite_chart')
    app.button(key='news_loop_sample_next').click().run()
    assert not app.exception and app.slider[0].value == 6
    app.slider[0].set_value(62).run()
    assert not app.exception
    assert app.button(key='news_loop_sample_recommend').disabled
    assert any('Game finished' in item.value for item in app.info)


JSC = Path('/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc')


@pytest.mark.skipif(not JSC.exists(), reason='system JavaScriptCore unavailable')
def test_exported_browser_bet_math_replay_and_live_recovery(tmp_path):
    html = (SAMPLE_OUTPUT/'demo.html').read_text()
    data = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', html, re.S)[1]
    scripts = '\n'.join(re.findall(r'<script>\s*(.*?)</script>', html, re.S))
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert '<html lang="en">' in html and 'type="file"' not in html
    assert 'Original cost' not in html and 'position-shares' not in html
    payload = json.loads(data)
    frames = [payload['pregame_steps'][0], payload['pregame_steps'][-1], payload['steps'][21], payload['steps'][-1]]
    expected=[]
    parlays=[]
    hedges=[]
    for i, frame in enumerate(frames):
        rows=[r for g in frame['slate'] for r in g['bets']]
        for stake in (10,100):
            expected += [{'frame':i,'id':r['id'],'stake':stake,'result':evaluate(r,stake)} for r in rows]
            for goal in ('profit','chance'):
                for legs in (2,3,4):
                    parlays.append({'frame':i,'stake':stake,'goal':goal,'legs':legs,'result':recommend_parlay(rows,stake,legs,goal)['parlay']})
            for r in frame['view']['bets']:
                if r['reference_current']:
                    hedges.append({'frame':i,'id':r['id'],'stake':stake,'result':hedge_bet(rows,r['id'],20,2,stake)})
    fixture = r'''
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.clientWidth=520;this.attrs={};this.value='';this.disabled=false;}
  setAttribute(k,v){this.attrs[k]=v;}
  appendChild(n){this.children.push(n);return n;}
  append(...n){this.children.push(...n);}
  replaceChildren(...n){this.children=n;}
  click(){if(!this.disabled&&this.onclick)return this.onclick();}
}
const elements={};IDS.forEach(id=>elements[id]=new Element('div'));
elements['loop-data'].textContent=DATA;
for(const [id,value] of Object.entries({stake:20,legs:2,goal:'profit',heldstake:20,heldodds:2}))elements[id].value=String(value);
globalThis.document={getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag)};
globalThis.window={addEventListener:()=>{}};globalThis.location={protocol:'file:'};
const timers=new Map();let timerId=0;globalThis.setInterval=(fn,ms)=>{timers.set(++timerId,{fn,ms});return timerId;};globalThis.clearInterval=id=>timers.delete(id);
let fetchFails=false,fetchCount=0,matchingView=null;
globalThis.fetch=async path=>{fetchCount++;if(fetchFails)throw Error('Connection interrupted.');return {ok:true,json:async()=>path.includes('/api/games')?{games:[{game_id:matchingView.game_id,home_team:matchingView.home_team,away_team:matchingView.away_team,phase:matchingView.phase==='pregame'?'pre':'in'}]}:matchingView};};
function check(v,message){if(!v)throw Error(message);}
function near(a,b,message){check(a===null?b===null:Math.abs(a-b)<1e-8,message);}
function allText(e){return e.textContent+' '+e.children.map(allText).join(' ');}
'''
    checks = r'''
const payload=JSON.parse(DATA),M=window.bettingMath,rows=i=>FRAMES[i].slate.flatMap(g=>g.bets);
for(const e of EXPECTED){const a=M.evaluate(rows(e.frame).find(r=>r.id===e.id),e.stake),b=e.result;for(const key of ['probability','model_odds','reference_odds','estimated_return','expected_profit','stress_profit'])near(a[key],b[key],'stake calculation parity '+key);check(a.eligible===b.eligible,'eligibility parity');}
for(const e of PARLAYS){const a=M.bestParlay(rows(e.frame),e.stake,e.legs,e.goal),b=e.result;check(!!a===!!b,'parlay availability');if(a){check(a.picks.map(r=>r.id).join()===b.picks.map(r=>r.id).join(),'best parlay parity');for(const key of ['probability','combined_odds','expected_profit','stress_profit','estimated_return'])near(a[key],b[key],'parlay math '+key);}}
for(const e of HEDGES){const a=M.hedge(rows(e.frame),e.id,20,2,e.stake);for(const key of ['amount','floor','before','if_original_wins','if_opposite_wins'])near(a[key],e.result[key],'hedge parity '+key);}
check(fetchCount===0,'offline demo has no API dependency');
check(elements.betoptions.children.length===3,'three common bet types');
check(elements.chartcard.hidden&&elements.probchart.children.length===0,'no pre-game curve');
check(elements.homechance.textContent!=='—'&&elements.awaychance.textContent!=='—','own pre-game estimates');
check(timers.size===1&&[...timers.values()][0].ms===1400,'demo updates automatically');
elements.play.click();check(timers.size===0,'pause demo');
elements.betoptions.children[0].children[1].children[0].click();
check(elements.picks.children.length===1&&elements.slipchance.textContent!=='—','click creates bet slip');
elements.total.click();check(elements.betoptions.children.length===1,'total filter');
elements.recommend.click();check(window.courtside.chosenRows()[0].kind==='total','recommend selected market');
elements.stake10.click();check(elements.stake.value===10&&elements.slipreturn.textContent!=='—','stake updates return');
elements.all.click();elements.parlay.click();elements.recommend.click();
check(window.courtside.chosenRows().length===2&&new Set(window.courtside.chosenRows().map(r=>r.game_id)).size===2,'recommended distinct-game parlay');
elements.legs.value=4;elements.recommend.click();check(window.courtside.chosenRows().length===4,'four-leg parlay');
elements.seek.oninput({target:{value:26}});check(allText(elements.score).includes('Q2 06:00')&&!elements.chartcard.hidden,'automatic in-play layout');
check(elements.events.children.length===2,'future incidents hidden');
elements.single.click();elements.restart.click();check(elements.chartcard.hidden,'restart before tip-off');
elements.play.click();for(let i=0;i<5;i++)[...timers.values()][0].fn();check(allText(elements.score).includes('Q1 12:00')&&!elements.chartcard.hidden,'auto hand-off at tip-off');
elements.next.click();check(allText(elements.score).includes('Q1 11:00'),'next update');
elements.seek.oninput({target:{value:62}});check(elements.actiontitle.textContent==='Game finished'&&elements.recommend.disabled,'final closes betting');
elements.play.click();for(let i=0;i<65&&timers.size;i++)[...timers.values()][0].fn();
check(elements.actiontitle.textContent==='Game finished'&&timers.size===0,'replay stops at final');
(async()=>{
 matchingView=JSON.parse(JSON.stringify(payload.steps[21].view));matchingView.simulation=false;matchingView.connection_note='Polymarket connected';
 await elements.live.click();check(fetchCount===2&&elements.betoptions.children.length===3,'automatic live connection');
 check([...timers.values()].some(t=>t.ms===5000),'five-second refresh target');
 elements.hedgeresult.textContent='Old hedge';fetchFails=true;await [...timers.values()][0].fn();
 check(elements.betoptions.children.length===0&&elements.actiontitle.textContent==='Waiting for live updates','failure clears actions');
 check(elements.hedgeresult.textContent===''&&elements.slipreturn.textContent==='—','old hedge and return clear');
 fetchFails=false;await [...timers.values()][0].fn();check(elements.betoptions.children.length===3,'automatic recovery');
 elements.demo.click();check(![...timers.values()].some(t=>t.ms===5000),'demo stops live polling');
 fetchFails=true;await elements.live.click();check(timers.size===1,'initial failure retries');
 matchingView=JSON.parse(JSON.stringify(payload.pregame_steps[0].view));matchingView.pre_curve=[];fetchFails=false;
 await [...timers.values()][0].fn();check(elements.chartcard.hidden&&elements.probchart.children.length===0,'live pre-game is an estimate page');
 const heldGame=elements.games.value;
 const realNow=Date.now;Date.now=()=>realNow()+31000;
 matchingView=JSON.parse(JSON.stringify(payload.steps[0].view));
 await [...timers.values()][0].fn();check(elements.games.value===heldGame&&!elements.chartcard.hidden&&elements.matchstage.textContent.startsWith('In-play'),'same game auto-switches without a phase click');
 Date.now=realNow;elements.demo.click();print('Betting UI: model math, two phases, parlays, replay and reconnect passed');
})().catch(e=>{print(e.stack);quit(1);});
'''
    source='const IDS='+json.dumps(ids)+';const DATA='+json.dumps(data)+';const FRAMES='+json.dumps(frames)+';const EXPECTED='+json.dumps(expected)+';const PARLAYS='+json.dumps(parlays)+';const HEDGES='+json.dumps(hedges)+';\n'+fixture+scripts+checks
    path=tmp_path/'betting-ui.js';path.write_text(source)
    run=subprocess.run([str(JSC),str(path)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'reconnect passed' in run.stdout


@pytest.mark.skipif(not JSC.exists(), reason='system JavaScriptCore unavailable')
def test_hosted_page_starts_live_polling_without_a_click(tmp_path):
    html=(SAMPLE_OUTPUT/'demo.html').read_text()
    data=re.search(r'<script type="application/json" id="loop-data">(.*?)</script>',html,re.S)[1]
    scripts='\n'.join(re.findall(r'<script>\s*(.*?)</script>',html,re.S))
    ids=re.findall(r'\bid="([^"]+)"',html)
    fixture=r'''class Element{constructor(){this.children=[];this.textContent='';this.value='';this.clientWidth=520;this.attrs={};}appendChild(n){this.children.push(n);return n;}replaceChildren(...n){this.children=n;}setAttribute(k,v){this.attrs[k]=v;}click(){return this.onclick();}}
const E={};IDS.forEach(id=>E[id]=new Element());E['loop-data'].textContent=DATA;
Object.assign(E.stake,{value:'20'});Object.assign(E.legs,{value:'2'});Object.assign(E.goal,{value:'profit'});
globalThis.document={getElementById:id=>E[id],createElement:()=>new Element(),createElementNS:()=>new Element()};
globalThis.window={addEventListener:()=>{}};globalThis.location={protocol:'http:'};
const timers=new Map();let n=0;globalThis.setInterval=(fn,ms)=>{timers.set(++n,{fn,ms});return n;};globalThis.clearInterval=id=>timers.delete(id);
const board=JSON.parse(DATA).pregame_steps[0].view;let calls=0;
globalThis.fetch=async path=>{calls++;return {ok:true,json:async()=>path.includes('/api/games')?{games:[{game_id:board.game_id,home_team:board.home_team,away_team:board.away_team,phase:'pre'}]}:board};};
'''
    checks=r'''(async()=>{for(let i=0;i<30;i++)await Promise.resolve();
if(calls!==2||!E.chartcard.hidden||!E.matchstage.textContent.startsWith('Pre-game'))throw Error('hosted page must start in the actual live match phase');
if(![...timers.values()].some(t=>t.ms===5000)||[...timers.values()].some(t=>t.ms===1400))throw Error('live polling must start automatically without replay');
if(E.homechance.textContent==='—')throw Error('own win chance must be visible before tip-off');
print('Hosted page starts live automatically');})().catch(e=>{print(e.stack);quit(1);});
'''
    path=tmp_path/'hosted.js';path.write_text('const IDS='+json.dumps(ids)+';const DATA='+json.dumps(data)+';'+fixture+scripts+checks)
    run=subprocess.run([str(JSC),str(path)],capture_output=True,text=True,timeout=15)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'starts live automatically' in run.stdout
