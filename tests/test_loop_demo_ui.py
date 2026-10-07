"""Execute the English Streamlit app and the actual exported browser scripts."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from agents.loop_demo import SAMPLE_OUTPUT
from agents.market_analysis import compare_routes
from agents.match_view import hedge_options


def test_streamlit_customer_replay_and_position_controls():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/'loop_demo_app.py', default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'Courtside'
    assert app.radio[0].value == 'Demo replay'
    assert not app.get('file_uploader')
    assert app.metric[0].label == 'Suggested spend'
    app.number_input(key='news_loop_sample_budget').set_value(10.).run()
    assert not app.exception
    app.checkbox(key='news_loop_sample_position_enabled').check().run()
    assert not app.exception
    assert any(m.label == 'Lower game-result P/L' for m in app.metric)
    app.button(key='news_loop_sample_reset').click().run()
    assert not app.exception and app.slider[0].value == 0
    app.button(key='news_loop_sample_next').click().run()
    assert not app.exception and app.slider[0].value == 1
    payload = json.loads((SAMPLE_OUTPUT/'result.json').read_text())
    live = [r for r in payload['steps'] if r['phase'] == 'inplay']
    app.slider[0].set_value(len(live)-1).run()
    assert not app.exception and any(s.value == 'Game finished' for s in app.subheader)
    assert app.metric[0].value == '$0.00'


JSC = Path('/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc')


@pytest.mark.skipif(not JSC.exists(), reason='system JavaScriptCore is not installed')
def test_actual_html_replay_math_and_automatic_live_failure_recovery(tmp_path):
    html = (SAMPLE_OUTPUT/'demo.html').read_text()
    data = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', html, re.S)[1]
    script = '\n'.join(re.findall(r'<script>\s*(.*?)</script>', html, re.S))
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert '<html lang="en">' in html and 'type="file"' not in html
    assert 'auditbody' not in html and 'scenario-sigma' not in html
    fixture = r'''
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.clientWidth=520;this.attrs={};this.value='';this.checked=false;}
  setAttribute(k,v){this.attrs[k]=v;}
  appendChild(n){this.children.push(n);return n;}
  append(...n){this.children.push(...n);}
  replaceChildren(...n){this.children=n;}
  click(){if(this.onclick)return this.onclick();}
}
const elements={};IDS.forEach(id=>elements[id]=new Element('div'));
elements['loop-data'].textContent=DATA;
for(const [id,value]of Object.entries({budget:100,'position-side':'home','position-shares':100,'position-cost':60}))elements[id].value=String(value);
globalThis.document={getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag)};
globalThis.window={addEventListener:()=>{}};globalThis.location={protocol:'file:'};
const timers=new Map();let timerId=0;globalThis.setInterval=(fn,ms)=>{timers.set(++timerId,{fn,ms});return timerId;};globalThis.clearInterval=id=>timers.delete(id);
globalThis.URLSearchParams=class{constructor(data){this.data=data;}set(k,v){this.data[k]=v;}toString(){return Object.entries(this.data).map(([k,v])=>k+'='+encodeURIComponent(v)).join('&');}};
let fetchFails=false,fetchCount=0,matchingView=null;
globalThis.fetch=async path=>{fetchCount++;if(fetchFails)throw Error('Connection interrupted.');return{ok:true,json:async()=>path.includes('/api/games')?{games:[{game_id:'1',label:'ORL @ BOS'}]}:matchingView};};
function check(condition,message){if(!condition)throw Error(message);}
'''
    checks = r'''
const live=JSON.parse(DATA).steps;
for(const expected of EXPECTED){
  const s=live[expected.index].snapshot,actual=window.newsLoopAnalysis.compare(s,s.market_quotes,expected.budget,expected.buffer,.002);
  check(actual.candidate===expected.result.candidate&&actual.decision===expected.result.decision,'Python/browser route decision parity');
  for(let i=0;i<2;i++){const a=actual.routes[i],b=expected.result.routes[i];check(a.ev===null?b.ev===null:Math.abs(a.ev-b.ev)<1e-8,'EV parity');check(a.robust===null?b.robust_ev===null:Math.abs(a.robust-b.robust_ev)<1e-8,'stress EV parity');}
}
for(const expected of HEDGES){const s=live[expected.index].snapshot,actual=window.matchMath.hedgeOptions(s,s.market_quotes,expected.position,expected.budget);check(actual.length===expected.result.length,'hedge option count');for(let i=0;i<actual.length;i++){check(actual[i].kind===expected.result[i].kind,'hedge kind');for(const field of ['shares','added_cost','floor','if_held_wins','if_other_wins'])check(Math.abs(actual[i][field]-expected.result[i][field])<1e-8,'hedge payout parity: '+field);}}
check(fetchCount===0,'demo works offline');
check(elements.oddsrows.children.length===2 && elements.probchart.children.length===1,'two outcomes and one chart');
check(elements.clock.textContent==='Q2 06:00','starts at a meaningful player incident');
check(elements.actiontitle.textContent.startsWith('Buy '),'demo illustrates a simulated candidate');
check(elements.events.children.length===2,'only observed injury reports visible');
elements['position-enabled'].checked=true;elements['position-enabled'].onchange();
check(elements.hedgeoutcomes.children.length===1 && elements.resultlabel.textContent==='Lower game-result P/L','position-aware hedge');
elements.restart.click();check(elements.clock.textContent==='Q1 12:00','restart at tip-off');
check(elements.events.children.length===1&&elements.events.children[0].textContent.startsWith('No new'),'future news hidden');
elements['position-enabled'].checked=false;elements.next.click();check(elements.clock.textContent==='Q1 11:00','next match update');
elements.seek.oninput({target:{value:live.length-1}});check(elements.clock.textContent==='Final'&&elements.actiontitle.textContent==='Game finished','settlement ends action');
elements.play.click();for(let i=0;i<live.length+2&&timers.size;i++)[...timers.values()][0].fn();
check(elements.clock.textContent==='Final'&&timers.size===0,'play ends at final');
(async()=>{
  matchingView=JSON.parse(JSON.stringify(live[21].view));matchingView.simulation=false;matchingView.connection_note='Polymarket connected';
  await elements.live.click();
  check(elements.modebadge.textContent==='LIVE'&&fetchCount===2,'live auto-discovers and fetches without upload');
  check([...timers.values()].some(t=>t.ms===5000),'live five-second refresh');
  fetchFails=true;await [...timers.values()][0].fn();
  check(elements.oddsrows.children.length===0&&elements.score.textContent==='— – —','old quotes and scores clear on failed refresh');
  check(elements.actiontitle.textContent==='Waiting for live data'&&elements.actionamount.textContent==='$0.00','no stale action survives');
  fetchFails=false;await [...timers.values()][0].fn();check(elements.oddsrows.children.length===2,'live connection recovers automatically');
  elements.demo.click();check(timers.size===0&&elements.modebadge.textContent==='SIMULATION','demo/live modes isolated');
  fetchFails=true;await elements.live.click();check(timers.size===1&&elements.actiontitle.textContent==='Waiting for live data','initial connection failure also retries');
  elements.demo.click();
  print('English customer dashboard: math parity, replay, hedge, automatic API polling and reconnect passed');
})().catch(e=>{print(e.stack);quit(1);});
'''
    payload = json.loads(data)
    snapshots = [r['snapshot'] for r in payload['steps']]
    expected = [{'index': i, 'budget': budget, 'buffer': buffer, 'result': compare_routes(s, s['market_quotes'], budget, buffer)}
                for i, s in enumerate(snapshots) for budget in (10, 100) for buffer in (0, 5)]
    hedges = [{'index': i, 'position': {'side': side, 'shares': 100, 'cost': 60}, 'budget': budget,
               'result': hedge_options(s, s['market_quotes'], {'side': side, 'shares': 100, 'cost': 60}, budget)}
              for i, s in enumerate(snapshots) for side in ('home', 'away') for budget in (10, 1000)]
    source = 'const IDS='+json.dumps(ids)+';const DATA='+json.dumps(data)+';const EXPECTED='+json.dumps(expected)+';const HEDGES='+json.dumps(hedges)+';\n'+fixture+script+checks
    path = tmp_path/'demo-test.js'
    path.write_text(source)
    run = subprocess.run([str(JSC), str(path)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout+run.stderr
    assert 'reconnect passed' in run.stdout
