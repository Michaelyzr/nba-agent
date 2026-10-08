(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('loop-data').textContent),$=id=>document.getElementById(id),M=window.bettingMath;
  let source='demo',phase='pregame',category='all',style='single',stepIndex=0,selected=[],slate=[],current=null;
  let replayTimer=null,refreshTimer=null,busy=false,version=0,catalog=[],catalogAt=0,poolInitialized=false,slipMessage=null;
  const included=new Set(),money=x=>x==null?'—':'$'+x.toFixed(2),pct=x=>x==null?'—':(100*x).toFixed(1)+'%',odds=x=>x==null?'—':x.toFixed(2);
  const node=(tag,text)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;return n;};
  $('forecastcard').after($('alertcard'));
  const demoRead = new Set();
  let shownAlerts=[],ackBusy=false,alertScope='';
  const alertLabels={injury_status_change:'Player status',probability_shift:'Win probability',market_move:'Market movement',news_conflict:'Conflicting reports',source_degraded:'News source degraded',source_recovered:'News source recovered',quote_stale:'Quote stale / unavailable',quote_recovered:'Quote recovered'};
  function alertRows(){return (current?.alerts||[]).map(a=>({...a,acknowledged:a.acknowledged||(source==='demo'&&demoRead.has(a.event_id))}));}
  function renderAlerts(){
    const scope=source+':'+(current?.game_id||'');
    if(scope!==alertScope){$('alertfeedback').textContent='';alertScope=scope;}
    const rows=alertRows(),type=$('alerttype').value;
    $('alerttype').replaceChildren(node('option','All types'));$('alerttype').firstChild.value='all';
    for(const t of [...new Set(rows.map(a=>a.type))].sort()){const o=node('option',alertLabels[t]||t);o.value=t;$('alerttype').appendChild(o);}
    $('alerttype').value=rows.some(a=>a.type===type)?type:'all';
    $('alertcount').textContent=rows.filter(a=>!a.acknowledged).length+' unread · '+rows.length+' total';
    $('alertstatus').textContent=!current?'Waiting for this match’s alert history.':source==='demo'?'Synthetic replay · local alerts only. No external notifications.':
      (phase==='pregame'?'Monitoring pregame changes. ':'Pregame history retained · in-play alerts are not enabled. ')+
      (current.alert_notifications_enabled?'Webhook enabled for eligible alerts.':'Webhook off · alerts are saved locally.');
    if(current?.alert_error)$('alertstatus').textContent+=' Alert processing failed on the latest poll; history is retained.';
    shownAlerts=rows.filter(a=>($('alertseverity').value==='all'||a.severity===$('alertseverity').value)&&($('alerttype').value==='all'||a.type===$('alerttype').value)&&(!$('alertunread').checked||!a.acknowledged)).reverse().slice(0,200);
    $('alertlist').replaceChildren();
    for(const a of shownAlerts){
      const item=node('article');item.className='alert-item '+(['low','medium','high'].includes(a.severity)?a.severity:'medium')+(a.acknowledged?' read':'');
      const head=node('div');head.className='alert-heading';head.appendChild(node('h3',a.title));head.appendChild(node('span',(a.severity||'medium').toUpperCase()+' · '+(a.acknowledged?'Read':'Unread')));item.appendChild(head);
      item.appendChild(node('p',a.message));
      item.appendChild(node('small',(alertLabels[a.type]||a.type)+' · '+new Date(a.observed_at).toLocaleString()));
      const p=a.payload||{};
      item.appendChild(node('small','Source: '+(p.source||p.conflict?.selected_source||(a.type.startsWith('market')||a.type.startsWith('quote')?'Polymarket':'Pregame model / retained evidence'))));
      if(source==='live')item.appendChild(node('small','Webhook: '+({pending:'Pending',sent:'Sent',failed:'Failed',not_sent:'Not sent'}[a.delivery?.status]||'Not sent')));
      const details=node('details');details.appendChild(node('summary','Trigger conditions & evidence'));
      details.appendChild(node('pre',JSON.stringify({event_id:a.event_id,observed_at:a.observed_at,...p},null,2)));item.appendChild(details);
      $('alertlist').appendChild(item);
    }
    if(!shownAlerts.length)$('alertlist').appendChild(node('p',rows.length?'No alerts match these filters.':!current?'Waiting for alert data.':'No pregame alerts recorded yet.'));
    $('alertread').disabled=ackBusy||!shownAlerts.some(a=>!a.acknowledged);
    $('alertdownload').disabled=!rows.length;
  }
  for(const id of ['alertseverity','alerttype','alertunread'])$(id).onchange=renderAlerts;
  $('alertread').onclick=async()=>{
    const ids=shownAlerts.filter(a=>!a.acknowledged).map(a=>a.event_id),gid=current?.game_id,epoch=version;
    if(!ids.length||!gid)return;
    if(source==='demo'){ids.forEach(id=>demoRead.add(id));renderAlerts();return;}
    ackBusy=true;renderAlerts();
    try{
      const response=await fetch(base+'/api/alerts/acknowledge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({game_id:gid,event_ids:ids})});
      if(!response.ok)throw Error('Could not mark alerts read. Please retry.');
      if(source==='live'&&epoch===version&&current?.game_id===gid){(current.alerts||[]).forEach(a=>{if(ids.includes(a.event_id))a.acknowledged=true;});$('alertfeedback').textContent='Read state saved.';}
    }catch(e){if(epoch===version)$('alertfeedback').textContent=e.message;}
    finally{ackBusy=false;renderAlerts();}
  };
  $('alertdownload').onclick=()=>{
    const rows=alertRows();if(!rows.length)return;
    const body=rows.map(a=>{if(source==='demo'){const {delivery,...local}=a;return local;}return a;}).map(a=>JSON.stringify(a)).join('\n')+'\n';
    const url=URL.createObjectURL(new Blob([body],{type:'application/x-ndjson'})),link=node('a');link.href=url;link.download='pregame-alerts-'+current.game_id+'.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  function stake(){const s=Number($('stake').value);if(!Number.isFinite(s)||s<1||s>100000)throw Error('Enter a stake between $1 and $100,000.');return s;}
  const timeline=[...D.pregame_steps,...D.steps],frames=()=>timeline,index=()=>stepIndex;
  function stopReplay(){if(replayTimer)clearInterval(replayTimer);replayTimer=null;}
  function activeRows(){return current?(current.bets||[]).filter(r=>category==='all'||r.kind===category):[];}
  function allRows(){return slate.filter(g=>included.has(g.game_id)).flatMap(g=>g.bets||[]).filter(r=>category==='all'||r.kind===category);}
  function chosenRows(){return selected.map(id=>slate.flatMap(g=>g.bets||[]).find(r=>r.id===id)).filter(Boolean);}
  function setMessage(title,reason){slipMessage={title,reason};$('actiontitle').textContent=title;$('actionreason').textContent=reason;}
  function fillGames(games,desired){
    $('games').replaceChildren();
    for(const game of games){const o=node('option',game.away_team+' @ '+game.home_team);o.value=game.game_id;$('games').appendChild(o);}
    $('games').value=games.some(g=>g.game_id===desired)?desired:games.length?games[0].game_id:'';
  }
  function draw(){
    const host=$('probchart');host.replaceChildren();if(!current)return;
    if(phase==='pregame')return;
    const rows=current.curve.map(r=>({minute:r.minute,p:r.probability}));
    const W=Math.max(310,host.clientWidth),H=210,L=36,T=25,B=25,end=Math.max(48,...rows.map(r=>r.minute));
    const x=m=>L+m/end*(W-L-12),y=p=>H-B-p*(H-T-B),svg=(tag,attrs)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,String(v));return e;};
    const root=svg('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Our model win chance over time'});
    const text=(value,xx,yy)=>{const e=svg('text',{x:xx,y:yy,'font-size':11,fill:'#70817c'});e.textContent=value;root.appendChild(e);};
    for(const p of [0,.5,1]){root.appendChild(svg('line',{x1:L,x2:W-12,y1:y(p),y2:y(p),stroke:'#dbe5df'}));text(Math.round(p*100)+'%',0,y(p)+4);}
    if(phase==='inplay')for(let q=0;q<4;q++)text('Q'+(q+1),x(q*12+6)-8,15);
    const valid=rows.filter(r=>r.p!=null);
    root.appendChild(svg('path',{d:valid.map((r,i)=>(i?'L':'M')+x(r.minute)+','+y(r.p)).join(' '),fill:'none',stroke:'#08775c','stroke-width':2.4}));
    for(const event of current.events||[]){const at=current.curve.find(r=>r.clock===event.clock);if(!at||phase==='pregame')continue;const dot=svg('circle',{cx:x(at.minute),cy:y(at.probability),r:4,fill:'#fff',stroke:'#08775c'}),title=svg('title');title.textContent=event.clock+' · '+event.headline;dot.appendChild(title);root.appendChild(dot);}
    host.appendChild(root);
  }
  function renderMarkets(){
    $('betoptions').replaceChildren();
    const types=['moneyline','spread','total'];
    for(const kind of types){const rows=activeRows().filter(r=>r.kind===kind);if(!rows.length)continue;
      const group=node('div');group.className='bet-group';group.appendChild(node('h3',{moneyline:'Winner · full game',spread:'Point spread · full game',total:'Total points · full game'}[kind]));
      const choices=node('div');choices.className='choices';
      for(const row of rows){const r=M.evaluate(row,stake()),button=node('button');button.className='choice'+(selected.includes(row.id)?' selected':'');button.setAttribute('aria-pressed',selected.includes(row.id));
        button.disabled=row.probability==null;
        const head=node('div');head.className='choice-head';head.appendChild(node('span',row.label));head.appendChild(node('span',pct(row.probability)+' chance'));button.appendChild(head);
        const prices=node('div');prices.className='comparison-values';
        const own=node('div');own.appendChild(node('small','Our model odds'));own.appendChild(node('strong',odds(r.model_odds)));prices.appendChild(own);
        const ref=node('div');ref.appendChild(node('small','Polymarket odds'));ref.appendChild(node('strong',odds(row.reference_quote?.ask?1/row.reference_quote.ask:null)));prices.appendChild(ref);button.appendChild(prices);
        const detail=node('div');detail.className='helper';detail.textContent=!r.reference_current&&row.reference_quote?'Last seen · waiting for a fresh price':row.reference_status;button.appendChild(detail);
        button.onclick=()=>{selectPick(row);};choices.appendChild(button);
      }
      group.appendChild(choices);$('betoptions').appendChild(group);
    }
    if(!activeRows().length)$('betoptions').appendChild(node('p','Waiting for model and market options.'));
  }
  function selectPick(row){
    if(row.probability==null){setMessage('Waiting for our model','Current game data is required before evaluating this pick.');return;}
    if(style==='single')selected=[row.id];else{
      if(selected.includes(row.id))selected=selected.filter(id=>id!==row.id);
      else{selected=selected.filter(id=>!slate.flatMap(g=>g.bets).some(r=>r.id===id&&r.game_id===row.game_id));if(selected.length>=4){setMessage('Four picks maximum','Remove one pick before adding another game.');return;}selected.push(row.id);}
    }
    slipMessage=null;renderMarkets();renderSlip();
  }
  function renderPool(){
    $('gamepool').replaceChildren();
    for(const game of slate){const button=node('button',game.away_team+' @ '+game.home_team);button.setAttribute('aria-selected',included.has(game.game_id));button.onclick=()=>{if(included.has(game.game_id))included.delete(game.game_id);else included.add(game.game_id);renderPool();};$('gamepool').appendChild(button);}
  }
  function renderSlip(){
    $('picks').replaceChildren();const rows=chosenRows();
    for(const row of rows){const p=node('div');p.className='pick';const label=node('div',row.label);label.appendChild(node('small',row.match+' · '+row.market));p.appendChild(label);const remove=node('button','×');remove.className='remove';remove.setAttribute('aria-label','Remove '+row.label);remove.onclick=()=>{selected=selected.filter(id=>id!==row.id);slipMessage=null;renderMarkets();renderSlip();};p.appendChild(remove);$('picks').appendChild(p);}
    if(!rows.length)$('picks').appendChild(Object.assign(node('p','Tap a betting option, or find the best match for your goal.'),{className:'empty'}));
    let result=null;
    if(style==='single'&&rows.length)result=M.evaluate(rows[0],stake());
    if(style==='parlay'&&rows.length>=2)result=M.parlay(rows,stake());
    $('slipchance').textContent=pct(result?.probability);$('slipodds').textContent=odds(style==='parlay'?result?.combined_odds:result?.reference_odds);
    $('slipreturn').textContent=money(result?.estimated_return);$('slipev').textContent=money(result?.expected_profit);
    $('oddslabel').textContent=style==='parlay'?'Combined reference odds':'Polymarket reference odds';$('returnlabel').textContent=style==='parlay'?'Illustrative return if all win':'Return if it wins';
    $('recommend').textContent=style==='parlay'?'Find my best parlay':'Find my best single bet';$('parlaycontrols').hidden=style!=='parlay';
    $('recommend').disabled=!current||current.quote_state==='final';
    if(!slipMessage){
      let title='Choose your bet',reason='Tap an option yourself, or let our model compare the available choices.';
      if(style==='parlay'&&rows.length<2&&rows.length){title='Add another game';reason='A parlay needs at least two picks from different games.';}
      else if(result){title=result.stress_profit>0?'Model value pick':'Wait or choose another bet';reason=result.stress_profit>0?'Our estimated chance beats the reference price after costs and a model stress buffer.':'The current reference price does not show positive value after costs and a model stress buffer.';if(result.reference_current===false){title='Waiting for a current reference';reason='Our probability is available; the reference is older or missing. No current betting recommendation.';}}
      else if(rows.length){title='Waiting for a current reference';reason='The selected picks need matching, current prices before we can estimate a return.';}
      if(current?.quote_state==='final'){title='Game finished';reason='Betting is closed. Replay an earlier update or choose another game.';}
      $('actiontitle').textContent=title;$('actionreason').textContent=reason;
    }else{$('actiontitle').textContent=slipMessage.title;$('actionreason').textContent=slipMessage.reason;}
    $('slipnote').textContent=style==='parlay'?'Different games assumed independent. This return uses reference odds, not an actual parlay quote. Confirm the combined price before betting.':'Return includes your stake; average profit is a model estimate, not a guaranteed outcome.';
    if(result&&result.unused_stake>1e-6)$('slipnote').textContent+=' Available reference depth covers only '+money(result.cost)+' of your stake.';
    if(result?.research_only!==false&&rows.length)$('slipnote').textContent+=' Research model; estimates are not yet calibrated for betting.';
  }
  function render(view){
    $('hedgeresult').textContent='';
    current=view;$('awayname').textContent=view.away_team;$('homename').textContent=view.home_team;$('score').replaceChildren();
    const score=view.score;$('score').appendChild(node('span',score?score.away_score+' – '+score.home_score:'vs'));
    $('score').appendChild(node('small',view.quote_state==='final'?'Final':view.clock));
    $('updatestatus').textContent=source==='demo'?'Demo · simulated scores and reference prices':'Live data · refresh target 5 seconds · '+new Date(view.as_of).toLocaleTimeString();
    $('slipbadge').textContent=source==='demo'?'DEMO':'MODEL ESTIMATE';
    $('matchstage').textContent=view.quote_state==='final'?'Final':phase==='pregame'?'Pre-game · auto-updating':'In-play · auto-updating';
    $('forecasttitle').textContent=phase==='pregame'?'Our estimated win chance':'Our live win chance';
    $('tipstatus').textContent=phase==='pregame'?'Tip-off '+(view.tip_time?new Date(view.tip_time).toLocaleString():'pending')+' · switches automatically when the game starts.':'The page follows the score, clock and latest match news automatically.';
    $('awayforecastname').textContent=view.away_team;$('homeforecastname').textContent=view.home_team;
    const win=view.bets.find(r=>r.kind==='moneyline'&&r.side==='home')?.probability;
    $('homechance').textContent=pct(win);$('awaychance').textContent=pct(win==null?null:1-win);
    $('forecastnote').textContent=phase==='pregame'?'Estimated from our historical model and retrieved pre-game news.':'Estimated from our model using the current score, time remaining and retrieved news.';
    if(view.quote_state==='final')$('forecastnote').textContent='Final score confirmed. Forecasts and betting recommendations are closed.';
    $('chartcard').hidden=phase==='pregame';
    $('connection').textContent=view.connection_note||(source==='demo'?'Polymarket-format prices are simulated.':'Model predictions are independent of Polymarket reference prices.');
    $('footnote').textContent=source==='demo'?'Synthetic replay. Our model supplies all probabilities; market prices are simulated references.':'Our model supplies all probabilities. Polymarket is a price reference. Research estimates; no bets are placed.';
    if(view.news_health==='degraded')$('footnote').textContent+=' Player-news coverage is incomplete.';
    $('marketlink').href=(view.bets.find(r=>r.reference_quote?.url)?.reference_quote.url)||'https://polymarket.com/sports/nba';
    $('modelscore').textContent=view.own_model?'Projected total '+view.own_model.total_mean.toFixed(1):'Model data pending';
    $('charttitle').textContent='Win chance · Q1–Q4';
    $('chartnote').textContent='Home-team win chance from our score, clock and news model. Incident markers show the first observed game clock.';
    $('newsstatus').textContent=(view.news_health==='degraded'?'Some news feeds are unavailable. ':'')+'News checked '+new Date(view.as_of).toLocaleTimeString()+'.';
    $('events').replaceChildren();for(const e of [...view.events].reverse().slice(0,20)){
      const item=node('div');item.className='feed-item';
      const when=e.published_at?new Date(e.published_at).toLocaleString():'';
      item.appendChild(node('small',(e.clock||'Pre-game')+' · '+(e.synthetic?'Simulated update':e.source)+(when?' · '+when:'')));
      item.appendChild(node('p',e.headline));
      if(e.selected&&e.impact_pp!=null)item.appendChild(node('small',view.home_team+' win chance '+(e.impact_pp>=0?'+':'')+e.impact_pp.toFixed(2)+' pp'));
      else item.appendChild(node('small',e.news_only?'Retrieved news · no automatic probability adjustment':'No additional model change'));
      if(!e.synthetic&&/^https:\/\//.test(e.url||'')){const link=node('a','Read source ↗');link.href=e.url;link.target='_blank';link.rel='noopener noreferrer';item.appendChild(node('div')).appendChild(link);}
      $('events').appendChild(item);
    }
    if(!view.events.length)$('events').appendChild(node('p','No new match news retrieved yet. Monitoring continues.'));
    $('replaycontrols').hidden=source!=='demo';$('seek').max=frames().length-1;$('seek').value=index();$('progress').textContent=(index()+1)+' / '+frames().length;$('next').disabled=index()===frames().length-1;$('play').textContent=replayTimer?'Pause demo':'Play demo';
    $('hedgesection').hidden=view.quote_state==='final';
    const held=$('heldpick').value;$('heldpick').replaceChildren();const empty=node('option','Choose your original bet');empty.value='';$('heldpick').appendChild(empty);
    for(const row of view.bets){const o=node('option',row.label);o.value=row.id;$('heldpick').appendChild(o);}$('heldpick').value=view.bets.some(r=>r.id===held)?held:'';
    renderPool();renderMarkets();renderSlip();draw();renderAlerts();
  }
  function showDemo(){
    try{const previous=$('games').value;slate=frames()[index()].slate;phase=frames()[index()].view.phase;fillGames(slate,previous);if(!poolInitialized){slate.forEach(g=>included.add(g.game_id));poolInitialized=true;}
      const valid=new Set(slate.flatMap(g=>g.bets).map(r=>r.id));selected=selected.filter(id=>valid.has(id));render(slate.find(g=>g.game_id===$('games').value)||slate[0]);
    }catch(e){setMessage('Check your inputs',e.message);}
  }
  function recommend(){
    try{if(style==='single'){const pick=M.single(activeRows(),stake(),$('goal').value);if(!pick){setMessage('No clear value yet','No positive-value choice in this market after costs. Try another bet type or wait.');return;}selected=[pick.id];slipMessage={title:'Your model pick: '+pick.label,reason:'Best '+($('goal').value==='chance'?'win chance':'estimated profit')+' among the positive-value options in your chosen market.'};}
      else{const result=M.bestParlay(allRows(),stake(),Number($('legs').value),$('goal').value);if(!result){setMessage('No suitable parlay yet','Choose more games or fewer legs. Each pick needs a current reference and positive model value.');return;}selected=result.picks.map(r=>r.id);slipMessage={title:result.picks.length+'-leg model parlay',reason:'Best '+($('goal').value==='chance'?'estimated win chance':'estimated profit')+' within your selected games, bet types and leg count.'};}renderMarkets();renderSlip();
    }catch(e){setMessage('Check your choices',e.message);}
  }
  function clearLive(message){current=null;slate=[];renderAlerts();$('hedgeresult').textContent='';$('recommend').disabled=true;$('betoptions').replaceChildren();$('score').replaceChildren(node('small','Waiting for current data'));$('probchart').replaceChildren();$('homechance').textContent=$('awaychance').textContent='—';$('awayname').textContent=$('homename').textContent='—';$('homeforecastname').textContent=$('awayforecastname').textContent='Model data pending';$('events').replaceChildren(node('p','Waiting for the latest match news.'));$('newsstatus').textContent='News refresh interrupted; retrying automatically.';$('picks').replaceChildren(node('p','Waiting for current betting options.'));for(const id of ['slipchance','slipodds','slipreturn','slipev'])$(id).textContent='—';$('connection').textContent=message;setMessage('Waiting for live updates','Refreshing our model and the reference prices.');}
  const base=location.protocol==='file:'?'http://127.0.0.1:8766':'';
  async function api(path){const response=await fetch(base+path,{cache:'no-store'}),data=await response.json();if(!response.ok)throw Error(data.error||'Live data unavailable.');return data;}
  async function refresh(){
    if(source!=='live')return;
    if(current){slipMessage=null;try{renderMarkets();renderSlip();}catch(e){setMessage('Check your stake',e.message);}}
    if(busy)return;busy=true;const epoch=version;
    try{
      if(Date.now()-catalogAt>=30000||!catalog.length){catalog=(await api('/api/games')).games;catalogAt=Date.now();}
      if(source!=='live'||epoch!==version)return;
      const available=catalog;
      const previous=$('games').value;fillGames(available,previous);
      if(!available.length){clearLive('No NBA games available right now. Try the demo; discovery will retry automatically.');return;}
      if(previous&&previous!==$('games').value){selected=[];slipMessage=null;}
      // For a parlay, refresh a bounded slate rather than relying on saved quotes.
      const chosen=available.find(g=>g.game_id===$('games').value);
      const wanted=style==='parlay'?[chosen,...available.filter(g=>g!==chosen).slice(0,11)]:[chosen];
      const boards=await Promise.all(wanted.map(g=>api('/api/betting?game_id='+encodeURIComponent(g.game_id))));
      if(source!=='live'||epoch!==version)return;
      const view=boards.find(g=>g.game_id===$('games').value)||boards[0];
      phase=view.phase;
      slate=boards;if(!poolInitialized){slate.forEach(g=>included.add(g.game_id));poolInitialized=true;}
      const ids=new Set(slate.flatMap(g=>g.bets).map(r=>r.id));selected=selected.filter(id=>ids.has(id));
      slipMessage=null;render(view);
    }catch(e){if(source==='live'&&epoch===version)clearLive(e.message+' Retrying automatically.');}finally{busy=false;}
  }
  for(const kind of ['all','moneyline','spread','total'])$(kind).onclick=()=>{category=kind;for(const k of ['all','moneyline','spread','total'])$(k).setAttribute('aria-selected',k===kind);slipMessage=null;try{renderMarkets();renderSlip();}catch(e){setMessage('Check your inputs',e.message);}};
  for(const s of ['single','parlay'])$(s).onclick=()=>{style=s;selected=[];slipMessage=null;version++;if(s==='parlay'){included.clear();poolInitialized=false;}for(const k of ['single','parlay'])$(k).setAttribute('aria-selected',k===s);if(source==='demo')showDemo();else{refresh();renderSlip();}};
  $('games').onchange=()=>{slipMessage=null;if(style==='single')selected=[];if(source==='demo')showDemo();else{version++;clearLive('Updating this match…');refresh();}};
  $('stake').oninput=()=>{slipMessage=null;try{renderMarkets();renderSlip();}catch(e){setMessage('Check your stake',e.message);for(const id of ['slipchance','slipodds','slipreturn','slipev'])$(id).textContent='—';}};
  for(const n of [10,20,50])$('stake'+n).onclick=()=>{$('stake').value=n;$('stake').oninput();};
  $('goal').onchange=()=>{slipMessage=null;renderSlip();};$('recommend').onclick=recommend;
  $('protect').onclick=()=>{try{const result=M.hedge(current?.bets||[],$('heldpick').value,Number($('heldstake').value),Number($('heldodds').value),stake());$('hedgeresult').textContent=result.floor>result.before?'Consider '+result.label+' for '+money(result.amount)+'. Lower game-result profit improves from '+money(result.before)+' to '+money(result.floor)+'. '+(result.balanced?'The two completed-game payouts are balanced.':'This is a partial hedge; some exposure remains.')+' Confirm the final fill and cancellation rules.':'Keeping your current bet has a better lower payout at these prices.';}catch(e){$('hedgeresult').textContent=e.message;}};
  function startReplay(){
    if(replayTimer)return;
    replayTimer=setInterval(()=>{stepIndex=Math.min(stepIndex+1,timeline.length-1);if(stepIndex===timeline.length-1)stopReplay();slipMessage=null;showDemo();},1400);
  }
  $('seek').oninput=e=>{stopReplay();stepIndex=Number(e.target.value);slipMessage=null;showDemo();};
  $('next').onclick=()=>{stopReplay();stepIndex=Math.min(stepIndex+1,timeline.length-1);slipMessage=null;showDemo();};
  $('restart').onclick=()=>{stopReplay();stepIndex=0;slipMessage=null;showDemo();};
  $('play').onclick=()=>{if(replayTimer){stopReplay();showDemo();return;}if(stepIndex===timeline.length-1)stepIndex=0;startReplay();showDemo();};
  $('demo').onclick=()=>{stopReplay();stepIndex=0;version++;source='demo';selected=[];slipMessage=null;included.clear();poolInitialized=false;if(refreshTimer)clearInterval(refreshTimer);refreshTimer=null;$('demo').setAttribute('aria-selected',true);$('live').setAttribute('aria-selected',false);startReplay();showDemo();};
  $('live').onclick=async()=>{stopReplay();version++;source='live';catalogAt=0;catalog=[];selected=[];slipMessage=null;included.clear();poolInitialized=false;$('demo').setAttribute('aria-selected',false);$('live').setAttribute('aria-selected',true);$('replaycontrols').hidden=true;clearLive('Connecting to our model and live reference prices…');if(!refreshTimer)refreshTimer=setInterval(refresh,5000);await refresh();};
  window.addEventListener('resize',draw);window.courtside={recommend,chosenRows};
  if(location.protocol==='file:'){startReplay();showDemo();}else{$('live').click();}
})();
