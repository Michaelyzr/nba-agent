/* Local bet-slip math. Probabilities arrive from our model, never from the reference. */
window.bettingMath = (() => {
  'use strict';
  function fill(q, stake, cap=Infinity) {
    if(!q || !Number.isFinite(q.fee_rate) || q.fee_rate<0 || q.fee_rate>1) return null;
    let left=stake,n=0,fees=0;
    const exponent=q.fee_exponent??1,levels=[...(q.asks||[])].sort((a,b)=>a.price-b.price);
    if(!Number.isFinite(exponent)||exponent<.01||exponent>5)return null;
    for(const r of levels) {
      if(!Number.isFinite(r.price)||r.price<.000001||r.price>.999999||!Number.isFinite(r.size)||r.size<0)return null;
      const fee=q.fee_rate*Math.pow(r.price*(1-r.price),exponent),unit=r.price+fee+.002,qty=Math.min(r.size,left/unit,Math.max(0,cap-n));
      if(qty<=1e-8)continue;
      n+=qty;fees+=qty*fee;left-=qty*unit;
    }
    const cost=stake-left;
    if(!n || cost<(q.minimum_notional||0) || n<(q.minimum_shares||0))return null;
    return {n,cost,unused:Math.max(0,left),odds:n/cost,fees};
  }
  function evaluate(row,stake=20) {
    if(!Number.isFinite(stake)||stake<1||stake>100000)throw Error('Enter a stake between $1 and $100,000.');
    row={...row};
    if(!row.simulation){const times=[row.model_as_of,row.reference_quote?.observed_at,row.reference_quote?.updated_at];if(times.some(t=>{const age=(Date.now()-Date.parse(t))/1000;return !Number.isFinite(age)||age<0||age>15;}))row.reference_current=false;}
    const f=fill(row.reference_quote,stake),p=row.probability,usable=!!f&&row.reference_current&&p!=null;
    const profit=usable?f.n*p-f.cost:null,stress=usable?f.n*Math.max(0,p-.02)-f.cost:null;
    return {...row,stake,reference_odds:f?f.odds:null,estimated_return:usable?f.n:null,cost:f?f.cost:null,
      unused_stake:f?f.unused:stake,expected_profit:profit,stress_profit:stress,edge_pp:usable?(p-1/f.odds)*100:null,
      eligible:!!(usable&&stress>0&&p-.02-1/f.odds>=.01)};
  }
  function single(rows,stake,goal) {
    const options=rows.map(r=>evaluate(r,stake)).filter(r=>r.eligible);
    options.sort((a,b)=>goal==='chance'?b.probability-a.probability||b.stress_profit-a.stress_profit:b.stress_profit-a.stress_profit||b.probability-a.probability);
    return options[0]||null;
  }
  function parlay(rows,stake) {
    if(rows.length<2||rows.length>4||new Set(rows.map(r=>r.game_id)).size!==rows.length)throw Error('Choose two to four picks, one per game.');
    const picks=rows.map(r=>evaluate(r,stake));
    if(picks.some(r=>!r.reference_current||!r.reference_odds||r.probability==null))return null;
    const probability=picks.reduce((p,r)=>p*r.probability,1),combined_odds=picks.reduce((o,r)=>o*r.reference_odds,1),stress=picks.reduce((p,r)=>p*Math.max(0,r.probability-.02),1);
    return {picks,probability,combined_odds,model_odds:probability?1/probability:null,estimated_return:stake*combined_odds,
      expected_profit:stake*(probability*combined_odds-1),stress_profit:stake*(stress*combined_odds-1),stake};
  }
  function bestParlay(rows,stake,legs,goal) {
    const groups=new Map();
    for(const row of rows)if(evaluate(row,stake).eligible){if(!groups.has(row.game_id))groups.set(row.game_id,[]);groups.get(row.game_id).push(row);}
    if(groups.size>12)throw Error('Choose at most twelve games.');
    const values=[...groups.values()];let best=null;
    function visit(start,chosen) {
      if(chosen.length===legs){const result=parlay(chosen,stake);if(!result||result.stress_profit<=0)return;
        const a=goal==='chance'?result.probability:result.stress_profit,b=best?(goal==='chance'?best.probability:best.stress_profit):-Infinity;
        const second=goal==='chance'?result.stress_profit:result.probability,previous=best?(goal==='chance'?best.stress_profit:best.probability):-Infinity;
        if(a>b||a===b&&second>previous)best=result;return;}
      for(let i=start;i<=values.length-(legs-chosen.length);i++)for(const row of values[i])visit(i+1,[...chosen,row]);
    }
    visit(0,[]);return best;
  }
  function hedge(rows,id,amount,original,budget) {
    if(!Number.isFinite(amount)||amount<=0||!Number.isFinite(original)||original<=1||!Number.isFinite(amount*original)||!Number.isFinite(budget)||budget<1||budget>100000)throw Error('Check your original stake, placed odds and hedge budget.');
    const row=rows.find(r=>r.id===id),other=row?rows.find(r=>r.contract_id===row.contract_id&&r.game_id===row.game_id&&r.side!==row.side):null;
    if(!other||!evaluate(other,budget).reference_current)throw Error('A current price for the exact opposite bet is required.');
    const f=fill(other.reference_quote,budget,amount*original);
    if(!f)throw Error('Not enough current reference depth for this hedge.');
    const a=amount*original-amount-f.cost,b=f.n-amount-f.cost;
    return {label:other.label,amount:f.cost,floor:Math.min(a,b),before:-amount,if_original_wins:a,if_opposite_wins:b,balanced:Math.abs(f.n-amount*original)<1e-6};
  }
  return {evaluate,single,parlay,bestParlay,hedge};
})();
