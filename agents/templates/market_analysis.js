/* Read-only binary winner math shared by replay controls. */
window.newsLoopAnalysis = (() => {
  'use strict';
  function walk(q, budget, extra, maxShares = Infinity) {
    const rate = Number(q.fee_rate), exponent = Number(q.fee_exponent ?? 1);
    if (q.fee_rate == null || !Number.isFinite(rate) || rate < 0 || rate > 1 || !Number.isFinite(exponent) || exponent < .01 || exponent > 5) throw Error('Invalid fee schedule.');
    if (!Number.isFinite(budget) || budget < 0 || !Number.isFinite(extra) || extra < 0 || maxShares < 0 || Number.isNaN(maxShares)) throw Error('Invalid budget or share limit.');
    const levels = (q.asks || []).map(r => ({price: Number(r.price), size: Number(r.size)}));
    if (levels.some(r => !Number.isFinite(r.price) || r.price < .000001 || r.price > .999999 || !Number.isFinite(r.size) || r.size < 0)) throw Error('Invalid order book.');
    levels.sort((a, b) => a.price - b.price);
    let left = budget, shares = 0, notional = 0, fees = 0;
    for (const r of levels) {
      const fee = rate * Math.pow(r.price * (1-r.price), exponent);
      const quantity = Math.min(r.size, left / (r.price+fee+extra), Math.max(0, maxShares-shares));
      if (quantity <= 1e-8) continue;
      shares += quantity; notional += quantity*r.price; fees += quantity*fee;
      left -= quantity*(r.price+fee+extra);
    }
    const cost = budget-left;
    return {shares, cost, unused_budget: Math.max(0, left), fees, notional, effective_price: shares ? cost/shares : null, vwap: shares ? notional/shares : null, depth_limited: left > 1e-6};
  }
  function quoteProblems(q, s, side) {
    if (!q || typeof q !== 'object' || Array.isArray(q)) return ['No matching order book.'];
    const problems = [];
    for (const field of ['updated_at', 'observed_at']) {
      const age = (Date.parse(s.as_of)-Date.parse(q[field]))/1000;
      if (!Number.isFinite(age) || age < 0 || age > 15) problems.push('Price unavailable or stale.');
    }
    if (String(q.game_id) !== String(s.game_id)) problems.push('Different game.');
    if (q.kind !== 'moneyline' || !q.includes_overtime || !q.rules_verified) problems.push('Unverified winner rules.');
    if (!q.active) problems.push('Market unavailable.');
    if (!q.fee_verified) problems.push('Unverified fees.');
    if (q.side !== side || !q.condition_id || !q.token_id) problems.push('Incorrect outcome token.');
    if (q.bid != null && q.ask != null && q.bid >= q.ask) problems.push('Crossed or locked order book.');
    return problems;
  }
  function compare(s, quotes, budget, buffer, extra) {
    const routes = [], report = s.report || {};
    const mismatch = quotes.home && quotes.away && (quotes.home.condition_id !== quotes.away.condition_id || quotes.home.token_id === quotes.away.token_id);
    for (const side of ['home', 'away']) {
      const q = quotes[side], p = s.p_home == null ? null : side === 'home' ? s.p_home : 1-s.p_home;
      const problems = quoteProblems(q, s, side);
      if (mismatch) problems.push('Different binary markets.');
      if (p == null || !['live', 'pregame'].includes(s.quote_state)) problems.push('No current prediction.');
      let fill = null;
      if (!problems.length) {
        try {
          fill = walk(q, budget, extra);
          if (!fill.shares) problems.push('No available ask depth.');
          else if (fill.cost < Number(q.minimum_notional || 0) || fill.shares < Number(q.minimum_shares || 0)) problems.push('Below minimum order.');
        } catch(e) { problems.push(e.message); }
      }
      const row = {side, p, problems, fill, eligible: false, ev: null, robust: null, roi: null, edge: null};
      if (!problems.length) {
        row.ev = fill.shares*p-fill.cost;
        row.robust = fill.shares*Math.max(0,p-buffer/100)-fill.cost;
        row.roi = row.ev/fill.cost; row.edge = (p-fill.effective_price)*100;
        row.eligible = row.robust > 0 && Math.max(0,p-buffer/100)-fill.effective_price >= .01;
        const e = report.news_effect, before = e ? (side === 'home' ? e.p_home_before : 1-e.p_home_before) : p;
        row.beforeEV = fill.shares*before-fill.cost; row.newsEV = fill.shares*(p-before);
      }
      routes.push(row);
    }
    const candidates = routes.filter(r=>r.eligible).sort((a,b)=>b.robust-a.robust), candidate = candidates.length ? candidates[0].side : 'wait', reasons = [];
    if (report.synthetic || Object.values(quotes).some(q=>q && q.synthetic)) reasons.push('Simulation.');
    if (!s.model_trained || s.calibration_status !== 'calibrated_heldout') reasons.push('Model validation pending.');
    if (s.heldout_validation && s.heldout_validation.development_only) reasons.push('Research model.');
    if (s.quote_quality === 'provisional' || s.freshness === 'source_timestamp_unverified') reasons.push('Provisional forecast or unverified score time.');
    if (s.errors && s.errors.length || s.news_health === 'degraded') reasons.push('Incomplete live coverage.');
    return {routes, candidate, decision: reasons.length ? 'wait' : candidate, reasons};
  }
  return {compare, walk, quoteProblems};
})();
