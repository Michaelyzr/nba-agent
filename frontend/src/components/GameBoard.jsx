import { useState } from 'react';
import { date, number, percent, request } from '../api.js';
import InjuryImpact from './InjuryImpact.jsx';

function Empty({ children }) {
  return <div className="empty"><div className="court" aria-hidden="true"><span /></div>{children}</div>;
}

export function MarketForm({ game, action }) {
  const [provider, setProvider] = useState('manual');
  const [selection, setSelection] = useState('home');
  const [confirmed, setConfirmed] = useState(false);
  function submit(event) {
    event.preventDefault();
    const f = new FormData(event.currentTarget);
    const body = { provider, selection };
    if (provider === 'polymarket') Object.assign(body, { token_id: f.get('token'), mapping_confirmed: confirmed });
    else Object.assign(body, { bid_price: Number(f.get('bid')), ask_price: Number(f.get('ask')), liquidity: Number(f.get('liquidity')) });
    action(() => request(`/intelligence/games/${game.id}/markets`, { method: 'POST', body }), '报价已保存；重新运行 Agent 可生成最新判断。');
  }
  return <form className="market-form" onSubmit={submit}>
    <div className="section-title"><h3>市场报价</h3><span>胜者合约</span></div>
    <div className="form-grid">
      <label>数据来源<select value={provider} onChange={e => setProvider(e.target.value)}><option value="manual">手动研究报价</option><option value="polymarket">Polymarket 实时报价</option></select></label>
      <label>对应方向<select value={selection} onChange={e => setSelection(e.target.value)}><option value="home">{game.home.abbreviation} 主队胜</option><option value="away">{game.away.abbreviation} 客队胜</option></select></label>
      {provider === 'manual' ? <>
        <label>Best bid<input name="bid" type="number" step="0.001" min="0.001" max="0.999" required placeholder="0.550" /></label>
        <label>Best ask<input name="ask" type="number" step="0.001" min="0.001" max="0.999" required placeholder="0.570" /></label>
        <label>最优卖价深度 ($)<input name="liquidity" type="number" min="0" step="0.01" required placeholder="1000" /></label>
      </> : <>
        <label className="wide">胜者方向的 Token ID<input name="token" pattern="[0-9]+" required placeholder="输入对应球队胜出的 CLOB Token ID" /></label>
        <label className="check wide"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} required />我已核对比赛日期、对阵、胜者合约及 Token 对应方向。</label>
      </>}
    </div>
    <p className="muted">手动报价仅供比较。Polymarket 报价通过公开订单簿读取，合约映射由你核对。</p>
    <button className="button secondary" type="submit">{provider === 'manual' ? '保存报价' : '读取订单簿'}</button>
  </form>;
}

function PredictionDetail({ game, runResult }) {
  const prediction = runResult?.prediction || game.prediction;
  if (!prediction) return <Empty><h3>先建立比赛预测</h3><p>运行 Agent 后查看模型概率、数据依据和市场判断。</p></Empty>;
  const f = prediction.features;
  return <>
    <div className="section-title"><h3>模型概率</h3><span>{prediction.model_version} · 未校准</span></div>
    <div className="probability"><div><small>{game.home.abbreviation} · 主队</small><strong>{percent(prediction.home_probability)}</strong></div><div className="away"><small>{game.away.abbreviation} · 客队</small><strong>{percent(prediction.away_probability)}</strong></div></div>
    <div className="probability-bar" role="img" aria-label={`主队胜率 ${percent(prediction.home_probability)}`}><span style={{ width: percent(prediction.home_probability) }} /></div>
    <div className="detail-meta"><span>预测于 {date(prediction.as_of)}</span><span>北京时间</span></div>
    <div className="section-title spaced"><h3>概率依据</h3><span>相对 50% 的百分点贡献</span></div>
    {prediction.factors.map(factor => <div className="factor" key={factor.name}><span>{factor.name}</span><strong>{factor.value >= 0 ? '+' : ''}{(factor.value * 100).toFixed(1)} pp</strong></div>)}
    <div className="history-grid"><div><small>主队历史</small><strong>{f.home_history_games} 场</strong></div><div><small>客队历史</small><strong>{f.away_history_games} 场</strong></div><div><small>主队近 10 场</small><strong>{percent(f.home_last10_win_rate)}</strong></div><div><small>客队近 10 场</small><strong>{percent(f.away_last10_win_rate)}</strong></div></div>
    <div className="section-title spaced"><h3>市场比较与风控</h3><span>报价 + 成本缓冲</span></div>
    {['home', 'away'].map(selection => {
      const signal = runResult?.signals?.find(s => s.selection === selection);
      const decision = signal || game.decisions?.[selection];
      const quote = game.markets[selection];
      return <div className="decision" key={selection}>
        <div className="section-title"><strong>{game[selection].abbreviation} 胜</strong><span className={decision?.status === 'paper_signal' ? 'success-text' : 'muted'}>{decision?.status === 'paper_signal' ? '达到研究阈值' : '未达到研究阈值'}</span></div>
        <div className="quote-row"><span>Ask <b>{number(quote?.ask_price)}</b></span><span>成本后价差 <b>{percent(decision?.edge)}</b></span></div>
        {quote ? <small className="muted">{quote.provider} · {date(quote.observed_at)} · 卖价深度 ${number(quote.liquidity)}</small> : null}
        <ul>{decision?.reasons?.map(reason => <li key={reason}>{reason.replace('模拟交易信号', '研究信号').replace('模拟信号检查', '研究阈值检查')}</li>)}</ul>
      </div>;
    })}
    <InjuryImpact impact={prediction.injury_impact} game={game} />
    <div className="section-title spaced"><h3>新闻与伤病证据</h3><span>按状态保留 · 可追溯来源</span></div>
    <p className="muted">{prediction.news_policy ? `普通新闻窗口：${prediction.news_policy.recent_hours} 小时；长期事件按后续状态保留。长期未更新的报告仅供复核，不能确认当前伤病状态。` : '这是历史预测快照；重新运行 Agent 可应用最新的新闻筛选规则。'}</p>
    {prediction.events?.length ? prediction.events.map((event, i) => <div className="event" key={event.id || i}><strong>{event.title}</strong><small>{event.context_status || '历史快照'} · {event.event_type} · {event.player_status || '未标注状态'} · {date(event.published_at)}</small>{event.source_url && /^https?:\/\//.test(event.source_url) ? <a href={event.source_url} target="_blank" rel="noreferrer">{event.source} ↗</a> : <span>{event.source}</span>}</div>) : <p className="muted">数据库内暂无符合筛选规则的相关事件，不能据此认定无伤病。</p>}
    <details className="limitations"><summary>模型边界与执行记录</summary><ul>{prediction.limitations.map(line => <li key={line}>{line}</li>)}</ul>{prediction.pipeline?.map(step => <p key={step.step}><b>{step.step}</b> — {step.detail}</p>)}</details>
  </>;
}

export default function GameBoard({ games, action, busy }) {
  const [selected, setSelected] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const game = games.find(g => g.id === selected) || games[0];
  function choose(id) { setSelected(id); setRunResult(null); }
  function run() {
    action(async () => {
      const result = await request(`/intelligence/games/${game.id}/run`, { method: 'POST' });
      setRunResult(result);
      return result;
    }, 'Agent 已完成预测与风控留痕。');
  }
  if (!games.length) return <Empty><h2>等待比赛数据</h2><p>在「数据与设置」中同步球队及赛程，未来比赛会显示在这里。</p><p className="muted">不使用虚构比赛或预设预测结果。</p></Empty>;
  return <div className="game-layout">
    <section className="game-list" aria-label="比赛列表">
      <div className="section-title"><h2>赛程</h2><span>{games.length} 场 · 北京时间</span></div>
      {games.map(g => <button className={`game-row ${game.id === g.id ? 'selected' : ''}`} onClick={() => choose(g.id)} key={g.id}>
        <small>{date(g.tipoff_time)} · {g.season_type || '类型未知'}</small>
        <div><strong>{g.home.abbreviation}</strong><span>vs</span><strong>{g.away.abbreviation}</strong></div>
        <footer><span>{g.home.name}</span><b>{g.prediction ? percent(g.prediction.home_probability) : '待预测'}</b></footer>
      </button>)}
    </section>
    <section className="game-detail" key={game.id}>
      <div className="detail-heading"><div><p className="muted">{date(game.tipoff_time)} · {game.season}</p><h2>{game.home.abbreviation} <span>vs</span> {game.away.abbreviation}</h2><p>{game.home.name} / {game.away.name}</p></div><button className="button" disabled={busy || !['Regular Season', 'Playoffs'].includes(game.season_type)} onClick={run}>{busy ? '处理中…' : '运行 Agent ↗'}</button></div>
      {!['Regular Season', 'Playoffs'].includes(game.season_type) ? <p className="notice">季前赛或未知类型：首版不生成胜率。</p> : null}
      <PredictionDetail game={game} runResult={runResult} />
      <MarketForm game={game} action={(operation, message) => action(async () => { const result = await operation(); setRunResult(null); return result; }, message)} />
    </section>
  </div>;
}
