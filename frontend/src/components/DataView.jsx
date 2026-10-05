import { useState } from 'react';
import { number, percent, request } from '../api.js';

export default function DataView({ overview, config, season, action, busy }) {
  const [syncResult, setSyncResult] = useState(null);
  const [statsType, setStatsType] = useState('Regular Season');
  const statsLabel = { 'Regular Season': '常规赛', Playoffs: '季后赛', 'Pre Season': '季前赛' }[statsType];
  function sync(path) { action(async () => { const r = await request(path, { method: 'POST' }); setSyncResult(r); return r; }, r => r.rows_received == null ? '数据同步完成。' : r.rows_received === 0
    ? '来源未返回球员统计，未新增数据。请检查赛季、比赛类型或稍后重试。'
    : `球员统计同步完成：新增 ${r.created} 条，更新 ${r.updated} 条，跳过 ${r.skipped} 条。${r.skipped > 0 ? '部分记录未匹配，请查看下方同步结果并补齐比赛、球员或球队目录。' : ''}`); }
  return <section>
    <div className="intro-row"><div><h2>数据管线与运行配置</h2><p>先同步球队，再分别同步历史赛季和当前赛季赛程。NBA 接口可能因网络限制超时。</p></div><a className="text-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API 文档 ↗</a></div>
    <div className="data-counts">{[['球队', 'teams'], ['比赛', 'games'], ['新闻事件', 'news_events'], ['比赛预测', 'game_predictions'], ['市场快照', 'market_snapshots']].map(([label, key]) => <div key={key}><small>{label}</small><strong>{overview?.[key] ?? '—'}</strong></div>)}</div>
    <div className="section-title spaced"><h3>官方数据同步</h3><span>NBA Stats · 同步目标 {season}</span></div>
    <p className="muted">同步赛季沿用「模型分析」中的选择；需要补齐其他赛季时，请先到「模型分析」更改赛季。</p>
    <div className="sync-row"><div><strong>01 / 球队目录</strong><p>导入 NBA 球队编号、名称和缩写。</p></div><button className="button secondary" disabled={busy} onClick={() => sync('/nba/sync/teams')}>同步球队</button></div>
    <div className="sync-row"><div><strong>02 / 赛程与赛果</strong><p>同步所选赛季的开赛时间、比赛类型、状态与比分。</p></div><button className="button secondary" disabled={busy} onClick={() => sync(`/nba/sync/games?season=${season}`)}>同步 {season} 赛程</button></div>
    <div className="sync-row"><div><strong>03 / 历史常规赛赛果</strong><p>从 LeagueGameLog 补齐所选赛季已结束比赛。缺少开赛时间时，按次日 00:00（纽约）保守估计。</p></div><button className="button secondary" disabled={busy} onClick={() => sync(`/intelligence/sync/results?season=${season}`)}>同步 {season} 赛果</button></div>
    <div className="sync-row"><div><strong>04 / 球员目录</strong><p>伤病实体核验需要 NBA 球员编号；ESPN 编号不作为 NBA 编号使用。</p></div><button className="button secondary" disabled={busy} onClick={() => sync('/nba/sync/players')}>同步球员</button></div>
    <div className="sync-row"><div><strong>05 / 当前阵容</strong><p>更新球员与当前球队关系，避免交易后把伤病映射到旧球队。</p></div><button className="button secondary" disabled={busy} onClick={() => sync('/nba/sync/rosters')}>同步阵容</button></div>
    <div className="sync-row"><div><strong>06 / 球员逐场统计</strong><p>按所选赛季和比赛类型同步分钟、得分等数据。赛程/赛果与球员统计分别同步；需先有对应比赛、球员和球队。</p></div><div className="stats-sync-controls"><label>统计比赛类型<select aria-label="球员统计同步比赛类型" value={statsType} disabled={busy} onChange={e => setStatsType(e.target.value)}><option value="Regular Season">常规赛</option><option value="Playoffs">季后赛</option><option value="Pre Season">季前赛</option></select></label><button className="button secondary" disabled={busy} onClick={() => sync(`/nba/sync/player-game-stats?season=${season}&season_type=${encodeURIComponent(statsType)}`)}>同步{statsLabel}统计</button></div></div>
    <p className="muted">在「新闻与伤病」查看定时采集、原文审核和模型训练。伤病修正需要完整赛前样本及通过时间验证的模型。</p>
    {syncResult ? <details className="limitations" open><summary>最近一次同步结果</summary><pre>{JSON.stringify(syncResult, null, 2)}</pre></details> : null}
    <div className="section-title spaced"><h3>模型与风险阈值</h3><span>通过 backend/.env 配置</span></div>
    <div className="table-scroll"><table><tbody>{[
      ['概率引擎', 'Elo v1 · 参数未经训练，未校准'], ['最低球队历史样本', `${config?.min_team_history ?? '—'} 场 / 队`],
      ['最低成本后价差', percent(config?.signal_min_edge)], ['最大报价时效', `${config?.market_max_age_seconds ?? '—'} 秒`],
      ['最低最优卖价深度', `$${number(config?.market_min_liquidity)}`], ['最大买卖价差', number(config?.market_max_spread)],
      ['每份额成本缓冲', number(config?.paper_cost_buffer)],
      ['OpenAI 功能', config?.openai_configured ? '已配置 · 新闻提取与分析对话' : '未配置 · Elo 和结构化伤病采集不依赖密钥'],
    ].map(([label, value]) => <tr key={label}><th>{label}</th><td>{value}</td></tr>)}</tbody></table></div>
  </section>;
}
