import { useState } from 'react';
import { date, number, request } from '../api.js';

const reviewNames = { pending: '待审核', approved: '已核验', rejected: '已排除' };
const sourceNames = { nba_official: 'NBA 官方 PDF', espn_injuries: 'ESPN 伤病', espn_rss: 'ESPN 新闻摘要' };

export default function NewsView({ status, events, action, busy, season }) {
  const [filter, setFilter] = useState('pending');
  const [operation, setOperation] = useState(null);
  const [document, setDocument] = useState(null);
  const rows = (events || []).filter(e => filter === 'all' || e.review === filter);
  function run(path, body, message) {
    action(async () => { const result = await request(path, { method: 'POST', body }); setOperation(result); return result; }, message);
  }
  function preference(key, value) {
    action(() => request('/evidence/preferences', { method: 'PUT', body: { ...status.preferences, [key]: value } }), '自动任务设置已更新。');
  }
  return <section>
    <div className="intro-row"><div><h2>新闻采集与伤病状态</h2><p>官方报告与 ESPN 状态按原始发布时间记录。新闻模型提取事实，审核后才进入状态和特征。</p></div><button className="button" disabled={busy} onClick={() => run(`/evidence/collect?season=${season}`, undefined, '可用来源已采集；各来源状态见下方。')}>立即采集 ↗</button></div>
    <div className="metric-strip three"><div><small>来源版本</small><strong>{status?.documents ?? '—'}</strong></div><div><small>待审核事件</small><strong>{status?.pending ?? '—'}</strong></div><div><small>赛前特征快照</small><strong>{status?.snapshots ?? '—'}</strong></div></div>
    <div className="news-controls">
      <label className="check"><input type="checkbox" disabled={busy || !status} checked={status?.preferences?.auto_collect ?? false} onChange={e => preference('auto_collect', e.target.checked)} />自动采集 · 每 {Math.round((status?.poll_seconds || 900) / 60)} 分钟</label>
      <label className="check"><input type="checkbox" disabled={busy || !status?.openai_configured} checked={status?.preferences?.auto_extract ?? false} onChange={e => preference('auto_extract', e.target.checked)} />自动模型提取 · 会消耗已配置的 OpenAI API 额度</label>
      <button className="button secondary" disabled={busy || !status?.openai_configured} onClick={() => run('/evidence/extract', undefined, r => `已提取 ${r.processed} 篇摘要，新增 ${r.events_added} 条待审核事件。`)}>提取待处理新闻</button>
    </div>
    <p className="muted">采集任务仅在后端运行时执行。自动模型提取默认关闭；每轮最多提取 {status?.extract_limit ?? 3} 篇 RSS 摘要。短摘要可能遗漏细节，审核时请核对来源全文。</p>
    <div className="section-title spaced"><h3>最近采集状态</h3><span>{date(status?.last_collection?.finished_at)} · 北京时间</span></div>
    {!status?.last_collection ? <p className="muted">尚未执行采集；未读取到报告不代表没有伤病。</p> : <div className="table-scroll"><table><thead><tr><th>来源</th><th>结果</th><th>说明</th></tr></thead><tbody>{Object.entries(status.last_collection.sources).map(([key, source]) => <tr key={key}><th>{sourceNames[key]}</th><td>{source.status === 'ok' ? '已读取' : '读取失败'}</td><td>{source.status === 'ok' ? `${source.parsed ?? 0} 条原文记录` : `${source.http_status ? `HTTP ${source.http_status} · ` : ''}${source.detail}`}</td></tr>)}</tbody></table></div>}
    <form className="report-import" onSubmit={e => { e.preventDefault(); run('/evidence/reports', { url: new FormData(e.currentTarget).get('url') }, '官方报告已归档并核验；重新运行比赛 Agent 更新特征。'); }}>
      <label>官方目录无法访问时，可导入 NBA 官方伤病 PDF 链接<input name="url" type="url" required placeholder="https://ak-static.cms.nba.com/referee/injury/Injury-Report_...pdf" /></label><button className="button secondary" disabled={busy}>导入官方 PDF</button>
    </form>
    <div className="section-title spaced"><h3>事件审核</h3><select aria-label="筛选事件审核状态" value={filter} onChange={e => setFilter(e.target.value)}><option value="pending">待审核</option><option value="approved">已核验</option><option value="rejected">已排除</option><option value="all">全部</option></select></div>
    <p className="notice">仅准确匹配球队和球员、证据可追溯的事件可批准。未匹配时先在「数据与设置」同步球员和阵容，再采集一次。重复旧报告不会刷新伤病有效时间。</p>
    {!rows.length ? <div className="empty"><h3>当前筛选下没有事件</h3><p>采集 ESPN 伤病列表，或使用模型提取新闻摘要后查看。</p></div> : rows.slice(0, 100).map(row => <article className="event review-event" key={row.id}>
      <div className="section-title"><strong>{row.data.player_name || row.data.team_name}</strong><span>{reviewNames[row.review]}</span></div>
      <small>{row.data.team_name} · {row.data.event_type} · {row.data.player_status} · {date(row.data.published_at || row.published_at)}</small>
      <p>{row.data.reason}</p><blockquote>{row.data.quote}</blockquote>
      <small>{row.data.match_error || '实体已匹配'}{row.data.minutes_limit != null ? ` · 分钟上限 ${number(row.data.minutes_limit)}` : ''}</small>
      <div className="event-actions"><a href={row.url} target="_blank" rel="noreferrer">{sourceNames[row.provider]} ↗</a><button className="text-link" disabled={busy} onClick={() => action(async () => { const d = await request(`/evidence/documents/${row.document_id}`); setDocument(d); return d; }, '已读取归档原文。')}>查看归档原文</button>
        {row.review === 'pending' ? <><button className="button secondary" disabled={busy || !!row.data.match_error} onClick={() => run(`/evidence/events/${row.id}/review`, { decision: 'approved' }, '事件已批准；重新运行 Agent 更新预测快照。')}>批准事件</button><button className="button secondary" disabled={busy} onClick={() => run(`/evidence/events/${row.id}/review`, { decision: 'rejected' }, '事件已排除。')}>排除事件</button></> : null}
      </div>
    </article>)}
    {document ? <details className="limitations" open><summary>归档原文 · {document.title}</summary><p>{date(document.published_at)} · 采集于 {date(document.observed_at)}</p><pre>{document.body}</pre></details> : null}
    <div className="section-title spaced"><h3>伤病胜率修正模型</h3><span>{status?.model?.status === 'validated' ? '已通过时间验证 · 未校准' : '等待真实赛前样本'}</span></div>
    <p className="muted">模型在 Elo 基线之外拟合双方伤病暴露差，使用时间顺序留出验证。至少 {status?.training_min_games ?? 200} 场完整的真实赛前样本；未同时改善 Brier 与 Log Loss 时保持 Elo。当前伤病不能倒填到过去比赛。</p>
    <button className="button secondary" disabled={busy} onClick={() => run('/evidence/train', undefined, r => `${r.detail}；可用样本 ${r.sample_size} 场。`)}>训练并验证伤病模型</button>
    {status?.last_training ? <details className="limitations" open><summary>最近训练结果 · {status.last_training.status}</summary><pre>{JSON.stringify(status.last_training, null, 2)}</pre></details> : null}
    {operation ? <details className="limitations"><summary>最近操作详情</summary><pre>{JSON.stringify(operation, null, 2)}</pre></details> : null}
  </section>;
}
