import { useEffect, useRef, useState } from 'react';
import { date, request, streamAnalysis } from '../api.js';

function sourceUrl(url) {
  try { const parsed = new URL(url); return ['http:', 'https:'].includes(parsed.protocol) && !parsed.username && !parsed.password ? parsed.href : null; }
  catch { return null; }
}
function inline(text, sources = []) {
  return text.split(/(\*\*[^*]+\*\*|\[[DW]\d+\])/g).map((part, i) => {
    if (part.startsWith('**')) return <strong key={i}>{inline(part.slice(2, -2), sources)}</strong>;
    const source = sources.find(s => `[${s.id}]` === part);
    if (source && sourceUrl(source.url)) return <a className="citation web-citation" key={i} href={sourceUrl(source.url)} title={source.title} target="_blank" rel="noopener noreferrer">{part}</a>;
    return /^\[D\d+\]$/.test(part) ? <span className="citation" key={i}>{part}</span> : part;
  });
}
// Only provider citation URLs become links; model HTML/Markdown remains text.
function ReportText({ text, sources }) {
  return <div className="report-text">{text.split('\n').map((line, i) => {
    if (/^#{1,4} /.test(line)) return <h3 key={i}>{inline(line.replace(/^#{1,4} /, ''), sources)}</h3>;
    if (/^[-*] /.test(line)) return <p className="report-bullet" key={i}>• {inline(line.slice(2), sources)}</p>;
    return line ? <p key={i}>{inline(line, sources)}</p> : <div className="report-space" key={i} />;
  })}</div>;
}

function exportReport(message) {
  const meta = message.meta;
  const sources = [...(meta?.sources || []), ...(message.web?.sources || [])];
  const text = [`# ${meta?.player?.name || meta?.team?.name || 'NBA'} 分析`,
    `赛季：${meta?.scope?.season} · ${meta?.scope?.season_type}  |  证据截止：${meta?.as_of}`,
    `模型：${meta?.model}  |  状态：${message.status === 'done' ? '已完成' : '部分结果'}`,
    `网页搜索：${meta?.web_search?.enabled ? message.web?.searched_at ? `已检索 ${message.web.searched_at}` : '未完成检索' : '关闭'}`,
    message.content, '## 数据来源', ...sources.map(s =>
      `- [${s.id}] ${s.title}${s.url && /^https?:\/\//.test(s.url) ? ` — ${s.url}` : ''} (${s.published_at || s.updated_at || '时间缺失'})`),
    '## 数据边界', ...(meta?.limitations || []).map(s => `- ${s}`)].join('\n\n');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
  const a = document.createElement('a'); a.href = url;
  a.download = `NBA-analysis-${new Date().toISOString().slice(0,10)}.md`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function AnalysisChat({ season, active, onSeasonResolved }) {
  const [catalog, setCatalog] = useState(null);
  const [teamId, setTeamId] = useState('');
  const [playerId, setPlayerId] = useState('');
  const [search, setSearch] = useState('');
  const [seasonType, setSeasonType] = useState('Regular Season');
  const [webSearch, setWebSearch] = useState(true);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [loading, setLoading] = useState(false);
  const [catalogVersion, setCatalogVersion] = useState(0);
  const controller = useRef(null);
  const pending = useRef(false);
  const generation = useRef(0);
  const historyPanel = useRef(null);
  const questionInput = useRef(null);
  const automaticSeason = useRef(null);
  const roster = catalog?.players.filter(p => p.team_id === teamId) || [];
  const players = roster.filter(p => p.name.toLowerCase().includes(search.toLowerCase()));
  const selectedTeam = catalog?.teams.find(t => t.id === teamId);
  const selectedPlayer = catalog?.players.find(p => p.id === playerId);

  useEffect(() => {
    if (!active) return;
    const abort = new AbortController();
    setLoading(true);
    request('/analysis/catalog', { signal: abort.signal }).then(data => {
      setCatalog(data); setError('');
      setTeamId(current => data.teams.some(t => t.id === current) ? current : data.teams[0]?.id || '');
    }).catch(e => { if (!abort.signal.aborted && e.status !== 422) setError(e.message); })
      .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [active, catalogVersion]);
  useEffect(() => {
    if (automaticSeason.current === season) { automaticSeason.current = null; return; }
    resetConversation();
  }, [season]);
  useEffect(() => () => { generation.current++; controller.current?.abort(); }, []);
  useEffect(() => {
    if (active && sending) historyPanel.current?.scrollIntoView({ block: 'start' });
  }, [active, sending]);
  useEffect(() => {
    if (active && sending && historyPanel.current) {
      historyPanel.current.scrollTop = historyPanel.current.scrollHeight;
    }
  }, [messages, active, sending]);
  useEffect(() => {
    if (active && !sending && messages.length) questionInput.current?.focus({ preventScroll: true });
  }, [active, sending, messages.length]);
  function resetConversation() {
    generation.current++; controller.current?.abort(); pending.current = false;
    setSending(false); setMessages([]); setError('');
  }
  function clear() { resetConversation(); setQuestion(''); }
  async function send(text, mode = 'chat') {
    if (pending.current || !text.trim() || !teamId) return;
    pending.current = true;
    const run = ++generation.current;
    const abort = new AbortController(); controller.current = abort;
    const history = messages.filter(m => m.role === 'user' || m.status === 'done')
      .slice(-8).map(m => ({ role: m.role, content: m.content.slice(0, 3500) }));
    const user = { role: 'user', content: text.trim().slice(0,5000) };
    const assistant = { role: 'assistant', content: '', status: 'streaming', meta: null };
    setMessages(current => [...current, user, assistant]); setQuestion(''); setError(''); setSending(true);
    const update = fn => { if (run === generation.current) setMessages(current => current.map((m,i) => i === current.length-1 ? fn(m) : m)); };
    try {
      await streamAnalysis({ team_id: teamId, player_id: playerId || null, season,
        season_type: seasonType, mode, web_search: webSearch, messages: [...history, user] }, {
        signal: abort.signal, onEvent: event => {
          if (event.type === 'meta' && run === generation.current) {
            update(m => ({ ...m, meta: event }));
            setTeamId(event.team.id); setPlayerId(event.player?.id || '');
            setSeasonType(event.scope.season_type); setSearch('');
            if (event.scope.season !== season && onSeasonResolved) {
              automaticSeason.current = event.scope.season;
              onSeasonResolved(event.scope.season);
            }
          }
          if (event.type === 'delta') update(m => ({ ...m, content: m.content + event.text }));
          if (event.type === 'web_status') update(m => ({ ...m, webStatus: event.status }));
          if (event.type === 'web_sources') update(m => ({ ...m, web: event, webStatus: event.status, content: event.text || m.content }));
          if (event.type === 'done') update(m => ({ ...m, status: 'done', usage: event.usage }));
        },
      });
    } catch (e) {
      if (run !== generation.current) return;
      update(m => ({ ...m, content: e.status === 422 ? e.message : m.content, status: abort.signal.aborted ? 'stopped' : e.status === 422 ? 'clarification' : 'error' }));
      if (!abort.signal.aborted && e.status !== 422) setError(e.message);
    } finally {
      if (run === generation.current) { pending.current = false; setSending(false); }
    }
  }

  return <section className="analysis-view">
    <div className="analysis-controls">
      <label>分析球队<select aria-label="分析球队" disabled={sending || loading} value={teamId} onChange={e => { resetConversation(); setTeamId(e.target.value); setPlayerId(''); setSearch(''); }}>
        {!catalog?.teams.length ? <option value="">请先同步球队目录</option> : null}
        {catalog?.teams.map(t => <option key={t.id} value={t.id}>{t.abbreviation} · {t.name}</option>)}
      </select></label>
      <label>比赛类型<select aria-label="分析比赛类型" disabled={sending} value={seasonType} onChange={e => { resetConversation(); setSeasonType(e.target.value); }}><option value="Regular Season">常规赛</option><option value="Playoffs">季后赛</option><option value="Pre Season">季前赛</option></select></label>
      <label>搜索当前阵容<input aria-label="搜索分析球员" disabled={sending} value={search} onChange={e => setSearch(e.target.value)} placeholder="输入球员英文姓名" /></label>
      <label>分析对象<select aria-label="分析球员" disabled={sending || loading} value={playerId} onChange={e => { resetConversation(); setPlayerId(e.target.value); }}><option value="">整支球队 · {roster.length} 名球员</option>
        {[...players, ...(selectedPlayer && !players.some(p => p.id === playerId) ? [selectedPlayer] : [])].map(p => <option key={p.id} value={p.id}>{p.name} · {p.position || '位置未同步'}</option>)}
      </select></label>
    </div>
    <div className="analysis-context"><div><strong>{selectedPlayer?.name || selectedTeam?.name || '请选择分析球队'}</strong><p>{season} · {seasonType === 'Regular Season' ? '常规赛' : seasonType === 'Playoffs' ? '季后赛' : '季前赛'} · 当前阵容与本地统计</p></div><div><span className={`status-dot ${catalog?.configured ? 'online' : ''}`} />{catalog?.configured ? `模型 ${catalog.model}` : '模型未配置'}<button className="text-link" disabled={sending || loading} onClick={() => setCatalogVersion(v => v+1)}>刷新目录</button></div></div>
    {!catalog?.configured && catalog ? <p className="notice">在 backend/.env 配置 OPENAI_API_KEY 后重启后端，即可使用分析对话。</p> : null}
    <div className="analysis-search"><label className="check"><input type="checkbox" aria-label="启用网页搜索" checked={webSearch} disabled={sending} onChange={e => setWebSearch(e.target.checked)} />网页搜索</label><span>{webSearch ? '补充新闻、伤病及缺失统计，附来源链接；会产生搜索费用。' : '仅使用已同步的本地证据。'}</span></div>
    <p className="muted analysis-boundary">问题中明确的对象、赛季和比赛类型优先于默认范围。开启网页搜索后，会按实际分析范围检索，优先官方与可靠媒体；网页结果不自动写入统计或预测。每次发送会将本地资料与近期对话交给已配置的 OpenAI API，产生调用费用。</p>
    <div className="analysis-toolbar"><div>
      <button className="button secondary" disabled={sending || !catalog?.configured || !teamId} onClick={() => send(selectedPlayer ? `请生成 ${selectedPlayer.name} 的球员分析报告，区分数据事实与评价，说明人员角色、近期表现及伤病证据。` : `请生成 ${selectedTeam.name} 的球队分析报告，评价阵容配置、表现、优劣势与伤病影响。`, selectedPlayer ? 'player_report' : 'team_report')}>生成{selectedPlayer ? '球员' : '球队'}报告 ↗</button>
      <button className="button secondary" disabled={sending || !catalog?.configured || !teamId} onClick={() => send(selectedPlayer ? '分析这名球员在当前球队的角色、优势与局限。缺失的数据请明确说明。' : '分析当前阵容的人员配置、位置分布、轮换深度和需要补强的环节。')}>人员配置分析</button>
    </div><button className="text-link" disabled={sending || !messages.length} onClick={clear}>新建对话</button></div>
    <div className="chat-history" ref={historyPanel} aria-label="模型分析对话" aria-busy={sending}>
      {!messages.length ? <div className="chat-empty"><span className="chat-symbol" aria-hidden="true">N</span><h2>围绕证据讨论球队与球员</h2><p>可询问球队评价、阵容配置、球员特点，或生成一份有来源的分析报告。</p><div className="chat-suggestions">{['这支球队目前有哪些优势和短板？', '哪些伤病或长期事件需要持续关注？', '现有数据还缺少哪些关键指标？'].map(text => <button key={text} disabled={!teamId || sending || !catalog?.configured} onClick={() => send(text)}>{text} ↗</button>)}</div></div> : messages.map((m,i) => <article className={`chat-message ${m.role}`} key={i}>
        <div className="chat-message-heading"><strong>{m.role === 'user' ? '你' : '分析助手'}</strong>{m.role === 'assistant' ? <span>{m.status === 'streaming' ? '正在生成…' : m.status === 'done' ? '已完成' : m.status === 'stopped' ? '已停止 · 部分结果' : m.status === 'clarification' ? '需要澄清' : '未完成'}</span> : null}</div>
        {m.role === 'assistant' && m.meta ? <p className="resolved-scope"><strong>已识别：</strong>{m.meta.player?.name || m.meta.team.name} · {m.meta.team.abbreviation} · {m.meta.scope.season} · {{ 'Regular Season':'常规赛', Playoffs:'季后赛', 'Pre Season':'季前赛' }[m.meta.scope.season_type]}{m.meta.routing?.changed ? '（已按问题更新范围）' : ''}</p> : null}
        {m.meta?.web_search?.enabled ? <p className="chat-search-status" role="status">{m.status === 'error' || m.status === 'stopped' ? '网页搜索或回答未完成' : m.webStatus === 'completed' ? `网页检索完成 · ${m.web.sources.length} 个引用来源` : m.webStatus === 'no_sources' ? '搜索已执行，未返回可引用网页来源；不能确认缺失的网页事实。' : m.webStatus === 'reading' ? '正在核对搜索结果…' : m.webStatus === 'searching' ? '正在搜索网页…' : '准备网页搜索…'}</p> : null}
        <ReportText sources={m.web?.sources} text={m.content || (m.role === 'assistant' && m.status === 'streaming' ? '正在识别问题、读取证据并等待模型回答…' : '没有生成完整回答。')} />
        {m.web?.sources?.length ? <div className="chat-web-sources"><h4>网页来源</h4><p>检索于 {date(m.web.searched_at)}（北京时间）；检索时间不等于新闻发布时间。</p><ul>{m.web.sources.map(s => <li key={s.id}>[{s.id}] {sourceUrl(s.url) ? <a href={sourceUrl(s.url)} target="_blank" rel="noopener noreferrer">{s.title} ↗</a> : s.title}</li>)}</ul></div> : null}
        {m.role === 'assistant' && m.meta ? <details className="chat-sources"><summary>证据与数据边界 · {m.meta.game_count} 场比赛 · {m.meta.roster_count} 名球员</summary><p>证据截止 {date(m.meta.as_of)}（北京时间）；本地记录更新时间不代表来源实时性。</p><ul>{m.meta.sources.map(s => <li key={s.id}><strong>[{s.id}]</strong> {s.title} · {date(s.published_at || s.updated_at)} {s.status || ''}{s.url && /^https?:\/\//.test(s.url) ? <a href={s.url} target="_blank" rel="noreferrer"> 原始来源 ↗</a> : null}</li>)}</ul><ul>{m.meta.limitations.map(s => <li key={s}>{s}</li>)}</ul></details> : null}
        {m.role === 'assistant' && m.meta && m.content && m.status !== 'streaming' ? <div className="chat-report-actions"><button className="text-link" onClick={() => exportReport(m)}>导出 Markdown 报告 ↓</button>{m.usage?.total_tokens ? <small>回答 Token：{m.usage.total_tokens}</small> : null}</div> : null}
      </article>)}
    </div>
    {error ? <div className="alert error" role="alert">{error}</div> : null}
    <form className="chat-composer" onSubmit={e => { e.preventDefault(); send(question); }}>
      <div className="chat-composer-heading"><strong>{messages.length ? '继续追问' : '输入问题'}</strong><small>{messages.length ? '沿用当前话题，无需新建对话' : '可询问球队、球员或生成分析报告'}</small></div>
      <label className="sr-only" htmlFor="analysis-question">分析问题</label><textarea ref={questionInput} id="analysis-question" value={question} disabled={sending} maxLength={5000} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(question); } }} placeholder={messages.length ? '继续这个话题，例如：展开解释刚才的第二点…' : '输入问题，例如：分析这支球队的轮换配置与潜在短板…'} rows={2} />
      <div><small>⌘ / Ctrl + Enter 发送 · 最近 8 条消息用于追问</small>{sending ? <button className="button secondary" type="button" onClick={() => controller.current?.abort()}>停止生成</button> : <button className="button" disabled={!question.trim() || !teamId || !catalog?.configured} type="submit">发送问题 ↗</button>}</div>
    </form>
  </section>;
}
