import { useCallback, useEffect, useRef, useState } from 'react';
import { request } from './api.js';
import GameBoard from './components/GameBoard.jsx';
import NewsView from './components/NewsView.jsx';
import DataView from './components/DataView.jsx';
import AnalysisChat from './components/AnalysisChat.jsx';

const views = [
  ['games', '比赛预测', '01', '用统计概率理解比赛，用市场报价检验差异。'],
  ['analysis', '模型分析', '02', '讨论球队与球员，基于证据生成研究报告。'],
  ['news', '新闻与伤病', '03', '采集证据，核验状态，量化出场影响。'],
  ['data', '数据与设置', '04', '管理数据来源，查看模型与风险边界。'],
];

export default function App() {
  const [view, setView] = useState('games');
  const [analysisSeason, setAnalysisSeason] = useState('2026-27');
  const [games, setGames] = useState([]);
  const [config, setConfig] = useState(null);
  const [overview, setOverview] = useState(null);
  const [evidenceStatus, setEvidenceStatus] = useState(null);
  const [evidenceEvents, setEvidenceEvents] = useState([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const pending = useRef(false);
  const generation = useRef(0);
  const alive = useRef(true);

  const load = useCallback(async () => {
    const id = ++generation.current;
    setLoading(true);
    const configRequest = request('/intelligence/config');
    const results = await Promise.allSettled([
      configRequest,
      configRequest.then(config => request(`/intelligence/games?season=${config.season}`)),
      request('/intelligence/overview'),
      request('/evidence/status'), request('/evidence/events?limit=500'),
    ]);
    if (!alive.current || id !== generation.current) return;
    setApiOnline(results[0].status === 'fulfilled');
    [setConfig, setGames, setOverview, setEvidenceStatus, setEvidenceEvents].forEach((setter, i) => {
      if (results[i].status === 'fulfilled') setter(results[i].value);
      else setter(i === 1 || i === 4 ? [] : null);
    });
    setError(results.find(r => r.status === 'rejected')?.reason?.message || '');
    setLoading(false);
  }, []);

  useEffect(() => { alive.current = true; load(); return () => { alive.current = false; generation.current++; }; }, [load]);

  async function action(operation, success) {
    if (pending.current) return;
    pending.current = true;
    setBusy(true); setError(''); setMessage('');
    try {
      const result = await operation();
      await load();
      setMessage(typeof success === 'function' ? success(result) : success || '操作完成。');
    } catch (e) { setError(e.message); }
    finally { pending.current = false; setBusy(false); }
  }
  const current = views.find(v => v[0] === view);
  return <div className="app-shell" aria-busy={busy}>
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); setView('games'); }}><span className="brand-symbol" aria-hidden="true">N</span><div>NBA <strong>Intelligence</strong></div></a>
      <p className="sidebar-label">研究工作台</p>
      <nav aria-label="主导航">{views.map(([key, label, index]) => <button key={key} className={view === key ? 'active' : ''} aria-current={view === key ? 'page' : undefined} onClick={() => { setView(key); setMessage(''); }}><span>{index}</span>{label}<b>↗</b></button>)}</nav>
      <div className="sidebar-bottom"><span className={`status-dot ${apiOnline ? 'online' : ''}`} />{apiOnline ? 'API 已连接' : 'API 未连接'}<p>数据与模型分析<br />Elo v1 · 未校准</p><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">接口文档 ↗</a></div>
    </aside>
    <main>
      <header className="topbar"><span>NBA / {current[1]}</span><div>{view === 'analysis' ? <label className="season-label">赛季<select aria-label="选择赛季" value={analysisSeason} disabled={busy} onChange={e => { setAnalysisSeason(e.target.value); setMessage(''); }}>{!['2026-27','2025-26','2024-25','2023-24'].includes(analysisSeason) ? <option>{analysisSeason}</option> : null}<option>2026-27</option><option>2025-26</option><option>2024-25</option><option>2023-24</option></select></label> : null}<button className="refresh" aria-label="刷新数据" disabled={busy || loading} onClick={load}>↻</button></div></header>
      <div className="workspace">
        <div className="page-heading"><div><h1>{current[1]}</h1><p>{current[3]}</p></div><span className="research-label">NBA RESEARCH</span></div>
        {error ? <div className="alert error" role="alert"><strong>请求未完成</strong><span>{error}</span><button onClick={load} disabled={loading}>刷新数据</button></div> : null}
        {message ? <div className="alert success" role="status">{message}</div> : null}
        {loading && !games.length ? <div className="loading" role="status">正在读取数据…</div> : null}
        {view === 'games' ? <GameBoard key={config?.season} games={games} action={action} busy={busy} /> : null}
        <div hidden={view !== 'analysis'}><AnalysisChat season={analysisSeason} active={view === 'analysis'} onSeasonResolved={setAnalysisSeason} /></div>
        {view === 'data' ? <DataView overview={overview} config={config} action={action} busy={busy} season={analysisSeason} /> : null}
        {view === 'news' ? <NewsView status={evidenceStatus} events={evidenceEvents} action={action} busy={busy || !config} season={config?.season} /> : null}
        <footer className="workspace-footer"><span>NBA Intelligence</span><span>比赛预测 · 新闻证据 · 球队与球员分析</span></footer>
      </div>
    </main>
  </div>;
}
