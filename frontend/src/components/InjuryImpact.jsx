import { number, percent } from '../api.js';

const statuses = { out: '确认缺阵', questionable: '出场存疑', doubtful: '大概率缺阵', probable: '大概率出场', available: '可出场', unknown: '状态待复核' };

export default function InjuryImpact({ impact, game }) {
  if (!impact) return null;
  return <section className="injury-impact">
    <div className="section-title spaced"><h3>伤病影响量化</h3><span>{impact.model_applied ? '模型已修正胜率' : '出场情景 · 尚未修正胜率'}</span></div>
    <p className="muted">{impact.model_reason}。Elo 基线 {percent(impact.base_home_probability)}；主队胜率变动 {(impact.probability_change * 100).toFixed(2)} 个百分点。</p>
    <div className="table-scroll"><table><thead><tr><th>球队</th><th>缺阵分钟</th><th>不确定分钟</th><th>明确受限分钟</th><th>缺少统计 / 待复核</th></tr></thead><tbody>{['home', 'away'].map(side => <tr key={side}><th>{game[side].abbreviation}</th><td>{number(impact.teams[side].out_minutes)}</td><td>{number(impact.teams[side].uncertain_minutes)}</td><td>{number(impact.teams[side].limited_minutes)}</td><td>{impact.teams[side].missing_stats} / {impact.teams[side].review_required}</td></tr>)}</tbody></table></div>
    {impact.players.length ? <div className="table-scroll"><table><thead><tr><th>球员</th><th>状态</th><th>通常分钟</th><th>出场分钟情景</th><th>统计场数</th></tr></thead><tbody>{impact.players.map(player => <tr key={player.player_id}><th>{player.player_name}<small className="muted">{game[player.side].abbreviation}</small></th><td>{statuses[player.status] || player.status}{player.needs_review ? ' · 待复核' : ''}{player.missing_minutes_limit ? ' · 限制未注明分钟' : ''}</td><td>{number(player.baseline_minutes)}</td><td>{player.minutes_range ? player.minutes_range.map(number).join(' – ') : '统计不足'}</td><td>{player.stats_games}</td></tr>)}</tbody></table></div> : <p className="muted">暂无可核验的相关球员状态，不能据此判断双方健康。</p>}
    <details className="limitations"><summary>量化口径与数据覆盖</summary><p>双方官方报告：{impact.official_both_teams ? '已读取' : '尚不完整'}；可训练快照：{impact.training_eligible ? '是' : '否'}</p><ul>{impact.limitations.map(line => <li key={line}>{line}</li>)}</ul></details>
  </section>;
}
