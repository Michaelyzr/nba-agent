"""Stats-only, as-of game features and empirical score distributions.

No current-game box scores, reconstructed inactive lists, or market prices are
features. Completion times inherit the source table's final_at assumptions.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = 'game-distribution-v1'
SIDE_FEATURES = ('off5', 'off10', 'off30', 'def5', 'def10', 'def30', 'margin5',
                 'margin10', 'margin30', 'margin_std10', 'total_std10', 'win10',
                 'rest', 'b2b', 'count', 'elo', 'form')
FEATURES = [f'{k}_{op}' for k in SIDE_FEATURES for op in ('diff', 'sum')]
FEATURES += ['expected_margin', 'expected_total']


def game_rows(games, lead_minutes=30, targets=None):
    """Build features with only final_at <= as_of; targets may have custom as_of.

    Raw results enter team histories only once completed. Features are built once
    for all candidates. Targets' scores, when present, are labels only.
    """
    g = games.copy()
    for c in ('tip_time', 'final_at'):
        g[c] = pd.to_datetime(g[c], utc=True)
    if g.game_id.duplicated().any():
        raise ValueError('game_id must be unique')
    if (g.final_at < g.tip_time).any():
        raise ValueError('final_at must not precede tip_time')
    t = g.copy() if targets is None else targets.copy()
    t['tip_time'] = pd.to_datetime(t.tip_time, utc=True)
    t['as_of'] = (pd.to_datetime(t.as_of, utc=True) if 'as_of' in t
                  else t.tip_time - pd.Timedelta(minutes=lead_minutes))
    if t.as_of.isna().any() or (t.as_of >= t.tip_time).any():
        raise ValueError('as_of must be present and before tip-off')
    events = g.dropna(subset=['home_pts', 'away_pts', 'final_at']).sort_values(['final_at', 'game_id'])
    events = list(events.itertuples())
    history, elo = defaultdict(list), defaultdict(lambda: 1500.)
    pos, result = 0, []

    def side(team, tip):
        records = history[team][-30:]
        if records:
            a = np.array([(r[1], r[2]) for r in records])
            rest = min(10., max(0., (tip - records[-1][0]).total_seconds() / 86400))
        else:
            a, rest = np.array([[110., 110.]]), 10.
        margin, total = a[:, 0] - a[:, 1], a.sum(axis=1)
        f = {f'{name}{w}': float(a[-w:, j].mean()) for j, name in enumerate(('off', 'def'))
             for w in (5, 10, 30)}
        f.update({f'margin{w}': float(margin[-w:].mean()) for w in (5, 10, 30)})
        f.update(margin_std10=float(margin[-10:].std()), total_std10=float(total[-10:].std()),
                 win10=float((margin[-10:] > 0).mean()) if records else .5, rest=rest,
                 b2b=float(rest < 1.5), count=min(len(history[team]), 30), elo=elo[team],
                 form=float(margin[-5:].mean() - margin.mean()))
        return f

    for r in t.sort_values(['as_of', 'game_id']).itertuples():
        while pos < len(events) and events[pos].final_at <= r.as_of:
            e = events[pos]
            p = 1 / (1 + 10 ** ((elo[e.away_team_id] - elo[e.home_team_id] - 65) / 400))
            delta = 20 * (float(e.home_pts > e.away_pts) - p)
            elo[e.home_team_id] += delta
            elo[e.away_team_id] -= delta
            history[e.home_team_id].append((e.tip_time, e.home_pts, e.away_pts))
            history[e.away_team_id].append((e.tip_time, e.away_pts, e.home_pts))
            pos += 1
        h, a = side(r.home_team_id, r.tip_time), side(r.away_team_id, r.tip_time)
        f = {f'{k}_{op}': h[k] - a[k] if op == 'diff' else h[k] + a[k]
             for k in SIDE_FEATURES for op in ('diff', 'sum')}
        hs, aws = (h['off10'] + a['def10']) / 2, (a['off10'] + h['def10']) / 2
        hp, ap = getattr(r, 'home_pts', np.nan), getattr(r, 'away_pts', np.nan)
        result.append(dict(f, expected_margin=hs-aws, expected_total=hs+aws,
                           game_id=r.game_id, tip_time=r.tip_time, as_of=r.as_of,
                           home_team_id=r.home_team_id, away_team_id=r.away_team_id,
                           margin=hp-ap, total=hp+ap,
                           home_win=float(hp > ap) if pd.notna(hp) and pd.notna(ap) else np.nan,
                           season_type=getattr(r, 'season_type', 'unknown')))
    return pd.DataFrame(result)


class ScoreDistribution:
    """Predict a center; estimate uncertainty from disjoint calibration residuals.

    Shared sorted residuals make arbitrary thresholds cheap. This first version
    assumes constant residual distribution and does not claim joint score modeling.
    """
    def __init__(self, target, kind='ridge', strength=100.):
        if target not in ('margin', 'total'):
            raise ValueError('target must be margin or total')
        self.target, self.kind, self.strength = target, kind, strength
        self.name = f'{target}-{kind}-{strength:g}'

    def fit(self, rows):
        if self.kind == 'ridge':
            self.model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=self.strength))
        elif self.kind == 'boost':
            self.model = HistGradientBoostingRegressor(max_iter=int(self.strength), max_leaf_nodes=7,
                         min_samples_leaf=60, l2_regularization=10., learning_rate=.04,
                         early_stopping=False, random_state=7606)
        elif self.kind != 'rolling':
            raise ValueError('unknown model kind')
        if self.kind != 'rolling':
            self.model.fit(rows[FEATURES], rows[self.target])
        return self

    def center(self, rows):
        if self.kind == 'rolling':
            return rows['expected_' + self.target].to_numpy(float)
        return self.model.predict(rows[FEATURES])

    def calibrate(self, rows):
        if len(rows) < 20:
            raise ValueError('at least 20 calibration games required')
        self.residuals = np.sort(rows[self.target].to_numpy() - self.center(rows))
        self.residuals = self.residuals[np.isfinite(self.residuals)]
        if len(self.residuals) < 20:
            raise ValueError('at least 20 finite calibration residuals required')
        self.calibration_games = len(self.residuals)
        return self

    def cdf(self, rows, boundary):
        z = np.asarray(boundary) - self.center(rows)
        # Smoothed empirical CDF: finite tails avoid certainty from a small sample.
        rank = np.searchsorted(self.residuals, z, side='right')
        return (rank + .5) / (len(self.residuals) + 1)

    def probability(self, rows, threshold, operator='>', side='home'):
        if operator not in ('>', '>=') or side not in ('home', 'away'):
            raise ValueError('unsupported operator or side')
        if not np.isfinite(threshold):
            raise ValueError('threshold must be finite')
        k = np.floor(threshold) + 1 if operator == '>' else np.ceil(threshold)
        if self.target == 'total':
            if side != 'home':
                raise ValueError('total has no away side')
            return np.ones(len(rows)) if k <= 0 else 1 - self.cdf(rows, k - .5)
        # Completed NBA games cannot tie. Remove mass assigned to integer margin 0.
        neg, pos = self.cdf(rows, -.5), self.cdf(rows, .5)
        zero = pos - neg
        if side == 'home':
            mass = 1 - self.cdf(rows, k - .5) - (zero if k <= 0 else 0)
        else:
            mass = self.cdf(rows, -k + .5) - (zero if k <= 0 else 0)
        return np.clip(mass / (1 - zero), 0., 1.)

    def quantiles(self, rows, levels=(.1, .25, .5, .75, .9)):
        q = self.center(rows)[:, None] + np.quantile(self.residuals, levels)[None, :]
        return np.maximum(q, 0) if self.target == 'total' else q
