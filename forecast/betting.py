"""Own-model full-game score distribution. Market prices never enter this model."""
import math
from statistics import NormalDist

import pandas as pd

from data_sources.inplay_types import utc
from forecast.inplay import remaining_seconds

N = NormalDist()


def historical_distribution(games, game, as_of):
    cutoff = min(utc(game.tip_time), utc(as_of))
    rows = games.dropna(subset=['home_pts', 'away_pts']).copy()
    rows = rows[(pd.to_datetime(rows.final_at, utc=True) < cutoff) & (rows.game_id.astype(str) != str(game.game_id))]
    rows = rows.sort_values('final_at').tail(600)
    if len(rows) < 30:
        return None
    total, margin = rows.home_pts+rows.away_pts, rows.home_pts-rows.away_pts
    league = float(total.mean()/2)
    def rating(team):
        recent = rows[(rows.home_team_id == team) | (rows.away_team_id == team)].tail(30)
        if recent.empty:
            return 0., 0.
        own = recent.home_pts.where(recent.home_team_id == team, recent.away_pts)
        allowed = recent.away_pts.where(recent.home_team_id == team, recent.home_pts)
        shrink = len(recent)/(len(recent)+10)
        return float((own.mean()-league)*shrink), float((allowed.mean()-league)*shrink)
    ho, hd = rating(game.home_team_id); ao, ad = rating(game.away_team_id)
    return {'total_mean': float(total.mean()+.5*(ho+ao+hd+ad)),
            'margin_sd': max(10., min(25., float(margin.std()))),
            'total_sd': max(12., min(35., float(total.std()))), 'history_games': len(rows),
            'history_cutoff': cutoff.isoformat(), 'model': 'own-history-score-distribution-prototype'}


def distribution(base, snapshot):
    if base is None or snapshot.get('p_home') is None or snapshot.get('quote_state') not in ('pregame', 'live'):
        return None
    p = float(snapshot['p_home'])
    if not math.isfinite(p) or not 0 < p < 1:
        return None
    score = snapshot.get('score')
    fraction = min(1., max(1/2880, remaining_seconds(score)/2880)) if score else 1.
    margin_sd = float(snapshot.get('model_parameters', {}).get('sigma', base['margin_sd']))*math.sqrt(fraction)
    # Existing independent win model anchors margin; all total features are own scores/history.
    mean_margin = N.inv_cdf(p)*margin_sd
    news_total = -sum(abs(float(e.get('home_margin_adjustment', 0))) for e in snapshot.get('player_effects', []))
    total_mean = base['total_mean']
    if score:
        observed = score['home_score']+score['away_score']
        elapsed = max(0., 1-fraction)
        weight = .5*elapsed
        pace = observed/max(elapsed, .05)
        total_mean = observed+fraction*((1-weight)*total_mean+weight*pace)+news_total
    else:
        # Pregame probability movement is a prototype news signal for scoring uncertainty.
        initial = (snapshot.get('baseline') or {}).get('p_home', p)
        news_total = -abs((N.inv_cdf(p)-N.inv_cdf(min(.999,max(.001,initial))))*margin_sd)
        total_mean += news_total
    return {**base, 'margin_mean': mean_margin, 'margin_sd': margin_sd,
            'total_mean': max(float((score or {}).get('home_score', 0)+(score or {}).get('away_score', 0)), total_mean),
            'total_sd': base['total_sd']*math.sqrt(fraction), 'p_home': p,
            'research_only': True, 'as_of': snapshot['as_of']}


def probability(model, kind, side, line=None):
    if model is None:
        return None
    if kind == 'moneyline':
        return model['p_home'] if side == 'home' else 1-model['p_home']
    if kind == 'spread':
        p = N.cdf((model['margin_mean']+float(line))/model['margin_sd'])
        return p if side == 'home' else 1-p
    if kind == 'total':
        p = N.cdf((model['total_mean']-float(line))/model['total_sd'])
        return p if side == 'over' else 1-p
    raise ValueError('Unsupported bet type')
