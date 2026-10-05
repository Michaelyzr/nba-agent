"""Opt-in numerical forecasts. No exchange orders and no silent model promotion.

Winner/spread probabilities share the same margin distribution. The richer
logistic winner is exposed separately for model comparison, not substituted
into the distribution. Bundles must be trusted, locally trained pickle files.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.game_distribution import FEATURES, VERSION, game_rows


class MarketForecaster:
    def __init__(self, bundle, player_forecaster=None):
        self.bundle = bundle
        self.player_forecaster = player_forecaster

    @classmethod
    def load(cls, path, player_forecaster=None):
        with Path(path).open('rb') as stream:
            return cls(pickle.load(stream), player_forecaster)

    def predict(self, view, game, kind, threshold=None, operator='>', side='home',
                player_id=None, participation_rule=None, out=(), includes_overtime=True):
        base = {'game_id': game.game_id, 'as_of': view.now.isoformat(), 'version': VERSION,
                'kind': kind, 'operator': operator, 'threshold': threshold, 'side': side,
                'includes_overtime': includes_overtime, 'external_sources_applied': []}
        available_at = self.bundle.get('metadata', {}).get('available_at')
        if available_at is None:
            raise ValueError('Bundle requires training/calibration availability metadata')
        if pd.Timestamp(view.now) < pd.Timestamp(available_at):
            raise ValueError('Model calibration outcomes were not available at this cutoff')
        if not includes_overtime:
            return dict(base, status='unsupported', reason='Only full-game results including overtime are modeled')
        if pd.Timestamp(view.now) >= pd.Timestamp(game.tip_time):
            raise ValueError('Forecast cutoff must precede tip-off')
        if operator not in ('>', '>=') or side not in ('home', 'away'):
            raise ValueError('Unsupported operator or side')
        if kind == 'player_points':
            if self.player_forecaster is None:
                return dict(base, status='unavailable', reason='No existing player model supplied')
            if player_id is None or threshold is None or not np.isfinite(threshold):
                raise ValueError('Player id and finite threshold required')
            if participation_rule not in ('void_if_not_play', 'no_if_not_play'):
                return dict(base, status='unsupported', reason='Specify non-participation settlement rule')
            if player_id in set(out):
                return dict(base, status='ok', p_play=0., p_yes_if_play=None, p_yes=0.,
                            p_void=1. if participation_rule == 'void_if_not_play' else 0.,
                            p_no=0. if participation_rule == 'void_if_not_play' else 1.,
                            external_sources_applied=['out_override'])
            k = np.floor(threshold) + 1 if operator == '>' else np.ceil(threshold)
            f = self.player_forecaster.player(view, game, player_id, 'pts', lines=[float(k-.5)], out=out)
            if f is None:
                return dict(base, status='unavailable', reason='No player history/model')
            from forecast.baselines import QUANTILES, prob_at_least
            q = [f['quantiles'].get(f'p{int(x*100)}') for x in QUANTILES]
            pplay = f['p_play']
            if any(x is None for x in q):
                return dict(base, status='unavailable', reason='No conditional player distribution')
            conditional = 1. if k <= 0 else prob_at_least(q, k)
            pvoid = 1-pplay if participation_rule == 'void_if_not_play' else 0.
            return dict(base, status='ok', model=f['model'], p_play=pplay,
                        p_yes_if_play=conditional, p_yes=pplay*conditional, p_void=pvoid,
                        p_no=(pplay*(1-conditional) if pvoid else 1-pplay*conditional),
                        external_sources_applied=['out_override'] if out else [],
                        caveat='Existing player model; absence features and probability calibration not repaired by this adapter')
        if kind not in ('winner','spread','total'):
            return dict(base,status='unsupported',reason='Season simulation is not implemented')
        if out:
            return dict(base,status='unsupported',reason='Stats-only game model has no validated news adjustment')
        target = pd.DataFrame([{'game_id':game.game_id,'tip_time':game.tip_time,'as_of':view.now,
                               'home_team_id':game.home_team_id,'away_team_id':game.away_team_id}])
        rows=game_rows(view.games(),targets=target)
        target_name='total' if kind=='total' else 'margin'
        m=self.bundle[target_name+'_selected']
        if kind=='winner': threshold,operator=0.,'>'
        if threshold is None: raise ValueError('Threshold required')
        p=float(m.probability(rows,threshold,operator,side)[0])
        result=dict(base,status='ok',threshold=threshold,operator=operator,model=m.name,
                    probability=p,quantiles=dict(zip(['p10','p25','p50','p75','p90'],map(float,m.quantiles(rows)[0]))),
                    distribution_target='home_margin' if target_name == 'margin' else 'total',
                    calibration_games=m.calibration_games)
        if kind=='winner':
            rich=float(self.bundle['win'].predict_proba(rows[FEATURES])[0,1])
            result['comparison_logit_probability']=rich if side=='home' else 1-rich
        return result
