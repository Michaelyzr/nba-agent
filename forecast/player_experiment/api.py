"""Explicit experiment inference. Existing forecast.api and its pickles are untouched."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.baselines import _qcols, QUANTILES
from forecast.player_experiment.features import build_rows
from forecast.player_experiment.model import points_probabilities


class PlayerExperiment:
    def __init__(self, bundle):
        self.bundle = bundle

    @classmethod
    def load(cls, path, variant='enriched'):
        """Load only a trusted, locally trained experiment artifact."""
        with Path(path).open('rb') as stream:
            return cls(pickle.load(stream)[variant])

    def predict(self, player_games, games, targets, thresholds=(10,15,20,25,30,35)):
        """Targets require game_id/player_id/team_id/as_of; outputs assume participation.

        Feature builder must see full history (or the appropriate as-of subset).
        Calibration never changes minutes or raw points quantile diagnostics.
        """
        if 'as_of' not in targets or targets.empty:
            raise ValueError('Nonempty targets with explicit as_of timestamps required')
        thresholds = np.asarray(thresholds, dtype=float)
        if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all():
            raise ValueError('Finite nonempty one-dimensional thresholds required')
        cutoff = pd.to_datetime(targets.as_of, utc=True)
        if (cutoff < pd.Timestamp(self.bundle['available_at'])).any():
            raise ValueError('Model calibration outcomes were unavailable at requested cutoff')
        rows = build_rows(player_games,games,targets=targets)
        pred = self.bundle['model'].predict(rows)
        raw = points_probabilities(pred,thresholds)
        calibrated = self.bundle['calibrator'].predict(raw)
        # Nonnegative points have a known support boundary, not a learned tail.
        raw[:,thresholds<=0] = 1.
        calibrated[:,thresholds<=0] = 1.
        result = []
        for i,row in rows.iterrows():
            result.append({
                'game_id':row.game_id, 'player_id':int(row.player_id), 'as_of':row.as_of.isoformat(),
                'model':self.bundle['model'].name, 'conditional_on':'played',
                'status':'ok' if row.games_prior >= 5 else 'limited_history',
                'prior_games':int(row.games_prior),
                'raw_quantiles':{t:{f'p{int(q*100)}':float(pred.loc[i,c])
                                    for q,c in zip(QUANTILES,_qcols(t))} for t in ('pts','min')},
                'events':[{'operator':'>=','threshold':float(t),
                           'raw_probability_given_play':float(raw[i,j]),
                           'calibrated_probability_given_play':float(calibrated[i,j])}
                          for j,t in enumerate(thresholds)],
                'external_sources_applied':[],
                'calibration_scope':'Player-points event probabilities only; quantiles above are raw',
            })
        return result
