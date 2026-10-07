"""Chronological calibration mechanics only; fixtures do not validate NBA accuracy."""
import numpy as np
import pandas as pd
import pytest

from forecast.inplay import FEATURES
from forecast.inplay_training import chronological_parts, fit_calibrated, game_weights


def training():
    rows = []
    for i in range(24):
        tip = pd.Timestamp('2026-01-01T00:00Z')+pd.Timedelta(days=i)
        win = i % 2
        for minute in range(12):
            rows.append({'game_id': str(i), 'tip_time': tip, 'final_at': tip+pd.Timedelta(hours=3),
                         'as_of': tip+pd.Timedelta(minutes=minute), 'home_win': win,
                         'scaled_margin': (1 if win else -1)*(5+minute),
                         'scaled_prior': .1 if win else -.1, 'scaled_news': 0., 'scaled_possession': 0.})
    return pd.DataFrame(rows)


def test_splits_group_games_and_exclude_games_crossing_a_cutoff():
    f = training()
    f.loc[f.game_id=='7', 'final_at'] = pd.Timestamp('2026-01-09T01:00Z')
    train, cal, test = chronological_parts(f, '2026-01-09', '2026-01-17')
    assert len(set(train.game_id)) == 7 and '7' not in set(cal.game_id)
    assert not set(train.game_id)&set(cal.game_id) and not set(cal.game_id)&set(test.game_id)
    f.loc[0, 'as_of'] = f.loc[0, 'final_at']
    with pytest.raises(ValueError, match='precede'):
        chronological_parts(f, '2026-01-09', '2026-01-17')


@pytest.mark.parametrize('candidate', ['logistic', 'hgb'])
def test_calibrated_candidates_score_only_future_games_and_keep_direction(candidate):
    f = training()
    model = fit_calibrated(f, '2026-01-09', '2026-01-17', candidate)
    validation = model.validation
    assert validation['train_games'] == validation['calibration_games'] == validation['test_games'] == 8
    assert validation['development_only'] and validation['calibrated']
    assert 0 <= validation['brier'] <= 1 and validation['log_loss'] >= 0
    assert model.name == f'inplay-{candidate}-calibrated'
    low = pd.DataFrame([dict(zip(FEATURES, [-12, 0, 0, 0]))])
    high = pd.DataFrame([dict(zip(FEATURES, [12, 0, 0, 0]))])
    assert model.fitted.predict_proba(high)[0, 1] > model.fitted.predict_proba(low)[0, 1]


def test_game_weight_not_poll_frequency_determines_metrics():
    f = pd.DataFrame({'game_id': ['one']*50+['two']*2})
    weight = game_weights(f)
    assert np.sum(weight[:50]) == pytest.approx(np.sum(weight[50:]))


def test_reversed_calibration_is_rejected():
    f = training()
    mask = (f.tip_time >= pd.Timestamp('2026-01-09T00:00Z')) & (f.tip_time < pd.Timestamp('2026-01-17T00:00Z'))
    f.loc[mask, 'home_win'] = 1-f.loc[mask, 'home_win']
    with pytest.raises(ValueError, match='reversed'):
        fit_calibrated(f, '2026-01-09', '2026-01-17')
