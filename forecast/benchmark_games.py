"""Chronological stats-only benchmark. Never overwrites legacy model artifacts.

python -m forecast.benchmark_games --source sample --out runs/game-baseline-v1
Selection: pre-Jan validation only. Residual calibration: January 2026.
Regression test: Feb-Apr 12, 2026; this period already has published repo results.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.game_distribution import FEATURES, VERSION, ScoreDistribution, game_rows


def probability_scores(y, p):
    return {'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])),
            'accuracy': float(np.mean((p >= .5) == y)), 'games': len(y)}


def paired_interval(y, challenger, baseline, seed=7606):
    """Game-paired bootstrap of the Brier improvement (positive favors challenger)."""
    delta = (baseline-y)**2 - (challenger-y)**2
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    return {'brier_improvement': float(delta.mean()), 'paired_game_bootstrap_95': np.quantile(means, [.025,.975]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['sample', 'frozen'], default='sample')
    ap.add_argument('--out', type=Path, default=Path('runs/game-baseline-v1'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = Path('data') / args.source / 'games.parquet'
    games = pd.read_parquet(path)
    start = time.perf_counter()
    rows = game_rows(games)
    feature_seconds = time.perf_counter() - start
    train = rows[rows.tip_time < '2025-10-01']
    val = rows[(rows.tip_time >= '2025-10-01') & (rows.tip_time < '2026-01-01')]
    fit = rows[rows.tip_time < '2026-01-01']
    cal = rows[(rows.tip_time >= '2026-01-01') & (rows.tip_time < '2026-02-01')]
    test_end = pd.Timestamp('2026-04-13', tz='America/New_York').tz_convert('UTC')
    test = rows[(rows.tip_time >= '2026-02-01') & (rows.tip_time < test_end)]
    if min(map(len, (train,val,cal,test))) < 20:
        raise ValueError('insufficient games in one of the fixed chronological partitions')
    rows.to_parquet(args.out / 'features.parquet', index=False)
    selections, metrics, predictions, models = [], [], test[['game_id','tip_time','home_win','margin','total']].copy(), {}
    # The existing stat-only win model, retrained without hindsight absence inputs.
    from forecast.history import History
    from forecast.win import training_rows
    pg = pd.read_parquet(path.with_name('player_games.parquet'))
    legacy = training_rows(History(pg, games), games).set_index('game_id')
    cols = ['rating_diff','margin10_diff','rest_diff','b2b_diff']
    bfit = legacy.loc[fit.game_id]
    baseline = LogisticRegression(C=1.,max_iter=1000).fit(bfit[cols],bfit.home_win)
    bp = baseline.predict_proba(legacy.loc[test.game_id,cols])[:,1]
    metrics.append(dict(model='legacy-logit-stats-only',target='home_win',**probability_scores(test.home_win,bp)))
    predictions['legacy-logit-stats-only'] = bp
    models['legacy_logit'] = baseline
    # Reproduce the legacy as-coded model as a diagnostic, not the clean baseline.
    from forecast.win import WinModel
    original_train = legacy[legacy.tip_time < '2026-02-01']
    original = WinModel().fit(original_train)
    op = original.predict(legacy.loc[test.game_id])
    metrics.append(dict(model='legacy-as-coded-hindsight-diagnostic',target='home_win',**probability_scores(test.home_win,op)))
    predictions['legacy-as-coded-hindsight-diagnostic'] = op
    winner = None
    for c in (.01,.1,1.,10.):
        m = make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(C=c,max_iter=2000))
        m.fit(train[FEATURES],train.home_win)
        score = brier_score_loss(val.home_win,m.predict_proba(val[FEATURES])[:,1])
        selections.append(dict(target='home_win',kind='logit',strength=c,validation_brier=score))
        if winner is None or score < winner[0]: winner=(score,c,m)
    _,c,m = winner
    m.fit(fit[FEATURES],fit.home_win)
    p=m.predict_proba(test[FEATURES])[:,1]
    models['win'] = m
    metrics.append(dict(model=f'rich-logit-{c}',target='home_win',**probability_scores(test.home_win,p)))
    predictions['rich-logit'] = p
    improvement=paired_interval(test.home_win.to_numpy(),p,bp)
    for target, lines in [('margin',[-10.5,-4.5,0.,4.5,10.5]),('total',[205.5,215.5,225.5,235.5,245.5])]:
        best=None
        for kind,strength in [('rolling',0),('ridge',10),('ridge',100),('ridge',1000),('boost',100),('boost',250)]:
            m=ScoreDistribution(target,kind,strength).fit(train)
            score=mean_absolute_error(val[target],m.center(val))
            selections.append(dict(target=target,kind=kind,strength=strength,validation_mae=score))
            if best is None or score < best[0]: best=(score,kind,strength)
        for label,kind,strength in [('baseline','rolling',0),('selected',best[1],best[2])]:
            m=ScoreDistribution(target,kind,strength).fit(fit).calibrate(cal)
            models[target+'_'+label]=m
            quantiles=m.quantiles(test)
            y=test[target].to_numpy()
            diff=y[:,None]-quantiles
            qs=np.array([.1,.25,.5,.75,.9])
            metrics.append(dict(model=m.name,role=label,target=target,mae=mean_absolute_error(y,quantiles[:,2]),
                                pinball=float(np.maximum(qs*diff,(qs-1)*diff).mean()),
                                coverage80=float(((y>=quantiles[:,0])&(y<=quantiles[:,-1])).mean()),games=len(y)))
            for line in lines:
                prob=m.probability(test,line)
                metrics.append(dict(model=m.name,role=label,target=f'{target}>{line}',**probability_scores(y>line,prob)))
                predictions[f'{target}_{label}>{line}']=prob
    models['metadata'] = {'version': VERSION, 'available_at': games.loc[games.game_id.isin(cal.game_id), 'final_at'].max().isoformat(),
                          'fit_before': '2026-01-01T00:00:00Z', 'calibration_before': '2026-02-01T00:00:00Z'}
    pd.DataFrame(selections).to_csv(args.out/'validation.csv',index=False)
    pd.DataFrame(metrics).to_csv(args.out/'metrics.csv',index=False)
    predictions.to_csv(args.out/'predictions.csv',index=False)
    manifest={'version':VERSION,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'split_counts':{k:len(v) for k,v in [('train',train),('validation',val),('refit',fit),('calibration',cal),('test',test)]},
              'splits': {'train_before':'2025-10-01T00:00:00Z', 'validation_before':'2026-01-01T00:00:00Z',
                         'calibration_before':'2026-02-01T00:00:00Z', 'test_end_exclusive':test_end.isoformat()},
              'artifact':models['metadata'], 'feature_seconds':feature_seconds,'elapsed_seconds':time.perf_counter()-start,
              'win_improvement_vs_clean_legacy':improvement,
              'note':'Regression benchmark, not new untouched test. No market prices or reconstructed absences. '
                     'Distribution thresholds are fixed diagnostic lines, not historical trading returns.'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'models.pkl').write_bytes(pickle.dumps(models))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
