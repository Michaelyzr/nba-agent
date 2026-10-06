"""Run isolated feature/calibration ablations on two predeclared temporal windows.

python -m forecast.player_experiment.benchmark --source sample --out runs/player-features-v1
Artifacts are isolated, and no existing Forecaster registry/default is changed.
"""
import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss

from nba_agent_paths import DATA, RUNS
from forecast.baselines import evaluate
from forecast.player_experiment.features import BASE_FEATURES, ENRICHED_FEATURES, build_rows
from forecast.player_experiment.model import PlayerBoost, PointsCalibrator, points_probabilities

THRESHOLDS = np.array([10, 15, 20, 25, 30, 35])
FOLDS = [('2025', '2024-10-01', '2025-01-01', '2025-02-01', '2025-04-14'),
         ('2026', '2025-10-01', '2026-01-01', '2026-02-01', '2026-04-13')]


def probability_metrics(y, p):
    p = np.asarray(p)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Nonfinite or invalid probability output')
    return {'brier': float(brier_score_loss(np.asarray(y).ravel(), p.ravel())),
            'log_loss': float(log_loss(np.asarray(y).ravel(), np.clip(p.ravel(), 1e-6, 1-1e-6), labels=[0,1]))}


def game_bootstrap(rows, y, baseline, candidate):
    delta = np.mean((baseline-y)**2 - (candidate-y)**2, axis=1)
    groups = pd.DataFrame({'game_id': rows.game_id.to_numpy(), 'delta': delta}).groupby('game_id').delta.agg(['sum','count'])
    a = groups.to_numpy()
    rng = np.random.default_rng(7606)
    sampled = rng.integers(0, len(a), size=(2000, len(a)))
    draws = a[sampled].sum(axis=1)
    return {'brier_gain': float(delta.mean()), 'game_cluster_95': np.quantile(draws[:,0]/draws[:,1], [.025,.975]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['sample', 'frozen'], default='sample')
    ap.add_argument('--out', type=Path, default=RUNS / 'player-features-v1')
    ap.add_argument('--iterations', type=int, default=150)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new output directory to preserve previous experiment evidence')
    args.out.mkdir(parents=True, exist_ok=True)
    folder = DATA / args.source
    pg, games = pd.read_parquet(folder/'player_games.parquet'), pd.read_parquet(folder/'games.parquet')
    start = time.perf_counter()
    rows = build_rows(pg, games)
    feature_seconds = time.perf_counter()-start
    rows.to_parquet(args.out/'features.parquet', index=False)
    print(f'Built {len(rows)} conditional player-game rows in {feature_seconds:.2f}s', flush=True)
    metrics, validation, reliability, threshold_scores, summaries, folds = [], [], [], [], [], {}
    configs = {'baseline': BASE_FEATURES, 'enriched': ENRICHED_FEATURES}
    for name, val_start, cal_start, test_start, end in FOLDS:
        val_time, cal_time, test_time = [pd.Timestamp(x, tz='UTC') for x in (val_start,cal_start,test_start)]
        # Every fitted label must have finished before the next period begins.
        initial = rows[rows.final_at < val_time]
        val = rows[(rows.as_of >= val_time) & (rows.final_at < cal_time)]
        fit = rows[rows.final_at < cal_time]
        cal = rows[(rows.as_of >= cal_time) & (rows.final_at < test_time)]
        test = rows[(rows.as_of >= test_time) & (rows.tip_time < pd.Timestamp(end,tz='America/New_York').tz_convert('UTC'))]
        if min(map(len,(initial,val,fit,cal,test))) < 100:
            raise ValueError('Insufficient data in a chronological partition')
        folds[name] = {label: {'rows':len(x),'games':int(x.game_id.nunique())} for label,x in
                       [('initial',initial),('validation',val),('refit',fit),('calibration',cal),('test',test)]}
        y = (test.pts.to_numpy()[:,None] >= THRESHOLDS).astype(int)
        yc = (cal.pts.to_numpy()[:,None] >= THRESHOLDS).astype(int)
        outputs, bundles = {}, {}
        for label, features in configs.items():
            print(f'{name}: {label}, validation fit ({len(initial)} rows)', flush=True)
            model = PlayerBoost(features,args.iterations,name=f'player-{label}-v1')
            model.fit(initial)
            vp = model.predict(val)
            vs = evaluate(vp,val).assign(fold=name,model=label)
            validation += vs.to_dict('records')
            print(f'{name}: {label}, final fit ({len(fit)} rows)', flush=True)
            model.fit(fit)
            predict_start = time.perf_counter()
            pred = model.predict(test)
            inference_seconds = time.perf_counter()-predict_start
            metrics += evaluate(pred,test).assign(fold=name,model=label,rows=len(test),
                                                  games=test.game_id.nunique(),predict_seconds=inference_seconds).to_dict('records')
            cp, raw = points_probabilities(model.predict(cal),THRESHOLDS), points_probabilities(pred,THRESHOLDS)
            calibrator = PointsCalibrator().fit(cp,yc)
            calibrated = calibrator.predict(raw)
            bundles[label] = {'model':model,'calibrator':calibrator,
                              'available_at':cal.final_at.max().isoformat(), 'thresholds':THRESHOLDS.tolist(),
                              'source':'stats_only','conditional_on':'played','feature_version':1}
            for variant,prob in [(label,raw),(label+'_calibrated',calibrated)]:
                outputs[variant] = prob
                summaries.append(dict(fold=name,variant=variant,**probability_metrics(y,prob)))
                for j,threshold in enumerate(THRESHOLDS):
                    threshold_scores.append(dict(fold=name,variant=variant,threshold=int(threshold),
                                                 **probability_metrics(y[:,j],prob[:,j])))
                bins = np.minimum((prob.ravel()*10).astype(int),9)
                for bucket in range(10):
                    mask=bins==bucket
                    if mask.any():
                        reliability.append(dict(fold=name,variant=variant,bin=bucket,events=int(mask.sum()),
                                                mean_probability=float(prob.ravel()[mask].mean()),
                                                observed_rate=float(y.ravel()[mask].mean())))
            print(f'{name}: {label}, Brier {probability_metrics(y,raw)["brier"]:.6f} '
                  f'-> calibrated {probability_metrics(y,calibrated)["brier"]:.6f}',flush=True)
        comparisons = {}
        for b,c in [('baseline','enriched'),('baseline','baseline_calibrated'),
                    ('enriched','enriched_calibrated'),('baseline','enriched_calibrated')]:
            comparisons[f'{c}_vs_{b}'] = game_bootstrap(test,y,outputs[b],outputs[c])
        folds[name]['comparisons'] = comparisons
        folds[name]['calibrators'] = {k:{'slope':v['calibrator'].slope,'intercept':v['calibrator'].intercept,
                                         'identity_fallback':v['calibrator'].identity_fallback} for k,v in bundles.items()}
        folds[name]['boundaries'] = dict(validation_start=val_start,calibration_start=cal_start,
                                         test_start=test_start,test_end_eastern_exclusive=end)
        with (args.out/f'models-{name}.pkl').open('wb') as stream: pickle.dump(bundles,stream)
        records = test[['game_id','player_id','tip_time','as_of','pts','min']].copy()
        for variant,prob in outputs.items():
            for j,threshold in enumerate(THRESHOLDS): records[f'{variant}_ge{threshold}']=prob[:,j]
        records.to_parquet(args.out/f'predictions-{name}.parquet',index=False)
    for filename,data in [('distribution_metrics',metrics),('validation',validation),('probability_metrics',summaries),
                          ('threshold_metrics',threshold_scores),('reliability',reliability)]:
        pd.DataFrame(data).to_csv(args.out/f'{filename}.csv',index=False)
    manifest = dict(feature_seconds=feature_seconds,elapsed_seconds=time.perf_counter()-start,
                    source=args.source,iterations=args.iterations,folds=folds,
                    versions={'python':platform.python_version(),'numpy':np.__version__,
                              'pandas':pd.__version__,'sklearn':sklearn.__version__},
                    hashes={f:hashlib.sha256((folder/f'{f}.parquet').read_bytes()).hexdigest() for f in ['games','player_games']},
                    caveats=['All events conditional on played; no DNP/void pricing.',
                             'Fixed diagnostic thresholds, not actual exchange lines.',
                             'Both seasons are retrospective temporal evaluations; 2026 overlaps published results.',
                             'Calibrated probability metrics do not describe uncalibrated quantile intervals.',
                             'Bootstrap clusters by game, not season or player across games.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(f'Finished in {manifest["elapsed_seconds"]:.1f}s; no default models changed.',flush=True)


if __name__=='__main__': main()
