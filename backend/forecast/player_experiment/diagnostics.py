"""Post-run stratified scoring; does not select or refit models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.player_experiment.benchmark import THRESHOLDS, probability_metrics, game_bootstrap


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    features=pd.read_parquet(args.run/'features.parquet')[['game_id','player_id','min_avg10','games_prior']]
    metrics=[]
    gains=[]
    for name in ['2025','2026']:
        rows=pd.read_parquet(args.run/f'predictions-{name}.parquet').merge(
            features,on=['game_id','player_id'],validate='one_to_one')
        for label,mask in [('all',np.ones(len(rows),dtype=bool)),
                           ('prior_minutes_at_least_20',rows.min_avg10>=20),
                           ('prior_minutes_below_20',rows.min_avg10<20),
                           ('fewer_than_5_prior_games',rows.games_prior<5)]:
            r=rows.loc[mask]
            if r.empty: continue
            y=(r.pts.to_numpy()[:,None]>=THRESHOLDS).astype(int)
            preds={v:r[[f'{v}_ge{t}' for t in THRESHOLDS]].to_numpy()
                   for v in ['baseline','enriched','baseline_calibrated','enriched_calibrated']}
            for variant,p in preds.items():
                metrics.append(dict(fold=name,slice=label,variant=variant,rows=len(r),games=r.game_id.nunique(),
                                    **probability_metrics(y,p)))
            change=game_bootstrap(r,y,preds['baseline'],preds['enriched_calibrated'])
            gains.append(dict(fold=name,slice=label,brier_gain=change['brier_gain'],
                              ci_low=change['game_cluster_95'][0],ci_high=change['game_cluster_95'][1]))
    pd.DataFrame(metrics).to_csv(args.run/'slice_metrics.csv',index=False)
    pd.DataFrame(gains).to_csv(args.run/'slice_gains.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__=='__main__': main()
