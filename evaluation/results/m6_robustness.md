# M6 robustness grid

Pre-declared grid, same splits, loss, early stopping and static features as `forecast/impact.py`; 3 seeds each. Rows: train 7826 (602 games), validation 1638, test 6508 (501 games). Gap = MAE of the 3-seed average median prediction minus the zero-move MAE, in cents (negative beats the martingale); 95% CI resamples games (day-clustered CI in the CSV). R² is against predicting zero.

## Test, decision times

| Configuration | MAE (¢) | Zero-move MAE (¢) | Gap [95% CI, games] | Gap per seed (min – max) | R² vs zero |
| --- | --- | --- | --- | --- | --- |
| GRU-16, 6 h @ 15 min (shipped) | 0.862 | 0.833 | +0.029 [+0.017, +0.040] | +0.025 – +0.052 | -0.0039 |
| GRU-64, 6 h @ 15 min | 0.853 | 0.833 | +0.020 [+0.009, +0.030] | +0.024 – +0.029 | -0.0013 |
| GRU-128, 6 h @ 15 min | 0.863 | 0.833 | +0.029 [+0.020, +0.039] | +0.025 – +0.043 | +0.0001 |
| GRU-16, 12 h @ 15 min | 0.863 | 0.833 | +0.030 [+0.018, +0.042] | +0.022 – +0.053 | -0.0045 |
| GRU-64, 12 h @ 15 min | 0.850 | 0.833 | +0.017 [+0.008, +0.024] | +0.015 – +0.023 | +0.0041 |
| GRU-16, 24 h @ 1 h | 0.860 | 0.833 | +0.026 [+0.016, +0.038] | +0.021 – +0.043 | +0.0062 |
| GRU-64, 24 h @ 1 h | 0.854 | 0.833 | +0.021 [+0.012, +0.029] | +0.021 – +0.030 | +0.0094 |
| MLP, static features only | 0.852 | 0.833 | +0.018 [+0.010, +0.026] | +0.021 – +0.023 | +0.0053 |
| 1D CNN-64, 12 h @ 15 min | 0.870 | 0.833 | +0.036 [+0.027, +0.045] | +0.029 – +0.050 | -0.0066 |

## Test, all rows

| Configuration | MAE (¢) | Zero-move MAE (¢) | Gap [95% CI, games] | Gap per seed (min – max) | R² vs zero |
| --- | --- | --- | --- | --- | --- |
| GRU-16, 6 h @ 15 min (shipped) | 1.372 | 1.346 | +0.026 [+0.018, +0.034] | +0.022 – +0.041 | +0.0036 |
| GRU-64, 6 h @ 15 min | 1.364 | 1.346 | +0.018 [+0.012, +0.024] | +0.019 – +0.024 | +0.0021 |
| GRU-128, 6 h @ 15 min | 1.366 | 1.346 | +0.020 [+0.014, +0.025] | +0.016 – +0.028 | +0.0026 |
| GRU-16, 12 h @ 15 min | 1.372 | 1.346 | +0.027 [+0.018, +0.035] | +0.017 – +0.045 | +0.0012 |
| GRU-64, 12 h @ 15 min | 1.360 | 1.346 | +0.015 [+0.010, +0.019] | +0.010 – +0.022 | +0.0008 |
| GRU-16, 24 h @ 1 h | 1.369 | 1.346 | +0.023 [+0.015, +0.032] | +0.019 – +0.032 | +0.0029 |
| GRU-64, 24 h @ 1 h | 1.361 | 1.346 | +0.015 [+0.010, +0.021] | +0.016 – +0.021 | +0.0038 |
| MLP, static features only | 1.358 | 1.346 | +0.013 [+0.008, +0.017] | +0.012 – +0.016 | +0.0021 |
| 1D CNN-64, 12 h @ 15 min | 1.370 | 1.346 | +0.024 [+0.019, +0.030] | +0.021 – +0.032 | +0.0007 |

## Conclusion

No configuration beats the zero-move baseline. 18 of 18 gap CIs (configuration x row set) lie entirely above zero, i.e. significantly worse than predicting no move. The best on decision times is GRU-64, 12 h @ 15 min at +0.017¢ [+0.008, +0.024].
Rows where any configuration predicts a move larger than the 5.5¢ needed to cover spread and fee: at most 0 of 771 decision-time rows.
