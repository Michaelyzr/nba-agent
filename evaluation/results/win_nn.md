# M4-NN: neural win models vs M4 and the market

Trained on 3365 games before 2026-02-01 (early stopping on 1 Dec – 31 Jan, then refit on all training games for the chosen epoch count), scored on the 501 test games (1 Feb – 12 Apr 2026) with absences known, as M4. MLP: M4's six difference features → hidden → 1. GRU: shared GRU over each team's last 10 completed games plus the six features. Hidden size and learning rate chosen by mean validation BCE (table at the end); three seeds each; the forecast is the 3-seed mean. CIs resample game-days (2000 replicates, seed 7606). "Anchor + X shift" = market mid 24 h before tip + model X's (absences known − absences ignored) shift: the estimate the deterministic agent trades on.

## Forecast accuracy

| Predictor | Brier | Log loss | Acc. | Brier − M4 [95% CI] | Log loss − M4 [95% CI] | Brier − market 1 h [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| M4 (logistic regression) | 0.1827 | 0.5476 | 74.1% | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0186 [+0.0107, +0.0271] |
| M4-NN MLP (3-seed mean) | 0.1842 | 0.5507 | 73.3% | +0.0015 [-0.0056, +0.0080] | +0.0032 [-0.0133, +0.0183] | +0.0201 [+0.0112, +0.0284] |
| M4-NN GRU (3-seed mean) | 0.1864 | 0.5575 | 73.9% | +0.0036 [-0.0017, +0.0086] | +0.0099 [-0.0035, +0.0217] | +0.0222 [+0.0142, +0.0302] |
| Anchor + M4 shift (agent's estimate) | 0.1663 | 0.5040 | 75.6% | -0.0164 [-0.0232, -0.0093] | -0.0436 [-0.0607, -0.0250] | +0.0022 [-0.0020, +0.0066] |
| Anchor + MLP shift | 0.1632 | 0.4972 | 75.6% | -0.0195 [-0.0291, -0.0101] | -0.0504 [-0.0776, -0.0198] | -0.0009 [-0.0063, +0.0048] |
| Anchor + GRU shift | 0.1642 | 0.4975 | 75.2% | -0.0185 [-0.0272, -0.0101] | -0.0501 [-0.0724, -0.0277] | +0.0000 [-0.0043, +0.0047] |
| Market 24 h before tip | 0.1676 | 0.5072 | 75.0% | -0.0151 [-0.0224, -0.0079] | -0.0404 [-0.0573, -0.0236] | +0.0035 [-0.0002, +0.0075] |
| Market 1 h before tip | 0.1641 | 0.4973 | 75.0% | -0.0186 [-0.0271, -0.0107] | -0.0502 [-0.0700, -0.0316] | +0.0000 [+0.0000, +0.0000] |

## Seed variability (single-seed models)

| Model | Brier mean ± std | Brier range | Log loss mean ± std | Best epochs |
| --- | --- | --- | --- | --- |
| MLP | 0.1846 ± 0.0016 | 0.1835 – 0.1864 | 0.5516 ± 0.0036 | 31, 17, 29 |
| GRU | 0.1869 ± 0.0014 | 0.1857 – 0.1885 | 0.5584 ± 0.0028 | 5, 7, 5 |

## Anchor estimates vs the agent's current estimate (anchor + M4 shift)

| Predictor | Brier − agent's estimate [95% CI] | Log loss − agent's estimate [95% CI] |
| --- | --- | --- |
| Anchor + MLP shift | -0.0032 [-0.0079, +0.0013] | -0.0068 [-0.0237, +0.0131] |
| Anchor + GRU shift | -0.0022 [-0.0058, +0.0012] | -0.0065 [-0.0159, +0.0029] |

## Validation grid (1 Dec – 31 Jan, fit on earlier games)

Logistic regression on the same features and slice: validation BCE 0.6660.

| Model | Hidden | Learning rate | Validation BCE mean ± std (3 seeds) | Chosen |
| --- | --- | --- | --- | --- |
| MLP | 8 | 0.001 | 0.6614 ± 0.0058 |  |
| MLP | 8 | 0.0003 | 0.6626 ± 0.0066 |  |
| MLP | 16 | 0.001 | 0.6604 ± 0.0029 |  |
| MLP | 16 | 0.0003 | 0.6602 ± 0.0024 | yes |
| GRU | 8 | 0.001 | 0.6608 ± 0.0014 | yes |
| GRU | 8 | 0.0003 | 0.6609 ± 0.0016 |  |
| GRU | 16 | 0.001 | 0.6633 ± 0.0032 |  |
| GRU | 16 | 0.0003 | 0.6620 ± 0.0027 |  |
