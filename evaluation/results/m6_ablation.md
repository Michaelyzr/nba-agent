Day-clustered bootstrap (evaluation/stats.py, 2,000 resamples of game-days). CLV per contract against the mid at tip; CLV $ = CLV x contracts.

| period | setup | trades | mean CLV [95% CI] | CLV $ | P&L [95% CI] | ROI | rules active |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | M6 agent, no learning | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% | 0 |
| test | M6 agent + learning | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% | 0 |
| test | M6, edge threshold off (diagnostic) | 501 | -0.0048 [-0.0064, -0.0035] | -133.97 | -652.34 [-1641.55, +268.30] | -6.6% |  |
| test | Anchor agent, no learning | 129 | -0.0035 [-0.0065, +0.0000] | -35.25 | -247.27 [-718.30, +264.93] | -9.7% | 0 |
| test | Raw M4 agent (no anchor), no learning | 366 | -0.0047 [-0.0063, -0.0031] | -172.12 | -2087.26 [-3082.52, -1045.04] | -28.8% | 0 |
| test | Anchor agent + learning (existing run) | 74 | -0.0022 [-0.0067, +0.0031] | -10.05 | -32.02 [-340.39, +280.58] | -2.2% | 5 |
| test | Never trade | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% |  |
| holdout | M6 agent, no learning | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% | 0 |
| holdout | M6 agent + learning | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% | 0 |
| holdout | M6, edge threshold off (diagnostic) | 87 | -0.0040 [-0.0061, -0.0017] | -16.97 | +109.20 [-250.26, +510.23] | +6.4% |  |
| holdout | Anchor agent, no learning | 6 | +0.0050 [-0.0050, +0.0200] | +0.06 | -73.87 [-175.80, +26.28] | -62.0% | 0 |
| holdout | Raw M4 agent (no anchor), no learning | 59 | -0.0052 [-0.0083, -0.0026] | -23.93 | +449.55 [-55.34, +1171.25] | +38.4% | 0 |
| holdout | Anchor agent + learning | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% | 5 |
| holdout | Never trade | 0 | — | +0.00 | +0.00 [+0.00, +0.00] | +0.0% |  |
