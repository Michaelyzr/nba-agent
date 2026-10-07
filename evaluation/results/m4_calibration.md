# M4 recalibration

Calibrators fit on 2889 out-of-sample M4 predictions for games from Nov 2023 to 31 Jan 2026 (rolling origin: M4 refit before each month on all earlier games). Scored on the 501 test games (1 Feb – 12 Apr 2026) with the default M4 (trained on games before 1 Feb). Platt: slope 0.93, intercept -0.02 on the logit (slope > 1 stretches M4 away from 50%). Brier-difference CIs resample game-days (2000 replicates).

## Forecast accuracy on the 501 test games

| Predictor | Brier | Log loss | Accuracy | Brier − raw M4 [95% CI] | Brier − market 1 h [95% CI] |
| --- | --- | --- | --- | --- | --- |
| M4 raw, absences known | 0.1827 | 0.5476 | 74.1% | +0.0000 [+0.0000, +0.0000] | +0.0186 [+0.0107, +0.0271] |
| M4 Platt, absences known | 0.1846 | 0.5523 | 74.5% | +0.0019 [+0.0011, +0.0027] | +0.0205 [+0.0125, +0.0288] |
| M4 isotonic, absences known | 0.1841 | 0.5510 | 74.9% | +0.0014 [-0.0009, +0.0036] | +0.0200 [+0.0123, +0.0282] |
| M4 Feb–Apr Platt, absences known | 0.1815 | 0.5429 | 72.3% | -0.0012 [-0.0034, +0.0011] | +0.0174 [+0.0090, +0.0262] |
| Anchor estimate, raw M4 shift | 0.1663 | 0.5040 | 75.6% | -0.0164 [-0.0232, -0.0093] | +0.0022 [-0.0020, +0.0066] |
| Anchor estimate, Platt M4 shift | 0.1664 | 0.5040 | 75.6% | -0.0163 [-0.0232, -0.0093] | +0.0023 [-0.0018, +0.0065] |
| Anchor estimate, isotonic M4 shift | 0.1659 | 0.5055 | 74.9% | -0.0168 [-0.0237, -0.0096] | +0.0018 [-0.0023, +0.0060] |
| Anchor estimate, Feb–Apr Platt M4 shift | 0.1665 | 0.5049 | 75.8% | -0.0162 [-0.0231, -0.0089] | +0.0024 [-0.0020, +0.0070] |
| Market 24 h before tip | 0.1676 | 0.5072 | 75.0% | -0.0151 [-0.0224, -0.0079] | +0.0035 [-0.0002, +0.0075] |
| Market 1 h before tip | 0.1641 | 0.4973 | 75.0% | -0.0186 [-0.0271, -0.0107] | +0.0000 [+0.0000, +0.0000] |
| Market at tip | 0.1633 | 0.4956 | 75.8% | -0.0195 [-0.0279, -0.0115] | -0.0009 [-0.0023, +0.0005] |

Sensitivity (chronological split): Platt fit only on January 2026 (M4 fit before 1 Jan, 233 games) gives slope 0.74 and test Brier 0.1904 for M4 with absences known. Diagnostic only (uses test outcomes): a Platt map fit on the test games themselves has slope 1.55 and Brier 0.1771, the best any monotone-logit recalibration could do; it is still worse than the market.

## Why the pooled calibrators do not help: M4's calibration changes with the season

Out-of-sample Platt slope by season phase (slope > 1 means M4 is too close to 50%, < 1 too far). The "Feb–Apr Platt" calibrator above is fit on the earlier seasons' Feb–Apr rows only; it was chosen after the pooled calibrators failed, so treat it as exploratory.

| Season | Phase | Games | Platt slope | Brier |
| --- | --- | --- | --- | --- |
| 2023-24 | Feb–Apr | 561 | 0.99 | 0.2148 |
| 2023-24 | Oct–Jan | 231 | 0.68 | 0.2223 |
| 2024-25 | Feb–Apr | 563 | 1.27 | 0.1953 |
| 2024-25 | Oct–Jan | 712 | 0.96 | 0.2114 |
| 2025-26 | Feb–Apr (test period) | 549 | 1.33 | 0.1831 |
| 2025-26 | Oct–Jan | 730 | 0.78 | 0.2227 |

## Trading with calibrated M4 (Platt unless marked)

Day-clustered bootstrap, evaluation/stats.py. Raw-M4 rows are the existing runs.

| Period | Setup | M4 | Trades | P&L after fees [95% CI] | Mean CLV [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Test 1 Feb–12 Apr | Never trade | — | 0 | +0 [+0, +0] | — |
| Test 1 Feb–12 Apr | Raw model, no agent | raw M4 (existing run) | 366 | -2087 [-3083, -1045] | -0.0047 [-0.0063, -0.0031] |
| Test 1 Feb–12 Apr | Raw model, no agent | Platt M4 | 370 | -2346 [-3461, -1263] | -0.0046 [-0.0062, -0.0030] |
| Test 1 Feb–12 Apr | Raw model, no agent | isotonic M4 | 372 | -1830 [-2772, -869] | -0.0045 [-0.0062, -0.0028] |
| Test 1 Feb–12 Apr | Anchor agent, no learning | raw M4 (existing run) | 129 | -247 [-718, +265] | -0.0035 [-0.0065, +0.0000] |
| Test 1 Feb–12 Apr | Anchor agent, no learning | Platt M4 | 126 | -251 [-720, +254] | -0.0035 [-0.0065, +0.0000] |
| Test 1 Feb–12 Apr | Anchor agent + learning | raw M4 (existing run) | 74 | -32 [-340, +281] | -0.0022 [-0.0067, +0.0031] |
| Walk-forward 1 Nov–12 Apr | Never trade | — | 0 | +0 [+0, +0] | — |
| Walk-forward 1 Nov–12 Apr | Raw model, no agent | raw M4 (existing run) | 711 | -2475 [-3995, -845] | -0.0046 [-0.0057, -0.0035] |
| Walk-forward 1 Nov–12 Apr | Raw model, no agent | Platt M4 | 722 | -2732 [-4379, -1010] | -0.0045 [-0.0056, -0.0034] |
| Walk-forward 1 Nov–12 Apr | Anchor agent, no learning | raw M4 (existing run) | 276 | -407 [-1324, +553] | -0.0048 [-0.0068, -0.0027] |
| Walk-forward 1 Nov–12 Apr | Anchor agent, no learning | Platt M4 | 270 | -401 [-1315, +559] | -0.0049 [-0.0069, -0.0028] |
| Walk-forward 1 Nov–12 Apr | Anchor agent + learning | raw M4 (existing run) | 157 | -25 [-569, +564] | -0.0050 [-0.0079, -0.0018] |

## Paired P&L differences on the same game-days

| Period | Setup | Comparison | P&L difference [95% CI] | p (one-sided, > 0) |
| --- | --- | --- | --- | --- |
| Test 1 Feb–12 Apr | Raw model, no agent | Platt M4 − never trade | -2346 [-3461, -1263] | 1.000 |
| Test 1 Feb–12 Apr | Raw model, no agent | Platt M4 − same setup with raw M4 | -259 [-664, +251] | 0.879 |
| Test 1 Feb–12 Apr | Raw model, no agent | isotonic M4 − never trade | -1830 [-2772, -869] | 1.000 |
| Test 1 Feb–12 Apr | Raw model, no agent | isotonic M4 − same setup with raw M4 | +258 [-145, +696] | 0.122 |
| Test 1 Feb–12 Apr | Anchor agent, no learning | Platt M4 − never trade | -251 [-720, +254] | 0.839 |
| Test 1 Feb–12 Apr | Anchor agent, no learning | Platt M4 − same setup with raw M4 | -3 [-118, +84] | 0.610 |
| Walk-forward 1 Nov–12 Apr | Raw model, no agent | Platt M4 − never trade | -2732 [-4379, -1010] | 0.999 |
| Walk-forward 1 Nov–12 Apr | Raw model, no agent | Platt M4 − same setup with raw M4 | -257 [-746, +331] | 0.809 |
| Walk-forward 1 Nov–12 Apr | Anchor agent, no learning | Platt M4 − never trade | -401 [-1315, +559] | 0.805 |
| Walk-forward 1 Nov–12 Apr | Anchor agent, no learning | Platt M4 − same setup with raw M4 | +6 [-99, +102] | 0.453 |
