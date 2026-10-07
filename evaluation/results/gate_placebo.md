# Placebo gate audit

Pre-registered in `docs/preregistration_gate.md` (H1, H2). Base: no-learning market-anchor agent, 2026-02-01–2026-04-12 (**deadline deviation:** test window, not the walk-forward season). Review every 7th market day after 21 market days of history (7 review points). At each point: the reviewer's own pick, all 11 templates (= exact pass rate of a uniformly random template) and 5 random-slice rules (seeded hash buckets, width = share of selection trades the reviewer's rule hit, else 0.25), each gated under split, split-edge, legacy. Rules only remove trades, so they are applied to the base fills exactly, without replaying. Pass-rate CIs: Wilson 95%. Differences and CLV $ effects: bootstrap over review points, 2000 replicates, seed 7606.

## Pass rates

| Gate | Rule kind | Rules gated | Accepted | Pass rate [95% CI] | Mean Δ CLV $ on test days | Mean forward Δ CLV $ (next 7 days) |
| --- | --- | --- | --- | --- | --- | --- |
| split | reviewer | 7 | 4 | 57.1% [25.0%, 84.2%] | +3.38 | +3.48 |
| split | template | 77 | 30 | 39.0% [28.8%, 50.1%] | +2.20 | +2.11 |
| split | random_slice | 35 | 20 | 57.1% [40.9%, 72.0%] | +3.04 | +1.42 |
| split-edge | reviewer | 7 | 2 | 28.6% [8.2%, 64.1%] | +3.38 | +3.48 |
| split-edge | template | 77 | 8 | 10.4% [5.4%, 19.2%] | +2.20 | +2.11 |
| split-edge | random_slice | 35 | 5 | 14.3% [6.3%, 29.4%] | +2.44 | +1.84 |
| legacy | reviewer | 7 | 2 | 28.6% [8.2%, 64.1%] | +6.47 | +2.93 |
| legacy | template | 77 | 5 | 6.5% [2.8%, 14.3%] | +2.88 | +2.11 |
| legacy | random_slice | 35 | 0 | 0.0% [0.0%, 9.9%] | +2.71 | +1.83 |

## Bootstrap comparisons

| Comparison | Estimate [95% CI] |
| --- | --- |
| H1: reviewer pass rate, split − legacy | +28.6% [-28.6%, +71.4%] |
| H1: reviewer pass rate, split-edge − legacy | +0.0% [-42.9%, +42.9%] |
| H2 (split): reviewer − random template pass rate | +18.2% [-1.3%, +36.4%] |
| H2 (split): reviewer − random slice pass rate | +0.0% [-28.6%, +28.6%] |
| H2 (split-edge): reviewer − random template pass rate | +18.2% [-2.6%, +45.5%] |
| H2 (split-edge): reviewer − random slice pass rate | +14.3% [+0.0%, +34.3%] |
| H2 (legacy): reviewer − random template pass rate | +22.1% [+0.0%, +51.9%] |
| H2 (legacy): reviewer − random slice pass rate | +28.6% [+0.0%, +71.4%] |
| H2: random-slice pass rate, split − legacy | +57.1% [+34.3%, +80.0%] |
| H2: random-slice pass rate, split-edge − legacy | +14.3% [+0.0%, +34.3%] |
| Forward CLV $, accepted − rejected rules (split, all kinds) | -0.76 [-2.59, +1.23] |
| Forward CLV $, accepted − rejected rules (split-edge, all kinds) | -2.12 [-4.16, +1.22] |
| Forward CLV $, accepted − rejected rules (legacy, all kinds) | -2.39 [-4.38, -0.69] |
| Δ CLV $ on test days of accepted reviewer rules (split) | +6.55 [+4.46, +9.03] |
| Δ CLV $ on test days of accepted reviewer rules (split-edge) | +8.20 [+6.36, +10.05] |
| Δ CLV $ on test days of accepted reviewer rules (legacy) | +12.63 [+7.91, +17.36] |

Reading the table: H1 is supported if the split − legacy reviewer CI lies below 0. H2 is supported if the reviewer − random CIs include 0 under legacy. If the accepted − rejected forward CLV $ CI includes 0, the gate cannot tell lessons from noise.

