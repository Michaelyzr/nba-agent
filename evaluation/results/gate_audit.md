# Gate audit

Pre-registered in `docs/preregistration_gate.md`. Day-clustered bootstrap, 2000 replicates, seed 7606. **Reported window (deadline deviation):** test 2026-02-01–2026-04-12 (not clean; learning arms start empty unless `--warm-dev`). Walk-forward 2025-11-01–2026-04-12 is optional via `--period wf`.

## Summaries

| Period | Setup | Rules proposed / accepted | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | Worst day | Days < −$100 | Kill-switch trips (enforced) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | Legacy gate + learning | 24 / 0 | 129 | -0.0035 [-0.0065, +0.0000] | -35 [-60, -12] | -247 [-718, +265] | -83 | 0 | 0 |
| test | No learning (gate never runs) | 0 / 0 | 129 | -0.0035 [-0.0065, +0.0000] | -35 [-60, -12] | -247 [-718, +265] | -83 | 0 | 0 |
| test | Never trade | 0 / 0 | 0 | – | +0 [+0, +0] | +0 [+0, +0] | +0 | 0 | 0 |

## Paired comparisons

| Period | Comparison | Metric | Difference [95% CI] | p (one-sided) |
| --- | --- | --- | --- | --- |
| test | Legacy gate + learning − Never trade | mean_clv | -0.0035 [-0.0065, +0.0000] | 0.990 |
| test | Legacy gate + learning − Never trade | clv_dollars | -35 [-60, -12] | 0.997 |
| test | Legacy gate + learning − Never trade | pnl | -247 [-718, +265] | 0.831 |

## Hypotheses


The kill switch counts only games already final when an order is placed, so a day can still end below −$100 when games tip together and lose after the last order. The replay confirms platform paper orders automatically (a replay simplification); retail orders need an explicit confirmation callable.

