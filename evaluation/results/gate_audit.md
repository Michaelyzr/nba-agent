# Gate audit

Pre-registered in `docs/preregistration_gate.md`. Day-clustered bootstrap, 2000 replicates, seed 7606. **Reported window (deadline deviation):** test 2026-02-01–2026-04-12 (not clean; learning arms start empty unless `--warm-dev`). Walk-forward 2025-11-01–2026-04-12 is optional via `--period wf`.

## Summaries

| Period | Setup | Rules proposed / accepted | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | Worst day | Days < −$100 | Kill-switch trips (enforced) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | Legacy gate + learning | 24 / 0 | 129 | -0.0035 [-0.0065, +0.0000] | -35 [-60, -12] | -247 [-718, +265] | -83 | 0 | 0 |
| test | Split gate + learning | 11 / 5 | 53 | -0.0050 [-0.0101, +0.0019] | -18 [-34, -2] | -96 [-437, +247] | -54 | 0 | 0 |
| test | Split gate + learning + kill switch | 11 / 5 | 53 | -0.0050 [-0.0101, +0.0019] | -18 [-34, -2] | -96 [-437, +247] | -54 | 0 | 0 |
| test | No learning (gate never runs) | 0 / 0 | 129 | -0.0035 [-0.0065, +0.0000] | -35 [-60, -12] | -247 [-718, +265] | -83 | 0 | 0 |
| test | Never trade | 0 / 0 | 0 | – | +0 [+0, +0] | +0 [+0, +0] | +0 | 0 | 0 |

## Paired comparisons

| Period | Comparison | Metric | Difference [95% CI] | p (one-sided) |
| --- | --- | --- | --- | --- |
| test | Split gate + learning − Legacy gate + learning | mean_clv | -0.0015 [-0.0053, +0.0028] | 0.755 |
| test | Split gate + learning − Legacy gate + learning | clv_dollars | +18 [+1, +34] | 0.020 |
| test | Split gate + learning − Legacy gate + learning | pnl | +151 [-238, +541] | 0.222 |
| test | Split gate + learning + kill switch − Split gate + learning | mean_clv | +0.0000 [+0.0000, +0.0000] | – |
| test | Split gate + learning + kill switch − Split gate + learning | clv_dollars | +0 [+0, +0] | – |
| test | Split gate + learning + kill switch − Split gate + learning | pnl | +0 [+0, +0] | – |
| test | Split gate + learning − No learning (gate never runs) | mean_clv | -0.0015 [-0.0053, +0.0028] | 0.755 |
| test | Split gate + learning − No learning (gate never runs) | clv_dollars | +18 [+1, +34] | 0.020 |
| test | Split gate + learning − No learning (gate never runs) | pnl | +151 [-238, +541] | 0.222 |
| test | Split gate + learning − Never trade | mean_clv | -0.0050 [-0.0101, +0.0019] | 0.970 |
| test | Split gate + learning − Never trade | clv_dollars | -18 [-34, -2] | 0.980 |
| test | Split gate + learning − Never trade | pnl | -96 [-437, +247] | 0.703 |
| test | Legacy gate + learning − Never trade | mean_clv | -0.0035 [-0.0065, +0.0000] | 0.990 |
| test | Legacy gate + learning − Never trade | clv_dollars | -35 [-60, -12] | 0.997 |
| test | Legacy gate + learning − Never trade | pnl | -247 [-718, +265] | 0.831 |
| test | Split gate + learning + kill switch − Never trade | mean_clv | -0.0050 [-0.0101, +0.0019] | 0.970 |
| test | Split gate + learning + kill switch − Never trade | clv_dollars | -18 [-34, -2] | 0.980 |
| test | Split gate + learning + kill switch − Never trade | pnl | -96 [-437, +247] | 0.703 |

## Hypotheses

- **H1** (held-out gate days → fewer rules pass): split accepted 5/11 proposed rules, legacy 0/24. See `gate_placebo.md` for the pass-rate CI.
- **H3** (kill switch cuts the worst day, mean CLV unchanged): worst day -54 with the switch vs -54 without; days below −$100 0 vs 0; 0 enforced trips; mean-CLV difference +0.0000 [+0.0000, +0.0000].

The kill switch counts only games already final when an order is placed, so a day can still end below −$100 when games tip together and lose after the last order. The replay confirms platform paper orders automatically (a replay simplification); retail orders need an explicit confirmation callable.

