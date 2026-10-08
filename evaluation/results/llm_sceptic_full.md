# Arm D (reasoner tool agent + sceptic) vs arm A (anchor), full test period

**Complete result: 64 of 64 game-days, 771 of 771 decision points** (every decision point on a scored day; no subsampling). D comes from the finished full-sample `deepseek-reasoner` runs (test-reasoner-full, test-reasoner-full-b); days they do not cover are replayed from the LLM cache only and count only if every call on them was cached. A is the full-sample anchor run restricted to the same days. Staked includes fees. Replay: $20 stake, caps $50/$100/$300, fills at the ask plus the Kalshi fee, min edge 4¢, no learning. 95% CIs from a day-clustered bootstrap (2000 replicates, seed 7606).

| Arm | Trades | Staked | P&L [95% CI] | Return on staked | Return on $1,000 | CLV $ [95% CI] | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 129 | $2,645 | -247 [-718, +265] | -9.3% | -24.7% | -35 [-60, -12] | 43% |
| D | 4 | $83 | +88 [-83, +372] | +107.2% | +8.8% | +4 [-2, +16] | 25% |

Paired D − A on the same days: pnl +335.76 [-220.37, +855.23]; clv_dollars +39.59 [+14.68, +66.22]

Sceptic: 105 reviews, 4 approved, 101 vetoed. Mean CLV per contract: vetoed -0.0044 vs kept +0.0075.

Long-shot dependence (price ≤ 20¢): D P&L excluding long shots -32 (1 long-shot trades); A -47 (21).

Kept D trades:

| Date | Game | Bought | Price | Won | P&L | CLV / contract | CLV $ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-06 | MEM @ POR | MEM yes | 0.29 | lost | -20.71 | -0.015 | -1.02 |
| 2026-02-26 | POR @ CHI | CHI yes | 0.40 | lost | -20.84 | +0.005 | +0.25 |
| 2026-03-20 | GSW @ DET | GSW no | 0.67 | won | +9.12 | +0.005 | +0.15 |
| 2026-03-23 | IND @ ORL | IND yes | 0.14 | won | +120.92 | +0.035 | +4.97 |

DeepSeek spend (token estimate at list price, all reasoner calls in the cache): $3.23 since the full run started, $8.68 in total.
