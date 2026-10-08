# Arm D (reasoner tool agent + sceptic) vs arm A (anchor): extra window 1 Dec 2025 - 31 Jan 2026, and pooled

**Why this window.** There are no recorded game prices for 2024-25 (Kalshi coverage in `data/frozen/prices.parquet` starts 20 Oct 2025; the Polymarket/venue tables cover only the test period), so the agent cannot trade that season. The extra window is the earlier part of 2025-26 instead: 60 game-days, 1 Dec 2025 - 31 Jan 2026 (October is skipped: prices start mid-month and there is little injury news).

**Leakage handling.** The production M4 and M6 are trained on data up to 1 Feb / 15 Jan 2026, which covers this window, so they are not used here. M4 is refit on the 2,936 games tipping before 1 Dec 2025 and M6 on Kalshi-era rows before 20 Nov 2025 (early stopping on 20-30 Nov), with the refit M4 feeding M6's news features (`evaluation/walkforward_models.py`, loaded through `NBA_MODEL_DIR`). The LLM prompts, the 4¢ minimum edge and the sceptic were fixed before this run; the only earlier contact with the window is a two-day Gemini prompt check on 5-6 Jan 2026.

Same replay settings as the full test run: `deepseek-reasoner`, every decision point (no subsampling), $20 stake, caps $50/$100/$300, fills at the ask plus the Kalshi fee, min edge 4¢, no learning. Staked includes fees; a win is a trade with positive P&L. 95% CIs from a day-clustered bootstrap (2000 replicates, seed 7606); the pooled bootstrap resamples days from both windows.

| Window | Arm | Days | Decision points | Trades | Staked | P&L [95% CI] | Return on staked | Return on $1,000 | CLV $ [95% CI] | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dec 2025 - Jan 2026 (walk-forward) | A | 60 | 630 | 95 | $1,959 | -85 [-663, +533] | -4.3% | -8.5% | -26 [-42, -9] | 39% |
| Dec 2025 - Jan 2026 (walk-forward) | D | 60 | 630 | 4 | $83 | -50 [-133, +13] | -60.1% | -5.0% | +3 [-0, +7] | 25% |
| Test, Feb - Apr 2026 | A | 64 | 771 | 129 | $2,645 | -247 [-718, +265] | -9.3% | -24.7% | -35 [-60, -12] | 45% |
| Test, Feb - Apr 2026 | D | 64 | 771 | 4 | $83 | +88 [-83, +372] | +107.2% | +8.8% | +4 [-2, +16] | 50% |
| Pooled | A | 124 | 1401 | 224 | $4,604 | -332 [-1,101, +508] | -7.2% | -33.2% | -61 [-91, -33] | 42% |
| Pooled | D | 124 | 1401 | 8 | $165 | +39 [-157, +349] | +23.4% | +3.9% | +7 [-1, +20] | 38% |

Sceptic (arm D):

| Window | Reviewed | Approved | Vetoed | Mean CLV / contract, vetoed | Mean CLV / contract, kept |
| --- | --- | --- | --- | --- | --- |
| Dec 2025 - Jan 2026 | 77 | 4 | 73 | -0.0044 | +0.0125 |
| Test, Feb - Apr 2026 | 105 | 4 | 101 | -0.0044 | +0.0075 |

Paired D − A on the same days (day-clustered bootstrap, one-sided p for D > A):

| Window | P&L difference [95% CI] | p | CLV $ difference [95% CI] | p |
| --- | --- | --- | --- | --- |
| Dec 2025 - Jan 2026 | +35.06 [-577.81, +627.04] | 0.473 | +29.06 [+12.10, +45.46] | <0.001 |
| Test, Feb - Apr 2026 | +335.76 [-220.37, +855.23] | 0.115 | +39.59 [+14.68, +66.22] | 0.003 |
| Pooled | +370.82 [-496.38, +1,194.38] | 0.170 | +68.66 [+38.58, +100.20] | <0.001 |

Kept D trades, Dec 2025 - Jan 2026:

| Date | Game | Bought | Price | Won | P&L | CLV / contract | CLV $ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-05 | DEN @ ATL | ATL yes | 0.38 | lost | -20.62 | +0.025 | +1.30 |
| 2025-12-13 | NYK @ ORL | NYK no | 0.36 | lost | -20.69 | +0.025 | +1.38 |
| 2025-12-28 | SAC @ LAL | LAL no | 0.19 | lost | -21.09 | +0.005 | +0.53 |
| 2026-01-17 | BOS @ ATL | ATL no | 0.60 | won | +12.64 | -0.005 | -0.17 |

DeepSeek spend for this window (account balance before minus after): $1.99 ($32.92 → $30.93).
