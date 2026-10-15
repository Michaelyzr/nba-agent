# Sceptic robustness: rules, stability, memorisation

Original approve/reject sceptic (`SCEPTIC_SYSTEM` + `ToolAgent._sceptic` parse, fail closed) on its 182 reviewed proposals (labelled set from `evaluation/sceptic_balanced.py`; CLV $ of the counterfactual $20 fill). Original: 8 approvals, kept CLV $+7.38; all 182 sum to $-43.84. Script: `evaluation/sceptic_robustness.py`.

## Test 3: one-line rules at a matched keep count of 8

Each rule keeps the top 8 by its continuous score (fixed-seed random tie-break). Random percentile = share of 20,000 random 8-subsets of the 182 with lower kept CLV $.

| Filter | Kept CLV $ | Mean CLV (c) | Overlap with LLM | Random pct | Random mean [95%] |
|---|---|---|---|---|---|
| LLM sceptic (original approvals) | +7.38 | +1.00 | 8 | 100 | -1.94 [-6.54, +3.51] |
| skip if moved against (keep most toward) | -0.72 | -0.06 | 0 | 71 | -1.94 [-6.54, +3.51] |
| skip if moved toward >= 3c (keep least toward) | -5.26 | -1.19 | 0 | 8 | -1.94 [-6.54, +3.51] |
| skip stale news (keep freshest) | +0.20 | +0.00 | 0 | 82 | -1.94 [-6.54, +3.51] |
| skip if 6h move against (keep most 6h toward) | -1.18 | -0.13 | 0 | 65 | -1.94 [-6.54, +3.51] |
| keep largest gap after fees | -5.45 | -1.00 | 0 | 7 | -1.94 [-6.54, +3.51] |

## Test 1: stability (3 fresh draws, roles `sceptic_rep1..3`)

All 182 proposals (the 60-proposal stratified subset, seed 7606, was run first).

| Draw | Approvals (subset) | Original 8 re-approved | Approvals scaled to 182 | Kept CLV $ (subset) | Kept CLV $ scaled to 182 | Agreement with original | Invalid |
|---|---|---|---|---|---|---|---|
| draw orig | 8 | 8 | 8.0 | +7.38 | +7.38 | 100% | 0 |
| draw rep1 | 5 | 1 | 5.0 | +1.71 | +1.71 | 94% | 0 |
| draw rep2 | 5 | 2 | 5.0 | +2.05 | +2.05 | 95% | 0 |
| draw rep3 | 5 | 1 | 5.0 | +6.04 | +6.04 | 94% | 1 |

Fleiss' kappa over 4 draws: **0.15**; mean pairwise agreement 95%; unanimous on 91% of proposals. Fresh draws' kept CLV $ (scaled): mean +3.27, range +1.71 to +6.04. Fresh-draw approval rate: 17% on the original approvals vs 2.1% on the original rejects. Fresh-draw kept CLV $ vs random same-size subsets of the evaluated rows (unweighted, so only meaningful on the full set): percentiles 93, 95, 100.

Per-proposal approval rate across the 4 draws (original + 3), approvals and any proposal approved at least once:

| Game | Date | Original | CLV $ | Approve rate |
|---|---|---|---|---|
| POR @ CHI | 2026-02-26 | approve | +0.25 | 0.50 |
| SAC @ LAL | 2025-12-28 | approve | +0.53 | 0.50 |
| IND @ ORL | 2026-03-23 | approve | +4.97 | 0.50 |
| MIL @ WAS | 2026-01-29 | reject | +0.66 | 0.50 |
| WAS @ MIA | 2026-03-10 | reject | +1.11 | 0.50 |
| NYK @ ORL | 2025-12-13 | approve | +1.38 | 0.50 |
| DEN @ ATL | 2025-12-05 | approve | +1.30 | 0.25 |
| GSW @ DET | 2026-03-20 | approve | +0.15 | 0.25 |
| IND @ SAC | 2026-03-10 | reject | +0.17 | 0.25 |
| DEN @ OKC | 2026-02-27 | reject | -0.33 | 0.25 |
| MEM @ POR | 2026-02-07 | reject | -0.37 | 0.25 |
| NYK @ BOS | 2026-02-08 | reject | -0.17 | 0.25 |
| MEM @ POR | 2026-02-06 | approve | -1.02 | 0.25 |
| BOS @ ATL | 2026-01-17 | approve | -0.17 | 0.25 |
| PHX @ DET | 2026-01-15 | reject | -0.33 | 0.25 |
| PHI @ ORL | 2026-01-09 | reject | +0.00 | 0.25 |
| IND @ CHI | 2026-04-01 | reject | +0.17 | 0.25 |

## Test 2a: outcome recall (deepseek-flash, thinking off, temperature 0)

20 games (8 approved + 12 random reviewed). Answered (not 'unknown'): 0%. Winner correct: 0% of all, nan% of answered (chance 50%). Exact score: 0%; median |score error| nan points.

| Game | Date | Approved | Actual | Said winner | Said score | Correct |
|---|---|---|---|---|---|---|
| DEN @ ATL | 2025-12-05 | yes | DEN 134-133 | UNKNOWN | nan | no |
| NYK @ ORL | 2025-12-13 | yes | NYK 132-120 | UNKNOWN | nan | no |
| SAC @ LAL | 2025-12-28 | yes | LAL 101-125 | UNKNOWN | nan | no |
| BOS @ ATL | 2026-01-17 | yes | BOS 132-106 | UNKNOWN | nan | no |
| MEM @ POR | 2026-02-06 | yes | POR 115-135 | UNKNOWN | nan | no |
| POR @ CHI | 2026-02-26 | yes | POR 121-112 | UNKNOWN | nan | no |
| GSW @ DET | 2026-03-20 | yes | DET 101-115 | UNKNOWN | nan | no |
| IND @ ORL | 2026-03-23 | yes | IND 128-126 | UNKNOWN | nan | no |
| IND @ CHA | 2026-04-03 |  | CHA 108-129 | UNKNOWN | nan | no |
| MIA @ IND | 2026-03-29 |  | IND 118-135 | UNKNOWN | nan | no |
| PHI @ ORL | 2026-01-09 |  | PHI 103-91 | UNKNOWN | nan | no |
| MIA @ PHX | 2026-01-25 |  | MIA 111-102 | UNKNOWN | nan | no |
| CHA @ BKN | 2026-03-31 |  | CHA 117-86 | UNKNOWN | nan | no |
| TOR @ POR | 2026-01-23 |  | TOR 110-98 | UNKNOWN | nan | no |
| DAL @ ORL | 2026-03-05 |  | ORL 114-115 | UNKNOWN | nan | no |
| MEM @ ORL | 2026-01-15 |  | ORL 111-118 | UNKNOWN | nan | no |
| MIN @ LAC | 2026-02-26 |  | MIN 94-88 | UNKNOWN | nan | no |
| DET @ WAS | 2026-03-17 |  | DET 130-117 | UNKNOWN | nan | no |
| IND @ SAC | 2026-03-10 |  | SAC 109-114 | UNKNOWN | nan | no |
| NYK @ DET | 2026-02-06 |  | DET 80-118 | UNKNOWN | nan | no |

## Test 2b: blinded sceptic (role `sceptic_blind`)

Team codes and names -> Team A / Team B, player names -> Player k, player ids -> neutral ids, dates -> Day X, news ids -> news-item; all other numbers kept.

- Blinded approvals on the 60-proposal subset: 4 (original: 8; fresh unblinded draws: 1, 4, 2).
- Original 8 approved when blinded: 2 of 8.
- Agreement blinded vs original: 87%; blinded vs unblinded fresh draws (mean): 92%; unblinded draws with each other (mean): 93%.
- Blinded kept CLV $: +1.01 on the subset (-1.08 scaled to 182).
- Invalid blinded replies: 0.

## Reading

**Not memorised, and a weak real signal; the +$7.38 is a lucky draw.** Fresh draws of the same prompt re-approve only 1-2 of the original 8 (Fleiss' kappa 0.15), keep 5 trades each and earn +$1.71, +$2.05 and +$6.04 (93rd-100th percentile vs random 5-subsets; mean +$3.27, less than half of +$7.38). Approvals still concentrate on the original approvals (17% vs 2.1% per draw), so the sceptic ranks proposals better than random but picks which of them it approves almost at random. No one-line rule matches it at 8 kept (best +$0.20). The model cannot name the winner of any of the 20 games, and blinding teams, players and dates changes verdicts no more than resampling does (92% vs 93% agreement), so team or outcome knowledge is not driving the approvals.

## Spend

DeepSeek balance 29.35 USD before, 27.86 USD after: $1.49 for 606 sceptic calls (546 stability draws on all 182 proposals + 60 blinded) and 20 recall questions; 0 API errors.
