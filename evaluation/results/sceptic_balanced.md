# Approach A: balanced scored sceptic vs the current approve/reject sceptic

**Setup.** The current sceptic (deepseek-reasoner, one call) argues only that the edge is priced in and answers approve/reject. The balanced sceptic (`sceptic_mode="balanced"` in `agents/tool_agent.py`, opt-in; the default is unchanged) states the strongest case for and against the trade and returns `score` = its probability that the trade beats the closing line after fees; it approves iff score ≥ τ. Invalid output or banned wording scores 0 (fail closed). It saw exactly the user prompt the current sceptic saw (read back from the LLM cache); only the system prompt and the last line differ.

**Labelled set.** Every proposal the current sceptic reviewed: 182 (77 in Dec 2025 - Jan 2026 with walk-forward refit models, 105 in the Feb - Apr 2026 test window). Each gets a counterfactual fill at its decision time from `Replay._fill`/`_settle`: $20 stake at the ask, CLV per contract = side's mid at tip − price paid (the fee is in P&L, not in CLV, as for real fills); CLV $ = CLV × contracts. Recomputed CLV matches the real fills of the 8 approved trades.

**Scores.** deepseek-reasoner (the model of the current sceptic runs), one pass per proposal, streamed (attempt capped at 180 s, no token within 60 s = failed attempt, 4 retries). 35 replies spent the whole 8,192-token budget reasoning and returned no answer; those were re-asked once with max_tokens 32,768 (any still invalid score 0, fail closed). τ is tuned on one window to maximise CLV $ of the kept set with at least 5 kept (ties → higher τ), then fixed on the other window. 95% CIs: day-clustered bootstrap (2000 replicates, seed 7606); random filter: 20,000 random subsets of the same size; percentile = share of random subsets with lower CLV $. Wrong veto = vetoed with CLV > 0; wrong approve = kept with CLV ≤ 0.

- Dec 2025 - Jan 2026: 77 proposals, keep-all CLV $ -15.66, 23 with CLV > 0, **oracle ceiling CLV $ +14.10**; current sceptic kept 4.
- Feb - Apr 2026 (test): 105 proposals, keep-all CLV $ -28.18, 32 with CLV > 0, **oracle ceiling CLV $ +25.77**; current sceptic kept 4.
- Both windows: oracle ceiling CLV $ +39.87 (55 of 182 proposals).

Chosen τ: reasoner tuned on Dec 2025 - Jan 2026 → τ = 0.600; reasoner tuned on Feb - Apr 2026 (test) → τ = 0.500

AUC of the score vs CLV > 0 (95% day-clustered CI): reasoner Feb - Apr 2026 (test) 0.59 [0.48, 0.71]; reasoner Dec 2025 - Jan 2026 0.45 [0.33, 0.60]; reasoner pooled 0.53 [0.45, 0.61]

## tune prev -> test test

| Filter | τ | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Random same size: mean [95% band] | Pctile vs random | Wrong veto | Wrong approve | CLV $ after fee | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all |  | 105 | 100% | -28.18 [-47.33, -7.42] | -0.39 [-0.57, -0.20] | -32.52 [-49.34, -16.76] | -28.18 [-28.18, -28.18] | 50 | – | 70% | -121.72 | -536.98 [-1084.43, +26.75] |
| Current sceptic (approve/reject) |  | 4 | 4% | +4.35 [-2.04, +15.55] | +0.75 [-1.50, +3.50] | – | -1.07 [-4.68, +3.88] | 98 | 29% | 25% | +0.87 | +88.49 [-82.97, +371.88] |
| Oracle (CLV > 0) |  | 32 | 30% | +25.77 [+13.73, +40.16] | +0.98 [+0.72, +1.28] | +21.43 [+12.50, +31.50] | -8.56 [-17.73, +1.18] | 100 | 0% | 0% | -1.02 | -180.65 [-478.10, +192.38] |
| Balanced reasoner, tau tuned on prev | 0.600 | 1 | 1% | +0.17 [+0.00, +0.50] | +0.50 [+0.50, +0.50] | -4.18 [-15.41, +2.31] | -0.26 [-2.00, +2.38] | 75 | 30% | 0% | -0.39 | +12.97 [+0.00, +38.91] |
| Balanced reasoner, tau = 0.5 (untuned) | 0.500 | 5 | 5% | +0.08 [-2.06, +2.39] | +0.10 [-0.90, +1.00] | -4.26 [-16.23, +2.99] | -1.35 [-5.31, +3.75] | 77 | 29% | 40% | -3.70 | -25.59 [-120.65, +59.83] |

## tune test -> test prev

| Filter | τ | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Random same size: mean [95% band] | Pctile vs random | Wrong veto | Wrong approve | CLV $ after fee | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all |  | 77 | 100% | -15.66 [-27.58, -4.29] | -0.35 [-0.67, -0.04] | -18.70 [-30.01, -8.08] | -15.66 [-15.66, -15.66] | 50 | – | 70% | -79.60 | -46.31 [-598.80, +581.64] |
| Current sceptic (approve/reject) |  | 4 | 5% | +3.04 [-0.17, +7.33] | +1.25 [-0.50, +2.50] | – | -0.80 [-3.61, +1.92] | 100 | 27% | 25% | -0.41 | -49.76 [-125.13, +12.64] |
| Oracle (CLV > 0) |  | 23 | 30% | +14.10 [+7.93, +20.71] | +1.13 [+0.79, +1.46] | +11.06 [+5.81, +16.74] | -4.67 [-10.31, +0.89] | 100 | 0% | 0% | -5.09 | +55.20 [-274.35, +453.91] |
| Balanced reasoner, tau tuned on test | 0.500 | 6 | 8% | -3.01 [-6.36, -0.25] | -1.08 [-1.50, -0.25] | -6.05 [-11.09, -1.83] | -1.22 [-4.60, +2.04] | 14 | 31% | 83% | -7.70 | -51.50 [-137.26, +36.98] |
| Balanced reasoner, tau = 0.5 (untuned) | 0.500 | 6 | 8% | -3.01 [-6.36, -0.25] | -1.08 [-1.50, -0.25] | -6.05 [-11.09, -1.83] | -1.22 [-4.60, +2.04] | 14 | 31% | 83% | -7.70 | -51.50 [-137.26, +36.98] |

## pooled leave-one-window-out

| Filter | τ | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Random same size: mean [95% band] | Pctile vs random | Wrong veto | Wrong approve | CLV $ after fee | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all |  | 182 | 100% | -43.84 [-66.84, -21.49] | -0.37 [-0.54, -0.21] | -51.22 [-71.34, -31.99] | -43.84 [-43.84, -43.84] | 50 | – | 70% | -201.32 | -583.29 [-1343.78, +285.42] |
| Current sceptic (approve/reject) |  | 8 | 4% | +7.38 [-0.65, +19.69] | +1.00 [-0.21, +2.17] | – | -1.93 [-6.57, +3.62] | 100 | 28% | 25% | +0.45 | +38.73 [-153.80, +351.27] |
| Oracle (CLV > 0) |  | 55 | 30% | +39.87 [+26.39, +54.51] | +1.05 [+0.83, +1.26] | +32.49 [+21.81, +44.29] | -13.30 [-24.12, -2.18] | 100 | 0% | 0% | -6.11 | -125.45 [-570.91, +408.60] |
| Balanced reasoner, tau from the other window |  | 7 | 4% | -2.85 [-6.52, -0.08] | -0.86 [-1.50, +0.00] | -10.23 [-22.39, -1.40] | -1.71 [-6.08, +3.61] | 31 | 30% | 71% | -8.10 | -38.53 [-129.23, +51.96] |

Calibration, reasoner, both windows (score vs realised share with positive CLV):

| Score bucket | n | Mean score | Share CLV > 0 | Share CLV $ − fee > 0 | Mean CLV ¢ |
| --- | --- | --- | --- | --- | --- |
| 0-0.2 | 105 | 0.12 | 30% | 9% | -0.38 |
| 0.2-0.4 | 57 | 0.24 | 30% | 4% | -0.39 |
| 0.4-0.5 | 9 | 0.43 | 33% | 22% | +0.00 |
| 0.5-0.6 | 5 | 0.53 | 40% | 0% | -0.30 |
| 0.6-0.8 | 5 | 0.69 | 40% | 0% | -0.40 |
| 0.8-1 | 1 | 0.82 | 0% | 0% | -2.50 |

reasoner score distribution: mean 0.20, sd 0.14, min 0.05, max 0.82, 11 of 182 at ≥ 0.5.

## Reading

Out of sample (each window filtered with τ from the other), the balanced sceptic keeps 7 proposals for CLV $ -2.85 (percentile 31 among random filters of the same size), and -10.23 [-22.39, -1.40] vs the current sceptic. Pooled AUC of the score vs CLV > 0 is 0.53 [0.45, 0.61]: the score does not rank proposals better than chance, so it does not beat the current approve/reject sceptic.

DeepSeek spend for this experiment (account balance before minus after): $0.31 ($30.19 → $29.88).
