# Truncation check on the original sceptic

Every review by the original approve/reject deepseek-reasoner sceptic (fail closed) in both walk-forward windows, classified from the trace and the cached reply. max_tokens = 8192 (REASONER_MAX_TOKENS); truncated = no valid verdict with tokens_out >= 98% of the cap. API errors are never cached, so an invalid review without a cache entry is counted as an API error or timeout. No API calls.

## Classification

| Window | Reviewed | Approve | Reject (genuine) | Fail: truncated | Fail: API error/timeout | Fail: parse error | Fail: banned wording | Vetoed | of which fail-closed |
|---|---|---|---|---|---|---|---|---|---|
| Dec 2025 - Jan 2026 | 77 | 4 | 73 | 0 | 0 | 0 | 0 | 73 | 0 |
| Feb - Apr 2026 (test) | 105 | 4 | 99 | 2 | 0 | 0 | 0 | 101 | 2 |
| Pooled | 182 | 8 | 172 | 2 | 0 | 0 | 0 | 174 | 2 |

Corrected veto statements:

- Dec 2025 - Jan 2026: of 73 vetoes, 73 were genuine rejects and 0 were fail-closed (0 truncated, 0 API error/timeout, 0 parse error, 0 banned wording).
- Feb - Apr 2026 (test): of 101 vetoes, 99 were genuine rejects and 2 were fail-closed (2 truncated, 0 API error/timeout, 0 parse error, 0 banned wording).
- Pooled: of 174 vetoes, 172 were genuine rejects and 2 were fail-closed (2 truncated, 0 API error/timeout, 0 parse error, 0 banned wording).

Fail-closed reviews:

| Window | Date | Game | Side | Category | tokens_out | answer chars | CLV (c) | P&L $ |
|---|---|---|---|---|---|---|---|---|
| test | 2026-02-04 | BOS @ HOU | HOU yes | fail: truncated | 8192 | 0 | +0.5 | -19.83 |
| test | 2026-03-10 | IND @ SAC | SAC yes | fail: truncated | 8192 | 0 | +0.5 | +12.97 |

## Mean CLV per contract by category

Counterfactual $20 fills at the decision time; mean over proposals in cents per contract with a day-clustered bootstrap 95% CI (2000 draws); CLV $ and settled P&L are sums.

| Window | Group | n | Mean CLV (c) [95% CI] | CLV $ | P&L $ |
|---|---|---|---|---|---|
| Dec 2025 - Jan 2026 | approve | 4 | +1.25 [-0.25, +2.50] | +3.04 | -49.76 |
| Dec 2025 - Jan 2026 | reject (genuine) | 73 | -0.44 [-0.75, -0.11] | -18.70 | +3.45 |
| Dec 2025 - Jan 2026 | fail-closed (all) | 0 | – [–, –] | +0.00 | +0.00 |
| Dec 2025 - Jan 2026 | all vetoes (genuine + fail-closed) | 73 | -0.44 [-0.75, -0.11] | -18.70 | +3.45 |
| Feb - Apr 2026 (test) | approve | 4 | +0.75 [-1.00, +2.75] | +4.35 | +88.49 |
| Feb - Apr 2026 (test) | reject (genuine) | 99 | -0.45 [-0.63, -0.29] | -32.82 | -618.61 |
| Feb - Apr 2026 (test) | fail-closed (all) | 2 | +0.50 [+0.50, +0.50] | +0.30 | -6.86 |
| Feb - Apr 2026 (test) | all vetoes (genuine + fail-closed) | 101 | -0.44 [-0.61, -0.27] | -32.52 | -625.47 |
| Pooled | approve | 8 | +1.00 [-0.12, +2.13] | +7.38 | +38.73 |
| Pooled | reject (genuine) | 172 | -0.45 [-0.63, -0.29] | -51.52 | -615.16 |
| Pooled | fail-closed (all) | 2 | +0.50 [+0.50, +0.50] | +0.30 | -6.86 |
| Pooled | all vetoes (genuine + fail-closed) | 174 | -0.44 [-0.61, -0.28] | -51.22 | -622.02 |

The 35 truncations reported by Approach A (sceptic_balanced) were replies to the balanced prompt; the original review was also truncated for 1 of them. The original prompt rarely exhausted the budget.

## Re-ask of truncated reviews

Not run: 2 fail-closed reviews in total, below the threshold of 10; the fail-closed share is too small to change the veto statistics.
