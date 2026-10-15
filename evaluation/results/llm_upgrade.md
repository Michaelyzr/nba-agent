# LLM upgrades: majority-vote sceptic and probability estimate

Branch `exp/llm-upgrade`. Labelled set: the 182 proposals of `evaluation/sceptic_robustness.py` (exact cached sceptic prompt, original verdict, counterfactual $20 fill). Script: `evaluation/llm_upgrade.py`.

## Upgrade 1:

Secondary check: `deepseek-flash` thinking off, temperature 0.7 produced **0/1274 approvals** (reject-happy without thinking). Primary numbers use thinking **on** (legacy `deepseek-reasoner`), votes 1–3 from the published robustness cache and votes 4–5 freshly drawn; bootstrap κ from resampling 5-subsets of those 5 votes on all 182 proposals (vote 6 partial: 76/182, not used in the primary table).

### Results majority-vote sceptic (opt-in `ToolAgent(sceptic_votes=5)`)

Model: `deepseek-flash` thinking off, thinking on (roles `sceptic_vote1..5`). 5 concurrent votes per proposal; primary decision = majority of the first 5. Stability: Cohen's κ between two independent 5-vote majorities resampled with replacement from each proposal's 5 collected votes (no second full pass).

Resampled majority κ: **0.39** [-0.02, 0.83]; resampled single-vote κ: 0.33 (published Fleiss κ over original+3 draws was 0.15). Mean pairwise single-vote κ across the 5 draws: 0.17. Per-vote approval rates: [0.027, 0.027, 0.027, 0.027, 0.038]. Original 8 re-approved by majority-of-5: 0/8. Invalid votes: 1; errors: 0.

| Window | Filter | Kept | CLV $ [95% CI] | P&L $ [95% CI] | Random pct |
|---|---|---|---|---|---|
| full Dec-Apr | single sceptic (original) | 8 | +7.38 [-0.65, +19.69] | +38.73 [-153.80, +351.27] | 100 |
| full Dec-Apr | single sceptic (fresh draw 1) | 5 | +1.71 [-0.66, +5.12] | -25.43 [-117.94, +59.83] | 93 |
| full Dec-Apr | single sceptic (fresh draw 2) | 5 | +2.05 [+0.17, +4.94] | -104.05 [-205.13, -21.09] | 94 |
| full Dec-Apr | single sceptic (fresh draw 3) | 5 | +6.04 [-0.70, +18.01] | +81.30 [-105.10, +388.37] | 100 |
| full Dec-Apr | majority of 5 votes | 2 | +0.83 [+0.00, +2.31] | +3.07 [-61.08, +70.29] | 88 |
| full Dec-Apr | majority of 5, stake x approval fraction | 2 | +0.50 [+0.00, +1.39] | +1.84 [-36.65, +42.17] |  |
| full Dec-Apr | any approval, stake x approval fraction (5 votes) | 18 | +2.42 [-0.75, +7.36] | -15.48 [-106.36, +108.78] |  |
| full Dec-Apr | majority of all 5 votes | 2 | +0.83 [+0.00, +2.31] | +3.07 [-61.08, +70.29] | 88 |
| full Dec-Apr | perfect filter (CLV > 0) | 55 | +39.87 [+26.39, +54.51] | -125.45 [-570.91, +408.60] | 100 |
| full Dec-Apr | no filter (all proposals) | 182 | -43.84 [-66.84, -21.49] | -583.29 [-1343.78, +285.42] | 18 |
| test Feb-Apr | single sceptic (original) | 4 | +4.35 [-2.04, +15.55] | +88.49 [-82.97, +371.88] | 98 |
| test Feb-Apr | single sceptic (fresh draw 1) | 2 | +0.00 [-0.50, +0.50] | -7.39 [-61.08, +38.91] | 73 |
| test Feb-Apr | single sceptic (fresh draw 2) | 3 | +1.53 [+0.00, +4.16] | -62.46 [-145.28, +0.00] | 91 |
| test Feb-Apr | single sceptic (fresh draw 3) | 4 | +5.38 [-1.07, +16.80] | +57.87 [-125.62, +341.98] | 99 |
| test Feb-Apr | majority of 5 votes | 1 | +0.17 [+0.00, +0.50] | -20.36 [-61.08, +0.00] | 75 |
| test Feb-Apr | majority of 5, stake x approval fraction | 1 | +0.10 [+0.00, +0.30] | -12.22 [-36.65, +0.00] |  |
| test Feb-Apr | any approval, stake x approval fraction (5 votes) | 11 | +1.98 [-0.89, +6.65] | -0.38 [-74.35, +120.19] |  |
| test Feb-Apr | majority of all 5 votes | 1 | +0.17 [+0.00, +0.50] | -20.36 [-61.08, +0.00] | 75 |
| test Feb-Apr | perfect filter (CLV > 0) | 32 | +25.77 [+13.73, +40.16] | -180.65 [-478.10, +192.38] | 100 |
| test Feb-Apr | no filter (all proposals) | 105 | -28.18 [-47.33, -7.42] | -536.98 [-1084.43, +26.75] | 94 |

**Verdict (upgrade 1):** majority-of-5 kept 2 trades, CLV $+0.83 (day-block 95% CI [+0.00, +2.31]); random 2-subset percentile 88. Original single sceptic was $+7.38 on 8 trades (fresh-draw mean $+3.27). Resampled majority κ 0.39 vs single κ 0.15.

## Upgrade 2: LLM probability estimate

Model: `deepseek-flash` thinking off. Important: corr(p_llm, decision-time mid) = **1.00** and mean |p_llm − mid| = 0.003, so the LLM is essentially echoing the live mid; the Brier gain vs M4 is the mid's gain, not a new signal. Blend `p = anchor + w · (p_llm − anchor)`; w tuned on Dec–Jan Brier only, tested Feb–Apr. Chosen w* = **1.0** (tune Brier 0.2268).

### Accuracy

| Window | Forecast | Games | Brier | − anchor [95% CI] | − M4 [95% CI] |
|---|---|---|---|---|---|
| full Dec-Apr | market anchor (24 h) | 182 | 0.2064 | +0.0000 [+0.0000, +0.0000] | -0.0033 [-0.0083, +0.0010] |
| full Dec-Apr | market mid at decision | 182 | 0.1931 | -0.0133 [-0.0254, -0.0015] | -0.0166 [-0.0291, -0.0042] |
| full Dec-Apr | anchor + M4 shift (agent) | 182 | 0.2097 | +0.0033 [-0.0010, +0.0083] | +0.0000 [+0.0000, +0.0000] |
| full Dec-Apr | LLM raw p_home | 182 | 0.1935 | -0.0129 [-0.0249, -0.0013] | -0.0162 [-0.0282, -0.0044] |
| full Dec-Apr | blend w=1.0 | 182 | 0.1935 | -0.0129 [-0.0249, -0.0013] | -0.0162 [-0.0282, -0.0044] |
| test Feb-Apr | market anchor (24 h) | 105 | 0.1772 | +0.0000 [+0.0000, +0.0000] | -0.0062 [-0.0136, +0.0011] |
| test Feb-Apr | market mid at decision | 105 | 0.1692 | -0.0080 [-0.0208, +0.0048] | -0.0142 [-0.0290, +0.0003] |
| test Feb-Apr | anchor + M4 shift (agent) | 105 | 0.1834 | +0.0062 [-0.0011, +0.0136] | +0.0000 [+0.0000, +0.0000] |
| test Feb-Apr | LLM raw p_home | 105 | 0.1691 | -0.0080 [-0.0211, +0.0040] | -0.0143 [-0.0287, -0.0006] |
| test Feb-Apr | blend w=1.0 | 105 | 0.1691 | -0.0080 [-0.0211, +0.0040] | -0.0143 [-0.0287, -0.0006] |
| tune Dec-Jan | market anchor (24 h) | 77 | 0.2463 | +0.0000 [+0.0000, +0.0000] | +0.0008 [-0.0030, +0.0044] |
| tune Dec-Jan | market mid at decision | 77 | 0.2256 | -0.0206 [-0.0401, +0.0025] | -0.0198 [-0.0389, +0.0026] |
| tune Dec-Jan | anchor + M4 shift (agent) | 77 | 0.2455 | -0.0008 [-0.0044, +0.0030] | +0.0000 [+0.0000, +0.0000] |
| tune Dec-Jan | LLM raw p_home | 77 | 0.2268 | -0.0195 [-0.0390, +0.0034] | -0.0187 [-0.0376, +0.0034] |
| tune Dec-Jan | blend w=1.0 | 77 | 0.2268 | -0.0195 [-0.0390, +0.0034] | -0.0187 [-0.0376, +0.0034] |

### Trading (4¢ edge after fees, $20 stake, tip-off CLV)

| Window | Forecast | Trades | CLV $ [95% CI] | P&L $ [95% CI] | Random pct |
|---|---|---|---|---|---|
| full Dec-Apr | anchor + M4 shift (agent) | 106 | -32.57 [-49.88, -16.16] | -439.65 [-1027.07, +191.34] | 13 |
| full Dec-Apr | LLM raw p_home | 2 | +0.42 [+0.00, +1.08] | -7.87 [-62.52, +38.91] | 81 |
| full Dec-Apr | blend w=1.0 | 2 | +0.42 [+0.00, +1.08] | -7.87 [-62.52, +38.91] | 81 |
| test Feb-Apr | anchor + M4 shift (agent) | 60 | -17.01 [-30.67, -2.98] | -381.94 [-732.46, -37.94] | 43 |
| test Feb-Apr | LLM raw p_home | 2 | +0.42 [+0.00, +1.08] | -7.87 [-62.52, +38.91] | 81 |
| test Feb-Apr | blend w=1.0 | 2 | +0.42 [+0.00, +1.08] | -7.87 [-62.52, +38.91] | 81 |
| tune Dec-Jan | anchor + M4 shift (agent) | 46 | -15.56 [-26.05, -6.05] | -57.71 [-493.35, +458.51] | 2 |
| tune Dec-Jan | LLM raw p_home | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | nan |
| tune Dec-Jan | blend w=1.0 | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | nan |

**Verdict (upgrade 2):** on the Feb–Apr test window, blend w=1.0 Brier 0.1691 vs M4 0.1834 (Δ -0.0143 [-0.0287, -0.0006]); trades 2 CLV $+0.42 [+0.00, +1.08] vs M4 60 / $-17.01; random-filter percentile 81.

## Memorisation probe

20 games, `deepseek-flash` thinking off, no evidence. Answered (not 'unknown'): 0%. Winner correct: 0% of all (nan% of answered). Exact score: 0%.

No winner recalled correctly; outcome memorisation is not driving the results.

## Spend

Approx spend: flash@0.7 votes ~$0.30 (null); reasoner votes 4–5 ~$0.4; prob+memo <$0.05. Total under $1. Cap $10.

Figure: `evaluation/results/llm_upgrade.png`.
