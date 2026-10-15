# Approach C: a learned trade filter vs the current LLM sceptic

**Question.** Can a cheap learned filter on decision-time features (no LLM calls) pick, among proposed trades, the ones that beat the closing line, better than random and better than the current deepseek-reasoner sceptic?

**Pools** (kept separate; counterfactual $20 fill at each decision time, CLV = side's mid at tip − price paid, CLV $ = CLV × contracts, fee in P&L only):

- *Sceptic set* — every proposal the current sceptic reviewed: 182 (77 Dec 2025 – Jan 2026 with walk-forward refit models, 105 Feb – Apr 2026 test). Recomputed fills match the 8 real fills of approved proposals.
- *Anchor pool* — the deterministic anchor agent's candidate at every decision point of the same windows (its trade where it traded, else its best option) with gap after fees > 0.02: 537 (225 Dec – Jan, 312 Feb – Apr), of which 224 are arm A's real trades (95 + 129; recomputed fills match all of them) and 313 are near misses or later decision points of a game already held.

- Sceptic set, Dec 2025 - Jan 2026: n = 77, keep-all CLV $ -15.66, 23 with CLV > 0, oracle ceiling CLV $ +14.10.
- Sceptic set, Feb - Apr 2026 (test): n = 105, keep-all CLV $ -28.18, 32 with CLV > 0, oracle ceiling CLV $ +25.77.
- Anchor pool, Dec 2025 - Jan 2026: n = 225, keep-all CLV $ -63.95, 62 with CLV > 0, oracle ceiling CLV $ +39.97.
- Anchor pool, Feb - Apr 2026 (test): n = 312, keep-all CLV $ -124.54, 82 with CLV > 0, oracle ceiling CLV $ +51.27.

**Features** (as of the decision time, side-relative): `side_price`, `gap`, `model_shift`, `market_move`, `move_6h`, `news_age_hours`, `has_news`, `hours_to_tip`, `is_fav`, `home_side`, `spread`, `log_volume`, `b2b_side`, `b2b_opp`, `n_out_side`, `n_out_opp`; for the sceptic set also `llm_gap` (analyst estimate − price − fee) and `llm_minus_model` (analyst estimate − anchor model). Labels: CLV > 0 (classifiers), CLV $ (ridge).

**Leakage checks.** (1) No label-derived column is a feature (checked by name). (2) For 40 random anchor-pool decision points the features were recomputed on tables truncated at the decision time (quotes and news up to then, box scores of games already tipped, no settlements): identical. (3) LR trained on permuted labels: out-of-sample AUC 0.50 (2.5–97.5%: 0.45–0.58).

**Models.** L2 logistic regression (standardised, C = 0.1), gradient boosting (depth 2, 60 trees, learning rate 0.05, min leaf 20, L2 1.0), ridge on CLV $ (α = 30.0); hyperparameters fixed in advance, not tuned. The keep threshold τ is chosen on the training window only, from day-grouped 5-fold out-of-fold scores (max CLV $ kept, at least 5 kept on the sceptic set / 10 on the anchor pool). "Top k (matched)" keeps the k highest scores, k = the current filter's count in that test window. CIs: day-clustered bootstrap (2000 replicates, seed 7606; AUC 500); random filter: 20,000 random subsets of the same size, percentile = share with lower CLV $.

## AUC of the learned scores vs CLV > 0 (out of sample, 95% day-clustered CI)

| Pool | Model | Dec–Jan (trained Feb–Apr) | Feb–Apr (trained Dec–Jan) | Pooled LOWO |
| --- | --- | --- | --- | --- |
| sceptic | LR, anchor pool | 0.61 [0.48, 0.74] | 0.51 [0.41, 0.62] | 0.55 [0.47, 0.62] |
| sceptic | GBM, anchor pool | 0.55 [0.41, 0.70] | 0.53 [0.41, 0.64] | 0.54 [0.46, 0.63] |
| sceptic | Ridge on CLV $, anchor pool | 0.48 [0.30, 0.66] | 0.60 [0.49, 0.70] | 0.55 [0.47, 0.64] |
| sceptic | LR, sceptic set + LLM features | 0.50 [0.34, 0.63] | 0.51 [0.39, 0.64] | 0.51 [0.40, 0.60] |
| anchor | LR, anchor pool | 0.58 [0.52, 0.64] | 0.52 [0.45, 0.59] | 0.54 [0.49, 0.59] |
| anchor | GBM, anchor pool | 0.50 [0.42, 0.59] | 0.59 [0.51, 0.66] | 0.55 [0.49, 0.60] |
| anchor | Ridge on CLV $, anchor pool | 0.53 [0.45, 0.61] | 0.56 [0.48, 0.63] | 0.55 [0.48, 0.60] |
| anchor, rolling monthly | LR, anchor pool | | | 0.46 [0.40, 0.51] |
| anchor, rolling monthly | GBM, anchor pool | | | 0.50 [0.44, 0.56] |
| anchor, rolling monthly | Ridge on CLV $, anchor pool | | | 0.53 [0.47, 0.59] |

## Sceptic set (like-for-like with the current sceptic)

### train Dec 2025 - Jan 2026 -> test Feb - Apr 2026 (test)

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 105 | 100% | -28.18 [-47.33, -7.42] | -0.39 [-0.57, -0.20] | -32.52 [-49.34, -16.76] | +0.00 [+0.00, +0.00] | -28.18 [-28.18, -28.18] | 50 | -109% | -536.98 [-1084.43, +26.75] |
| Current sceptic (approve/reject) | 4 | 4% | +4.35 [-2.04, +15.55] | +0.75 [-1.50, +3.50] | – | +32.52 [+16.76, +49.34] | -1.07 [-4.68, +3.88] | 98 | 17% | +88.49 [-82.97, +371.88] |
| Oracle (CLV > 0) | 32 | 30% | +25.77 [+13.73, +40.16] | +0.98 [+0.72, +1.28] | +21.43 [+12.50, +31.50] | +53.95 [+40.15, +68.91] | -8.56 [-17.73, +1.18] | 100 | 100% | -180.65 [-478.10, +192.38] |
| Rule: skip if the market moved against the side | 26 | 25% | -2.13 [-12.94, +12.16] | -0.19 [-0.70, +0.38] | -6.48 [-14.33, +2.10] | +26.04 [+9.93, +42.83] | -7.00 [-15.40, +2.26] | 85 | -8% | -20.86 [-388.80, +406.70] |
| Rule: gap <= cut tuned on train | 18 | 17% | -4.89 [-15.51, +8.96] | -0.61 [-1.18, +0.10] | -9.24 [-16.97, -2.35] | +23.29 [+7.11, +39.59] | -4.85 [-12.12, +3.32] | 51 | -19% | +76.68 [-254.66, +488.07] |
| LR, anchor pool, tau from train | 6 | 6% | -2.86 [-6.83, +0.54] | -0.83 [-1.50, -0.17] | -7.21 [-19.81, +0.41] | +25.31 [+4.97, +44.34] | -1.59 [-5.96, +4.02] | 31 | -11% | -69.85 [-172.84, +18.22] |
| LR, anchor pool, top 4 (matched to current) | 4 | 4% | -1.68 [-5.13, +1.39] | -0.75 [-1.50, +0.50] | -6.03 [-18.52, +1.61] | +26.49 [+5.73, +45.21] | -1.07 [-4.68, +3.88] | 40 | -7% | -28.64 [-104.25, +49.44] |
| GBM, anchor pool, tau from train | 5 | 5% | -0.53 [-1.91, +0.49] | -0.10 [-0.50, +0.50] | -4.88 [-16.12, +1.89] | +27.64 [+6.80, +46.85] | -1.35 [-5.31, +3.75] | 69 | -2% | -40.25 [-123.49, +26.48] |
| GBM, anchor pool, top 4 (matched to current) | 4 | 4% | -0.71 [-2.07, +0.27] | -0.25 [-0.50, +0.50] | -5.05 [-16.29, +1.68] | +27.47 [+6.75, +46.62] | -1.07 [-4.68, +3.88] | 63 | -3% | -19.69 [-89.97, +40.00] |
| Ridge on CLV $, anchor pool, tau from train | 6 | 6% | -1.42 [-4.20, +0.75] | -0.83 [-1.90, +1.00] | -5.76 [-17.78, +1.44] | +26.76 [+5.77, +46.79] | -1.59 [-5.96, +4.02] | 57 | -6% | +22.46 [-44.33, +89.37] |
| Ridge on CLV $, anchor pool, top 4 (matched to current) | 4 | 4% | -1.24 [-3.15, +0.13] | -1.00 [-1.50, +0.50] | -5.58 [-17.22, +1.28] | +26.94 [+6.26, +46.42] | -1.07 [-4.68, +3.88] | 50 | -5% | +10.91 [-52.62, +77.74] |
| LR, sceptic set + LLM features, tau from train | 17 | 16% | -0.48 [-12.44, +13.91] | -0.12 [-0.73, +0.54] | -4.82 [-13.44, +3.73] | +27.70 [+14.67, +41.28] | -4.57 [-11.63, +3.63] | 85 | -2% | -95.75 [-333.78, +222.42] |
| LR, sceptic set + LLM features, top 4 (matched to current) | 4 | 4% | -2.45 [-7.20, +1.17] | -0.63 [-1.50, +0.50] | -6.79 [-18.96, +1.71] | +25.73 [+5.15, +44.68] | -1.07 [-4.68, +3.88] | 23 | -9% | -29.11 [-105.23, +46.86] |

### train Feb - Apr 2026 (test) -> test Dec 2025 - Jan 2026

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 77 | 100% | -15.66 [-27.58, -4.29] | -0.35 [-0.67, -0.04] | -18.70 [-30.01, -8.08] | +0.00 [+0.00, +0.00] | -15.66 [-15.66, -15.66] | 50 | -111% | -46.31 [-598.80, +581.64] |
| Current sceptic (approve/reject) | 4 | 5% | +3.04 [-0.17, +7.33] | +1.25 [-0.50, +2.50] | – | +18.70 [+8.08, +30.01] | -0.80 [-3.61, +1.92] | 100 | 22% | -49.76 [-125.13, +12.64] |
| Oracle (CLV > 0) | 23 | 30% | +14.10 [+7.93, +20.71] | +1.13 [+0.79, +1.46] | +11.06 [+5.81, +16.74] | +29.76 [+20.96, +39.35] | -4.67 [-10.31, +0.89] | 100 | 100% | +55.20 [-274.35, +453.91] |
| Rule: skip if the market moved against the side | 19 | 25% | +2.88 [-3.31, +9.01] | +0.24 [-0.32, +0.80] | -0.16 [-5.22, +4.37] | +18.54 [+8.69, +28.82] | -3.88 [-9.24, +1.46] | 99 | 20% | +59.81 [-163.10, +317.00] |
| Rule: model_shift >= cut tuned on train | 7 | 9% | -2.46 [-5.70, +0.00] | -0.71 [-1.30, +0.10] | -5.50 [-10.43, -1.44] | +13.20 [+2.46, +24.18] | -1.41 [-4.98, +2.14] | 28 | -17% | -43.76 [-132.48, +49.77] |
| LR, anchor pool, tau from train | 1 | 1% | -0.14 [-0.42, +0.00] | -0.50 [-0.50, -0.50] | -3.18 [-7.47, +0.00] | +15.52 [+4.25, +27.30] | -0.19 [-1.80, +1.38] | 62 | -1% | +7.98 [+0.00, +23.94] |
| LR, anchor pool, top 4 (matched to current) | 4 | 5% | -0.64 [-3.22, +1.08] | -0.38 [-1.50, +1.00] | -3.68 [-8.48, +0.24] | +15.02 [+4.37, +25.78] | -0.80 [-3.61, +1.92] | 54 | -5% | -28.09 [-80.84, +9.44] |
| GBM, anchor pool, tau from train | 16 | 21% | -2.42 [-9.32, +3.93] | -0.12 [-0.95, +0.72] | -5.45 [-12.58, +0.66] | +13.24 [+3.06, +22.90] | -3.24 [-8.29, +1.73] | 63 | -17% | -106.28 [-251.78, +44.33] |
| GBM, anchor pool, top 4 (matched to current) | 4 | 5% | -1.12 [-4.03, +1.45] | -0.50 [-2.50, +2.50] | -4.15 [-9.12, +0.16] | +14.54 [+3.26, +26.42] | -0.80 [-3.61, +1.92] | 41 | -8% | -24.03 [-94.17, +27.44] |
| Ridge on CLV $, anchor pool, tau from train | 4 | 5% | -0.59 [-2.13, +0.38] | +0.00 [-0.17, +0.50] | -3.62 [-8.19, -0.24] | +15.08 [+3.73, +27.14] | -0.80 [-3.61, +1.92] | 56 | -4% | -0.16 [-14.64, +14.16] |
| Ridge on CLV $, anchor pool, top 4 (matched to current) | 4 | 5% | -0.59 [-2.13, +0.38] | +0.00 [-0.17, +0.50] | -3.62 [-8.19, -0.24] | +15.08 [+3.73, +27.14] | -0.80 [-3.61, +1.92] | 56 | -4% | -0.16 [-14.64, +14.16] |
| LR, sceptic set + LLM features, tau from train | 6 | 8% | -0.91 [-2.84, +0.25] | -0.17 [-0.50, +0.50] | -3.95 [-8.67, -0.40] | +14.75 [+3.50, +26.72] | -1.22 [-4.60, +2.04] | 57 | -6% | -12.44 [-70.70, +26.86] |
| LR, sceptic set + LLM features, top 4 (matched to current) | 4 | 5% | -0.90 [-2.69, +0.00] | -0.25 [-0.25, -0.25] | -3.93 [-8.52, -0.47] | +14.77 [+3.36, +26.98] | -0.80 [-3.61, +1.92] | 47 | -6% | -25.14 [-75.42, +0.00] |

### pooled leave-one-window-out

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 182 | 100% | -43.84 [-66.84, -21.49] | -0.37 [-0.54, -0.21] | -51.22 [-71.34, -31.99] | +0.00 [+0.00, +0.00] | -43.84 [-43.84, -43.84] | 50 | -110% | -583.29 [-1343.78, +285.42] |
| Current sceptic (approve/reject) | 8 | 4% | +7.38 [-0.65, +19.69] | +1.00 [-0.21, +2.17] | – | +51.22 [+31.99, +71.34] | -1.93 [-6.57, +3.62] | 100 | 19% | +38.73 [-153.80, +351.27] |
| Oracle (CLV > 0) | 55 | 30% | +39.87 [+26.39, +54.51] | +1.05 [+0.83, +1.26] | +32.49 [+21.81, +44.29] | +83.71 [+67.76, +100.52] | -13.30 [-24.12, -2.18] | 100 | 100% | -125.45 [-570.91, +408.60] |
| Rule: skip if the market moved against the side | 45 | 25% | +0.74 [-12.33, +15.43] | -0.01 [-0.38, +0.37] | -6.64 [-16.29, +2.61] | +44.59 [+26.35, +64.34] | -10.85 [-20.77, -0.35] | 99 | 2% | +38.95 [-409.79, +529.92] |
| LR, anchor pool, tau from the other window | 7 | 4% | -3.00 [-7.05, +0.34] | -0.79 [-1.33, -0.21] | -10.38 [-22.97, -1.32] | +40.84 [+18.33, +63.74] | -1.71 [-6.08, +3.61] | 29 | -8% | -61.87 [-165.64, +34.06] |
| LR, anchor pool, top 8 (matched to current) | 8 | 4% | -3.66 [-8.35, +0.29] | -0.87 [-1.50, +0.00] | -11.04 [-23.60, -1.78] | +40.18 [+17.71, +63.00] | -1.93 [-6.57, +3.62] | 24 | -9% | -57.30 [-149.98, +28.61] |
| GBM, anchor pool, tau from the other window | 21 | 12% | -2.95 [-10.10, +3.16] | -0.12 [-0.75, +0.54] | -10.33 [-24.93, -0.37] | +40.89 [+19.33, +63.31] | -5.02 [-12.28, +3.06] | 71 | -7% | -146.53 [-314.78, +24.12] |
| GBM, anchor pool, top 8 (matched to current) | 8 | 4% | -0.72 [-3.19, +1.63] | -0.12 [-1.08, +0.90] | -8.10 [-20.56, +0.29] | +43.12 [+20.65, +66.22] | -1.93 [-6.57, +3.62] | 71 | -2% | -43.49 [-143.58, +38.20] |
| Ridge on CLV $, anchor pool, tau from the other window | 10 | 5% | -2.00 [-5.30, +0.46] | -0.50 [-1.42, +0.50] | -9.38 [-21.46, -0.69] | +41.84 [+19.02, +65.97] | -2.43 [-7.66, +3.64] | 58 | -5% | +22.30 [-42.03, +89.37] |
| Ridge on CLV $, anchor pool, top 8 (matched to current) | 8 | 4% | -1.82 [-4.19, +0.02] | -0.50 [-1.50, +0.10] | -9.20 [-21.32, -0.90] | +42.02 [+19.51, +65.77] | -1.93 [-6.57, +3.62] | 54 | -5% | +10.75 [-51.69, +76.91] |
| LR, sceptic set + LLM features, tau from the other window | 23 | 13% | -1.39 [-12.94, +12.43] | -0.13 [-0.58, +0.37] | -8.77 [-18.44, +0.96] | +42.46 [+25.01, +60.49] | -5.52 [-13.27, +2.98] | 84 | -3% | -108.19 [-356.73, +219.56] |
| LR, sceptic set + LLM features, top 8 (matched to current) | 8 | 4% | -2.37 [-7.44, +1.17] | -0.31 [-0.75, +0.17] | -9.75 [-22.46, -0.31] | +41.47 [+18.96, +64.16] | -1.93 [-6.57, +3.62] | 44 | -6% | -25.26 [-110.95, +57.53] |

## Anchor pool

### train Dec 2025 - Jan 2026 -> test Feb - Apr 2026 (test)

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 312 | 100% | -124.54 [-173.83, -79.76] | -0.54 [-0.76, -0.33] | -89.29 [-126.88, -55.73] | +0.00 [+0.00, +0.00] | -124.54 [-124.54, -124.54] | 50 | -243% | -1391.49 [-2212.66, -579.92] |
| Arm A's own trades (gap > 0.04) | 129 | 41% | -35.25 [-58.87, -11.52] | -0.35 [-0.65, +0.01] | – | +89.29 [+55.73, +126.88] | -51.51 [-68.51, -34.77] | 97 | -69% | -247.27 [-743.13, +264.94] |
| Oracle (CLV > 0) | 82 | 26% | +51.27 [+33.48, +73.30] | +1.13 [+0.82, +1.50] | +86.52 [+63.49, +111.76] | +175.81 [+134.41, +222.06] | -32.66 [-47.99, -17.57] | 100 | 100% | -131.44 [-467.59, +229.77] |
| Rule: skip if the market moved against the side | 35 | 11% | -1.77 [-9.23, +5.02] | -0.16 [-0.45, +0.23] | +33.47 [+7.77, +59.05] | +122.77 [+77.10, +174.15] | -13.94 [-25.10, -3.29] | 99 | -3% | -204.76 [-393.22, -30.05] |
| Rule: market_move >= cut tuned on train | 25 | 8% | +2.45 [-1.90, +7.24] | +0.04 [-0.33, +0.50] | +37.70 [+12.37, +62.73] | +126.99 [+80.71, +177.88] | -10.02 [-19.64, -1.01] | 100 | 5% | -188.39 [-334.82, -50.70] |
| LR, anchor pool, tau from train | 4 | 1% | -0.23 [-1.99, +1.20] | +0.00 [-1.50, +1.50] | +35.01 [+11.40, +58.73] | +124.30 [+79.99, +172.96] | -1.61 [-5.90, +2.10] | 78 | -0% | +39.30 [+2.68, +97.99] |
| LR, anchor pool, top 129 (matched to current) | 129 | 41% | -53.84 [-88.60, -22.22] | -0.68 [-1.14, -0.24] | -18.59 [-46.01, +2.53] | +70.70 [+43.89, +96.90] | -51.51 [-68.51, -34.77] | 40 | -105% | -565.59 [-1038.18, -107.30] |
| GBM, anchor pool, tau from train | 5 | 2% | -0.53 [-5.65, +4.60] | -0.40 [-3.75, +3.23] | +34.71 [+12.43, +57.00] | +124.00 [+80.87, +170.20] | -1.98 [-6.60, +2.20] | 76 | -1% | +58.15 [+12.64, +112.26] |
| GBM, anchor pool, top 129 (matched to current) | 129 | 41% | -41.74 [-74.30, -13.53] | -0.52 [-0.91, -0.12] | -6.50 [-39.49, +22.54] | +82.80 [+54.28, +112.20] | -51.51 [-68.51, -34.77] | 87 | -81% | -393.81 [-996.52, +264.34] |
| Ridge on CLV $, anchor pool, tau from train | 23 | 7% | -1.92 [-3.76, -0.45] | -0.37 [-0.70, -0.10] | +33.32 [+9.85, +57.22] | +122.61 [+78.11, +171.26] | -9.20 [-18.53, -0.44] | 95 | -4% | +83.62 [+43.20, +130.92] |
| Ridge on CLV $, anchor pool, top 129 (matched to current) | 129 | 41% | -24.62 [-49.60, -2.72] | -0.53 [-1.06, -0.08] | +10.63 [-17.17, +33.10] | +99.92 [+64.77, +135.07] | -51.51 [-68.51, -34.77] | 100 | -48% | +226.88 [-205.12, +624.39] |

### train Feb - Apr 2026 (test) -> test Dec 2025 - Jan 2026

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 225 | 100% | -63.95 [-90.02, -37.97] | -0.48 [-0.69, -0.27] | -37.92 [-56.53, -19.66] | +0.00 [+0.00, +0.00] | -63.95 [-63.95, -63.95] | 50 | -160% | -9.76 [-1057.32, +1031.78] |
| Arm A's own trades (gap > 0.04) | 95 | 42% | -26.03 [-40.36, -10.61] | -0.49 [-0.85, -0.15] | – | +37.92 [+19.66, +56.53] | -27.01 [-39.22, -14.85] | 56 | -65% | -84.82 [-653.87, +537.18] |
| Oracle (CLV > 0) | 62 | 28% | +39.97 [+26.71, +54.74] | +1.24 [+0.91, +1.63] | +65.99 [+52.20, +81.22] | +103.92 [+83.22, +126.08] | -17.63 [-28.60, -6.66] | 100 | 100% | +494.15 [-183.26, +1281.84] |
| Rule: skip if the market moved against the side | 15 | 7% | +0.60 [-2.36, +4.47] | -0.03 [-0.50, +0.54] | +26.63 [+10.83, +41.12] | +64.55 [+38.67, +90.29] | -4.31 [-10.40, +1.92] | 94 | 2% | +76.53 [-56.01, +220.10] |
| Rule: market_move >= cut tuned on train | 11 | 5% | +1.23 [-1.28, +4.78] | +0.23 [-0.33, +0.87] | +27.26 [+11.80, +41.75] | +65.19 [+39.41, +91.00] | -3.14 [-8.41, +2.32] | 95 | 3% | +101.33 [-10.98, +230.39] |
| LR, anchor pool, tau from train | 3 | 1% | +0.31 [-0.34, +1.40] | +0.33 [-0.50, +1.50] | +26.33 [+11.03, +40.67] | +64.26 [+38.41, +90.17] | -0.85 [-3.76, +2.05] | 81 | 1% | +2.03 [-50.12, +45.64] |
| LR, anchor pool, top 95 (matched to current) | 95 | 42% | -16.37 [-32.67, +0.84] | -0.31 [-0.65, +0.05] | +9.66 [-3.78, +24.40] | +47.58 [+27.88, +66.47] | -27.01 [-39.22, -14.85] | 96 | -41% | -66.32 [-640.38, +548.96] |
| GBM, anchor pool, tau from train | 5 | 2% | -0.54 [-1.43, +0.24] | -0.20 [-0.50, +0.25] | +25.49 [+9.80, +40.06] | +63.41 [+37.27, +89.53] | -1.43 [-5.15, +2.35] | 70 | -1% | +71.53 [-20.26, +180.72] |
| GBM, anchor pool, top 95 (matched to current) | 95 | 42% | -24.51 [-39.65, -9.18] | -0.51 [-0.84, -0.13] | +1.51 [-16.79, +18.74] | +39.44 [+17.25, +61.55] | -27.01 [-39.22, -14.85] | 66 | -61% | -432.03 [-976.44, +108.02] |
| Ridge on CLV $, anchor pool, tau from train | 1 | 0% | -0.16 [-0.48, +0.00] | -0.50 [-0.50, -0.50] | +25.87 [+10.46, +40.20] | +63.79 [+37.97, +89.71] | -0.30 [-2.20, +1.67] | 60 | -0% | +11.63 [+0.00, +34.89] |
| Ridge on CLV $, anchor pool, top 95 (matched to current) | 95 | 42% | -9.81 [-25.67, +7.01] | -0.24 [-0.60, +0.16] | +16.22 [+3.05, +28.90] | +54.14 [+32.79, +75.90] | -27.01 [-39.22, -14.85] | 100 | -25% | -296.94 [-671.47, +101.03] |

### pooled leave-one-window-out

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 537 | 100% | -188.49 [-243.11, -133.35] | -0.52 [-0.67, -0.37] | -127.22 [-169.38, -86.27] | +0.00 [+0.00, +0.00] | -188.49 [-188.49, -188.49] | 50 | -207% | -1401.25 [-2720.70, -48.87] |
| Arm A's own trades (gap > 0.04) | 224 | 42% | -61.28 [-90.71, -32.72] | -0.41 [-0.64, -0.18] | – | +127.22 [+86.27, +169.38] | -78.61 [-99.68, -57.78] | 95 | -67% | -332.09 [-1127.44, +440.27] |
| Oracle (CLV > 0) | 144 | 27% | +91.24 [+66.98, +117.00] | +1.18 [+0.92, +1.45] | +152.51 [+126.25, +184.36] | +279.73 [+230.79, +330.83] | -50.56 [-69.60, -31.98] | 100 | 100% | +362.71 [-400.34, +1209.75] |
| Rule: skip if the market moved against the side | 50 | 9% | -1.17 [-9.02, +6.32] | -0.12 [-0.37, +0.18] | +60.10 [+31.11, +90.22] | +187.32 [+131.68, +243.85] | -17.55 [-29.92, -5.50] | 100 | -1% | -128.23 [-359.68, +107.44] |
| LR, anchor pool, tau from the other window | 7 | 1% | +0.07 [-1.84, +1.91] | +0.14 [-0.65, +1.00] | +61.35 [+33.14, +90.47] | +188.56 [+134.34, +243.26] | -2.46 [-7.56, +2.25] | 86 | 0% | +41.33 [-26.68, +118.21] |
| LR, anchor pool, top 224 (matched to current) | 224 | 42% | -70.86 [-113.61, -33.76] | -0.51 [-0.81, -0.23] | -9.58 [-43.50, +20.03] | +117.63 [+86.36, +147.66] | -78.61 [-99.68, -57.78] | 77 | -78% | -658.92 [-1364.44, +54.80] |
| GBM, anchor pool, tau from the other window | 10 | 2% | -1.07 [-6.07, +3.87] | -0.30 [-1.94, +1.36] | +60.20 [+32.23, +89.15] | +187.42 [+133.54, +240.76] | -3.49 [-9.46, +2.12] | 80 | -1% | +129.68 [+28.29, +254.21] |
| GBM, anchor pool, top 224 (matched to current) | 224 | 42% | -74.09 [-113.53, -37.71] | -0.51 [-0.77, -0.23] | -12.81 [-52.82, +25.29] | +114.40 [+78.08, +149.27] | -78.61 [-99.68, -57.78] | 66 | -81% | -979.49 [-1791.89, -138.79] |
| Ridge on CLV $, anchor pool, tau from the other window | 24 | 4% | -2.08 [-3.88, -0.56] | -0.37 [-0.70, -0.11] | +59.19 [+30.90, +87.88] | +186.41 [+132.12, +241.27] | -8.42 [-17.50, +0.10] | 93 | -2% | +95.25 [+48.22, +149.04] |
| Ridge on CLV $, anchor pool, top 224 (matched to current) | 224 | 42% | -39.82 [-67.62, -15.20] | -0.45 [-0.75, -0.17] | +21.45 [-6.06, +48.00] | +148.67 [+105.14, +191.60] | -78.61 [-99.68, -57.78] | 100 | -44% | -62.61 [-611.82, +506.79] |

### rolling monthly (2026-01, 2026-02, 2026-03, 2026-04)

| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] | Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share | P&L [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep all | 444 | 100% | -169.99 [-221.32, -119.89] | -0.56 [-0.72, -0.38] | -116.81 [-156.54, -80.23] | +0.00 [+0.00, +0.00] | -169.99 [-169.99, -169.99] | 50 | -237% | -1970.16 [-3036.76, -871.50] |
| Arm A's own trades (gap > 0.04) | 189 | 43% | -53.18 [-80.35, -26.05] | -0.40 [-0.65, -0.13] | – | +116.81 [+80.23, +156.54] | -72.30 [-91.59, -52.90] | 97 | -74% | -433.76 [-1051.56, +241.55] |
| Oracle (CLV > 0) | 117 | 26% | +71.85 [+50.30, +94.80] | +1.18 [+0.92, +1.48] | +125.04 [+97.15, +156.02] | +241.85 [+196.38, +290.67] | -44.81 [-62.16, -27.97] | 100 | 100% | -123.66 [-621.35, +432.12] |
| Rule: skip if the market moved against the side | 44 | 10% | -1.52 [-9.24, +5.03] | -0.11 [-0.39, +0.20] | +51.66 [+23.26, +79.89] | +168.47 [+117.66, +220.43] | -16.80 [-28.68, -5.32] | 100 | -2% | -157.82 [-375.33, +46.21] |
| LR, anchor pool, tau from earlier months | 28 | 6% | -12.38 [-22.08, -4.16] | -0.86 [-1.45, -0.37] | +40.80 [+12.02, +69.54] | +157.61 [+108.54, +209.91] | -10.74 [-20.51, -1.36] | 36 | -17% | +32.40 [-219.24, +289.05] |
| GBM, anchor pool, tau from earlier months | 23 | 5% | -11.63 [-20.75, -2.96] | -0.78 [-1.60, +0.04] | +41.55 [+16.91, +67.13] | +158.36 [+112.65, +207.66] | -8.84 [-17.86, -0.39] | 26 | -16% | -170.97 [-352.46, +8.35] |
| Ridge on CLV $, anchor pool, tau from earlier months | 42 | 9% | -10.94 [-19.94, -4.09] | -0.65 [-1.12, -0.30] | +42.24 [+14.15, +70.68] | +159.05 [+109.94, +212.11] | -16.08 [-27.59, -4.91] | 81 | -15% | -31.06 [-199.33, +124.15] |

## Verdict

- **Ranking.** Out of sample the learned scores barely rank proposals: pooled AUC 0.51–0.55, and no pooled CI excludes 0.5. For comparison, LR on permuted labels gives 0.50 (0.45–0.58).
- **Sceptic set (like-for-like).** The current sceptic keeps 8 of 182 for CLV $ +7.38 (random percentile 100). The best learned filter ("GBM, anchor pool, top 8 (matched to current)") keeps 8 for -0.72 (percentile 71), -8.10 [-20.56, +0.29] vs the current sceptic. 0 of 8 learned variants beat the current sceptic with a CI above zero; 6 are worse with a CI below zero. All of them beat keep-all, as any small filter does when keep-all loses.
- **One-feature rule.** "Skip if the market moved against the side" keeps 45 (25%) of the sceptic set for +0.74 (percentile 99, mean -0.01¢); on the anchor pool 50 (9%) for -1.17 (percentile 100) vs arm A's own -61.28. It removes losers about as well as anything learned, but the trades it keeps only break even on CLV; ML adds nothing over it.
- **Anchor pool, tuned τ.** With τ from the other window the learned filters keep 7–24 of 537 (CLV $ -2.08 to +0.07, random percentiles 80–93): the tuned threshold mostly learns "trade almost nothing", which beats arm A only because arm A's trades lose CLV on average.
- **Anchor pool, matched to arm A's keep rate** (top k by score, k = arm A's trades): LR, anchor pool -70.86 (percentile 77, -9.58 [-43.50, +20.03] vs arm A); GBM, anchor pool -74.09 (percentile 66, -12.81 [-52.82, +25.29] vs arm A); Ridge on CLV $, anchor pool -39.82 (percentile 100, +21.45 [-6.06, +48.00] vs arm A); arm A itself -61.28. Ridge on CLV $, anchor pool ranks the anchor pool better than random at this keep rate, i.e. it drops the worst losers, but the kept set still loses CLV.
- **Bottom line.** With n = 182 (sceptic set) / 537 (anchor pool) and these features, a learned filter does not beat random selection at the sceptic's keep rate and does not beat the current sceptic out of sample; on the larger anchor pool it can drop the worst losers at arm A's 42% keep rate, but the kept set does not reach positive CLV. The current sceptic's CLV $ +7.38 on 8 trades remains the best point estimate, but its own CI includes zero; nothing here is evidence of a reliable positive-CLV selector.

## Feature weights

Standardised LR coefficients (positive = more likely CLV > 0) on the anchor pool per window and pooled, GBM out-of-sample permutation importance (mean AUC drop, both directions), and the LR on the pooled sceptic set with LLM features.

| Feature | LR Dec–Jan | LR Feb–Apr | LR both | GBM perm. ΔAUC | LR sceptic set |
| --- | --- | --- | --- | --- | --- |
| side_price | +0.078 | +0.027 | +0.082 | +0.023 | +0.100 |
| gap | -0.025 | +0.016 | -0.013 | +0.006 | -0.003 |
| model_shift | -0.061 | +0.402 | +0.276 | -0.001 | +0.316 |
| market_move | -0.007 | +0.120 | +0.007 | -0.005 | +0.100 |
| move_6h | -0.071 | -0.059 | -0.025 | -0.003 | +0.135 |
| news_age_hours | +0.091 | +0.069 | +0.096 | -0.002 | +0.022 |
| has_news | -0.091 | -0.069 | -0.096 | -0.001 | -0.022 |
| hours_to_tip | +0.091 | +0.069 | +0.096 | +0.000 | +0.022 |
| is_fav | +0.028 | +0.068 | +0.032 | +0.000 | -0.038 |
| home_side | -0.158 | +0.016 | -0.054 | -0.001 | -0.031 |
| spread | -0.172 | -0.274 | -0.247 | +0.009 | -0.097 |
| log_volume | -0.064 | -0.001 | -0.026 | +0.014 | -0.247 |
| b2b_side | +0.022 | +0.150 | +0.091 | +0.000 | +0.357 |
| b2b_opp | -0.173 | -0.063 | -0.133 | +0.000 | -0.385 |
| n_out_side | -0.074 | -0.042 | -0.038 | +0.009 | -0.120 |
| n_out_opp | +0.144 | -0.053 | +0.051 | +0.000 | +0.020 |
| llm_gap | – | – | – | – | -0.201 |
| llm_minus_model | – | – | – | – | -0.065 |
