# Walk-forward evaluation, 2025-26 season

M4 retrained before each window on every earlier game; one replay per setup over the season (the learning agent starts with an empty notebook on 1 Nov and carries it forward). Stake $20 per order, offline rules. 95% CIs and one-sided p-values from a day-clustered bootstrap (2000 replicates, seed 7606). Never trade is $0 by construction.

## Season totals

| Window | Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | ROI [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Season, Nov–12 Apr | Full agent: anchor + learning | 157 | -0.0050 [-0.0079, -0.0018] | -44 [-67, -21] | -25 [-569, +564] | -0.8% [-18.0%, +19.2%] | 1.000 |
| Season, Nov–12 Apr | Agent: anchor, no learning | 276 | -0.0048 [-0.0068, -0.0027] | -84 [-114, -54] | -407 [-1,324, +553] | -7.5% [-24.2%, +10.3%] | 1.000 |
| Season, Nov–12 Apr | Raw model, no agent | 711 | -0.0046 [-0.0057, -0.0035] | -255 [-326, -188] | -2,475 [-3,995, -845] | -17.6% [-27.9%, -6.1%] | 1.000 |
| Season, Nov–12 Apr | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Season incl. play-offs | Full agent: anchor + learning | 157 | -0.0050 [-0.0078, -0.0019] | -44 [-67, -22] | -25 [-576, +574] | -0.8% [-18.3%, +19.5%] | 1.000 |
| Season incl. play-offs | Agent: anchor, no learning | 282 | -0.0046 [-0.0066, -0.0024] | -84 [-116, -50] | -481 [-1,336, +439] | -8.6% [-24.2%, +8.1%] | 1.000 |
| Season incl. play-offs | Raw model, no agent | 772 | -0.0046 [-0.0057, -0.0036] | -279 [-349, -209] | -2,041 [-3,723, -296] | -13.4% [-23.6%, -2.0%] | 1.000 |
| Season incl. play-offs | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |

## By month

| Window | Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | ROI [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nov | Full agent: anchor + learning | 38 | -0.0091 [-0.0147, -0.0053] | -18 [-26, -10] | -37 [-433, +398] | -4.9% [-50.3%, +66.6%] | 0.999 |
| Nov | Agent: anchor, no learning | 44 | -0.0090 [-0.0138, -0.0057] | -22 [-30, -14] | -32 [-456, +415] | -3.7% [-46.6%, +56.0%] | 1.000 |
| Nov | Raw model, no agent | 127 | -0.0044 [-0.0068, -0.0023] | -26 [-42, -7] | -134 [-607, +359] | -5.4% [-24.2%, +14.3%] | 1.000 |
| Nov | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Dec | Full agent: anchor + learning | 6 | -0.0033 [-0.0075, +0.0013] | -1 [-2, +1] | +110 [-12, +270] | +93.0% [-19.2%, +207.4%] | 0.949 |
| Dec | Agent: anchor, no learning | 35 | -0.0049 [-0.0100, +0.0008] | -8 [-17, +2] | +102 [-271, +541] | +14.7% [-43.7%, +73.2%] | 0.967 |
| Dec | Raw model, no agent | 87 | -0.0037 [-0.0062, -0.0011] | -17 [-31, -4] | -51 [-634, +495] | -2.9% [-35.3%, +30.3%] | 0.996 |
| Dec | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Jan | Full agent: anchor + learning | 36 | -0.0056 [-0.0124, +0.0006] | -13 [-25, -2] | -52 [-272, +193] | -7.3% [-41.4%, +27.6%] | 0.945 |
| Jan | Agent: anchor, no learning | 60 | -0.0050 [-0.0100, -0.0007] | -18 [-31, -6] | -186 [-563, +253] | -15.7% [-48.3%, +21.3%] | 0.972 |
| Jan | Raw model, no agent | 131 | -0.0048 [-0.0081, -0.0018] | -40 [-62, -17] | -131 [-956, +848] | -5.0% [-35.7%, +34.2%] | 0.999 |
| Jan | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Feb | Full agent: anchor + learning | 44 | -0.0010 [-0.0079, +0.0079] | -5 [-21, +11] | -127 [-375, +130] | -14.6% [-45.0%, +14.6%] | 0.563 |
| Feb | Agent: anchor, no learning | 52 | -0.0014 [-0.0074, +0.0057] | -8 [-24, +9] | -140 [-425, +208] | -13.7% [-45.8%, +19.2%] | 0.630 |
| Feb | Raw model, no agent | 124 | -0.0049 [-0.0083, -0.0018] | -30 [-55, -6] | -661 [-1,169, -192] | -26.9% [-48.3%, -7.6%] | 0.998 |
| Feb | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Mar | Full agent: anchor + learning | 24 | -0.0042 [-0.0075, -0.0012] | -4 [-7, -1] | +104 [-25, +233] | +22.0% [-5.5%, +47.8%] | 0.990 |
| Mar | Agent: anchor, no learning | 50 | -0.0030 [-0.0066, +0.0005] | -12 [-24, +1] | -125 [-421, +157] | -12.7% [-42.4%, +15.8%] | 0.949 |
| Mar | Raw model, no agent | 159 | -0.0038 [-0.0059, -0.0019] | -70 [-93, -48] | -926 [-1,699, -129] | -29.5% [-51.9%, -4.1%] | 1.000 |
| Mar | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Apr 1-12 | Full agent: anchor + learning | 9 | -0.0083 [-0.0125, +0.0050] | -4 [-8, +0] | -23 [-103, +58] | -12.9% [-55.7%, +58.6%] | 1.000 |
| Apr 1-12 | Agent: anchor, no learning | 35 | -0.0064 [-0.0109, -0.0025] | -16 [-30, -5] | -25 [-258, +216] | -3.6% [-42.5%, +26.3%] | 0.997 |
| Apr 1-12 | Raw model, no agent | 83 | -0.0063 [-0.0091, -0.0039] | -72 [-104, -40] | -573 [-909, -223] | -34.8% [-61.8%, -12.5%] | 1.000 |
| Apr 1-12 | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Play-offs | Full agent: anchor + learning | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Play-offs | Agent: anchor, no learning | 6 | +0.0050 [-0.0050, +0.0200] | +0 [-3, +4] | -74 [-176, +26] | -62.0% [-105.1%, +22.9%] | 0.203 |
| Play-offs | Raw model, no agent | 61 | -0.0053 [-0.0084, -0.0028] | -24 [-39, -12] | +434 [-79, +1,151] | +35.9% [-6.7%, +91.2%] | 0.999 |
| Play-offs | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |

## Training games per window

| Window start | Training games |
| --- | --- |
| 2025-10-21 | 2640 |
| 2025-11-01 | 2720 |
| 2025-12-01 | 2939 |
| 2026-01-01 | 3137 |
| 2026-02-01 | 3370 |
| 2026-03-01 | 3536 |
| 2026-04-01 | 3775 |
| 2026-04-13 | 3871 |
