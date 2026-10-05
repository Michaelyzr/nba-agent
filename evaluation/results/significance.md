# Significance tests

Day-clustered bootstrap, 2000 replicates, seed 7606. One-sided p-values: for single setups H1 is "mean CLV > 0"; for comparisons H1 is "first setup > second setup". A comparison against never trade is the setup's own total (never trade is $0).

CLV is measured against the mid at tip, but orders fill at the ask, so a trader with no edge over the closing price shows CLV of about minus half the spread (the median half-spread in the last six hours before tip is 0.005).

## Is mean CLV above zero?

| Period | Setup | Trades | Mean CLV [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- |
| Season, Nov–12 Apr | Full agent: anchor + learning | 157 | -0.0050 [-0.0079, -0.0018] | 1.000 |
| Season, Nov–12 Apr | Agent: anchor, no learning | 276 | -0.0048 [-0.0068, -0.0027] | 1.000 |
| Season, Nov–12 Apr | Raw model, no agent | 711 | -0.0046 [-0.0057, -0.0035] | 1.000 |
| Season incl. play-offs | Full agent: anchor + learning | 157 | -0.0050 [-0.0078, -0.0019] | 1.000 |
| Season incl. play-offs | Agent: anchor, no learning | 282 | -0.0046 [-0.0066, -0.0024] | 1.000 |
| Season incl. play-offs | Raw model, no agent | 772 | -0.0046 [-0.0057, -0.0036] | 1.000 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning | 0 | – | – |
| Holdout 13 Apr–14 Jun | Agent: anchor, no learning | 6 | +0.0050 [-0.0050, +0.0200] | 0.203 |
| Holdout 13 Apr–14 Jun | Raw model, no agent | 59 | -0.0052 [-0.0083, -0.0026] | 0.999 |
| Original test 1 Feb–12 Apr | Full agent: anchor + learning | 74 | -0.0022 [-0.0067, +0.0031] | 0.803 |
| Original test 1 Feb–12 Apr | Agent: anchor, no learning | 129 | -0.0035 [-0.0065, +0.0000] | 0.990 |
| Original test 1 Feb–12 Apr | Raw model, no agent | 366 | -0.0047 [-0.0063, -0.0031] | 1.000 |

## Paired comparisons on the same game-days

| Period | Comparison | Metric | Difference [95% CI] | p (one-sided) |
| --- | --- | --- | --- | --- |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Never trade | CLV $ | -44 [-67, -21] | 1.000 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Never trade | P&L | -25 [-569, +564] | 0.532 |
| Walk-forward, Season, Nov–12 Apr | Agent: anchor, no learning − Never trade | CLV $ | -84 [-114, -54] | 1.000 |
| Walk-forward, Season, Nov–12 Apr | Agent: anchor, no learning − Never trade | P&L | -407 [-1,324, +553] | 0.809 |
| Walk-forward, Season, Nov–12 Apr | Raw model, no agent − Never trade | CLV $ | -255 [-326, -188] | 1.000 |
| Walk-forward, Season, Nov–12 Apr | Raw model, no agent − Never trade | P&L | -2,475 [-3,995, -845] | 0.999 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Agent: anchor, no learning | mean CLV | -0.0002 [-0.0019, +0.0015] | 0.592 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Agent: anchor, no learning | CLV $ | +39 [+18, +61] | 0.001 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Agent: anchor, no learning | P&L | +382 [-304, +1,047] | 0.127 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Raw model, no agent | mean CLV | -0.0004 [-0.0031, +0.0026] | 0.595 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Raw model, no agent | CLV $ | +210 [+148, +280] | 0.000 |
| Walk-forward, Season, Nov–12 Apr | Full agent: anchor + learning − Raw model, no agent | P&L | +2,450 [+854, +3,984] | 0.002 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Never trade | CLV $ | -44 [-67, -22] | 1.000 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Never trade | P&L | -25 [-576, +574] | 0.522 |
| Walk-forward, Season incl. play-offs | Agent: anchor, no learning − Never trade | CLV $ | -84 [-116, -50] | 1.000 |
| Walk-forward, Season incl. play-offs | Agent: anchor, no learning − Never trade | P&L | -481 [-1,336, +439] | 0.857 |
| Walk-forward, Season incl. play-offs | Raw model, no agent − Never trade | CLV $ | -279 [-349, -209] | 1.000 |
| Walk-forward, Season incl. play-offs | Raw model, no agent − Never trade | P&L | -2,041 [-3,723, -296] | 0.994 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Agent: anchor, no learning | mean CLV | -0.0004 [-0.0022, +0.0014] | 0.676 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Agent: anchor, no learning | CLV $ | +39 [+17, +63] | 0.002 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Agent: anchor, no learning | P&L | +456 [-225, +1,111] | 0.083 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Raw model, no agent | mean CLV | -0.0004 [-0.0031, +0.0024] | 0.572 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Raw model, no agent | CLV $ | +235 [+167, +302] | 0.000 |
| Walk-forward, Season incl. play-offs | Full agent: anchor + learning − Raw model, no agent | P&L | +2,016 [+300, +3,720] | 0.007 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Never trade | CLV $ | +0 [+0, +0] | – |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Never trade | P&L | +0 [+0, +0] | – |
| Holdout 13 Apr–14 Jun | Agent: anchor, no learning − Never trade | CLV $ | +0 [-3, +4] | 0.488 |
| Holdout 13 Apr–14 Jun | Agent: anchor, no learning − Never trade | P&L | -74 [-176, +26] | 0.942 |
| Holdout 13 Apr–14 Jun | Raw model, no agent − Never trade | CLV $ | -24 [-38, -12] | 0.999 |
| Holdout 13 Apr–14 Jun | Raw model, no agent − Never trade | P&L | +450 [-55, +1,171] | 0.087 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Agent: anchor, no learning | mean CLV | -0.0050 [-0.0200, +0.0050] | 0.798 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Agent: anchor, no learning | CLV $ | -0 [-4, +3] | 0.512 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Agent: anchor, no learning | P&L | +74 [-26, +176] | 0.059 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Raw model, no agent | mean CLV | +0.0052 [+0.0026, +0.0083] | 0.002 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Raw model, no agent | CLV $ | +24 [+12, +38] | 0.001 |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning − Raw model, no agent | P&L | -450 [-1,171, +55] | 0.914 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Never trade | CLV $ | -10 [-27, +8] | 0.873 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Never trade | P&L | -32 [-340, +281] | 0.571 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Agent: anchor, no learning − Never trade | CLV $ | -35 [-60, -12] | 0.997 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Agent: anchor, no learning − Never trade | P&L | -247 [-718, +265] | 0.831 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Raw model, no agent − Never trade | CLV $ | -172 [-231, -118] | 1.000 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Raw model, no agent − Never trade | P&L | -2,087 [-3,083, -1,045] | 1.000 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Agent: anchor, no learning | mean CLV | +0.0013 [-0.0010, +0.0040] | 0.167 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Agent: anchor, no learning | CLV $ | +25 [+9, +44] | 0.006 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Agent: anchor, no learning | P&L | +215 [-167, +584] | 0.129 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Raw model, no agent | mean CLV | +0.0025 [-0.0017, +0.0075] | 0.150 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Raw model, no agent | CLV $ | +162 [+111, +217] | 0.000 |
| Original test 1 Feb–12 Apr (not a clean holdout) | Full agent: anchor + learning − Raw model, no agent | P&L | +2,055 [+1,031, +3,009] | 0.000 |
