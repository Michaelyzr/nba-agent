# Holdout: play-in and play-offs, 13 Apr – 14 Jun 2026

44 game-days, 87 games with Kalshi markets. Not looked at before this run; each setup run once. Default models (M4 trained on games before 1 Feb 2026). The full agent starts from the test-period notebook and keeps learning. Inactive-list news in the frozen data ends 07 May 2026; after that the agents see no news.

| Window | Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | ROI [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Holdout 13 Apr–14 Jun | Full agent: anchor + learning | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |
| Holdout 13 Apr–14 Jun | Agent: anchor, no learning | 6 | +0.0050 [-0.0050, +0.0200] | +0 [-3, +4] | -74 [-176, +26] | -62.0% [-105.1%, +22.9%] | 0.203 |
| Holdout 13 Apr–14 Jun | Raw model, no agent | 59 | -0.0052 [-0.0083, -0.0026] | -24 [-38, -12] | +450 [-55, +1,171] | +38.4% [-5.1%, +95.9%] | 0.999 |
| Holdout 13 Apr–14 Jun | Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – | – |

