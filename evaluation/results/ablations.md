| setup | fills | mean_clv | clv_se | pnl_after_fees | roi | brier_model | brier_market | max_drawdown | kill_switch_trips | rules_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full agent (learning) | 74 | -0.0022 | 0.0024 | -32.02 | -0.0219 | 0.2044 | 0.2 | 194.69 | 0 | 5.0 |
| Agent, development rules frozen | 129 | -0.0035 | 0.0015 | -247.27 | -0.0972 | 0.1882 | 0.1795 | 438.5 | 0 | 2.0 |
| Agent, no learning | 129 | -0.0035 | 0.0015 | -247.27 | -0.0972 | 0.1882 | 0.1795 | 438.5 | 0 | 0.0 |
| Agent, no learning, raw model (no market anchor) | 366 | -0.0047 | 0.0007 | -2087.26 | -0.2881 | 0.1849 | 0.1563 | 2325.65 | 10 | 0.0 |
| Plain model, no agent | 366 | -0.0047 | 0.0007 | -2087.26 | -0.2881 | 0.1849 | 0.1563 | 2325.65 | 10 | nan |
| Agent on win-rate placeholder | 246 | -0.0038 | 0.0009 | -429.09 | -0.0886 | 0.1696 | 0.1559 | 598.56 | 0 | 0.0 |
