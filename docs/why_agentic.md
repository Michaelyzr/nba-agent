# Why an agentic framework

The argument is from the structure of this problem and from our own measurements, not
from fashion.

1. **The hard decision is *when* to act.** Injury news arrives asynchronously, and the
   market moves before it is public: on the 193 test games where an absence moved M4 by at
   least a point, 91% of the price move happens before the inactive list, and about
   two-thirds (67% [47%, 85%]) before the first official NBA injury report
   (`injury_timing.md`). A static model scores a game once; it cannot decide when to look,
   when to wait, or that the news is already priced. The agent's trigger → investigate →
   decide loop is built for that.
2. **Tool use over as-of data.** The agent calls tools for prices, news, rotations and the
   models; every tool reads an as-of view, so no look-ahead is enforced by the tools, not
   by the prompt.
3. **Multi-step decisions with safety.** Checks, the market anchor, an edge threshold, risk
   caps ($50 / order, $100 / game, $300 / day), an enforced kill switch and blocked orders
   (9 / 9 planted violations) sit between a forecast and an order. With gambling harm in
   scope, these must be code the LLM cannot override.
4. **Market-graded learning.** Review → gate → rules, with the gate judged on held-out days
   by CLV $. The placebo audit shows what learning really does: random rules pass the split
   gate 57% of the time, the same as the reviewer's picks, so learning helps by trading
   less (full agent − no learning: CLV $ +39 [+18, +61], p = 0.001; P&L +382, p = 0.13).
5. **Human in the loop.** The Coach and the League turn the agent's analysis into user
   education: break-even after costs, paper calls, ranking by closing-line value.
6. **Orchestration.** One as-of clock ties the team's pregame news loop, in-play updates,
   trader and Coach together.

## Counter-argument: "a single classifier would do"

Our evidence says the opposite. Walk-forward season (Nov – 12 Apr, M4 retrained monthly,
day-clustered bootstrap, `significance.md`):

| Setup | Trades | P&L after fees [95% CI] |
| --- | --- | --- |
| The model alone (raw M4, trade on edge) | 711 | −$2,475 [−3,995, −845] |
| Agent: anchor, no learning | 276 | −$407 [−1,324, +553] |
| Full agent: anchor + learning | 157 | −$25 [−569, +564] |
| Never trade | 0 | $0 |

Full agent − raw model: **+$2,450 [+854, +3,984], p = 0.002**. A better classifier does not
close the gap: recalibrated M4 and the neural M4-NN do not beat the market's Brier
(`m4_calibration.md`, `win_nn.md`). The value comes from the decision process around the
model (anchoring to the market, thresholds, abstention, gated rules), which is exactly what
an agentic framework provides. It does not beat never trading, and we say so; that finding
is why the product educates and protects rather than tips.
