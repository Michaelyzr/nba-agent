# Hard Q&A: deep learning and "why agentic"

**Q1. Your deep-learning models don't beat the baselines. Where is the deep learning?**
In four places (`docs/deep_learning.md`). The M2 player GRU beats gradient boosting (pinball
1.55 vs 1.59 for points). M6, a GRU over the price stream with quantile outputs, is worse than
"no move" in 9 of 9 pre-declared configurations × 3 seeds. That is a well-powered null and
what an efficient market predicts. M4-NN ties logistic regression (Brier 0.184 vs 0.183); used
as the anchor's news shift it reaches 0.163, level with the market (0.164), not
significantly better. The LLM agent is a transformer used as a policy with tools. We report
nulls with CIs instead of tuning until something looks good.

**Q2. Why such small networks? Isn't that why they fail?**
The labels are per game: about 3,400 training games for the win model and about 600 for M6,
even though there are 3.9M price rows. We swept capacity (GRU-16/64/128, a CNN, an MLP). Bigger
was not better and every configuration stayed above the zero-move error. The hidden size was
picked on a validation slice only, never on test.

**Q3. Why an agent? A single classifier with a threshold would do.**
Our data says no. The model alone loses −$2,475 over 711 trades in the walk-forward season.
The decision process around it (market anchor, thresholds, abstention, gated rules) adds
+$2,450 [+854, +3,984], p = 0.002. A better classifier doesn't close the gap: neither
recalibration nor a neural net beats the market's Brier score. The problem is also
event-driven: 91% of the move comes before the inactive list and about two-thirds before the
first official report. So *when* to act is the decision, and a loop that triggers,
investigates and decides is the right shape.

**Q4. Does the learning loop actually learn anything?**
Honestly, it mostly learns to trade less. The placebo audit shows that random rules pass the
split gate 57% of the time, the same rate as the reviewer's rules. Learning improves CLV $
(+$39, p = 0.001) by cutting trades, not by finding edge. We report that as the result.

**Q5. Isn't Kalshi regulated by the CFTC, so the prices are fair?**
Regulated, yes: Kalshi has been a CFTC designated contract market since 2020, and it
self-certified sports contracts in January 2025 without the CFTC blocking them. But CFTC
oversight is about *market integrity*: surveillance, no manipulation, everyone sees the same order
book. It does not check that the price a retail user pays gives a fair chance of profit, there is
no best-execution duty on a direct exchange trade, and there is no federal responsible-gambling
obligation (sports contracts are open at 18). You still pay the taker fee
(0.07 × p × (1 − p) per contract) and the spread, and you trade against professionals, including
Kalshi's own affiliated market maker, which the CFTC has proposed to restrict in 2026. We measured
the result: about 5% cost per trade (1.92¢), and 91% of the news move before the public news.
"Fair" in the integrity sense, structurally tilted in the price sense. We make no accusation of
fraud. Sources: `docs/market_regulation.md`.

**Q6. Why not just use a sportsbook instead?**
It isn't a better deal, just a different one. A licensed sportsbook builds its margin into the odds:
US hold was 10.1% of handle in Q2 2026 (AGA), more than our ~5% exchange cost. In return, state law
gives you 21+ age limits, self-exclusion, responsible-gambling tools and advertising rules. Neither
venue lets an informed retail user win on average: the books' margin is larger, and the exchange's
price already holds the news. That is why our product teaches and lets people practise with play
money rather than recommending a venue. The regulatory gap is that exchange users get neither the
gambling protections of a sportsbook nor any oversight of the price they pay.
