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
