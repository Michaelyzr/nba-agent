# Deep learning in the system

Four neural components, each graded by the market's own prices. Numbers are on the
501 test games (1 Feb – 12 Apr 2026) unless stated; CIs resample game-days
(`evaluation/stats.py` convention, 2,000 replicates, seed 7606).

## Where deep learning sits

| Component | Model | Role in decisions | Result |
| --- | --- | --- | --- |
| M2 (`forecast/gru.py`) | GRU over each player's last 20 games + tonight's features → points and minutes quantiles (pinball loss) | Player forecasts in `forecast/api.py` (Coach, props) | Held-out pinball: points 1.55 vs GBM 1.59 vs rolling-10 1.64; minutes 1.68 vs 1.72 vs 1.97 (`m2_heldout.csv`). The one neural model that beats its baselines. |
| M4-NN (`forecast/win_nn.py`) | MLP on M4's six home-minus-away features; GRU variant over each team's last 10 games | Candidate win model / news shift for the agent's anchor estimate | See below: ties logistic M4 alone; as the anchor's news shift it is level with the market. |
| M6 (`forecast/impact.py`) | GRU-16 over 6 h of 15-min quotes + 18 static features → 10/50/90% quantiles of the move to tip (pinball loss) | Optional trading signal (`MarketAgent(impact=...)`) | Worse than "the price won't move" in all 9 configurations × 3 seeds of a pre-declared grid (`m6_robustness.md`); never predicts the 5.5¢ needed to trade, so the agent makes 0 trades. |
| LLM tool agent (`agents/tool_agent.py`) | Transformer used as a reasoning policy that picks as-of tools; code checks every number; a sceptic can veto | Alternative decision policy | Plain LLM: 5 trades in 304 decisions, CLV $ −1 (= never trading). Tool-agent run: `evaluation/results/llm_agent.md`. |

## M4-NN results (`evaluation/results/win_nn.md`, `python -m evaluation.win_nn_eval`)

Same rows (`forecast.win.training_rows`, absences known), same split (3,365 games before
1 Feb 2026) and same 501 test games as M4. Early stopping on 1 Dec – 31 Jan, then refit on
all training games for the chosen epoch count; hidden size and learning rate picked by
validation BCE only (MLP: 16 units, lr 3e-4; GRU: 8 units, lr 1e-3); three seeds, forecast =
3-seed mean.

| Predictor | Brier | Log loss | Brier − M4 [95% CI] | Brier − market 1 h [95% CI] |
| --- | --- | --- | --- | --- |
| M4 (logistic regression) | 0.1827 | 0.548 | — | +0.0186 [+0.0107, +0.0271] |
| M4-NN MLP | 0.1842 | 0.551 | +0.0015 [−0.0056, +0.0080] | +0.0201 [+0.0112, +0.0284] |
| M4-NN GRU | 0.1864 | 0.558 | +0.0036 [−0.0017, +0.0086] | +0.0222 [+0.0142, +0.0302] |
| Anchor + M4 shift (agent's estimate) | 0.1663 | 0.504 | −0.0164 | +0.0022 [−0.0020, +0.0066] |
| Anchor + MLP shift | 0.1632 | 0.497 | −0.0195 | −0.0009 [−0.0063, +0.0048] |
| Anchor + GRU shift | 0.1642 | 0.498 | −0.0185 | +0.0000 [−0.0043, +0.0047] |
| Market 1 h before tip | 0.1641 | 0.497 | −0.0186 | — |

Seed spread (single seeds): MLP 0.1846 ± 0.0016, GRU 0.1869 ± 0.0014. Anchor + MLP shift
vs the agent's current estimate: −0.0032 [−0.0079, +0.0013] Brier. Read: a neural net does
not beat logistic regression on six features and 3.4k games, and nothing beats the market;
the neural news shift is at best a small, non-significant improvement to the anchor. Not
yet replayed through the trading agent.

## Design choices

- **Why a GRU for price sequences (M6).** The inputs are an ordered, irregular stream of
  quotes; a GRU summarises recent drift, spread and volume with few parameters and handles
  missing steps with a mask channel. A 1D CNN and a static-feature MLP were run as controls
  and did no better.
- **Why quantiles and pinball loss.** A trade needs the move to clear ~5.5¢ of spread and
  fee, so the band matters as much as the median; monotone quantile heads (softplus
  increments) give a 10–90% band that can be used to abstain.
- **Why such small models.** There are 3.9M one-minute price rows, but they come from about
  1,300 games (≈600 in M6's training split) and 3.4k games for the win model; labels are
  per game, so the effective sample is games, not bars. Hidden sizes 8–16, dropout, weight
  decay and early stopping on a later time slice; capacity sweeps (GRU-16/64/128) did not help.
- **Why null results are expected and still informative.** A liquid market price is
  approximately a martingale and already aggregates public news, so "no move" and the
  market's own probability are strong baselines. A null with tight CIs, pre-declared grids
  and several seeds is evidence about the market, and it is what makes the product
  education rather than tips.

## Data plan and splits

Kalshi KXNBAGAME 1-minute quotes (Oct 2025 – Jun 2026) and ESPN games, box scores and
inactive lists (2023-24 to 2025-26). All features are as-of: history is queried by tip,
quotes and news only up to the decision time. Train before 1 Feb 2026 (M6: before 15 Jan,
validation 15–31 Jan), test 1 Feb – 12 Apr, play-offs a holdout scored once.

## Limitations and future work

- One season of market data; small test set (501 games, ~70 game-days), so CIs are wide.
- M4-NN is evaluated as a forecast only; a replay with its news shift in the agent is next.
- **Future work: M6's uncertainty as a trade filter** (abstain when the 10–90% band says the
  price is likely to move against the side, or the band is wide), scored offline on the
  saved trades with paired CLV $ CIs. Not done for this deadline.
