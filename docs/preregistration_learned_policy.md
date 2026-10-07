# Pre-registration: learned trade/pass policy (option G)

Written and committed on Wed 7 Oct 2026, **before** the test window (1 Feb – 12 Apr 2026) or the play-off
holdout was scored with this policy. Code: `forecast/policy.py`, `evaluation/learned_policy.py`.

## Question

Can a policy learned from data decide *whether* to trade and *which side* better than the hand-written
threshold rule ("trade when the anchored gap after fees exceeds 4 points")?

## Data and labels

- **Decision points:** every replay decision time (each news update in the six hours before tip, plus one hour
  before tip) for every game with a Kalshi game-winner market. Each market gives two options, buy yes at the ask
  or buy no at 1 − bid, so a game has four options per decision time.
- **Features (as-of only, from the `AsOf` view through the trader's own trigger → investigate → forecast →
  analyse steps):** gap after fees for that option, side price, market move toward the option since the anchor
  (24 h before tip), signed M4 news shift toward the option, news age (hours, capped at 6), has-news flag,
  hours to tip, spread, log volume over the last hour, anchored probability of the option, is-yes, is-home.
  M4 is the walk-forward model retrained before each month, the same as `runs/walkforward/`. Calibrated
  probabilities are not included (they were not available as a stable module when this was written).
- **Label:** closing-line value after costs, per contract: (mid at tip for that side) − (fill price) − (Kalshi
  fee per contract). Both sides' closes are recorded, so every option has a label (full information; supervised
  learning of action values, not off-policy RL).

## Windows

| Use | Dates |
| --- | --- |
| Fit (for selection) | 1 Nov 2025 – 14 Jan 2026 |
| Validation (for selection) | 15 – 31 Jan 2026 |
| Final training (refit with the chosen configuration) | 1 Nov 2025 – 31 Jan 2026 |
| **Test (main result)** | **1 Feb – 12 Apr 2026** |
| Play-off holdout (scored once by this policy; the agents' holdout has been seen before) | 13 Apr – 14 Jun 2026 |

No test or holdout row is used for fitting, selection or thresholds.

## Policy and selection

- **Models (value regression of net CLV on the features, standardised):** ridge (alpha 1, 10, 100), gradient
  boosting (depth 2 or 3, 200 iterations, learning rate 0.05, min leaf 40), and a small MLP (16 units with
  L2 0.01, or 32-16 units with L2 0.001; early stopping) as the deep-learning arm. Seed 7606.
- **Policy:** at each decision time take the option with the highest predicted net CLV; trade it if the
  prediction exceeds tau, at most once per game, $20 stake, through the same `Replay` (fills at ask plus fee,
  volume caps, risk caps). tau ∈ {0, 0.5, 1, 2, 3, 5} cents.
- **Selection rule:** choose the (model, tau) with the highest validation net CLV dollars (ties: fewer trades).
  **If no configuration has positive validation net CLV dollars, the learned policy is "never trade"** (learned
  abstention), and that is what is reported as the primary arm.
- **Diagnostic arms (pre-declared):** "forced to trade" = the best validation configuration with at least 10
  validation trades; "best MLP" = the best MLP configuration on validation. These exist to measure whether the
  learned scores rank trades at all, not as deployable policies.

## Comparison arms and statistics

Learned policy, forced, best MLP, the deterministic agent without learning, the full agent (anchor + learning)
and never trade, all on the same game-days. `evaluation/stats.py`: day-clustered bootstrap, 2,000 replicates,
seed 7606; paired differences on the same resampled days. Reported: trades, mean CLV per contract, CLV dollars,
P&L after fees, and a reliability table (deciles of predicted vs realised net CLV over test options).

## Hypotheses (and what we expect)

- **H1 (primary):** the learned policy beats the deterministic agent without learning on test CLV dollars
  (paired, one-sided, 95% CI above zero). *Expected: not supported.*
- **H2:** the learned policy beats never trade on test CLV dollars. *Expected: not supported; if validation
  chooses never trade the difference is exactly zero.*
- **H3 (diagnostic):** among test options, the top decile of predicted net CLV has realised net CLV above zero.
  *Expected: not supported.* A positive rank correlation alone is not evidence of edge, because the fee is a known
  function of price and the model can learn it.
- **H4 (diagnostic):** the forced policy beats the agent without learning on CLV dollars. *Expected: not supported.*

**Reason for expecting a null:** the label is M6's target (move to the close) minus half the spread and the fee.
M6 showed the move to tip is not predictable beyond zero. In a three-day pilot of the extractor (1–3 Nov,
training window) the largest net CLV of any option was +0.1 cents. A null is a valid, reportable result: it
says the learned policy should learn to pass.

## What would change the conclusion

Only H1 with a 95% CI above zero counts as "the learned policy passes the evidence gate". Anything else is
reported as a null, with the numbers. The holdout is run once after the test result is written, and is not used
to change anything.
