# Pre-registration: gate audit, placebo gate, kill switch and fractional Kelly

Written and committed on Wed 7 Oct 2026, before any of the runs below were made. It covers option B
("safety and gate audit") and option D ("fractional Kelly") in `docs/agentic_review.md`. Any
change made after the first run will be listed at the end under "Deviations", with the reason.

## What changes in the code

**Gate modes** (`MarketAgent(gate=...)`, `--gate` on the command line):

| Mode | Reviewer selects on | Gate tests on | Metric to accept |
| --- | --- | --- | --- |
| `legacy` (published numbers) | trades in the last 21 calendar days, up to and including day *d* | last 14 market game-days before *d* (overlaps the selection) | mean CLV per trade rises by ≥ 0.005 |
| `split` (new default) | trades on the last 7 market game-days, up to and including *d* | the 14 market game-days **before** those 7 (disjoint) | total CLV dollars (Σ clv × contracts) rise by ≥ $2.00 |
| `split-edge` | as `split` | as `split` | total edge over the entry mid (Σ (close mid − entry mid) × contracts) rises by ≥ $2.00 |

All modes keep: at least 3 changed trades (`LIMITS["min_cases"]`), at least 2 test days or the rule is
deferred, back-tests replay only days before the decision, and a rule is active only from the
time the gate decides.

We chose "select on the most recent 7 days, test on the 14 before" because the reviewer should
react to recent losses, while the gate keeps its 14-day test length. Both windows lie strictly
before the decision. We add `split-edge` because CLV dollars still reward trading less whenever
the average trade has negative CLV, which is true here (fills pay the spread). Edge over the
mid excludes the spread, so dropping trades at random leaves it unchanged in expectation.

**Kill switch** (`Replay(kill_switch=100)`): once the *realised* P&L of the day's fills is below
−$100, no further order fills that day (`risk_result = "kill_switch"`). The P&L is realised only for
games with `final_at <= now`. Unsettled or future results are never used. Every trip is logged. The
agent's own risk step sees the same flag, through the as-of view, and blocks too.

**Confirmation:** the replay trades the platform's own paper account (channel `platform`), and
`auto_confirm` is labelled as a replay simplification. Retail orders are never auto-confirmed:
`auto_confirm` drops them, so a retail channel needs an explicit confirmation callable.

**Fractional Kelly** (`--sizing kelly --kelly-fraction 0.25`, bankroll $1,000 fixed). We buy one
side at price *c*, with fee *φ* = 0.07·c·(1 − c) per contract, so the cost of a contract is *k* = c + φ
and it pays $1 if it wins. Spending a fraction *x* of bankroll *B* gives expected log growth

  g(x) = p·log(1 + x(1 − k)/k) + (1 − p)·log(1 − x),

and g′(x) = 0 gives **x\* = (p − k)/(1 − k) = gap/(1 − c − φ)**, where gap = p − c − φ is the agent's
existing edge after fees. The stake is fraction · x\* · B in total outlay, so the contract
notional is that times c/k. It is capped at the $50 order cap, and the replay's game ($100) and
day ($300) caps still apply. Kelly uses the same abstention rule as flat: trade only if
gap > min edge (0.04) and no rule skips. *p* is the market-anchored estimate. A second arm uses
Platt-calibrated M4 from `forecast/calibrate.py`, fit only on out-of-sample predictions before
1 Feb 2026, if that file is present.

## Windows and setups

- **Primary:** the walk-forward season, 1 Nov 2025 – 12 Apr 2026, with M4 retrained before each
  month (`evaluation/walkforward.py` models). Secondary: the same, including the play-offs to
  14 Jun.
- **Secondary, not clean:** the original test period, 1 Feb – 12 Apr 2026. The development
  notebook is learned on 1 Nov – 31 Jan with the same gate mode, then learning continues.
- Every setup uses $20 flat stakes unless stated, offline rules (no LLM) and the same fills, fees
  and caps.

## Statistics

`evaluation/stats.py`: day-clustered bootstrap, 2,000 replicates, seed 7606, 95% percentile CIs,
one-sided p-values. Paired comparisons use the same game-days. Pass rates get Wilson 95% CIs.
Differences in pass rates get a bootstrap over review points (2,000 replicates, seed 7606).

## Placebo gate (`evaluation/gate_placebo.py`)

- **Base agent:** the walk-forward anchor agent, no learning, empty notebook, over the primary
  window. Every rule is therefore gated against the same base.
- **Review points:** every 7th market game-day, starting at the first day with at least 21
  earlier market game-days, so that the split windows are full.
- **Rules gated at each review point:**
  - the reviewer's own pick under its `legacy` selection window and under its `split` selection
    window;
  - **every** one of the 11 reviewer templates. This gives the exact pass rate of a uniformly
    random template, so no sampling is needed.
  - 5 random-slice rules per mode. Each skips trades whose seeded hash bucket (md5 of ticker and
    decision time) falls in a random interval. The interval width is the share of
    selection-window trades the reviewer's rule hit, or 0.25 when the reviewer proposes nothing.
    The draws use seed 7606.
- **Outcomes per rule:**
  - accepted or not under `legacy`, `split` and `split-edge`;
  - the change in CLV $ (and mean CLV, edge $) on the gate's test window;
  - the change in CLV $ on the **next 7 market game-days** after the review point (forward,
    out-of-sample).

## Hypotheses and decision rules

- **H1:** with held-out gate days, fewer rules pass. Measures:
  - the reviewer's pass rate at the placebo review points under `split` versus `legacy`;
  - rules accepted in the full learning runs under `split` versus `legacy` (walk-forward and
    test period).

  Supported if the split rate is lower and the bootstrap CI of the difference excludes 0.
  Otherwise we report it as directional only.
- **H2:** under the legacy gate, the random-rule pass rate is similar to the reviewer's, so
  learning works only by reducing exposure. Supported if the CI of (reviewer − random template)
  and of (reviewer − random slice) pass rate includes 0. A further prediction: under `split` (CLV
  $), random slices pass more often than under `legacy`, because dropping any negative-CLV trades
  raises CLV $. Under `split-edge`, random slices pass at most about as often as under `legacy`.
  Second check: the forward CLV $ of accepted rules minus that of rejected rules. If its CI
  includes 0, the gate cannot tell lessons from noise.
- **H3:** the enforced kill switch reduces worst-day losses without changing mean CLV. Compare
  the full split agent with and without the kill switch: worst-day P&L, number of days below
  −$100, and paired mean-CLV difference. Supported if the worst day improves and the mean-CLV CI
  includes 0. We also report the P&L and CLV $ differences.
- **H4 (Kelly):** ¼-Kelly does not beat flat $20 on P&L or CLV $, because edges at the 4-point
  threshold are small and Kelly scales up the trades the market grades worst. Arms: no-learning
  anchor agent, flat vs ¼-Kelly (vs calibrated ¼-Kelly if available) vs never trade, with
  paired bootstrap and maximum drawdown. Reported as a null result unless Kelly − flat has a P&L
  CI above 0.

## Reported for every agent run

Rules proposed and accepted, trades, mean CLV, CLV $, P&L after fees, worst day, days below −$100
and enforced kill-switch trips. All are against never trade and against the legacy agent, with
the CIs above. Files: `evaluation/results/gate_audit.{md,csv}`, `gate_placebo.{md,csv}` and
`kelly.{md,csv}`.

## Deviations

None yet.
