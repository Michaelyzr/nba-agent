# Preregistration: Coach agent, simulated-user evaluation

Written and committed **before** `evaluation/coach_sim.py` was run on real data. Any deviation found
later is listed under "Deviations" at the bottom of `evaluation/results/coach_sim.md`.

## What this is (and is not)

This is a **simulation of compliance**, not a user study. Rule-based personas stand in for biased
retail users. With the Coach, a persona follows the Coach's nudge with a fixed probability. The
results show the Coach's *mechanism* (does following its nudges avoid costly trades, measured by
the market's own closing price?), not whether humans learn from it or follow it.

## Intervention

`agents/coach_agent.advise(snapshot, team, history)`, offline rule-based mode (no LLM), as
committed with this file. The intervention is fixed before the run. Before a pending pick it runs
as-of checks chosen by its planner: the user's habits, break-even after fees, the move since the
24 h anchor (priced in?), long-shot price, news freshness and the M4 news shift. It returns a nudge
in {`pass`, `caution`, `none`}. It never suggests a bigger stake or more trades.

## Slates (as-of information only)

- 20 League slates named `coach-sim-00` … `coach-sim-19`, seed `league.seed_for(name)`.
- Each has 10 decisions from the **test period** (2026-02-01 to 2026-04-12), from `league.choose_slate`.
- Each decision uses `league.decision_info` with the trained M4 forecaster (`models/`). That is the
  same as-of snapshot a League player sees, so there are no later prices, news or scores.
- Fills and settlement use `coach.paper_trade` (the recorded ask plus fee, capped by volume, settled
  at the final result). The closing price is the last quote before tip-off.

## Personas (fixed $20 stake per trade)

| persona | rule at each decision |
|---|---|
| price chaser | Back the side whose mid has moved most toward it since the 24 h anchor, if that move is at least 1 point; otherwise pass. |
| long-shot lover | Back the cheaper side if its ask is at or below 40¢; otherwise pass. |
| overtrader | Back the side with the larger raw-M4 gap (M4 probability minus break-even after fees) at every decision. |
| cautious | Back the side with the larger raw-M4 gap only when that gap is at least 10 points; otherwise pass. |

Plus the **never-trade** reference (zero on every metric).

## Arms

- **Without Coach:** the persona's rule as above.
- **With Coach, compliance c ∈ {0.5, 1.0}:** whenever the persona would trade, the Coach advises on
  that pick. Its inputs are the snapshot and the persona's earlier decisions in the same slate and
  arm, using decision-time facts only.
  - If the nudge is `pass` and u < c, the persona passes.
  - If the nudge is `caution` and u < c, the persona halves its stake to $10.
  - Otherwise it trades as planned.
  - u ~ Uniform(0, 1) comes from `numpy.random.default_rng` seeded by (7606, persona, slate, decision).
    The same u is used for both values of c, so c = 1.0 complies whenever c = 0.5 does.

## Metrics

For each persona and arm, totals over all 200 decisions:
- P&L after fees;
- CLV $ (CLV per contract × contracts);
- mean CLV per contract;
- trades;
- fees paid;
- the worst slate's P&L.

**Paired differences** (with Coach − without Coach) are computed on the same slates and decisions:
- 95% percentile bootstrap CIs over the 20 slates (2000 reps, seed 7606);
- for P&L, CLV $ and mean CLV, also the day-clustered paired bootstrap from `evaluation/stats.py`
  (`day_table` and `paired`) over the game-days in the slates.

**Was the nudge right?** In the without-Coach arm, every trade is labelled with the nudge the Coach
would have given (`pass`, `caution` or `none`). We report mean CLV per contract for each label, and
the difference `pass` − `none` with a slate bootstrap CI. The nudge is "right" if `pass`-nudged trades
have lower CLV than un-nudged ones.

## Primary outcome and hypotheses

- **Primary:** paired difference in CLV $ and in fees, at c = 1.0, for each persona.
- **H1 (mechanical):** the Coach lowers fees paid for every persona that trades. Passing or halving
  can only lower fees.
- **H2:** the Coach raises CLV $ for the price chaser, the long-shot lover and the overtrader, with
  the 95% CI above 0.
- **H3:** for the cautious persona the difference is small, with a CI that includes 0.
- **H4:** `pass`-nudged trades have lower mean CLV than un-nudged trades.
- **Secondary:**
  - P&L differences. We expect wide CIs: results are mostly luck over 200 decisions.
  - c = 0.5.
  - The worst slate.

No other personas, thresholds or slates will be tried and then selectively reported.

## Limitations stated in advance

- Personas are fixed rules. They do not learn across slates, and real users may ignore, resent or
  over-follow the Coach.
- The Coach and the cautious and overtrader personas both read the M4 snapshot. Agreement between
  them is partly by construction. We therefore judge outcomes by **market CLV**, not by the Coach's
  own estimate.
- Slates may share games, because seeds are drawn independently. The day-clustered CI accounts for
  shared nights.
- The LLM mode is not evaluated here; it only rewrites the message and does not change the nudge.
