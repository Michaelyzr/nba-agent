# Coach agent: simulated users with and without the Coach

**This is a simulation of compliance, not a user study.** Rule-based personas stand in for biased users and follow the Coach's nudge with probability c. Design, personas and hypotheses were pre-registered in `docs/preregistration_coach.md` before this run.

20 seeded League slates x 10 decisions from the test period (as-of information only), $20 per trade (halved to $10 when a caution nudge is followed). Fills at the recorded ask plus fee; CLV against the last price before tip-off. Paired: same persona, slates and compliance draws in every arm. CIs are 95% percentile bootstraps over slates (2000 reps); 'day CI' resamples game-days with `evaluation/stats.py`.

## Totals per persona and arm

| persona | arm | trades | P&L | CLV $ | mean CLV | fees | worst slate | nudged_pass | nudged_caution | complied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price chaser | without | 158 | −$152.13 | −$16.90 | -0.30¢ | +$86.66 | −$81.20 | 138.0 | 20.0 | 0.0 |
| price chaser | with c=0.5 | 87 | +$33.07 | −$4.87 | -0.21¢ | +$44.59 | −$82.01 | 138.0 | 20.0 | 82.0 |
| price chaser | with c=1 | 20 | −$41.32 | −$0.12 | -0.15¢ | +$3.77 | −$16.67 | 138.0 | 20.0 | 158.0 |
| long-shot lover | without | 152 | −$1,227.95 | −$92.52 | -0.42¢ | +$165.43 | −$210.59 | 109.0 | 42.0 | 0.0 |
| long-shot lover | with c=0.5 | 94 | −$432.77 | −$54.00 | -0.46¢ | +$91.35 | −$125.68 | 109.0 | 42.0 | 78.0 |
| long-shot lover | with c=1 | 43 | −$172.29 | −$20.77 | -0.69¢ | +$24.27 | −$52.13 | 109.0 | 41.0 | 150.0 |
| overtrader | without | 200 | −$697.84 | −$111.43 | -0.57¢ | +$184.42 | −$145.52 | 144.0 | 54.0 | 0.0 |
| overtrader | with c=0.5 | 123 | −$439.65 | −$62.83 | -0.67¢ | +$99.64 | −$136.20 | 144.0 | 52.0 | 103.0 |
| overtrader | with c=1 | 56 | −$107.51 | −$23.50 | -0.90¢ | +$28.37 | −$51.01 | 144.0 | 49.0 | 193.0 |
| cautious | without | 53 | −$238.99 | −$33.05 | -0.49¢ | +$54.25 | −$63.61 | 35.0 | 17.0 | 0.0 |
| cautious | with c=0.5 | 36 | −$178.90 | −$22.59 | -0.61¢ | +$32.52 | −$63.14 | 35.0 | 17.0 | 27.0 |
| cautious | with c=1 | 18 | −$124.28 | −$8.80 | -0.78¢ | +$9.96 | −$21.04 | 35.0 | 16.0 | 51.0 |
| never trade | reference | 0 | $0.00 | $0.00 | - | $0.00 | $0.00 | nan | nan | nan |

## Paired differences: with Coach − without Coach

| persona | arm | Δ P&L [slate CI] | Δ P&L day CI | Δ CLV $ [slate CI] | Δ CLV $ day CI | Δ mean CLV [slate CI] | Δ fees [slate CI] | Δ trades [slate CI] | Δ worst slate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price chaser | with c=0.5 | +$185.20 [−$99.69, +$485.38] | [−$217.04, +$570.14] | +$12.02 [+$2.17, +$22.35] | [−$0.91, +$25.08] | +0.10¢ [-0.10¢, +0.30¢] | −$42.07 [−$50.50, −$34.08] | -71 [-83, -60] | −$0.81 |
| price chaser | with c=1 | +$110.81 [−$201.36, +$431.34] | [−$477.76, +$640.47] | +$16.78 [−$3.50, +$33.37] | [−$3.82, +$39.83] | +0.15¢ [-0.19¢, +0.47¢] | −$82.89 [−$94.47, −$71.02] | -138 [-152, -125] | +$64.53 |
| long-shot lover | with c=0.5 | +$795.18 [+$473.93, +$1,115.28] | [+$411.57, +$1,152.07] | +$38.52 [+$23.23, +$54.53] | [+$20.51, +$55.32] | -0.05¢ [-0.19¢, +0.09¢] | −$74.08 [−$82.92, −$64.45] | -58 [-67, -48] | +$84.91 |
| long-shot lover | with c=1 | +$1,055.66 [+$303.40, +$1,746.38] | [+$388.02, +$1,690.41] | +$71.75 [+$53.18, +$89.97] | [+$42.50, +$102.02] | -0.27¢ [-0.57¢, +0.02¢] | −$141.16 [−$152.88, −$127.19] | -109 [-119, -98] | +$158.46 |
| overtrader | with c=0.5 | +$258.19 [−$232.15, +$723.34] | [−$357.47, +$782.89] | +$48.60 [+$30.56, +$68.43] | [+$30.12, +$70.26] | -0.09¢ [-0.28¢, +0.06¢] | −$84.78 [−$96.82, −$73.37] | -77 [-88, -67] | +$9.32 |
| overtrader | with c=1 | +$590.33 [−$128.01, +$1,165.50] | [−$182.98, +$1,312.44] | +$87.94 [+$65.07, +$111.47] | [+$56.69, +$119.61] | -0.33¢ [-0.84¢, +0.07¢] | −$156.05 [−$163.10, −$149.09] | -144 [-150, -138] | +$94.51 |
| cautious | with c=0.5 | +$60.09 [−$269.95, +$314.12] | [−$277.78, +$328.94] | +$10.46 [+$3.65, +$18.37] | [+$2.43, +$20.95] | -0.12¢ [-0.25¢, -0.00¢] | −$21.73 [−$29.26, −$14.04] | -17 [-24, -10] | +$0.47 |
| cautious | with c=1 | +$114.71 [−$369.03, +$520.45] | [−$349.67, +$530.93] | +$24.24 [+$13.93, +$35.78] | [+$11.29, +$38.67] | -0.29¢ [-0.53¢, +0.00¢] | −$44.29 [−$53.43, −$35.76] | -35 [-44, -27] | +$42.57 |

## Was the nudge right? (without-Coach trades, labelled by the nudge the Coach would have given)

Mean CLV per contract; 'pass − none' with a slate-bootstrap CI. 'Flagged' = pass or caution.

| persona | pass trades | pass mean CLV | caution trades | caution mean CLV | none trades | none mean CLV | pass - none | ci_low | ci_high | pass trades with CLV < 0 | flagged share | flagged with CLV < 0 | all trades with CLV < 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all personas | 426 | -0.36¢ | 133 | -0.76¢ | 4 | +1.00¢ | -1.36¢ | -5.95¢ | +0.23¢ | 72% | 99% | 75% | 75% |
| cautious | 35 | -0.34¢ | 17 | -0.79¢ | 1 | -0.50¢ | +0.16¢ | -0.09¢ | +0.35¢ | 74% | 98% | 79% | 79% |
| long-shot lover | 109 | -0.31¢ | 42 | -0.69¢ | 1 | -0.50¢ | +0.19¢ | -0.03¢ | +0.38¢ | 75% | 99% | 79% | 80% |
| overtrader | 144 | -0.44¢ | 54 | -1.03¢ | 2 | +2.50¢ | -2.94¢ | -6.04¢ | +0.16¢ | 74% | 99% | 77% | 76% |
| price chaser | 138 | -0.33¢ | 20 | -0.15¢ | 0 | - | - | - | - | 67% | 100% | 68% | 68% |

## Reading

- **Fees fall for every persona (H1 supported)** and **CLV $ rises** at c = 1 for the long-shot lover (+$71.75 [+$53.18, +$89.97]), the overtrader (+$87.94 [+$65.07, +$111.47]) and also the cautious persona; the price chaser's CI includes 0 at c = 1 (H2 partly supported; H3 not supported).
- **The gain comes from trading less, not trading better.** Mean CLV per contract does not improve (every Δ mean CLV CI includes or is below 0). Every persona's trades lose about the half-spread plus fee, so any rule that removes trades raises CLV $ toward never trading ($0).
- **How often was the nudge right?** 75% of trades the Coach flagged (pass or caution) had negative CLV, but so did 75% of all trades: the Coach flags 99% of trades, so it is barely discriminating. Only 4 trades got no nudge, too few to test H4 (pass-nudged CLV below un-nudged CLV).
- So the Coach is a well-behaved brake: it never encouraged a trade, its advice is correct on average because almost every retail trade here has negative CLV, and it cannot (and does not claim to) find edges.

## Limitations

- Personas are fixed rules; they do not learn across slates, and real users may ignore or over-follow the Coach. Compliance is assumed, not measured.
- The Coach and the overtrader and cautious personas read the same M4 snapshot, so some agreement is by construction; outcomes are therefore judged by market CLV, not by the Coach's own estimate.
- P&L over 200 decisions is mostly luck; CLV and fees are the more reliable outcomes.
- Slates may share games; the day-clustered CI accounts for shared nights.
- The optional LLM wording is not evaluated: it never changes the nudge.

## Deviations from the pre-registration

- None in design, personas, slates or metrics. Added after the run (descriptive only): the share of flagged trades with negative CLV against the base rate, because H4 could not be tested.
