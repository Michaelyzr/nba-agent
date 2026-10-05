# Presentation deck (Fri 9 Oct 2026)

**Narrative:** the deck opens with a three-part consumer problem. (1) Betting is mainstream and
people want to understand it. (2) Betting markets are structurally tilted against consumers; that
part is our own measurement. (3) Published research shows betting harms well-being. Then comes the
research question: can a consumer, even one armed with an AI agent and public news, beat the market?
Our agent is the evidence that the answer is no, so the product educates and protects consumers.
"Unfair" here means structural disadvantages (costs, speed, expertise), not manipulation or fraud.
We make no accusation against Kalshi.

`nba_agent_deck.pptx` is a 16:9 deck of 14 slides with speaker notes on every slide. It is
built by `build_deck.py` from the figures in `figures/`, which are copies of
`evaluation/results/*.png`, so this folder is self-contained.

## Rebuild

```bash
python -m evaluation.consumer_fairness      # costs, price discovery, long-shot check (needs data/frozen)
cp evaluation/results/{product_workflow,cost_burden,price_discovery,m4_vs_market,m6_vs_baselines,walkforward,longshot_calibration,agent_workflow}.png presentation/figures/
python presentation/build_deck.py           # writes presentation/nba_agent_deck.pptx
python presentation/build_deck.py --check   # also prints the estimated fit of every text box
```

Every number we measured is in the `K` dict at the top of `build_deck.py`, with its source. The
external research numbers on slide 2 are in a separate `EXT` dict, with their references.
The script warns, and exits with status 1, if a shape leaves the slide, a text box is
estimated to overflow, or a slide has no speaker notes.

**M4 Brier score:** the deck uses the README's definition, M4 0.186 against the market's 0.164,
both 1 hour before tip on 501 test games. Slide 7 footnotes the at-tip pair: M4 with the inactive
list 0.183, market at tip 0.163.

## Slides and timing (target: about 10 minutes including the demo)

The time limit has not been confirmed, so adjust the timings if needed.

| # | Slide | Time |
| --- | --- | --- |
| 1 | Title: Betting markets are tilted against consumers | 0:15 |
| 2 | The problem: three things consumers face (want to learn / markets tilted / betting causes harm) | 1:00 |
| 3 | Research question, what we built to test it, and the answer | 0:30 |
| 4 | What we built: a market-graded agent (`product_workflow.png`) | 0:20 |
| 5 | Evidence 1: costs set a hurdle before you start (`cost_burden.png`, with the selectivity funnel) | 0:40 |
| 6 | Evidence 2: the price moves before the news reaches you (`price_discovery.png`) | 0:30 |
| 7 | Evidence 3: the market beats our model (`m4_vs_market.png`) | 0:25 |
| 8 | Evidence 4: even a market-trained neural net finds nothing (`m6_vs_baselines.png`) | 0:35 |
| 9 | Evidence 5: tested with CIs, never trading wins (`walkforward.png`) | 0:45 |
| 10 | Our product: educate and protect the consumer (`agent_workflow.png`) | 0:30 |
| 11 | Coach: teach why most bets don't clear costs | 0:20 |
| 12 | Honest scope: what "unfair" means here (disclosure, long-shot result, external-research note) | 0:30 |
| 13 | Live demo: `streamlit run app.py` | 2:30 |
| 14 | Recommendations, next steps and Q&A | 0:25 |
| | **Total** | **about 9:15** |

That leaves about 45 seconds for transitions. The 5 minutes of Q&A follow. The old long-shot and
"who sits on the other side" slides were cut. The long-shot result is now a caveat on slide 12, and
`figures/longshot_calibration.png` is kept as a backup for Q&A. If time is short, shorten slide 4.
If the live demo fails, play the backup recording on slide 13.

## External research on slide 2 (not our measurement)

These were checked against the full texts in October 2026. The slide labels this pillar
"External research".

1. Wardle H, Degenhardt L, Marionneau V, et al. *The Lancet Public Health Commission on
   gambling.* Lancet Public Health 2024; 9: e950–94. doi:10.1016/S2468-2667(24)00167-1.
   - An estimated 46.2% of adults (95% CI 41.7–50.8) and 17.9% of adolescents gambled in the
     past year (380 samples, 68 countries).
   - About 448.7 million adults experience some risk gambling, and about 80 million have
     gambling disorder or problematic gambling.
   - Gambling disorder may affect 8.9% of adults who bet on sports.
   - The Commission concludes that gambling is a public-health threat.
2. Hollenbeck B, Larsen P, Proserpio D. *The Financial Consequences of Legalized Sports
   Gambling.* SSRN 4903302 (2024, revised 2025). Uses the UC Consumer Credit Panel (US).
   - 38 states legalised sports betting after the 2018 Supreme Court ruling.
   - Credit scores fall about 0.8 points with legal betting, and 2.75 points with online access.
   - Bankruptcies rise about 10% (9 per 100,000) with online access, roughly 30,000 a year.
   - Debt collections, debt consolidation and auto-loan delinquencies also rise.
3. Baker SR, Balthrop J, Johnson MJ, Kotter JD, Pisciotta K. *Gambling Away Stability: Sports
   Betting's Impact on Vulnerable Households.* NBER Working Paper 33108 (2024).
   doi:10.3386/w33108.
   - Each $1 of online sports betting reduces net investment by about $0.99.
   - Low-savings households see credit-card balances rise by about $368 (about 8%), and more
     overdrafts.

"Few understand prices, fees and spreads" on slide 2 is motivation, not a measured statistic.

## Evidence used

- **Costs** (`consumer_fairness.md`): a $20 order at tip − 60 min pays 1.92¢ per contract over
  the mid, which is 5.0% of the price, and 9–14% under 20¢. In the Feb–Apr window the full
  agent made +$36 at the mid and −$32 after costs. Funnel: 771 decision points → 153 with a gap
  over 4 points → 74 trades, of which 27 (36%) beat or matched the close.
- **Information speed** (`consumer_fairness.md`): on 193 test games, 91% of the news-direction
  move happened before the inactive list was public (+1.80 vs +0.18 points).
- **The market beats our model:** see the M4 Brier note above.
- **M6** (`m6_heldout.csv`, `m6_ablation.md`): on 771 test decision times, M6's mean error is
  0.867¢ against 0.833¢ for "price won't move", with R² ≈ 0 on test and holdout. After the list the
  price moves about 2.5% of M4's implied shift (coefficient 0.025). The M6 agent makes 0 trades;
  forced to trade, it made 501 trades and lost $652.
- **Walk-forward** (`walkforward_summary.md`, `significance.md`, `holdout.md`): Nov – 12 Apr,
  with M4 retrained monthly and a day-clustered bootstrap.
  - Mean CLV is below zero for every setup, and every 95% CI lies entirely below zero.
  - Full agent −$25 [−$569, +$564], p = 0.53 against never trading.
  - No learning −$407; raw model −$2,475 [−$3,995, −$845].
  - The anchor adds +$2,450 over the raw model (p = 0.002). Learning only means trading less
    (P&L gain p = 0.13).
  - Play-off holdout of 87 games: full agent 0 trades; no learning 6 trades, −$74; raw model
    +$450 with negative CLV, which is luck.
- **Long-shot bias: not clearly supported** (1 win in 66 against 7.4% implied, z = −1.8). Long
  shots still return −13.4% against −2.2% for heavy favourites, because of costs.
- **Disclosure (slide 12):** the market anchor and the reviewer were designed after seeing the
  Feb–Apr results, so that window is not a clean holdout. The walk-forward season and the
  play-off holdout are the honest checks.
