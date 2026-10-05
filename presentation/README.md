# Presentation deck (Fri 9 Oct 2026)

**Narrative:** sports prediction markets are structurally unfair to consumers. We built a
serious NBA trading agent to try to beat them, and it is our evidence. The product's job is to
protect and educate consumers. "Unfair" here means structural disadvantages (costs, speed,
expertise), not manipulation or fraud. We make no accusation against Kalshi.

`nba_agent_deck.pptx` is a 16:9 deck of 15 slides with speaker notes on every slide. It is
built by `build_deck.py` from the figures in `figures/`, which are copies of
`evaluation/results/*.png`, so this folder is self-contained.

## Rebuild

```bash
python -m evaluation.consumer_fairness      # costs, price discovery, long-shot check (needs data/frozen)
cp evaluation/results/{product_workflow,cost_burden,price_discovery,m4_vs_market,m6_vs_baselines,walkforward,longshot_calibration,agent_workflow}.png presentation/figures/
python presentation/build_deck.py           # writes presentation/nba_agent_deck.pptx
python presentation/build_deck.py --check   # also prints the estimated fit of every text box
```

Every number on the slides is in the `K` dict at the top of `build_deck.py`, with its source.
The script warns, and exits with status 1, if a shape leaves the slide, a text box is
estimated to overflow, or a slide has no speaker notes.

**M4 Brier score:** the deck uses the README's definition, M4 0.186 against the market's 0.164,
both 1 hour before tip on 501 test games. Slide 6 footnotes the at-tip pair: M4 with the inactive
list 0.183, market at tip 0.163.

## Slides and timing (target: about 10 minutes including the demo)

The time limit has not been confirmed, so adjust the timings if needed.

| # | Slide | Time |
| --- | --- | --- |
| 1 | Title: Can a consumer beat a sports prediction market? | 0:15 |
| 2 | The belief: "I follow the news, so I can beat the market" | 0:30 |
| 3 | What we built: a market-graded agent (`product_workflow.png`) | 0:20 |
| 4 | Evidence 1: costs set a hurdle before you start (`cost_burden.png`, with the selectivity funnel) | 0:45 |
| 5 | Evidence 2: the price moves before the news reaches you (`price_discovery.png`) | 0:30 |
| 6 | Evidence 3: the market beats our model (`m4_vs_market.png`) | 0:25 |
| 7 | Evidence 4: even a market-trained neural net finds nothing (`m6_vs_baselines.png`) | 0:40 |
| 8 | Evidence 5: tested with CIs, never trading wins (`walkforward.png`) | 0:45 |
| 9 | Long shots: where the data is weaker (`longshot_calibration.png`) | 0:20 |
| 10 | Context: who sits on the other side? (general knowledge, not measured) | 0:15 |
| 11 | Our product, reframed: protect the consumer first (`agent_workflow.png`) | 0:30 |
| 12 | Coach: teach why most bets don't clear costs | 0:20 |
| 13 | Honest scope: what "unfair" means here (with the research-process disclosure) | 0:30 |
| 14 | Live demo: `streamlit run app.py` | 2:30 |
| 15 | Recommendations, next steps and Q&A | 0:25 |
| | **Total** | **about 9:00** |

That leaves about a minute for transitions. The 5 minutes of Q&A follow. If time is short,
cut slides 9 and 10 (saves 0:35) or shorten slide 3. If the live demo fails, play the backup
recording on slide 14.

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
- **Asymmetry (slide 10)** is general industry knowledge and is labelled as not measured.
- **Disclosure (slide 13):** the market anchor and the reviewer were designed after seeing the
  Feb–Apr results, so that window is not a clean holdout. The walk-forward season and the
  play-off holdout are the honest checks.
