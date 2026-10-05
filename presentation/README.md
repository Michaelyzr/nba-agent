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
python -m evaluation.consumer_fairness      # cost, price-discovery and long-shot evidence (needs data/frozen)
cp evaluation/results/{cost_burden,price_discovery,longshot_calibration,product_workflow,agent_workflow,trade_funnel,headline_clv,m4_vs_market}.png presentation/figures/
python presentation/build_deck.py           # writes presentation/nba_agent_deck.pptx
python presentation/build_deck.py --check   # also prints the estimated fit of every text box
```

Every number on the slides is in the `K` dict at the top of `build_deck.py`, with its source.
When the statistical-rigour work or the M6 market-impact model changes a number, update `K` and
rebuild. The script warns, and exits with status 1, if a shape leaves the slide, a text box is
estimated to overflow, or a slide has no speaker notes.

**M4 Brier score:** the deck uses the README's definition, M4 0.186 against the market's 0.164,
both 1 hour before tip on 501 test games. Slide 7 footnotes the at-tip pair: M4 with the inactive
list 0.183, market at tip 0.163.

## Slides and a 10-minute timing

| # | Slide | Time |
| --- | --- | --- |
| 1 | Title: Can a consumer beat a sports prediction market? | 0:20 |
| 2 | The belief: "I follow the news, so I can beat the market" | 0:35 |
| 3 | What we built: a market-graded agent (`product_workflow.png`) | 0:25 |
| 4 | Evidence 1: costs set a hurdle before you start (`cost_burden.png`) | 0:40 |
| 5 | Even a selective agent rarely beats the closing price (`trade_funnel.png`) | 0:25 |
| 6 | Evidence 2: the price moves before the news reaches you (`price_discovery.png`) | 0:40 |
| 7 | Evidence 3: the market beats our model (`m4_vs_market.png`) | 0:30 |
| 8 | Evidence 4: long shots, where the data is weaker (`longshot_calibration.png`) | 0:25 |
| 9 | Evidence 5: not trading beat every strategy (`headline_clv.png`) | 0:35 |
| 10 | Context: who sits on the other side? (general knowledge, not measured) | 0:20 |
| 11 | Our product, reframed: protect the consumer first (`agent_workflow.png`) | 0:35 |
| 12 | Coach: teach why most bets don't clear costs | 0:20 |
| 13 | Honest scope: what "unfair" means here | 0:30 |
| 14 | Live demo: `streamlit run app.py` | 2:30 |
| 15 | Recommendations, next steps and Q&A | 0:30 |
| | **Total** | **about 9:20** |

That leaves about 40 seconds of slack. The 5 minutes of Q&A follow. If the live demo fails, play the backup recording on slide 14.
If time runs short, shorten slides 3, 10 and 12.

## Evidence used (from `evaluation/results/consumer_fairness.md`)

- **Costs:** a $20 order at tip − 60 min pays 1.92¢ per contract over the mid (5.0% of the
  price): half-spread 0.53¢ plus fee 1.39¢. That is 9–14% of the price for contracts under 20¢.
  The full agent made +$36 at the mid but lost $32 after $51 of fees and $17 of spread.
- **Information speed:** on 193 test games where the inactive list moved M4 by at least 1 point,
  the price had moved 1.80 points in the news direction before the list was public and 0.18
  points after it (91% before). The list is stamped tip − 30 min, which is our proxy.
- **The market beats our model:** see the M4 Brier note above.
- **Long-shot bias: not clearly supported.** Only the 0–10¢ bucket hints at it (1 win in 66
  against 7.4% implied, z = −1.8). Long shots at 25¢ or less still returned −13.4%, against
  −2.2% for heavy favourites, because costs are a bigger share of a cheap price.
- **Outcomes:** never trading $0; full agent −$32; anchor without learning −$247; raw model
  −$2,087; buying every contract all season −6.0% ROI.
- **Asymmetry (slide 10)** is general industry knowledge and is labelled as not measured.
