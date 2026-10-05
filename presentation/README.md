# Presentation deck (Fri 9 Oct 2026)

`nba_agent_deck.pptx` is a 16:9 deck of 15 slides with speaker notes on every slide.
It is built by `build_deck.py` from the figures in `figures/`. Those figures are
copies of `evaluation/results/*.png`, so this folder is self-contained.

## Rebuild

```bash
pip install python-pptx pillow
python presentation/build_deck.py           # writes presentation/nba_agent_deck.pptx
python presentation/build_deck.py --check   # also prints the estimated fit of every text box
```

The script warns, and exits with status 1, if a shape leaves the slide, a text box is
estimated to overflow, or a slide has no speaker notes. If you regenerate the figures in
`evaluation/results/`, copy them into `figures/` again and rebuild.

All numbers come from the main README ("Results") and `evaluation/results/`.

## Slides and a 10-minute timing

| # | Slide | Time |
| --- | --- | --- |
| 1 | Title | 0:20 |
| 2 | The problem: late news moves win probabilities | 0:40 |
| 3 | The whole product, end to end (`product_workflow.png`) | 0:40 |
| 4 | Models M1–M5: only the M4 win model drives trades | 0:40 |
| 5 | Our win model vs the market (`m4_vs_market.png`) | 0:30 |
| 6 | The agent loop: Decide, then nightly Review (`agent_workflow.png`) | 0:40 |
| 7 | How one trade is decided (`trade_flow.png`) | 0:30 |
| 8 | Worked example: CLE @ POR, 1 Feb 2026 (`trade_example.png`) | 0:30 |
| 9 | Learning: propose one rule a day, keep it only if it helps | 0:30 |
| 10 | Results on the test period (`headline_clv.png`) | 0:40 |
| 11 | Selectivity: from 771 decision points to 74 trades (`trade_funnel.png`) | 0:20 |
| 12 | Safety and honest caveats | 0:30 |
| 13 | Coach: learning how the market works | 0:20 |
| 14 | Live demo: `streamlit run app.py` | 2:30 |
| 15 | Conclusion, next steps and Q&A | 0:20 |
| | **Total** | **about 10:00** |

Slides 1–13 take about 6:50, the live demo 2:30 and the conclusion 0:20 (9:40), which
leaves about 20 seconds of slack. The 5 minutes of Q&A follow. If the live demo
fails, play the backup recording on slide 14. If time runs short, shorten slides 11 and 13.
