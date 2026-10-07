# Presentation deck (Fri 9 Oct 2026)

**Narrative: "Market-Graded Learning".** The market's closing price grades every component of the
agent: the models, the learned rules and the LLM. Only the components that defer to the market
survive (the market anchor, and gated learning that trades less); every component with more
freedom (the raw win model, the M6 neural net) fails. If a disciplined agent with our data cannot
beat the close, a consumer cannot either, so the product educates and protects consumers. The
deck opens with a three-part consumer problem: betting is mainstream, the markets are structurally
tilted (our measurement) and betting harms well-being (published research). "Unfair" here means
structural disadvantages (costs, speed, expertise), not manipulation or fraud. We make no
accusation against Kalshi. The framing follows `docs/agentic_review.md`, "How to present the agent".

`nba_agent_deck.pptx` is a 16:9 deck of 15 slides with speaker notes on every slide. It is
built by `build_deck.py` from the figures in `figures/`, which are copies of
`evaluation/results/*.png`, so this folder is self-contained. The rehearsal script with per-slide
timing is `script.md`.

## Rebuild

```bash
python -m evaluation.consumer_fairness      # costs, price discovery, long-shot check (needs data/frozen)
python -m evaluation.cumulative_pnl         # cumulative P&L chart from the saved runs in runs/
python -m evaluation.data_overview          # data table counts (data_overview.csv)
cp evaluation/results/{product_workflow,cost_burden,price_discovery,m4_vs_market,m6_vs_baselines,cumulative_pnl,agent_workflow}.png presentation/figures/
# figures/league_reveal.png is a screenshot of the League tab (streamlit run app.py, demo league friday-league)
python presentation/build_deck.py           # writes presentation/nba_agent_deck.pptx
python presentation/build_deck.py --check   # also prints the estimated fit of every text box and table
```

The build copies `injury_timing.png`, `m6_robustness.png` and `orchestrator.png` from
`evaluation/results/` into `figures/` automatically when they exist (or are newer). Slide 11 shows
`orchestrator.png` next to the table only if that file exists.

Every number we measured is in the `K` dict at the top of `build_deck.py`, with its source. The
data counts on slide 5 are in the `D` dict (measured by `evaluation/data_overview.py`), and the
external research numbers on slide 2 are in a separate `EXT` dict, with their references.
The script warns, and exits with status 1, if a shape leaves the slide, a text box or table is
estimated to overflow, or a slide has no speaker notes.

**Agentic upgrades (slide 11).** `K["agentic"]` (defined just after `K`) has one row per upgrade:
name, what it is, the result files in `evaluation/results/` it comes from, and the result (`None`
shows "pending"; a result starting "Not run" is shown muted). The numbers in the results are `K`
keys (`ga_*` gate audit, `pl_rate` placebo, `co_*` Coach, `llm_*` LLM agent), so updating a number
changes the table and the speaker notes together. The build prints which rows are filled, which
are pending, and which pending rows already have a result file. The slide closes with the
`TRADE_LESS` message, which is also the last line of the speaker notes.

**Filling in the LLM tool agent later (one line).** When `evaluation/results/llm_agent.md` has the
DeepSeek tool-agent arms, set one key in `K` and rebuild:

```python
"llm_tool": "8 trades, CLV $ −$2 [−5, +1] vs never; sceptic vetoed 3 of 9",   # was None ("pending")
```

The "LLM tool agent + sceptic" row then shows that text instead of a muted "pending", the speaker
notes say "The tool agent with the sceptic scored <your text>.", and the build stops listing it as
pending. Keep it to about 100 characters (two lines); `--check` warns if it does not fit. If the
plain-LLM or deterministic numbers change in the rerun, update `llm_plain_n`, `llm_plain_clv`,
`llm_det_n`, `llm_det_clv` and `llm_points` too, and the slide 11 line in `script.md`.

**M4 Brier score:** slide 8 uses the README's definition, M4 0.186 against the market's 0.164,
both 1 hour before tip on 501 test games, and footnotes the at-tip pair (M4 with the inactive
list 0.183, market at tip 0.163). The recalibration line on slide 9 uses the 0.183 definition
(absences known), as in `m4_calibration.md`.

## Slides and timing (target: about 10 minutes including the demo)

The time limit has not been confirmed, so adjust the timings if needed. Full wording: `script.md`.

| # | Slide | Time |
| --- | --- | --- |
| 1 | Title: Market-Graded Learning (the thesis in one sentence) | 0:15 |
| 2 | The problem: three things consumers face | 0:35 |
| 3 | Research question, what we built (`product_workflow.png`), and the answer | 0:30 |
| 4 | How the agent works: decide loop, learn loop, people, guardrails, market as grader (`agent_architecture.png`, from `make_agent_architecture.py`) | 0:45 |
| 5 | Market-Graded Learning: the closing price is the grader (component / test / verdict table) | 0:35 |
| 6 | Data: five public sources, split in time, no look-ahead | 0:25 |
| 7 | Evidence 1: costs, and a price that moves before the news (`cost_burden.png`, `price_discovery.png`) | 0:45 |
| 8 | Evidence 2: the market beats our model (`m4_vs_market.png`) | 0:20 |
| 9 | Deep learning: M6 GRU design, result, robustness grid; M4 recalibration (`m6_robustness.png`, `m6_vs_baselines.png`) | 0:45 |
| 10 | Evidence 3: money over time and the walk-forward CIs (`cumulative_pnl.png`, table) | 0:35 |
| 11 | Agentic upgrades: the market's answer is "trade less" (table, `orchestrator.png`, closing message) | 0:45 |
| 12 | Our product: protect, teach, practise (agent, Coach, League; `league_reveal.png`) | 0:30 |
| 13 | Why agentic (each box of slide 4 answers the problem) and team contributions | 0:35 |
| 14 | Live demo: `streamlit run app.py` | 2:00 |
| 15 | Recommendations, honest scope and Q&A | 0:30 |
| | **Total** | **about 9:50** |

That leaves about 10 seconds for transitions before the 5 minutes of Q&A.

Changes from the 17-slide version: the title now states the thesis and a new slide 4 shows the
market grading every component. The two data slides are merged (the timeline figure is dropped;
splits and rules are bullets). The walk-forward CI slide is merged into the cumulative P&L slide.
The product, Coach and League slides are one slide. Honest scope is merged into Recommendations.
The agent architecture slide 4 was added and the costs and information-speed slides were merged
into slide 7 (`injury_timing.png` moves to Q&A backup; its numbers stay in the bullets). New: the injury-report timing on slide 7, the deep-learning slide 9, agentic upgrades on slide 11
and team contributions on slide 13. `figures/walkforward.png`, `data_timeline.png` and
`longshot_calibration.png` are kept as backups for Q&A. If time is short, shorten slide 6 or skip
the League call in the demo. If the live demo fails, play the backup recording on slide 14.

## Data (slide 5)

Measured from `data/frozen/*.parquet` by `evaluation/data_overview.py` (`evaluation/results/data_overview.csv`).
All sources are public; the raw data is gitignored and not redistributed (`data/sample/` holds a small
subset). Rebuild with `python -m data_sources.espn` and
`python -m data_sources.kalshi download --start 2025-10-21 --end 2026-06-30 --series KXNBAGAME`.

| Table | Source | Rows | Coverage |
| --- | --- | --- | --- |
| `prices` | Kalshi public trade API, historical 1-minute candlesticks (bid, ask, volume), series KXNBAGAME | 3,938,368 | 20 Oct 2025 – 14 Jun 2026; 2,632 markets on 1,316 games (1,229 regular season, 87 play-in and play-off) |
| `markets`, `settlements` | Kalshi: market → game, and the yes / no outcome | 2,632 / 2,634 | 2025-26 |
| `games` | ESPN public site API: schedule, tip time, season type, score | 3,962 | 2023-24 to 2025-26 |
| `player_games`, `players` | ESPN box scores: minutes, points, shots per player | 85,422 / 818 | 3 seasons |
| `news` | ESPN inactive lists (who is out and why), stamped tip − 30 min | 3,371 | 1,772 games; last list 7 May 2026 |

Splits: M1–M4 train on games before 1 Feb 2026 (two earlier seasons plus 2025-26 to 31 Jan). Agent
development 1 Nov – 31 Jan; test 1 Feb – 12 Apr (501 games); walk-forward 1 Nov – 12 Apr with M4
retrained monthly; M6 trains before 15 Jan and validates on 15–31 Jan; holdout = the 87 play-in and
play-off games, 14 Apr – 13 Jun. Replay rules (`replay.py`): as-of view only, fills at the recorded ask
plus the Kalshi fee, quotes at most 5 minutes old, size at most 10% of the last hour's volume.

Prices are 1-minute candles rather than individual ticks. The price table has 2,640 tickers and the
settlement table 2,634; the 8 and 2 without a game market row are not used.

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
- **Injury-report timing** (`injury_timing.md`): on the same 193 games, about two-thirds of the
  move, 67% [47%, 85%], came before the first official NBA injury report listing the key absent
  player Out/Doubtful (report counted public 15 minutes after its slot; 49% with no lag). The
  report comes a median of 6.0 hours before the inactive list. 77 of the reports predate the 24 h
  anchor, so their pre-report share is zero by construction. Official reports only; beat reporters
  and social media can be earlier.
- **M4 recalibration** (`m4_calibration.md`): Brier 0.183 raw, 0.185 Platt, 0.184 isotonic, against
  0.164 for the market 1 h before tip. M4's out-of-sample Platt slope is below 1 in Oct–Jan and
  above 1 in Feb–Apr, so a pooled calibrator fits neither.
- **M6 robustness** (`m6_robustness.md`): 9 configurations (GRU 16/64/128 units, 6/12/24 h
  history, static-feature MLP, 1D CNN) × 3 seeds. All are worse than zero move; all 18 gap CIs lie
  above zero; best gap +0.017¢ (GRU-64, 12 h). None predicts the 5.5¢ move needed to trade.
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
- **Money over time** (`cumulative_pnl.csv`, from `evaluation/cumulative_pnl.py`): cumulative
  P&L after fees by game date, with never trading at $0 as the reference line. Test period
  1 Feb – 12 Apr: full agent −$32 (74 trades, lowest point −$131), no learning −$247 (129), raw
  model −$2,087 (366), M6 agent $0 (0 trades). Walk-forward Nov – 12 Apr: −$25 / −$407 / −$2,475,
  the same as `walkforward_summary.md`. The dotted lines are the cumulative closing-line value of
  the same fills (as in `headline_clv.png`), below zero for every setup. The raw model's steady
  slope is costs. Learning flattens the line by trading less, not by finding profit.
- **Long-shot bias: not clearly supported** (1 win in 66 against 7.4% implied, z = −1.8). Long
  shots still return −13.4% against −2.2% for heavy favourites, because of costs.
- **League (slide 12, `agents/league.py`):** a $1,000 play-money bankroll, stakes of $5–$50, a seeded
  slate of 10 real decision points from the test period, and the same fills as the agent (ask + fee).
  Players are ranked by mean CLV per contract once they have at least 5 trades. The skill-or-luck badge
  comes from the day-clustered bootstrap in `evaluation/stats.py`. The example leaderboard and the
  screenshot come from the demo league `friday-league`. Its two users' plays are illustrative, not a
  measurement; the house bots ran offline on the same slate. In it the raw-model bot is up +$15.27 with
  a mean CLV of −0.63¢, so its badge is "costs".
- **Team contributions (slide 13):** names are as the authors appear on the GitHub PRs
  (`K["team"]`); edit there if the team prefers full names.
- **Disclosure (slide 15):** the market anchor and the reviewer were designed after seeing the
  Feb–Apr results, so that window is not a clean holdout. The walk-forward season and the
  play-off holdout are the honest checks.
