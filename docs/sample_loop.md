# Courtside: choose a bet, compare the price

The English customer interface follows the selected match automatically from
**Pre-game** through **In-play** to **Final**; there is no manual phase switch.
Before tip-off it shows our estimated win chances, live model-versus-Polymarket
odds, the hedge/bet-slip section and retrieved news. The probability chart appears
only after the game starts. Latest match updates remain the final section.
Choose a game and **Moneyline**, **Spread** or **Total points**. Tap an outcome to
add it to **Your bet slip**, enter a stake, and compare our model's chance and fair
odds with Polymarket's reference odds. The slip shows return including stake and
estimated profit on average, rather than shares and acquisition cost.

Use **Find my best single bet** to rank the positive-value choices in the selected
market. Choose **Parlay**, include games, and select two, three or four legs to
find a combination. Recommendations can prioritize **Highest estimated profit**
or **Highest chance to win**. These are different objectives; the largest possible
payout does not necessarily have the best expected return. Users can also build
and remove picks themselves, with one pick per game to avoid treating correlated
same-game outcomes as independent.

## Open the dashboard

```bash
source .venv/bin/activate
python -m agents.dashboard_server
```

Open **http://127.0.0.1:8766**. The [portable demo website](samples/news_loop/demo.html)
works offline too. Its **Live data** button connects to this service on port 8766;
when using another port, open the service URL.

- **Demo** starts before tip-off and includes four games so users can try parlays.
  It auto-plays across tip-off and the four-quarter replay. From the start, Play, Next update
  and the progress slider replay observed updates and stop at final. Scores, news
  incidents and Polymarket-format quotes are synthetic and labeled accordingly.
- The local HTTP website opens in **Live data** and automatically discovers the
  current and upcoming ESPN schedule. Choosing a game determines its phase;
  the selected game stays selected when it starts. Model/news and reference
  fetches run in parallel, with a five-second refresh target. Reconnection is
  automatic; failures clear old recommendations, returns and hedge suggestions.
  Provider latency, access and source coverage still affect update speed.
- **Your stake** updates the depth-aware odds, return and estimated average profit.
  No file upload, hand-entered reference quote, wallet or order submission is needed.
- **Already placed a bet? Protect it**, available before and during play, accepts the original
  pick, amount bet and decimal odds when placed. It compares an opposite-outcome
  hedge in the exact same market and line. It shows the additional stake and lower
  completed-game profit, including whether the hedge is partial or balanced.

The same betting flow is available in Streamlit:

```bash
streamlit run loop_demo_app.py
# Or the main app's News loop sample tab:
streamlit run app.py
```

Streamlit connects directly to the Python controller. Automatic refresh requires
`st.fragment`; older versions refresh through interaction.

## Our model and the reference price

**All probabilities and fair odds come from our models.** Polymarket prices are
read only after our prediction and never enter probability features, priors or
news effects. Repricing a reference changes estimated value, not our forecast.

The existing PregameAgent supplies a historical prior with availability/news
updates. The independent InPlayAgent supplies score, clock and incident updates.
`forecast/betting.py` builds an own-history final-score distribution from completed
games before tip-off, excluding the current game's final result. It estimates
full-game spread and total probabilities at the **same line** as the reference.
The winner probability anchors the margin distribution; observed pace and
remaining time update totals, with prototype news adjustments. Half-point lines
avoid presenting a push as a loss. If a matching reference is unavailable, model
options remain visible but do not become betting recommendations.

These spread/total distributions and news weights are **research prototypes**,
not separately calibrated betting models. Win-loss priors, normal score tails,
pace blending and injury effects need validation on historical observations.
The page labels these estimates and does not invent accuracy for a replay.

Our fair decimal odds are `1 / model probability`. Comparison cards show Polymarket ask odds `1 / best ask` before fees. The bet slip
uses the winning payout divided by all-in entry cost at the user's stake. Calculations
walk actual ask depth and include verified price-dependent fees and a $0.002
per-unit cost reserve. Positive-value recommendations require a two percentage-point
probability stress buffer and at least one percentage point of remaining edge.
That buffer is an assumption, not a confidence interval. Profit on average is
probability times winning return minus actual cost; it is not guaranteed profit.

The parlay search considers every eligible combination in the selected slate
(up to twelve games), selected bet types and leg count. Different games are
assumed independent. It multiplies our model probabilities and the reference
single-bet odds as a **planning benchmark**, not an executable combined quote.
No combined-market API or actual parlay price is fetched. Confirm an actual
combined quote and its settlement rules before acting. Recommended combinations
maximize the selected metric within this bounded set, not every available market.

For an existing bet, the opposite hedge is capped at the original potential
payout and available budget/depth. It balances two completed-game outcomes:

```
Original pick wins: original stake × placed odds − original stake − hedge cost
Opposite pick wins: opposite payout − original stake − hedge cost
```

A partial hedge leaves exposure. Even an improved lower profit can remain negative.
Cancellation is a separate settlement case governed by the exact contract.
The model is an estimator, not a second trading venue, and no orders are placed.

## Automatic live API connection

`agents/dashboard_live.py` creates independent PregameAgent and InPlayAgent state
for each match under `runs/dashboard/`, with five-second caching shared across
viewers. Per-game locks serialize updates while different games can refresh in
parallel. Separate phase state prevents pre-game news from mutating the in-play
agent. Model observation time is retained independently of comparison time;
quotes and forecasts older than 15 seconds cannot produce a current recommendation.
Past in-play observations alone form each live chart; demo values never populate
live charts. Retrieved articles and X posts are displayed with publication time and
source links, including relevant news that does not create a model factor. News
is deduplicated and filtered by game, publication and observation time.

`data_sources/polymarket_books.py` uses public read-only market-data APIs:

1. Gamma `GET /events/slug/{slug}`, then exact catalog lookup when needed.
2. Match NBA teams, scheduled tip time within 60 seconds, full-game market type,
   condition ID and outcome tokens. No fuzzy pairing or quarter/half substitution.
3. Choose one most-liquid open line per type: winner, spread and total.
4. CLOB `POST /books` requests all selected outcome books in one **read-only** batch;
   response tokens/conditions are matched individually regardless of ordering.
5. `GET /clob-markets/{condition_id}` retrieves fee parameters. Unknown fees block
   comparisons. The existing single-winner reader remains available separately.

Winner contracts must explicitly state final score including overtime, postponement
through completion and 50-50 canceled-game resolution. Spread/total contracts must
match recognized full-game half-point rule templates and scoring thresholds; the
full-game type is treated as including overtime. Unknown rules, wrong dates,
closed/suspended markets, invalid books and missing outcome pairs are rejected.
Both book source time and local fetch time are checked: fetching an old book does
not make it current. Matched older prices may display **Last seen**, but cannot
produce an average-profit estimate, recommendation or hedge action.

Public market data needs no wallet or API token. X news coverage still requires
appropriate access and configured tokens; unavailable feeds are shown as incomplete
coverage. Local historical statistics come from complete `data/frozen/` inputs,
otherwise `data/sample/`. Update these records before evaluating live model accuracy.

References: [Gamma event lookup](https://docs.polymarket.com/api-reference/events/get-event-by-slug),
[batch order books](https://docs.polymarket.com/api-reference/market-data/get-order-books-request-body),
[fees](https://docs.polymarket.com/trading/fees),
[resolution](https://docs.polymarket.com/concepts/resolution).

## Reproduce and inspect the sample

```bash
python -m agents.loop_demo --output docs/samples/news_loop
```

The primary BOS–ORL match runs the actual independent agents: 5 pre-game polls and
58 in-play polls covering Q1–Q4, injury exit, ruled-out update, return/correction,
opponent ejection, a lower-priority conflict, the last pre-final second and final
settlement. Scores interpolate synthetic quarter totals, not real play-by-play.
Three additional matches use synthetic score projections from their own historical
priors to demonstrate cross-game parlays; they are not additional real live feeds.
Their synthetic prices lag past model estimates. No future scores or news are
used to calculate earlier forecasts.

Outputs include English `demo.html`, `report.md`, schema-v3 `result.json`,
`timeline.csv`, overview figures and independent agent logs. Forecast audits,
sensitivities and research details remain outside the main customer flow.
Generating the replay never fetches live prices or mutates existing run state.

For calibrated winner-model research, the existing training module offers logistic
regression and monotonic histogram gradient boosting, chronological whole-game
train/calibration/test splits, sigmoid calibration, Brier score and log loss:

```bash
python -m forecast.inplay_training \
  --training-data snapshots.parquet \
  --train-end 2026-02-01 --calibration-end 2026-03-01 \
  --candidate logistic --output models/inplay/logistic-calibrated.pkl
python -m agents.dashboard_server --model-file models/inplay/logistic-calibrated.pkl
```

Only load trusted model files. Exported candidates remain `development_only` until
independent validation and execution-cost testing; training alone does not prove
profitable betting or validate the spread/total extensions.
