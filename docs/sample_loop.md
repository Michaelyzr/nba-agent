# Courtside: match updates and market actions

The customer page is in English and focuses on four things: the score and game
clock, model versus Polymarket odds, a next action, and recent player updates.
The main view has one probability chart. Technical reports, raw inputs, model
comparisons and training remain available separately from the customer page.

## Open the dashboard

```bash
source .venv/bin/activate
python -m agents.dashboard_server
```

Open **http://127.0.0.1:8766**. Choose **Demo replay** or **Live games**.
The standalone [demo.html](samples/news_loop/demo.html) also works directly from
its file; its Live games button connects to this local service on port 8766.
When serving on a different port, open the service URL instead of the file.

- **Demo replay** needs no network, uploads, credentials or wallet. Scores,
  incidents and Polymarket-format prices are simulated. It initially shows
  Q2 06:00, when an injury update changes the model's estimated probability.
  From tip-off, Play, Next update and the progress slider replay all four quarters;
  the display uses information observed up to the selected step and stops at final.
- **Live games** discovers the current ESPN NBA schedule and automatically finds
  each selected game's Polymarket full-game winner market. It refreshes every five
  seconds while this mode is open. Failed connections retry; stale actions clear.
  This is HTTP polling; provider latency and news coverage still matter. Five
  seconds is a refresh target, not a guarantee of end-to-end detection latency.
- **Available budget** updates the entry-cost and action calculations. Enter
  **My position (optional)** to compare holding, buying the opposite outcome and
  selling held shares. Positions reset on match changes in the browser. No upload
  or hand-entered market quote is required. Nothing places trades.

Streamlit also presents the same English customer view:

```bash
streamlit run loop_demo_app.py
# The main app's News loop sample tab uses the shared view:
streamlit run app.py
```

Streamlit connects directly to the same Python controller; it does not require
the HTTP service. Automatic refresh needs a Streamlit release with `st.fragment`;
older releases can refresh by interacting with the page.

## How prices and actions work

**Our odds** are fair decimal odds `1 / model probability`. Polymarket pays one
dollar per winning share; its decimal equivalent is shares divided by the all-in
purchase cost. The table walks available asks for the chosen budget and includes
price-dependent fees and a $0.002 per-share cost reserve. The chart uses midpoint
prices for comparison, rather than treating them as available entry prices.

Without a held position, the page compares buying either outcome with waiting.
It ranks candidates with positive expected value after a two percentage-point
probability stress buffer and at least one percentage point of remaining edge.
The buffer is a modeling assumption, not a confidence interval. Demo actions are
explicitly hypothetical. Live buy signals require a validated model, current
matching quotes and healthy score/news coverage. The default model is a research
prototype, so live prices may be shown while the suggested action remains **Wait**.
Before tip-off, the independent in-play model waits for current game data;
Polymarket prices can already be displayed.

A model probability is an estimate, not a second venue on which a hedge can be
traded. With a held position, protection uses real outcome shares and current
bids/asks in the **same binary market**. Buying opposite shares is capped at held
shares and the available cash budget; selling is capped at owned shares and bid
depth. The page chooses the option that maximizes the lower P/L of the two
completed-game outcomes, breaking ties by added cash required. This protects a
chosen downside objective; it does not promise the highest eventual profit.

For `N` held shares with original cost `C`, and `H` opposite shares costing `K`:

```
Held team wins:   N - C - K
Other team wins:  H - C - K
```

Equal shares balance these two payouts. Partial hedges retain exposure; a better
lower payout can still be negative. Cancellation is a separate settlement case
and follows the particular market's rules. The optional explanation shows the
outcome table so users can review this tradeoff before acting externally.

## Automatic API connection

`agents/dashboard_live.py` creates a separate live InPlayAgent for each selected
game, with independent run state under `runs/dashboard/live-<id>/<game-id>/`.
It uses the existing NBA/ESPN score and news providers, with a dedicated
five-second media polling interval for this dashboard, polls the model and market
in parallel, retains past observations for the chart, and caches source fetches
for five seconds across viewers. Changing the budget or position recomputes the
view without rerunning cached providers. Pregame agent state stays separate.

`data_sources/polymarket_books.py` only calls public GET endpoints:

1. Gamma `/events/slug/{slug}` with the scheduled NBA matchup and Eastern date;
   if needed, scan the NBA event catalog for one exact match.
2. Match the two team names, scheduled tip time (within 60 seconds),
   `sportsMarketType=moneyline`, condition ID and outcome tokens. No fuzzy pairing.
3. CLOB `/clob-markets/{condition_id}` for fee parameters and `/book?token_id=...`
   for each outcome's bids, asks, depth and timestamps.

The reader recognizes the standard NBA description only when it states a final
score including overtime, postponement through completion, and 50-50 canceled-game
resolution. Unrecognized rules remain unverified and block price actions.
Closed/suspended markets, unknown or unsupported fees, mismatched game identity,
crossed books, future timestamps and prices older than 15 seconds are rejected.
Discovery metadata is cached for 20 seconds; order books are fetched every poll.
Both source time and local fetch time are retained: a newly fetched old book does
not become fresh. Old but otherwise matched prices may appear with **Last seen**
for reference; they never produce an entry edge or position-trade recommendation. The model observation time is retained separately from the
combined comparison time. If a refresh takes too long, the model becomes stale.

Public price queries require no API token or wallet. The existing X news feed
still needs `INPLAY_X_BEARER_TOKEN` and appropriate access; without it, news
coverage is marked degraded. This is live API integration, not a guarantee that
all news is seen immediately or that the prototype's probabilities are calibrated.
Local historical statistics come from `data/frozen/` when complete, otherwise
`data/sample/`. Refresh those inputs and calibrate models before evaluating live
forecast accuracy.

References: [Gamma event lookup](https://docs.polymarket.com/api-reference/events/get-event-by-slug),
[CLOB order books](https://docs.polymarket.com/api-reference/market-data/get-order-book),
[fees](https://docs.polymarket.com/trading/fees),
[resolution](https://docs.polymarket.com/concepts/resolution).

## Reproduce the sample and inspect the research outputs

```bash
python -m agents.loop_demo --output docs/samples/news_loop
```

This runs the actual PregameAgent and InPlayAgent with separate fresh temporary
state: 5 pregame polls and 58 in-play polls covering four quarters, injury exit,
ruled-out update, return/correction, opponent ejection, a lower-priority conflict,
the last pre-final second and final settlement. Scores interpolate synthetic
quarter totals; this is not actual NBA play-by-play. The historical records are
real local inputs; news, scores, source personas and all market quotes are synthetic.
Probability differences describe model effects at the same score and clock,
not verified causal impacts. Final settlement is excluded from forecast curves.

Outputs include the English `demo.html`, full `report.md`, schema-v3 `result.json`,
`timeline.csv`, overview figures and independent pregame/in-play logs. Full raw
inputs, forecast audits and parameter sensitivities remain in the exported data,
not in the primary customer page. Generating the sample never fetches live prices
or changes existing agent run state.

### Model research

The diffusion prototype uses a historical prior, score/time and estimated
remaining player availability. Fixed variance, replacement-value and injury
weights need calibration on real in-game observations. The training module offers
logistic regression and monotonic histogram gradient boosting with chronological,
whole-game train/calibration/test splits, sigmoid calibration, game-balanced
weights, Brier score, log loss and reliability bins:

```bash
python -m forecast.inplay_training \
  --training-data snapshots.parquet \
  --train-end 2026-02-01 --calibration-end 2026-03-01 \
  --candidate logistic --output models/inplay/logistic-calibrated.pkl
# --candidate hgb for monotonic boosting
python -m agents.dashboard_server --model-file models/inplay/logistic-calibrated.pkl
```

Only load trusted local model files. Exported models remain `development_only`
until further independent validation and market execution-cost testing. Training
success does not establish profitable trades. The page never invents accuracy or
calibration metrics for the synthetic replay.
