# Functional equivalence verification

Baseline: `Michaelyzr/nba-agent` main commit
`9944d71ed1a2e0a840753aef5b0904c9c68f559c`. Checked on 2026-10-06.

The baseline and the proposed integration were run separately against the same
inputs, Python 3.12 environment, dependency versions and controlled API responses.
The user's checkout was not changed during this verification.

- Untouched latest main: **205 tests passed**. Integrated layout: **207 tests
  passed**, including every original test and two storage/install regression cases.
- All frontend function/class bodies are unchanged from latest main. The complete
  Polymarket client, live provider, Agent adapter and updated pregame integration
  are byte-identical after relocation. Original test bodies are retained, with
  imports adjusted for the installed package layout.
  Production differences were reviewed: shared paths, compatible dotenv lookup,
  model/output path defaults and extraction of the synthetic policy fixture.
- Before/after results were exactly equal in 12 categories: trained M4
  coefficients and forecast scenarios; replay decisions, fills and agent traces;
  policy results; live market healthy/empty/unavailable states; player-experiment
  features, quantiles, raw probabilities and calibrated probabilities; the new
  live Agent tables, game matching, price history, AsOf quotes and failure gate.
- Independently trained M4 pickle files were byte-identical. The installed
  refactored API also loaded the original baseline model file successfully.
- Pregame and in-play replay CLIs ran from outside their checkouts. Every generated
  state, snapshot and evidence file matched after normalizing the storage root.
- Streamlit AppTest compared 10 main-page states: healthy, empty and unavailable
  live markets and manual refresh; outcome filtering; the historical seven-tab
  page; historical-night execution and trained-M4 pregame execution. Titles,
  messages, controls, metrics and table contents matched. Only storage roots and
  the intentionally random per-run UUID were normalized.
- The independent in-play monitor read the replay logs and refreshed without
  exceptions; displayed states, metrics and tables matched the original page.
- All **33** committed sample and evaluation artifacts remained byte-identical,
  including the isolated player experiment results.

Live API responses were controlled for reproducible comparison. This verifies
functional equivalence under the tested inputs, not all future external API
responses, credentials, operating systems or production data sets. No live orders
were executed. Generated QA data, weights and caches are excluded from the patch.
