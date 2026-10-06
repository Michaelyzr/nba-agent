# Refactor verification

Source: `Michaelyzr/nba-agent` main commit
`3977a679aa3401e8ad05381d9dad9c91bab47b86`. Checked on 2026-10-06.

- Full backend regression suite: **182 passed**, including the latest pregame,
  in-play, news-source and player-experiment tests.
- Backend editable installation and wheel build succeeded in a temporary
  Python 3.12 environment; `pip check` found no broken requirements.
- The wheel includes the news source JSON, policy fixtures and all three CLI
  entry points (`nba-agent`, `nba-pregame`, `nba-inplay`), and excludes tests.
- Storage regression cases imported the installed backend from outside the
  checkout, with both default paths and a custom `NBA_AGENT_ROOT`; the news
  catalog loaded all 30 teams from the moved package.
- All **9** planted policy violations were blocked by the evaluation CLI.
- Pregame replay CLI consumed sample news and stopped at tip-off. The independent
  in-play CLI completed its synthetic score/news replay through the final score.
  Both were launched outside the checkout, with no live network requests.
- Streamlit AppTest loaded all **7 main-page tabs**, exercised the historical
  night button and the pregame replay button, and loaded the independent in-play
  monitor with its three probability/odds metrics. The night used a constant
  win-model stub with real sample tables and real agent/replay code; pregame
  used its existing record baseline. Full model training was not repeated.
- SHA-256 checks confirmed that all **33** committed sample tables and evaluation
  artifacts were unchanged after relocation, including the player experiment.

The archive excludes generated QA runs, caches, package metadata and model
weights. Train models before starting the main demo in a fresh checkout.
This verification does not establish live source availability or model accuracy.
