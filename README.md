# NBA Game-Impact Agent

A LangGraph research agent for NBA late news, model forecasts, market comparison
and gated rule learning. The Streamlit demo replays historical nights and uses
paper transactions. See [the project plan and results](docs/project-plan.md).

This layout is based on `Michaelyzr/nba-agent` main commit
`3977a679aa3401e8ad05381d9dad9c91bab47b86`.

## Structure

```text
nba-agent/
├── frontend/
│   ├── app.py                 # Streamlit interface: seven existing tabs
│   ├── inplay_app.py          # Independent read-only in-play monitor
│   └── requirements.txt       # Streamlit and chart dependencies
├── backend/
│   ├── agents/                # Historical, pregame and in-play agent loops
│   ├── forecast/              # Models, in-play training and player experiment
│   ├── data_sources/          # NBA/news/market adapters and table schemas
│   ├── evaluation/            # Scoring, ablations and figure generation code
│   ├── tests/                 # Backend regression tests
│   ├── replay.py              # Historical fills and settlement
│   ├── llm.py                 # Gemini access and offline fallback
│   ├── nba_agent_paths.py     # Shared data, model and output locations
│   ├── pyproject.toml         # Backend package and dependencies
│   └── .env.example           # Optional backend configuration
├── data/sample/               # Committed replay inputs
├── reports/results/           # Committed evaluation tables and figures
├── docs/                      # Architecture and original project plan
├── models/                    # Generated, ignored model weights
├── runs/                      # Generated, ignored decisions and traces
├── rules/                     # Generated, ignored rule notebook
├── requirements.txt           # Installs both parts, from the repository root
└── pytest.ini                 # Discovers backend tests
```

The frontend calls the installed Python backend directly. This is a directory
and dependency refactor: the existing Streamlit UI and agent behavior are
preserved. It does not introduce a React frontend or an HTTP API.

## Setup and demo

Use Python 3.11 or 3.12. Run the following from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m forecast.train --source sample
streamlit run frontend/app.py
```

Trained weights are intentionally not committed. `Forecaster.load()` requires
the win model in `models/m4/win.pkl`; train before opening the demo.

For Gemini, create `backend/.env` from `backend/.env.example` and add your key.
Existing root `.env` files are still read; process environment variables take
precedence, followed by `backend/.env`, then the root `.env`. LLM CLI steps need
`--llm`. The existing demo uses offline rules and displays learning notebooks
from previous runs.

## News monitoring

The main demo includes the pregame news loop. The independent in-play monitor
reads runs written by its own agent:

```bash
python -m agents.inplay --mode replay --source sample --game-id 401811041 --initial-p-home 0.60 --name inplay-demo
streamlit run frontend/inplay_app.py
```

See the [pregame runner instructions](docs/project-plan.md#pregame-news-polling-and-fair-odds),
[in-play guide](docs/inplay.md), and [news source catalog](docs/news_sources.md).
Live X access uses `X_BEARER_TOKEN` for pregame and a separate
`INPLAY_X_BEARER_TOKEN` for in-play, configured in `backend/.env`.
The isolated [player experiment](backend/forecast/player_experiment/README.md)
and its committed results in `reports/results/player_features_v1/` are retained.

## Commands

Install the backend once, then module commands work from any directory:

```bash
python -m agents.graph --source sample --start 2026-02-03 --end 2026-02-03 --name sample-night
python -m agents.graph --source sample --start 2026-02-03 --end 2026-02-03 --llm
python -m agents.graph --forecaster record --draw
python -m evaluation.scorer --policy-tests
python -m evaluation.ablations
pytest -q
```

The committed market sample covers 2026-02-03, 2026-04-10 and 2026-04-12.
For other sample nights, choose a date shown in the demo. Evaluation defaults still use `data/frozen/`;
follow the original project plan to download the full evaluation inputs.

All modules use `backend/nba_agent_paths.py`, so launching from `frontend/`,
`backend/` or the root does not redirect outputs. Set `NBA_AGENT_ROOT` to an
absolute storage or checkout directory when using a wheel or external data.

See [the architecture and migration notes](docs/architecture.md).
