# Architecture and migration

## Responsibilities

| Location | Responsibility |
| --- | --- |
| `frontend/` | Streamlit inputs, session state, tables, charts and display |
| `backend/agents/` | Agent orchestration, policy checks, rule gate, briefs and coach |
| `backend/forecast/` | Training, features and numeric forecast distributions |
| `backend/data_sources/` | Downloading, parsing and timestamped data schemas |
| `backend/replay.py` | As-of views, paper fills and settlement |
| `backend/evaluation/` | Scoring and reproducible evaluation commands |
| `data/`, `models/`, `runs/`, `rules/` | Shared input and runtime storage |
| `reports/results/` | Evaluation artifacts, separated from Python source |

The Streamlit process imports the installed backend. An HTTP service and a
JavaScript client can be added later, but they are not part of this refactor.
The frontend retains the live Polymarket dashboard and the historical demo
with paper transactions.

## Changes from the original main branch

| Original location | New location |
| --- | --- |
| `app.py` | `frontend/app.py` |
| `inplay_app.py` | `frontend/inplay_app.py` |
| `agents/`, `forecast/`, `data_sources/` | Same directories under `backend/` |
| `replay.py`, `llm.py` | Same modules under `backend/` |
| `evaluation/*.py` | `backend/evaluation/*.py` |
| `evaluation/results/` | `reports/results/` |
| `tests/` | `backend/tests/` |
| `pyproject.toml`, `.env.example` | `backend/` |
| Original long `README.md` | `docs/project-plan.md` |

## Compatibility

- Run `python -m pip install -r requirements.txt` from the repository root.
  The editable backend installation replaces reliance on the root import path.
- Python module names remain `agents.*`, `forecast.*`, `data_sources.*`,
  `evaluation.*`, `llm` and `replay`. Existing model pickle class paths are
  preserved; keep model weights under the root `models/`.
- Run the demo with `streamlit run frontend/app.py`. Use `python -m replay`
  instead of `python replay.py`. Other module commands remain unchanged.
- Existing `data/`, `models/` and `runs/` stay where they were. Move prior
  `evaluation/results/` artifacts to `reports/results/` when migrating a
  checkout containing local generated results.
- `backend/.env` is preferred. Existing root `.env` remains supported.
- Shared storage paths are independent of the current working directory.
  Installed wheels should set `NBA_AGENT_ROOT` to the data checkout/storage root.
- The synthetic slate used by the Safety tab now lives in
  `backend/evaluation/fixtures.py`. Runtime evaluation code no longer imports
  the test suite, so the demo works with the installed backend package.

The pregame and in-play command entry points, source catalog package data,
player experiment modules, live Polymarket client/UI and committed results from
main `9944d71` are retained.
Live Polymarket modules retain the upstream adapter/provider and its Agent
context; current quotes always come from each refresh.
The main Streamlit page keeps seven tabs; the in-play monitor remains separate.

## Existing limits

Directory separation does not add live order execution, a daily-loss stop,
authentication, independently deployable HTTP services or a new UI. Those need
separate feature work. The Streamlit agent still defaults to offline rules and
loads learning history from saved runs.
