# NBA Agent backend

The installable Python backend contains the LangGraph agents, forecast models,
data adapters, replay engine and evaluation code. It is called by the Streamlit
frontend and by the existing command-line modules.

From the repository root:

```bash
python -m pip install -e './backend[dev]'
python -m agents.graph --source sample --start 2026-02-03 --end 2026-02-03
```

See [the project README](../README.md) for model training and the demo setup.
Runtime paths are defined in `nba_agent_paths.py`, and tests live in `tests/`.
Model serialization module names (`forecast.*`) remain unchanged.
