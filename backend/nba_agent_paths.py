"""Shared storage locations, independent of the current working directory.

Editable installs use the checkout root. Set NBA_AGENT_ROOT to the checkout or
storage root when installing a wheel or using an external data directory.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NBA_AGENT_ROOT", Path(__file__).resolve().parent.parent)).expanduser().resolve()
DATA = ROOT / "data"
FROZEN = DATA / "frozen"
RAW = DATA / "raw"
SAMPLE = DATA / "sample"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
NOTEBOOK = ROOT / "rules" / "notebook.json"
RESULTS = ROOT / "reports" / "results"
