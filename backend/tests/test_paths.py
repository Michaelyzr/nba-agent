"""Storage and package imports must survive launches outside the checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("external_storage", [False, True])
def test_installed_backend_uses_shared_storage_from_another_directory(tmp_path, external_storage):
    checkout = Path(__file__).resolve().parents[2]
    expected = tmp_path / "storage" if external_storage else checkout
    env = os.environ.copy()
    env.pop("NBA_AGENT_ROOT", None)
    if external_storage:
        env["NBA_AGENT_ROOT"] = str(expected)
    script = """
import json
from pathlib import Path
import data_sources
import replay
from agents import notebook
from forecast import api, baselines, inplay
from forecast.player_experiment import benchmark
from agents.inplay import INPLAY_RUNS
from data_sources.news_registry import NewsRegistry
import nba_agent_paths as paths

expected = Path(__import__('sys').argv[1])
assert paths.ROOT == expected
assert data_sources.ROOT == expected
assert data_sources.FROZEN == expected / 'data' / 'frozen'
assert data_sources.RAW == expected / 'data' / 'raw'
assert replay.RUNS == expected / 'runs'
assert notebook.NOTEBOOK == expected / 'rules' / 'notebook.json'
assert api.MODELS == expected / 'models'
assert baselines.MODEL_DIR == api.MODELS / 'm1'
assert paths.RESULTS == expected / 'reports' / 'results'
assert inplay.MODELS == expected / 'models'
assert INPLAY_RUNS == expected / 'runs' / 'inplay'
assert benchmark.DATA == expected / 'data'
assert benchmark.RUNS == expected / 'runs'
assert len(NewsRegistry.load().data['teams']) == 30
print(json.dumps({'root': str(paths.ROOT)}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(expected)], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["root"] == str(expected)
