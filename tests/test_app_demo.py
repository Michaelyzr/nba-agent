"""Demo page selection and a Streamlit AppTest smoke test of the historical page with the Coach agent.

The Live Markets page needs the network, so its default is checked through the pure helper only.
"""
import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _helper():
    """Load PAGES and resolve_default_page from app.py without running the Streamlit script."""
    tree = ast.parse(APP.read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "resolve_default_page")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PAGES" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(APP), "exec"), ns)
    return ns["resolve_default_page"], ns["PAGES"]


def test_default_page_is_live_unless_asked_for_history():
    resolve, pages = _helper()
    live, history = pages
    assert resolve("", "") == live                              # main's default is unchanged
    assert resolve("", "history") == history                    # APP_DEFAULT_PAGE=history
    assert resolve("history", "") == history                    # ?page=history
    assert resolve("live", "history") == live                   # the query parameter wins
    assert resolve("", "nonsense") == live


def test_history_page_renders_with_coach_agent(monkeypatch):
    if not (APP.parent / "models").exists():
        pytest.skip("needs trained models in models/")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APP_DEFAULT_PAGE", "history")
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "Historical replay and agent"
    assert any("Coach" in t.label for t in at.tabs)
    assert [t.label for t in at.tabs][:2] == ["Replayed night", "Graded by the market"]
    assert any(s.value == "Scoreboard: every setup graded by the market" for s in at.subheader)
    assert any(s.value.startswith("Coach agent") for s in at.subheader)
    assert any(list(df.value.columns) == ["step", "check", "why", "finding"] for df in at.dataframe)
