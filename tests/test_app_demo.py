"""Smoke tests for the Streamlit demo: default page override and Coach agent wiring."""
import os

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"


@pytest.fixture
def clear_default_page(monkeypatch):
    monkeypatch.delenv("APP_DEFAULT_PAGE", raising=False)


def test_live_markets_is_default_page(clear_default_page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "NBA Polymarket Live Markets"


def test_app_default_page_history_env(monkeypatch):
    monkeypatch.setenv("APP_DEFAULT_PAGE", "history")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Historical replay and agent"
    # Historical page exposes the Coach tab among others.
    assert any("Coach" in t.label for t in at.tabs)


def test_query_param_page_history(clear_default_page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["page"] = "history"
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Historical replay and agent"
