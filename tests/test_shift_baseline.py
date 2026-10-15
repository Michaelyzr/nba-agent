"""M4 news-shift baseline (absences known at the anchor) and the LLM pre-filter (offline, no network)."""
import re

import pandas as pd
import pytest

from agents.graph import ANCHOR_LEAD, MarketAgent, anchor_out
from agents.llm_client import CachedLLM, StubBackend
from agents.tool_agent import ToolAgent, heuristic_reply
from agents.tools import AsOfTools
from tests.test_agent_graph import NEWS_AT, TIP1, TIP2, make_tables
from tests.test_agent_graph import run as run_graph
from tests.test_tool_agent import _forecaster, tables_with_history, view_at
from tests.test_tool_agent import run as run_tools


def with_previous_list(tables, published_at=TIP1 - pd.Timedelta(minutes=30)):
    """Player 7 was also on the home team's inactive list for the previous game (g1)."""
    prev = pd.DataFrame({"news_id": ["n_prev"], "published_at": [published_at], "game_id": ["g1"], "player_id": [7],
                         "team_id": [1], "status": ["Out"], "source": ["test"], "url": [""],
                         "text": ["Player 7 listed out"]})
    tables["news"] = pd.concat([prev, tables["news"].assign(team_id=1)], ignore_index=True)
    return tables


def forecast_pairs(traces):
    detail = next(s["detail"] for s in traces[0]["trace"] if s["step"] == "forecast")
    return [(float(a), float(b)) for a, b in re.findall(r"(\d\.\d\d)->(\d\.\d\d)", detail)]


def test_absence_known_before_the_anchor_gives_no_shift():
    tables = with_previous_list(make_tables())
    _, fills, traces = run_graph(MarketAgent(shift_baseline="anchor"), tables)
    assert forecast_pairs(traces) and all(a == b for a, b in forecast_pairs(traces))
    assert fills.empty                                  # p = anchor, no edge left after the spread and fee


def test_new_absence_after_the_anchor_still_shifts():
    tables = make_tables()
    tables["news"] = tables["news"].assign(team_id=1)
    decisions, _, traces = run_graph(MarketAgent(shift_baseline="anchor"), tables)
    assert forecast_pairs(traces) and all(abs(a - b) > 0.09 for a, b in forecast_pairs(traces))
    assert "news shift -10.2 points" in decisions.iloc[0].reason


def test_no_shift_before_tonights_list_is_published():
    """Player 7 was out last game and tonight's list is not out yet: that is not news that he returns."""
    tables = with_previous_list(make_tables(with_news=False))
    view, game, now = view_at(tables)
    _, _, traces = run_graph(MarketAgent(shift_baseline="anchor"), tables)
    assert forecast_pairs(traces) and all(a == b for a, b in forecast_pairs(traces))
    m4 = AsOfTools(view, game, now, forecaster=_forecaster(), shift_baseline="anchor").m4_win_prob([])
    assert m4["shift"]["HOM"] == 0 and m4["before_out"] == []


def test_empty_baseline_reproduces_the_published_double_count():
    decisions, _, traces = run_graph(MarketAgent(shift_baseline="empty"), with_previous_list(make_tables()))
    assert "news shift -10.2 points" in decisions.iloc[0].reason


def test_default_shift_baseline_is_empty():
    assert MarketAgent().shift_baseline == "empty"
    decisions, _, _ = run_graph(MarketAgent(), with_previous_list(make_tables()))
    assert "news shift -10.2 points" in decisions.iloc[0].reason


def test_cli_default_shift_baseline_is_empty():
    import agents.graph as graph
    src = open(graph.__file__).read()
    assert re.search(r'"--shift-baseline", choices=SHIFT_BASELINES, default="empty"', src)


def test_anchor_out_ignores_news_published_after_the_anchor():
    tables = with_previous_list(make_tables(), published_at=TIP2 - ANCHOR_LEAD + pd.Timedelta(minutes=1))
    view, game, _ = view_at(tables)
    assert anchor_out(view, game) == []
    view, game, _ = view_at(with_previous_list(make_tables()))
    assert anchor_out(view, game) == [7]


def test_bad_shift_baseline_is_rejected():
    with pytest.raises(ValueError):
        MarketAgent(shift_baseline="tip")


def test_tool_m4_shift_uses_the_same_baseline():
    tables = with_previous_list(make_tables())
    view, game, now = view_at(tables)
    empty = AsOfTools(view, game, now, forecaster=_forecaster()).m4_win_prob([7])
    anchored = AsOfTools(view, game, now, forecaster=_forecaster(), shift_baseline="anchor").m4_win_prob([7])
    assert empty["shift"]["HOM"] == pytest.approx(-0.10) and "before_out" not in empty
    assert anchored["shift"]["HOM"] == 0 and anchored["before_out"] == [7]


def tool_agent(prefilter_edge=None):
    llm = CachedLLM(StubBackend(heuristic_reply, model="test-stub"), cache_dir=None)
    return ToolAgent(llm, sceptic=False, forecaster=_forecaster(), players={7: "Test Player"},
                     prefilter_edge=prefilter_edge)


def test_prefilter_off_by_default_and_passes_clear_edges():
    for edge in (None, 0.02):
        a = tool_agent(edge)
        _, fills, traces = run_tools(a, tables_with_history())
        assert len(fills) == 1 and traces[0]["status"] == "order"
        assert ("prefilter" in traces[0]) == (edge is not None)


def test_prefilter_skips_the_llm_below_the_threshold():
    a = tool_agent(0.50)
    _, fills, traces = run_tools(a, tables_with_history())
    assert fills.empty
    live = [t for t in traces if t["status"] not in ("idle", "held")]
    assert live and all(t["status"] == "prefilter" and t["llm"]["calls"] == 0 for t in live)
    assert a.llm.backend.calls == 0


def test_anchor_edge_matches_the_heuristic_stub():
    view, game, now = view_at(tables_with_history())
    best = AsOfTools(view, game, now, forecaster=_forecaster()).anchor_edge()
    assert best["team"] == "AWY" and best["gap"] > 0.04
