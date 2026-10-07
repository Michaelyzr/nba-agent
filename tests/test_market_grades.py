"""'Graded by the market' tab: loaders read the committed results, and the tab renders (AppTest) with and without them."""
from pathlib import Path

import pandas as pd

import market_grades as mg

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = """
import pandas as pd
import market_grades as mg
from pathlib import Path
tip = pd.Timestamp("2026-03-01 00:30", tz="UTC")
fills = pd.DataFrame({{"as_of": [tip - pd.Timedelta(hours=1)], "market_ticker": ["KX-HOME"], "side": ["yes"],
                      "contracts": [36], "price": [0.55], "fee": [0.62], "close_price": [0.625], "clv": [0.075],
                      "outcome": [1], "pnl": [15.58], "game_id": ["g1"]}})
ts = pd.date_range(tip - pd.Timedelta(hours=2), tip, freq="10min")
prices = lambda ticker, start, end: pd.DataFrame({{"ts": ts, "bid": 0.55, "ask": 0.57}}).query("@start <= ts <= @end")
mg.render(fills, "a test", tip, prices, Path({results!r}))
"""


def test_ci_text():
    assert mg.ci_text(-0.0035, -0.0065, 0.0, "cents") == "-0.35¢ [-0.65, +0.00]"
    assert mg.ci_text(-35.2, -60.3, -11.5, "dollars") == "−$35 [-60, -12]"
    assert mg.ci_text(None, None, None, "dollars") == "–"


def test_scoreboard_reads_results_and_keeps_tool_agent_pending():
    board, missing = mg.scoreboard()
    assert not missing
    assert {"Deterministic agent, no learning", "Never trade", "B. Plain LLM, one call (no tools)"} <= set(board.setup)
    tool = board[board.setup.str.contains("tool agent")].iloc[0]
    assert tool["CLV $ [95% CI]"] == mg.PENDING
    loop, rates = mg.learning_loop()
    assert set(loop.gate) <= {"accepted", "rejected"} and len(loop) == 7
    assert len(mg.coach_board())


def test_scoreboard_degrades_without_files(tmp_path):
    board, missing = mg.scoreboard(tmp_path)
    assert set(missing) == {"gate_audit.csv", "llm_agent.csv", "holdout.csv"}
    assert list(board.setup) == ["C/D. LLM tool agent (+ sceptic)"]
    assert mg.learning_loop(tmp_path) == (None, None)


def test_tab_renders_with_results():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(SCRIPT.format(results=str(mg.RESULTS)), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("closing-line value" in m.value for m in at.markdown)
    heads = [s.value for s in at.subheader]
    assert "Scoreboard: every setup graded by the market" in heads
    assert "The learning loop, graded by the market" in heads
    assert any("trading less" in s.value for s in at.success)
    assert not at.warning


def test_tab_renders_without_results(tmp_path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(SCRIPT.format(results=str(tmp_path)), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("gate_audit.csv" in w.value for w in at.warning)
