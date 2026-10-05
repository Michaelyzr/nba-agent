"""Agent graph on a tiny slate: news trade, checks retry, risk block, rule timing and the gate (offline, no LLM)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.graph import MarketAgent, OUT_MINUTE_VALUE
from agents.notebook import Notebook, matches, validate
from replay import Replay

TIP1 = pd.Timestamp("2026-02-01T00:30:00Z")
TIP2 = pd.Timestamp("2026-02-02T00:30:00Z")
HOME, AWAY = "SYN-g2-HOM", "SYN-g2-AWY"
NEWS_AT = TIP2 - pd.Timedelta(hours=2)


def make_tables(with_news=True, home_quote=(0.60, 0.62)):
    games = pd.DataFrame({
        "game_id": ["g1", "g2"], "date": ["2026-01-31", "2026-02-01"], "tip_time": [TIP1, TIP2],
        "final_at": [TIP1 + pd.Timedelta(hours=3), TIP2 + pd.Timedelta(hours=3)],
        "home_team_id": [1, 1], "away_team_id": [2, 2], "home_team": ["HOM", "HOM"],
        "away_team": ["AWY", "AWY"], "home_pts": [110, 101], "away_pts": [100, 99]})
    player_games = pd.DataFrame({"game_id": ["g1"], "player_id": [7], "team_id": [1], "min": [34.0], "pts": [25],
                                 "fga": [18], "fta": [6], "usage": [0.3], "started": [True]})
    news = pd.DataFrame({"news_id": ["n_out"], "published_at": [NEWS_AT], "game_id": ["g2"], "player_id": [7],
                         "status": ["Out"], "source": ["test"], "url": [""], "text": ["Player 7 listed out"]})
    if not with_news:
        news = news.iloc[0:0]
    markets = pd.DataFrame({"venue": "synthetic", "market_ticker": [HOME, AWAY], "kind": "game", "game_id": "g2",
                            "team": ["HOM", "AWY"], "player_id": None, "line": None, "title": ["HOM", "AWY"]})
    ts = pd.date_range(TIP2 - pd.Timedelta(hours=3), TIP2 + pd.Timedelta(hours=1), freq="1min")
    bid, ask = home_quote
    prices = pd.concat([
        pd.DataFrame({"venue": "synthetic", "market_ticker": HOME, "ts": ts, "bid": bid, "ask": ask, "volume": 500.0}),
        pd.DataFrame({"venue": "synthetic", "market_ticker": AWAY, "ts": ts, "bid": round(1 - ask, 2),
                      "ask": round(1 - bid, 2), "volume": 500.0})])
    settlements = pd.DataFrame({"market_ticker": [HOME, AWAY], "settled_at": TIP2 + pd.Timedelta(hours=3),
                                "outcome": [1, 0]})
    return {"games": games, "player_games": player_games, "news": news, "markets": markets,
            "prices": prices, "settlements": settlements}


def run(agent, tables=None):
    tables = tables or make_tables()
    decisions, fills = Replay(tables, agent.policy).run("2026-02-01", "2026-02-01")
    return decisions, fills, [t for t in agent.traces if t["phase"] == "decide"]


def steps(trace):
    return [s["step"] for s in trace["trace"]]


def test_graph_has_every_flowchart_step():
    nodes = set(MarketAgent().graph.get_graph().nodes)
    assert {"trigger", "investigate", "forecast", "analyse", "propose", "no_action", "checks", "risk", "confirm",
            "blocked", "deliver", "settle", "review", "gate", "save_rule", "reject_rule"} <= nodes


def test_anchored_agent_trades_only_the_news_shift():
    decisions, fills, traces = run(MarketAgent())
    d = decisions.iloc[0]
    p_home = 0.61 - OUT_MINUTE_VALUE * 34              # market mid before the news plus the model's shift
    assert d.p_model == pytest.approx(p_home if d.market_ticker == HOME else 1 - p_home, abs=0.01)
    assert "news shift -10.2 points" in d.reason


def test_news_trade_end_to_end():
    decisions, fills, traces = run(MarketAgent(anchor=False))
    assert len(decisions) == len(fills) == 1                 # one position per game, taken at the news
    d = decisions.iloc[0]
    assert d.as_of == NEWS_AT and d.citations == ["n_out"] and d.risk_result == "approved"
    p_home_before = traces[0]["trace"][2]["detail"]
    assert "HOM 0.62->0.52" in p_home_before
    p_home = 0.62 - OUT_MINUTE_VALUE * 34
    assert d.p_model == pytest.approx(p_home if d.market_ticker == HOME else 1 - p_home, abs=0.01)
    assert (d.market_ticker, d.side) in {(HOME, "no"), (AWAY, "yes")}
    assert steps(traces[0])[-4:] == ["checks", "risk", "confirm", "deliver"]
    assert traces[1]["status"] == "brief_only"               # LEAD decision: game already held


def test_no_news_no_gap_is_brief_only():
    decisions, fills, traces = run(MarketAgent(), make_tables(with_news=False))
    assert decisions.empty and fills.empty
    assert [t["status"] for t in traces] == ["brief_only"]
    assert "already priced in" in traces[0]["brief"]


@pytest.mark.parametrize("fault", ["future_citation", "number_mismatch"])
def test_check_failure_gets_one_retry(fault):
    decisions, fills, traces = run(MarketAgent(plant=fault))
    checks = [s["detail"] for s in traces[0]["trace"] if s["step"] == "checks"]
    assert checks[0].startswith("rejected (try 1 of 2)") and checks[1] == "passed"
    assert len(fills) == 1 and "planted-future-news" not in decisions.iloc[0].citations


def test_repeated_failure_is_blocked():
    decisions, fills, traces = run(MarketAgent(plant="lock_wording"))
    assert decisions.empty and fills.empty
    t = traces[0]
    assert t["status"] == "blocked" and steps(t).count("checks") == 2 and "Order blocked" in t["brief"]


def test_risk_limit_blocks_order_inside_graph():
    decisions, fills, traces = run(MarketAgent(stake=500.0))
    assert decisions.empty
    assert traces[0]["status"] == "blocked"
    assert any(s["step"] == "risk" and "over_order_cap" in s["detail"] for s in traces[0]["trace"])


def skip_rule(valid_from):
    return {"rule_id": "r001", "version": 1, "kind": "trader", "status": "active", "when": {"market_kind": "game"},
            "do": {"action": "skip_market", "params": {}}, "valid_from": valid_from, "expires_after_days": 45}


def test_rule_applies_only_after_valid_from():
    _, fills, _ = run(MarketAgent(Notebook([skip_rule(str(TIP2 + pd.Timedelta(days=1)))])))
    assert len(fills) == 1
    _, fills, traces = run(MarketAgent(Notebook([skip_rule(str(TIP1))])))
    assert fills.empty and "rule r001" in traces[0]["brief"]


def test_notebook_vocabulary_and_matching():
    assert validate({"when": {"mood": "bad"}, "do": {"action": "skip_market"}})
    assert validate({"when": {}, "do": {"action": "stake_scale", "params": {"scale": 2}}})
    assert not validate({"when": {"news_age_minutes_min": 30}, "do": {"action": "skip_market", "params": {}}})
    rule = {"when": {"market_kind": "game", "news_age_minutes_min": 30}}
    assert matches(rule, {"market_kind": "game", "news_age_minutes": 45})
    assert not matches(rule, {"market_kind": "game", "news_age_minutes": 10})
    assert not matches(rule, {"market_kind": "pts", "news_age_minutes": 45})


def gate_setup(with_rule_clv):
    """Three traded days; the reviewer sees six stale trades with negative closing-line value."""
    days = ["2026-01-29", "2026-01-30", "2026-01-31"]
    tips = [pd.Timestamp(f"{d}T00:30:00Z") for d in days]
    games = pd.DataFrame({"game_id": ["a", "b", "c"], "date": days, "tip_time": tips,
                          "final_at": [t + pd.Timedelta(hours=3) for t in tips]})
    tables = {"games": games, "markets": pd.DataFrame({"game_id": ["a", "b", "c"]})}
    fills = [{"decision_id": f"x{i}", "market_ticker": f"T{i}", "as_of": tips[0], "game_id": "c", "side": "yes",
              "p_model": 0.6, "price": 0.5, "contracts": 40, "fee": 0.7, "outcome": 0, "clv": -0.01, "pnl": -20.7}
             for i in range(6)]
    agent = MarketAgent()
    agent.situations = {(f"T{i}", tips[0]): {"news_age_minutes": 9999.0} for i in range(6)}
    without = pd.DataFrame({"clv": [-0.01] * 4 + [0.05], "pnl": [-5.0] * 4 + [10.0]})
    with_rule = pd.DataFrame({"clv": [with_rule_clv], "pnl": [10.0]})
    agent.backtest = lambda rp, nb, start, end: with_rule if len(nb.rules) else without
    rp = SimpleNamespace(t=tables, fills=fills)
    return agent, rp


@pytest.mark.parametrize("with_rule_clv, status", [(0.05, "active"), (-0.02, "rejected")])
def test_gate_keeps_only_rules_that_help_on_earlier_days(with_rule_clv, status):
    agent, rp = gate_setup(with_rule_clv)
    out = agent.graph.invoke({"phase": "review", "day": "2026-01-31", "replay": rp, "trace": []})
    assert [s["step"] for s in out["trace"]][:3] == ["settle", "review", "gate"]
    rule = agent.notebook.rules[0]
    assert rule["status"] == status and rule["do"]["action"] == "skip_market"
    assert rule["gate"]["backtest_days"] == ["2026-01-29", "2026-01-30"]       # earlier days only
    assert (rule["valid_from"] is not None) == (status == "active")
