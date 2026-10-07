"""Thin night orchestrator: routing order, no look-ahead, graceful degradation."""
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.orchestrator import ORDER, NightOrchestrator, _look_ahead, format_trace, msg, save_trace
from agents.graph import record_forecaster

TIP = pd.Timestamp("2026-03-10T23:30:00Z")
AS_OF = TIP - pd.Timedelta(minutes=60)


def make_tables():
    games = pd.DataFrame({
        "game_id": ["g1"], "date": ["2026-03-10"], "tip_time": [TIP],
        "final_at": [TIP + pd.Timedelta(hours=3)], "home_team_id": [1], "away_team_id": [2],
        "home_team": ["HOM"], "away_team": ["AWY"], "home_pts": [110], "away_pts": [100]})
    player_games = pd.DataFrame({"game_id": ["g1"], "player_id": [7], "team_id": [1], "min": [34.0], "pts": [25],
                                 "fga": [18], "fta": [6], "usage": [0.3], "started": [True]})
    news = pd.DataFrame({"news_id": ["n1"], "published_at": [AS_OF - pd.Timedelta(minutes=30)],
                         "game_id": ["g1"], "player_id": [7], "status": ["Out"], "source": ["test"],
                         "url": [""], "text": ["Player 7 out"]})
    markets = pd.DataFrame({"venue": "synthetic", "market_ticker": ["HOM-yes", "AWY-yes"], "kind": "game",
                            "game_id": "g1", "team": ["HOM", "AWY"], "player_id": None, "line": None,
                            "title": ["HOM", "AWY"]})
    ts = pd.date_range(TIP - pd.Timedelta(hours=3), TIP + pd.Timedelta(hours=1), freq="1min")
    prices = pd.concat([
        pd.DataFrame({"venue": "synthetic", "market_ticker": "HOM-yes", "ts": ts, "bid": 0.55, "ask": 0.57,
                      "volume": 500.0}),
        pd.DataFrame({"venue": "synthetic", "market_ticker": "AWY-yes", "ts": ts, "bid": 0.43, "ask": 0.45,
                      "volume": 500.0})])
    settlements = pd.DataFrame({"market_ticker": ["HOM-yes", "AWY-yes"],
                                "settled_at": TIP + pd.Timedelta(hours=3), "outcome": [1, 0]})
    return {"games": games, "player_games": player_games, "news": news, "markets": markets,
            "prices": prices, "settlements": settlements}


def fake_component(name, payload=None, fail=False):
    def fn(game, now, night):
        if fail:
            raise RuntimeError(f"{name} broke")
        return payload or {"ok": True, "as_of": str(now), "citations": [], "orders": []}
    return fn


@pytest.fixture
def orch(tmp_path):
    components = {c: fake_component(c) for c in ORDER}
    return NightOrchestrator(make_tables(), record_forecaster, out_dir=tmp_path, components=components)


def test_routing_order(orch):
    out = orch.run("2026-03-10")
    assert out["done"] == list(ORDER)
    routes = [m["receiver"] for m in out["messages"] if m["kind"] == "route" and m["sender"] == "supervisor"]
    assert routes[:4] == list(ORDER)
    assert out["messages"][-1]["kind"] == "done"


def test_no_lookahead_rejects_future_citation():
    news = make_tables()["news"]
    future = {"citations": ["n1"], "as_of": str(AS_OF - pd.Timedelta(hours=2))}
    # Move decision earlier than the news so the citation is in the future.
    assert _look_ahead(future, AS_OF - pd.Timedelta(hours=1), news) is not None
    assert _look_ahead({"as_of": str(AS_OF), "citations": ["n1"], "orders": []}, AS_OF, news) is None
    assert _look_ahead({"as_of": str(AS_OF + pd.Timedelta(minutes=1))}, AS_OF, news) is not None


def test_every_component_gets_the_same_as_of_time_never_later(tmp_path):
    seen = []

    def spy(name):
        def fn(game, now, night):
            seen.append((name, now, sorted(night)))
            return {"as_of": str(now), "citations": [], "orders": []}
        return fn

    orch = NightOrchestrator(make_tables(), record_forecaster, out_dir=tmp_path,
                             components={c: spy(c) for c in ORDER})
    orch.run("2026-03-10")
    assert [s[0] for s in seen] == list(ORDER)
    assert all(now == AS_OF and now < TIP for _, now, _ in seen)


def test_component_returning_future_data_is_rejected(tmp_path):
    late = {"as_of": str(TIP + pd.Timedelta(minutes=5)), "citations": [], "orders": []}
    components = {c: fake_component(c) for c in ORDER}
    components["pregame"] = fake_component("pregame", late)
    orch = NightOrchestrator(make_tables(), record_forecaster, out_dir=tmp_path, components=components)
    out = orch.run("2026-03-10")
    assert out["pregame"] == {} and "pregame" in out["failed"]
    assert any(m["kind"] == "error" and "look-ahead" in m["content"] for m in out["messages"])


def test_failing_component_degrades(tmp_path):
    components = {
        "pregame": fake_component("pregame", fail=True),
        "trader": fake_component("trader", {"status": "brief_only", "orders": [], "candidates": [],
                                           "citations": [], "as_of": str(AS_OF)}),
        "coach": fake_component("coach", {"explanation": ["ok"], "agent_panel": ["degraded"],
                                          "lessons": [], "coach_pick": None, "banned_words": [],
                                          "notice": ""}),
        "briefs": fake_component("briefs", {"platform": "p", "media": "m", "team": "t", "retail": "r"}),
    }
    orch = NightOrchestrator(make_tables(), record_forecaster, out_dir=tmp_path, components=components)
    out = orch.run("2026-03-10")
    assert "pregame" in out["failed"]
    assert set(out["done"]) == set(ORDER)           # supervisor still finishes the night
    assert out["pregame"] == {}                     # no successful pregame handoff
    assert "g1" in out["trader"] and "g1" in out["coach"]
    errors = [m for m in out["messages"] if m["kind"] == "error"]
    assert len(errors) == 1 and "pregame" in errors[0]["sender"]
    assert "night complete" in out["messages"][-1]["content"]


def test_empty_night(tmp_path):
    tables = make_tables()
    tables["games"] = tables["games"].iloc[0:0]
    orch = NightOrchestrator(tables, record_forecaster, out_dir=tmp_path,
                             components={c: fake_component(c) for c in ORDER})
    out = orch.run("2026-03-10")
    assert out["done"] == []
    assert out["messages"][-1]["kind"] == "done"
    assert "no games" in out["messages"][-1]["content"]


def test_save_trace_roundtrip(orch, tmp_path):
    out = orch.run("2026-03-10")
    path = save_trace(out, tmp_path / "night.json")
    loaded = path.read_text()
    assert "pregame" in loaded and "trader" in loaded
    assert "supervisor" in format_trace(out)
