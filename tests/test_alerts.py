"""Deterministic pregame alert detection, persistence and delivery guards."""
import json

import pandas as pd

from agents.alerts import AlertConfig, AlertDetector, AlertManager, AlertStore


T0 = pd.Timestamp("2026-02-01T20:00:00Z")


def snapshot(at=T0, **changes):
    base = {
        "phase": "update", "game_id": "g1", "as_of": at.isoformat(),
        "p_home": 0.60, "news_health": "ok", "errors": [], "new_news_ids": [],
        "factors": [], "conflicts": [],
    }
    base.update(changes)
    return base


def test_status_and_probability_alerts_have_threshold_payloads():
    detector = AlertDetector(AlertConfig())
    current = snapshot(
        p_home=0.54, new_news_ids=["n1"],
        factors=[{"player_id": 7, "news_id": "n1", "status": "out", "source": "nba_injury_report"}],
    )
    events = detector.detect(snapshot(), current)
    assert {e["type"] for e in events} == {"injury_status_change", "probability_shift"}
    injury = next(e for e in events if e["type"] == "injury_status_change")
    assert injury["severity"] == "high"
    assert injury["payload"]["news_id"] == "n1"
    probability = next(e for e in events if e["type"] == "probability_shift")
    assert probability["severity"] == "high"
    assert probability["payload"]["thresholds"]["medium_pp"] == 3.0


def test_small_probability_change_does_not_alert():
    detector = AlertDetector(AlertConfig())
    assert not [e for e in detector.detect(snapshot(), snapshot(p_home=0.575)) if e["type"] == "probability_shift"]


def test_market_move_and_quote_staleness_alerts():
    detector = AlertDetector(AlertConfig())
    old = snapshot(polymarket={"status": "OK", "age_seconds": 5, "markets": [
        {"market_ticker": "HOME", "team": "HOM", "midpoint": 0.55}
    ]})
    current = snapshot(
        at=T0 + pd.Timedelta(minutes=4),
        polymarket={"status": "WARNING", "age_seconds": 120, "markets": [
            {"market_ticker": "HOME", "team": "HOM", "midpoint": 0.61}
        ]},
    )
    events = detector.detect(old, current)
    assert {e["type"] for e in events} == {"quote_stale"}
    fresh = snapshot(
        at=T0 + pd.Timedelta(minutes=4),
        polymarket={"status": "OK", "age_seconds": 5, "markets": [
            {"market_ticker": "HOME", "team": "HOM", "midpoint": 0.61}
        ]},
    )
    market_events = detector.detect(old, fresh)
    assert {e["type"] for e in market_events} == {"market_move"}
    assert market_events[0]["severity"] == "high"


def test_no_market_provider_does_not_create_quote_alert():
    assert not [e for e in AlertDetector().detect(snapshot(), snapshot(p_home=0.60))
                 if e["type"] in {"quote_stale", "quote_recovered"}]


def test_store_cooldown_and_restart(tmp_path):
    manager = AlertManager(tmp_path, config=AlertConfig(), mode="replay", run_id="run1")
    first = manager.process(snapshot(), snapshot(p_home=0.54))
    second = manager.process(snapshot(p_home=0.54), snapshot(at=T0 + pd.Timedelta(minutes=1), p_home=0.50))
    assert len(first) == 1 and len(second) == 0
    assert (tmp_path / "alerts.jsonl").exists() and (tmp_path / "alert_state.json").exists()
    resumed = AlertManager(tmp_path, config=AlertConfig(), mode="replay", run_id="run1")
    third = resumed.process(snapshot(p_home=0.54), snapshot(at=T0 + pd.Timedelta(minutes=2), p_home=0.50))
    assert third == []
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["delivery"]["status"] == "not_sent" and not rows[0]["acknowledged"]
    assert AlertStore(tmp_path).acknowledge([rows[0]["event_id"]]) == 1
    assert json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()[0])["acknowledged"]


def test_replay_never_sends_webhook(monkeypatch, tmp_path):
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("replay must not send external notifications")

    monkeypatch.setattr("agents.alerts.requests.post", post)
    config = AlertConfig(enabled=True, notify_requested=True, webhook_url="https://example.test/hook")
    manager = AlertManager(tmp_path, config=config, mode="replay", run_id="replay")
    events = manager.process(snapshot(), snapshot(p_home=0.54))
    assert len(events) == 1 and events[0]["delivery"]["status"] == "not_sent"
    assert calls == []


def test_live_webhook_payload_and_failure_do_not_raise(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 204

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("agents.alerts.requests.post", post)
    config = AlertConfig(enabled=True, notify_requested=True, webhook_url="https://example.test/hook",
                         webhook_secret="secret")
    manager = AlertManager(tmp_path, config=config, mode="live", run_id="live")
    events = manager.process(snapshot(), snapshot(p_home=0.54))
    assert events[0]["delivery"]["status"] == "sent"
    assert calls and calls[0][1]["headers"]["X-NBA-Agent-Signature"].startswith("sha256=")


def test_pregame_agent_records_alerts(tmp_path):
    from agents.graph import record_forecaster
    from agents.pregame import PregameAgent
    from data_sources.live_news import TableNews
    from test_agent_graph import make_tables, NEWS_AT

    tables = make_tables()
    players = __import__("pandas").DataFrame({"player_id": [7], "player_name": ["Test Star"]})
    game = next(tables["games"].iloc[[1]].itertuples(index=False))
    agent = PregameAgent(tables, players, game, record_forecaster, TableNews(tables["news"]), tmp_path)
    agent.poll(NEWS_AT - pd.Timedelta(hours=1))
    result = agent.poll(NEWS_AT)["snapshot"]
    assert result["alert_count"] >= 1
    assert (tmp_path / "alerts.jsonl").exists()

