"""Brief grounding, event attribution, source evidence and an offline end-to-end demo."""
import json

import pandas as pd
import pytest
import requests

from agents.loop_demo import generate_demo
from test_inplay import Feed, NOW, agent, event, score
from test_pregame import setup_agent


def test_report_is_generated_from_forecast_and_keeps_source_evidence(tmp_path):
    feed = Feed([event()])
    s = agent(tmp_path, feed).poll(NOW)["snapshot"]
    report = s["report"]
    assert report["p_home"] == s["p_home"]
    assert report["home_decimal_odds"] == pytest.approx(1 / s["p_home"])
    assert report["new_evidence"][0]["player"] == "Test Star"
    assert "https://example.com/event" in report["markdown"]
    assert json.loads((tmp_path / "reports.jsonl").read_text())["report_id"] == report["report_id"]
    assert (tmp_path / "latest_report.md").read_text() == report["markdown"]


def test_score_change_is_not_mislabelled_as_new_news_effect(tmp_path):
    feed = Feed([event()])
    a = agent(tmp_path, feed)
    first = a.poll(NOW)["snapshot"]
    feed.current = score(NOW + pd.Timedelta(seconds=5), home_score=52, period=4, clock_seconds=120)
    second = a.poll(NOW + pd.Timedelta(seconds=5))["snapshot"]
    assert second["p_home"] != first["p_home"]
    assert second["report"]["new_event_effect_pp"] == pytest.approx(0)
    assert second["report"]["active_news_effect_pp"] < 0
    assert second["report"]["total_delta_pp"] != 0
    assert second["report"]["new_evidence"] == []
    assert second["report"]["news_effect"]["home_odds_delta"] == pytest.approx(0)
    assert second["report"]["news_effect"]["p_home_before"] != first["p_home"]


def test_return_incremental_effect_is_grounded_in_same_score(tmp_path):
    feed = Feed()
    a = agent(tmp_path, feed)
    baseline = a.poll(NOW)["snapshot"]
    feed.rows = [event()]
    injury = a.poll(NOW)["snapshot"]
    feed.rows.append(event("returned", at=NOW + pd.Timedelta(seconds=1)))
    back = a.poll(NOW + pd.Timedelta(seconds=1))["snapshot"]
    assert back["p_home"] == baseline["p_home"]
    assert back["report"]["new_event_effect_pp"] == pytest.approx((back["p_home"] - injury["p_home"]) * 100)


def test_stale_score_brief_does_not_publish_a_prediction(tmp_path):
    s = agent(tmp_path, Feed(current=score(at=NOW - pd.Timedelta(seconds=31)))).poll(NOW)["snapshot"]
    assert s["report"]["p_home"] is None
    assert s["report"]["new_event_effect_pp"] is None
    assert s["report"]["news_effect"] is None
    assert "暂不输出" in s["report"]["summary"]


def test_duplicate_raw_events_have_one_report_citation(tmp_path):
    row = event()
    s = agent(tmp_path, Feed([row, dict(row)])).poll(NOW)["snapshot"]
    assert len(s["report"]["new_evidence"]) == 1


def test_departure_and_later_prognosis_are_not_conflicting_facts(tmp_path):
    feed = Feed([event("left_injured"), event("injury_out", "espn_rss", NOW + pd.Timedelta(seconds=1))])
    s = agent(tmp_path, feed).poll(NOW + pd.Timedelta(seconds=1))["snapshot"]
    assert s["factors"][0]["status"] == "injury_out" and s["conflicts"] == []


def test_pregame_reports_have_an_independent_history_reference(tmp_path):
    a = setup_agent(tmp_path)
    s = a.poll(a.game.tip_time - pd.Timedelta(hours=2))["snapshot"]
    assert s["report"]["pipeline"] == "pregame"
    assert s["report"]["reference_p_home"] == s["baseline"]["p_home"]
    assert (tmp_path / "latest_report.md").exists()
    assert not (tmp_path / "inplay_state.json").exists()


def test_full_demo_uses_actual_agents_without_network_and_is_repeatable(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the synthetic demo must never request live sources")
    monkeypatch.setattr(requests.Session, "get", forbidden)
    monkeypatch.setattr(requests.Session, "post", forbidden)
    payload = generate_demo(tmp_path)
    again = generate_demo(tmp_path)
    assert payload == again and len(payload["steps"]) == 63
    steps = payload["steps"]
    assert all(step["snapshot"]["report"]["synthetic"] for step in steps)
    inplay = [r["snapshot"] for r in steps if r["phase"] == "inplay"]
    events = payload["timeline_events"]
    assert [e["clock"] for e in events] == ["Q2 08:00", "Q2 06:00", "Q3 09:00", "Q4 05:00", "Q4 05:00"]
    assert [e["home_delta_pp"] > 0 for e in events] == [False, False, True, True, False]
    assert events[-1]["home_delta_pp"] == 0 and inplay[events[-1]["index"]]["conflicts"]
    assert {s["score"]["period"] for s in inplay} == {1, 2, 3, 4}
    for q in range(1, 5):
        quarter = [s for s in inplay if s["score"]["period"] == q]
        assert quarter[0]["score"]["clock_seconds"] == 720
        assert quarter[-1]["score"]["clock_seconds"] == 0
    for e in events:
        snapshot = inplay[e["index"]]
        assert e["p_home_before"] == snapshot["p_home_before_new_events"]
        assert e["home_odds_before"] == pytest.approx(1 / e["p_home_before"])
        assert e["home_odds_after"] == pytest.approx(snapshot["home_decimal_odds"])
        assert e["home_odds_delta"] == pytest.approx(e["home_odds_after"] - e["home_odds_before"])
        assert e["away_delta_pp"] == -e["home_delta_pp"]
        assert e["time_basis"] == "synthetic_occurrence"
        audit = snapshot["forecast_audit"]
        assert sum(r["delta_pp"] for r in audit["decomposition"]) == pytest.approx(snapshot["report"]["total_delta_pp"])
        assert audit["scenarios"][0]["p_home"] == pytest.approx(snapshot["p_home"])
        assert snapshot["market_comparison"]["decision"] == "wait"
    assert all(pd.Timestamp(r["final_at"]) < pd.Timestamp(payload["original_data"]["history_cutoff"]) for r in payload["original_data"]["historical_games"])
    assert len(payload["original_data"]["player_games"]) <= 20
    from agents.forecast_audit import visible_original_data
    known = visible_original_data(payload, inplay[0])
    assert len(known["input_scores"]) == 1 and known["input_events"] == []
    assert visible_original_data(payload, inplay[-1])["input_events"]
    assert visible_original_data(payload, steps[0]["snapshot"])["input_scores"] == []
    assert inplay[-1]["report"]["quote_state"] == "final"
    assert inplay[-1]["report"]["new_event_effect_pp"] is None
    assert (tmp_path / "logs/pregame/state.json").exists()
    assert (tmp_path / "logs/inplay/inplay_state.json").exists()
    assert all((tmp_path / name).exists() for name in ("result.json", "report.md", "demo.html", "overview.png", "timeline.csv"))
    for phase in ("pregame", "inplay"):
        records = (tmp_path / "logs" / phase / "reports.jsonl").read_text().splitlines()
        assert len(records) == (5 if phase == "pregame" else 58)


def test_game_clock_positions_include_quarter_boundaries_and_overtime():
    from agents.game_timeline import elapsed_seconds, clock_label
    assert elapsed_seconds({"period": 1, "clock_seconds": 720}) == 0
    assert elapsed_seconds({"period": 2, "clock_seconds": 720}) == 720
    assert elapsed_seconds({"period": 4, "clock_seconds": 0}) == 2880
    assert elapsed_seconds({"period": 5, "clock_seconds": 300}) == 2880
    assert elapsed_seconds({"period": 6, "clock_seconds": 120}) == 3360
    assert clock_label({"period": 2, "clock_seconds": 480}) == "Q2 08:00"
    assert clock_label({"period": 5, "clock_seconds": 300}) == "OT1 05:00"


def test_pregame_general_news_is_visible_without_changing_probability(tmp_path):
    from data_sources.live_news import NewsBatch
    from agents.match_view import event_feed
    a=setup_agent(tmp_path)
    now=a.game.tip_time-pd.Timedelta(hours=2)
    good={'item_id':'article','game_id':a.game.game_id,'published_at':now,'observed_at':now,
          'source':'espn_rss','url':'https://example.com/preview','title':'Team announces warm-up schedule',
          'text':'Team announces warm-up schedule','review_required':True}
    items=[good,{**good,'item_id':'future','published_at':now+pd.Timedelta(seconds=1)},
           {**good,'item_id':'wrong-game','game_id':'other'}]
    a.provider=type('News',(),{'fetch':lambda self,*args:NewsBatch(items=items)})()
    s=a.poll(now)['snapshot']
    assert s['p_home']==s['baseline']['p_home']
    assert [r['item_id'] for r in s['retrieved_news']]==['article']
    feed=event_feed([{'snapshot':s}])
    assert feed[0]['headline']==good['title'] and feed[0]['url']==good['url']
    assert feed[0]['impact_pp'] is None and not feed[0]['selected']
    repeat=a.poll(now+pd.Timedelta(seconds=5))['snapshot']
    assert repeat['retrieved_news']==[]
    assert len(event_feed([{'snapshot':s},{'snapshot':repeat}]))==1


def test_inplay_general_news_is_visible_once_without_an_injury_effect(tmp_path):
    from agents.match_view import event_feed
    from data_sources.inplay_types import InPlayBatch
    feed=Feed()
    good={'item_id':'live-article','game_id':'g2','published_at':NOW,'observed_at':NOW,
          'source':'espn_rss','url':'https://example.com/live','title':'Coach comments on defensive scheme','text':'Match update'}
    feed.fetch=lambda *args:InPlayBatch(score=score(at=args[-1]),evidence=[good,{**good,'item_id':'future','published_at':NOW+pd.Timedelta(minutes=1)}])
    a=agent(tmp_path,feed)
    s=a.poll(NOW)['snapshot']
    assert [r['item_id'] for r in s['retrieved_news']]==['live-article']
    assert not s['factors'] and s['news_effect_pp']==0
    assert event_feed([{'snapshot':s}])[0]['headline']==good['title']
    repeat=a.poll(NOW+pd.Timedelta(seconds=5))['snapshot']
    assert repeat['retrieved_news']==[]
