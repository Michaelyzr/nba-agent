"""News-loop invariants: as-of, correction, idempotence, outage, stop and resume."""
import json

import pandas as pd
import pytest

from agents.graph import record_forecaster
from agents.pregame import PregameAgent, odds, run_loop
from data_sources.live_news import LiveNews, NewsBatch, TableNews, classify_sentence, rss_rows
from test_agent_graph import make_tables, TIP2, NEWS_AT


def setup_agent(tmp_path, provider=None, clock=None):
    tables = make_tables()
    players = pd.DataFrame({"player_id": [7], "player_name": ["Test Star"]})
    game = next(tables["games"].iloc[[1]].itertuples(index=False))
    agent = PregameAgent(tables, players, game, record_forecaster,
                         provider or TableNews(tables["news"]), tmp_path, clock=clock)
    return agent


def row(news_id="n", status="out", published=NEWS_AT, **extras):
    return {"news_id": news_id, "game_id": "g2", "player_id": 7, "published_at": published,
            "status": status, "source": "test", "url": "https://example.com/news", "text": status, **extras}


class Feed:
    def __init__(self, rows=(), errors=()):
        self.rows, self.errors, self.calls = list(rows), list(errors), []

    def fetch(self, game, players, roster, since, now):
        self.calls.append(now)
        return NewsBatch([dict(r) for r in self.rows], list(self.errors))


def test_baseline_out_repeat_then_return(tmp_path):
    feed = Feed()
    agent = setup_agent(tmp_path, feed)
    start = NEWS_AT - pd.Timedelta(hours=1)
    initial = agent.poll(start)["snapshot"]
    assert initial["p_home"] == pytest.approx(0.62, abs=0.001)
    feed.rows = [row()]
    out = agent.poll(NEWS_AT)["snapshot"]
    assert out["p_home"] == pytest.approx(initial["p_home"] - 0.003 * 34)
    repeat = agent.poll(NEWS_AT + pd.Timedelta(minutes=5))["snapshot"]
    assert repeat["p_home"] == out["p_home"] and repeat["delta_home"] == 0
    assert repeat["new_news_ids"] == []
    back_at = NEWS_AT + pd.Timedelta(minutes=10)
    feed.rows.append(row("back", "available", back_at))
    back = agent.poll(back_at)["snapshot"]
    assert back["p_home"] == initial["p_home"]
    assert back["p_home"] + back["p_away"] == 1
    assert back["home_decimal_odds"] == 1 / back["p_home"]
    assert len((tmp_path / "news.jsonl").read_text().splitlines()) == 2


def test_future_wrong_game_and_unmatched_player_never_enter_state(tmp_path):
    feed = Feed([row("future", published=NEWS_AT + pd.Timedelta(minutes=1)),
                 row("observed", observed_at=NEWS_AT + pd.Timedelta(minutes=1)),
                 row("wrong_game", game_id="another"), row("wrong_player", player_id=99)])
    # row's extras deliberately override schema values.
    agent = setup_agent(tmp_path, feed)
    s = agent.poll(NEWS_AT)["snapshot"]
    assert s["new_news_ids"] == [] and s["p_home"] == s["baseline"]["p_home"]


def test_latest_status_wins_and_official_wins_ties(tmp_path):
    feed = Feed([row("official", "out", source="nba_injury_report"), row("rss", "available", source="espn_rss")])
    agent = setup_agent(tmp_path, feed)
    s = agent.poll(NEWS_AT)["snapshot"]
    assert s["expected_lost_share"] == {"7": 1.0}
    feed.rows = [row("old", "available", NEWS_AT - pd.Timedelta(minutes=30))]
    s = agent.poll(NEWS_AT + pd.Timedelta(minutes=10))["snapshot"]
    assert s["expected_lost_share"] == {"7": 1.0}


def test_uncertain_status_and_minutes_limit_combine_without_double_counting(tmp_path):
    feed = Feed([row("q", "questionable"), row("limit", "minutes_limit", minutes_limit=20)])
    s = setup_agent(tmp_path, feed).poll(NEWS_AT)["snapshot"]
    assert s["expected_lost_share"]["7"] == pytest.approx(1 - 0.5 * 20 / 34)
    assert s["p_home"] < 0.62


def test_outage_preserves_factors_and_restart_deduplicates(tmp_path):
    feed = Feed([row()])
    agent = setup_agent(tmp_path, feed)
    first = agent.poll(NEWS_AT)["snapshot"]
    feed.rows, feed.errors = [], [{"source": "test", "error": "timeout"}]
    degraded = agent.poll(NEWS_AT + pd.Timedelta(minutes=5))["snapshot"]
    assert degraded["p_home"] == first["p_home"] and degraded["news_health"] == "degraded"
    feed.rows, feed.errors = [row()], []
    resumed = setup_agent(tmp_path, feed).poll(NEWS_AT + pd.Timedelta(minutes=10))["snapshot"]
    assert resumed["p_home"] == first["p_home"] and resumed["new_news_ids"] == []
    assert resumed["delta_home"] == 0
    with pytest.raises(ValueError, match="backward"):
        setup_agent(tmp_path, feed).poll(NEWS_AT)


def test_pregame_snapshot_records_fresh_polymarket_block(tmp_path):
    calls = []

    def market_provider(game, now):
        calls.append((game.game_id, now))
        return {"source": "polymarket", "is_live": True, "status": "OK",
                "source_state": "SOURCE DATA AVAILABLE", "fetched_at": now.isoformat(),
                "age_seconds": 0.0, "usable_for_future_analysis": True,
                "reasons": [], "markets": [{"token_id": "live-token", "bid": 0.4, "ask": 0.41}]}

    tables = make_tables()
    players = pd.DataFrame({"player_id": [7], "player_name": ["Test Star"]})
    game = next(tables["games"].iloc[[1]].itertuples(index=False))
    agent = PregameAgent(tables, players, game, record_forecaster, TableNews(tables["news"]),
                         tmp_path, market_provider=market_provider)
    result = agent.poll(NEWS_AT)["snapshot"]
    assert calls == [("g2", NEWS_AT)]
    assert result["polymarket"]["is_live"]
    assert result["polymarket"]["markets"][0]["token_id"] == "live-token"


def test_pregame_polymarket_failure_is_visible_and_does_not_crash(tmp_path):
    def fail(*_args):
        raise TimeoutError("offline")

    tables = make_tables()
    players = pd.DataFrame({"player_id": [7], "player_name": ["Test Star"]})
    game = next(tables["games"].iloc[[1]].itertuples(index=False))
    agent = PregameAgent(tables, players, game, record_forecaster, TableNews(tables["news"]),
                         tmp_path, market_provider=fail)
    result = agent.poll(NEWS_AT)["snapshot"]
    assert result["polymarket"]["status"] == "UNAVAILABLE"
    assert not result["polymarket"]["usable_for_future_analysis"]


def test_no_search_or_forecast_at_or_after_tip(tmp_path):
    feed = Feed([row()])
    agent = setup_agent(tmp_path, feed)
    assert agent.poll(TIP2)["stopped"]
    assert feed.calls == [] and not (tmp_path / "state.json").exists()


def test_network_request_crossing_tip_does_not_emit_update(tmp_path):
    feed = Feed([row()])
    agent = setup_agent(tmp_path, feed, clock=lambda: TIP2)
    out = agent.poll(TIP2 - pd.Timedelta(seconds=1))
    assert out["stopped"]
    snapshots = [json.loads(s) for s in (tmp_path / "snapshots.jsonl").read_text().splitlines()]
    assert [s["phase"] for s in snapshots] == ["baseline"]


def test_scheduler_polls_on_fixed_cadence_and_stops_at_tip(tmp_path):
    feed = Feed()
    agent = setup_agent(tmp_path, feed)
    start = TIP2 - pd.Timedelta(minutes=12)
    run_loop(agent, interval_seconds=300, start=start, replay_mode=True)
    assert feed.calls == [start, start + pd.Timedelta(minutes=5), start + pd.Timedelta(minutes=10)]


def test_live_scheduler_waits_in_short_chunks(tmp_path):
    feed = Feed()
    agent = setup_agent(tmp_path, feed)
    current = [TIP2 - pd.Timedelta(seconds=120)]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        current[0] += pd.Timedelta(seconds=seconds)
    run_loop(agent, interval_seconds=60, clock=lambda: current[0], sleeper=sleep)
    assert len(feed.calls) == 2 and max(sleeps) <= 30


@pytest.mark.parametrize("value", [0, 1, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(value):
    with pytest.raises(ValueError):
        odds(value)


def test_rss_requires_current_game_context_and_unambiguous_player():
    players = pd.DataFrame({"player_id": [7, 8], "player_name": ["Test Star", "Other Player"]})
    game = next(make_tables()["games"].iloc[[1]].itertuples(index=False))
    titles = ["Test Star ruled out tonight", "Test Star out for tomorrow's game",
              "Test Star and Other Player ruled out tonight", "Test Star could be ruled out tonight",
              "Test Star will have a 20-minute limit tonight", "Unknown Player ruled out tonight"]
    xml = "<rss><channel>" + "".join(
        f"<item><title>{title}</title><link>https://example.com/{i}</link>"
        "<pubDate>Sun, 01 Feb 2026 17:00:00 GMT</pubDate></item>" for i, title in enumerate(titles)) + "</channel></rss>"
    rows = rss_rows(xml, game, players, {7, 8}, NEWS_AT - pd.Timedelta(days=1), NEWS_AT)
    assert [r["status"] for r in rows] == ["out", "review", "review", "review", "review", "minutes_limit"]
    assert rows[-1]["minutes_limit"] == 20


def test_rss_verification_page_is_a_source_failure():
    with pytest.raises(ValueError, match="non-RSS"):
        rss_rows("<html><body>Verify your browser</body></html>", None, None, None, NEWS_AT, NEWS_AT)


@pytest.mark.parametrize("text", ["Test Star not ruled out tonight", "Test Star was out last night",
                                  "Test Star may return tonight", "Other Player ruled out; Test Star comments"])
def test_ambiguous_prose_is_review_only(text):
    assert classify_sentence(text, "Test Star")[0] == "review"


def test_table_provider_does_not_reveal_late_observation():
    frame = pd.DataFrame([row(observed_at=NEWS_AT + pd.Timedelta(minutes=5))])
    game = next(make_tables()["games"].iloc[[1]].itertuples(index=False))
    assert TableNews(frame).fetch(game, None, None, NEWS_AT - pd.Timedelta(days=1), NEWS_AT).rows == []


def test_missing_optional_values_and_invalid_limit_do_not_break_logging(tmp_path):
    feed = Feed([row("out", minutes_limit=float("nan")),
                 row("bad-limit", "minutes_limit", minutes_limit=float("nan"))])
    s = setup_agent(tmp_path, feed).poll(NEWS_AT)["snapshot"]
    assert s["p_home"] < 0.62 and s["news_health"] == "degraded"
    logged = [json.loads(line) for line in (tmp_path / "news.jsonl").read_text().splitlines()]
    assert len(logged) == 2 and logged[0]["minutes_limit"] is None


def test_forecaster_history_refreshes_when_a_game_finishes_on_the_same_day():
    from forecast.api import Forecaster
    from replay import AsOf
    f, tables = Forecaster(), make_tables()
    before = AsOf(tables, tables["games"].final_at.iloc[0] - pd.Timedelta(seconds=1), {})
    after = AsOf(tables, tables["games"].final_at.iloc[0], {})
    assert f.history(before).team_games(1, TIP2) == []
    assert f.history(after).team_games(1, TIP2) == ["g1"]


def test_official_adapter_matches_report_game_and_maps_names(monkeypatch):
    import data_sources.live_news as module
    from types import SimpleNamespace
    game = next(make_tables()["games"].iloc[[1]].itertuples(index=False))
    raw = [{"Matchup": "AWY@HOM", "GameDate": "02/01/2026", "PlayerName": "Star,Test",
            "CurrentStatus": "Out", "Reason": "Knee"},
           {"Matchup": "AWY@HOM", "GameDate": "02/02/2026", "PlayerName": "Star,Test",
            "CurrentStatus": "Available", "Reason": ""}]
    monkeypatch.setattr(module, "parse_pdf", lambda path: raw)
    provider = LiveNews(rss_urls=[], clock=lambda: NEWS_AT, x_enabled=False)
    provider.session.get = lambda *a, **k: SimpleNamespace(
        status_code=200, content=b"%PDF fake", raise_for_status=lambda: None)
    players = pd.DataFrame({"player_id": [7], "player_name": ["Test Star"]})
    batch = provider.fetch(game, players, {7}, NEWS_AT - pd.Timedelta(days=1), NEWS_AT)
    assert batch.errors == [] and len(batch.rows) == 1
    assert batch.rows[0]["player_id"] == 7 and batch.rows[0]["status"] == "out"
    assert batch.rows[0]["observed_at"] == NEWS_AT


def test_source_failures_logged_and_no_post_tip_http(monkeypatch):
    import requests
    game = next(make_tables()["games"].iloc[[1]].itertuples(index=False))
    provider = LiveNews(rss_urls=["https://www.espn.com/espn/rss/nba/news"], clock=lambda: NEWS_AT, x_enabled=False)
    calls = []
    def fail(url, **kwargs):
        calls.append(url)
        raise requests.Timeout("timeout")
    provider.session.get = fail
    players = pd.DataFrame({"player_id": [7], "player_name": ["Test Star"]})
    batch = provider.fetch(game, players, {7}, NEWS_AT - pd.Timedelta(days=1), NEWS_AT)
    assert len(batch.errors) == 2 and batch.rows == []
    provider.clock = lambda: TIP2
    provider.fetch(game, players, {7}, NEWS_AT - pd.Timedelta(days=1), TIP2)
    assert len(calls) == 2


def test_weighted_m4_features_match_binary_endpoints():
    from forecast.history import History
    from forecast.win import game_features
    tables = make_tables()
    game = next(tables["games"].iloc[[1]].itertuples(index=False))
    h = History(tables["player_games"].iloc[:1], tables["games"].iloc[:1])
    full = game_features(h, 1, 2, TIP2, out=[7])
    weighted = game_features(h, 1, 2, TIP2, availability={7: 1.0})
    half = game_features(h, 1, 2, TIP2, availability={7: 0.5})
    assert weighted == full
    assert half["missing_min_diff"] == pytest.approx(full["missing_min_diff"] / 2)
    assert half["missing_pts_diff"] == pytest.approx(full["missing_pts_diff"] / 2)
