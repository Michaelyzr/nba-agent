"""In-play temporal boundaries, event semantics, pricing and provider contracts."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from agents.inplay import InPlayAgent, run_loop
from data_sources.inplay import TableInPlay
from data_sources.inplay_news import classify, text_events
from data_sources.inplay_score import LiveScore, espn_batch, nba_batch, source_time
from data_sources.inplay_types import InPlayBatch, clock_seconds
from data_sources.inplay_x import InPlayX
from data_sources.news_registry import NewsRegistry
from forecast.inplay import InPlayWinModel, remaining_seconds
from evaluation.fixtures import make_tables, TIP2

NOW = TIP2 + pd.Timedelta(hours=1)


def setup():
    tables = make_tables()
    tables["player_games"] = pd.concat([tables["player_games"], pd.DataFrame({
        "game_id": ["g1"], "player_id": [8], "team_id": [2], "min": [30.0], "pts": [20],
        "fga": [15], "fta": [4], "usage": [0.2], "started": [True]})], ignore_index=True)
    raw = next(tables["games"].iloc[[1]].itertuples(index=False))
    game = SimpleNamespace(**{**raw._asdict(), "home_team": "BOS", "away_team": "LAL"})
    players = pd.DataFrame({"player_id": [7, 8], "player_name": ["Test Star", "Other Player"]})
    return tables, players, game


def score(at=NOW, **kwargs):
    return {"game_id": "g2", "phase": "live", "home_score": 50, "away_score": 49,
            "period": 2, "clock_seconds": 360, "observed_at": at, "updated_at": at,
            "source": "nba_live", "player_minutes": {"7": 20, "8": 18},
            "player_teams": {"7": 1, "8": 2}, "roster": [7, 8], **kwargs}


def event(status="injury_out", source="x:shamscharania", at=NOW, pid=7, **kwargs):
    return {"event_id": f"{source}-{status}-{at}-{pid}", "game_id": "g2", "player_id": pid,
            "status": status, "source": source, "published_at": at, "observed_at": at,
            "url": "https://example.com/event", "text": status, **kwargs}


class Feed:
    registry = NewsRegistry.load()
    def __init__(self, rows=(), current=None, errors=()):
        self.rows, self.current, self.errors, self.calls = list(rows), current or score(), list(errors), []
    def fetch(self, *args):
        self.calls.append(args[-1])
        return InPlayBatch(score=self.current, events=self.rows, errors=self.errors)
    def dump_state(self):
        return {"cursor": "feed-cursor"}
    def load_state(self, state):
        self.restored = state
    def close(self):
        self.closed = True


def agent(tmp_path, feed=None, **kwargs):
    tables, people, game = setup()
    return InPlayAgent(tables, people, game, feed or Feed(), tmp_path,
                       initial_p_home=0.62, **kwargs)


def test_waits_before_tip_and_never_queries_pregame_news(tmp_path):
    f = Feed()
    a = agent(tmp_path, f)
    s = a.poll(TIP2 - pd.Timedelta(seconds=1))["snapshot"]
    assert s["quote_state"] == "waiting_for_tip" and s["p_home"] is None and f.calls == []
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "snapshots.jsonl").exists()
    assert (tmp_path / "inplay_state.json").exists()


def test_game_score_and_clock_change_probabilities(tmp_path):
    f = Feed()
    a = agent(tmp_path, f)
    first = a.poll(NOW)["snapshot"]
    f.current = score(NOW + pd.Timedelta(seconds=5), home_score=60)
    second = a.poll(NOW + pd.Timedelta(seconds=5))["snapshot"]
    assert second["p_home"] > first["p_home"]
    assert second["p_home"] + second["p_away"] == 1
    assert second["home_decimal_odds"] == pytest.approx(1 / second["p_home"])
    f.current = score(NOW + pd.Timedelta(seconds=10), home_score=60, period=4, clock_seconds=30)
    late = a.poll(NOW + pd.Timedelta(seconds=10))["snapshot"]
    assert late["p_home"] > second["p_home"]


def test_injury_return_and_duplicate_events_do_not_compound(tmp_path):
    f = Feed()
    a = agent(tmp_path, f)
    baseline = a.poll(NOW)["snapshot"]
    f.rows = [event()]
    injury = a.poll(NOW)["snapshot"]
    repeat = a.poll(NOW)["snapshot"]
    assert injury["p_home"] < baseline["p_home"] and repeat["p_home"] == injury["p_home"]
    assert repeat["new_event_ids"] == []
    f.rows.append(event("returned", at=NOW + pd.Timedelta(seconds=1)))
    back = a.poll(NOW + pd.Timedelta(seconds=1))["snapshot"]
    assert back["p_home"] == baseline["p_home"]
    assert len((tmp_path / "inplay_events.jsonl").read_text().splitlines()) == 2


def test_away_removal_helps_home_and_late_impact_is_remaining_only(tmp_path):
    f = Feed([event(pid=8)])
    a = agent(tmp_path, f)
    early = a.poll(NOW)["snapshot"]
    assert early["news_margin"] > 0
    f.current = score(NOW + pd.Timedelta(seconds=5), period=4, clock_seconds=30)
    late = a.poll(NOW + pd.Timedelta(seconds=5))["snapshot"]
    assert 0 < late["news_margin"] < early["news_margin"]


def test_ejection_is_not_undone_by_ordinary_return_but_can_be_rescinded(tmp_path):
    f = Feed([event("ejected", "nba_live")])
    a = agent(tmp_path, f)
    a.poll(NOW)
    f.rows.append(event("returned", "nba_live", NOW + pd.Timedelta(seconds=1)))
    s = a.poll(NOW + pd.Timedelta(seconds=1))["snapshot"]
    assert s["factors"][0]["status"] == "ejected"
    f.rows.append(event("rescinded", "nba_live", NOW + pd.Timedelta(seconds=2)))
    s = a.poll(NOW + pd.Timedelta(seconds=2))["snapshot"]
    assert s["news_margin"] == 0 and s["factors"][0]["status"] == "rescinded"


def test_authority_conflict_preserved_and_restart_has_its_own_state(tmp_path):
    f = Feed([event("injury_out", "espn_rss"),
              event("returned", "x:shamscharania", NOW + pd.Timedelta(seconds=1))])
    a = agent(tmp_path, f)
    s = a.poll(NOW + pd.Timedelta(seconds=1))["snapshot"]
    assert s["factors"][0]["source"] == "espn_rss" and s["conflicts"]
    resumed = agent(tmp_path, Feed()).poll(NOW + pd.Timedelta(seconds=2))["snapshot"]
    assert resumed["p_home"] == s["p_home"] and resumed["conflicts"] == s["conflicts"]
    state = json.loads((tmp_path / "inplay_state.json").read_text())
    assert state["config"]["pipeline"] == "inplay-v1" and state["provider"]["cursor"] == "feed-cursor"


def test_new_injury_after_return_is_a_new_episode(tmp_path):
    f = Feed([event("returned", "espn_rss", NOW - pd.Timedelta(minutes=5)), event()])
    s = agent(tmp_path, f).poll(NOW)["snapshot"]
    assert s["factors"][0]["status"] == "injury_out" and s["news_margin"] < 0
    assert s["factors"][0]["confirmation"] == "secondary_only"


def test_explicit_prognosis_refines_departure_without_inventing_a_conflict(tmp_path):
    f = Feed([event("left_injured", "nba_live", NOW - pd.Timedelta(seconds=5)), event()])
    s = agent(tmp_path, f).poll(NOW)["snapshot"]
    assert s["factors"][0]["status"] == "injury_out"


def test_primary_return_can_correct_a_secondary_ejection_report(tmp_path):
    f = Feed([event("ejected", "x:shamscharania", NOW - pd.Timedelta(seconds=5)),
              event("returned", "espn_rss")])
    s = agent(tmp_path, f).poll(NOW)["snapshot"]
    assert s["factors"][0]["source"] == "espn_rss" and s["news_margin"] == 0


@pytest.mark.parametrize("row", [
    event(at=TIP2 - pd.Timedelta(seconds=1)),
    event(at=NOW + pd.Timedelta(seconds=1)),
    event(observed_at=NOW + pd.Timedelta(seconds=1)),
    event(pid=999), event(game_id="wrong"),
])
def test_pre_tip_future_wrong_game_and_unknown_player_are_rejected(tmp_path, row):
    s = agent(tmp_path, Feed([row])).poll(NOW)["snapshot"]
    assert s["factors"] == [] and s["new_event_ids"] == []


def test_stale_score_or_missing_feed_never_reprices_cached_state(tmp_path):
    f = Feed(current=score(updated_at=NOW - pd.Timedelta(seconds=31)))
    a = agent(tmp_path, f)
    stale = a.poll(NOW)["snapshot"]
    assert stale["quote_state"] == "stale_score" and stale["p_home"] is None
    f.current = score()
    assert a.poll(NOW)["snapshot"]["p_home"] is not None
    f.current = None
    f.rows = [event()]
    down = a.poll(NOW + pd.Timedelta(seconds=1))["snapshot"]
    assert down["quote_state"] == "score_unavailable" and down["home_decimal_odds"] is None


def test_invalid_or_regressed_score_cannot_price_cached_state(tmp_path):
    f = Feed()
    a = agent(tmp_path, f)
    a.poll(NOW)
    f.current = score(game_id="other")
    assert a.poll(NOW)["snapshot"]["p_home"] is None
    f.current = score(period=1)
    assert a.poll(NOW)["snapshot"]["p_home"] is None


def test_source_timestamp_unknown_is_explicit(tmp_path):
    s = agent(tmp_path, Feed(current=score(updated_at=None))).poll(NOW)["snapshot"]
    assert s["p_home"] is not None and s["freshness"] == "source_timestamp_unverified"
    assert s["score_source_age_seconds"] is None


def test_final_and_overtime_are_distinct_and_final_stops_requests(tmp_path):
    f = Feed(current=score(phase="live", period=4, clock_seconds=0, home_score=99, away_score=99))
    a = agent(tmp_path, f)
    tied = a.poll(NOW)["snapshot"]
    assert remaining_seconds(tied["score"]) == 300 and 0 < tied["p_home"] < 1
    f.current = score(phase="final", period=5, clock_seconds=0, home_score=103, away_score=101)
    final = a.poll(NOW)
    assert final["stopped"] and final["snapshot"]["p_home"] == 1
    assert final["snapshot"]["away_decimal_odds"] is None
    count = len(f.calls)
    a.poll(NOW + pd.Timedelta(seconds=1))
    assert len(f.calls) == count


def test_suspension_is_not_a_final_result(tmp_path):
    s = agent(tmp_path, Feed(current=score(phase="suspended"))).poll(NOW)["snapshot"]
    assert s["p_home"] is None and s["quote_state"] == "suspended"


def test_refuses_pregame_directory_and_resume_config_changes(tmp_path):
    (tmp_path / "state.json").write_text("{}")
    with pytest.raises(ValueError, match="pregame"):
        agent(tmp_path)
    (tmp_path / "state.json").unlink()
    agent(tmp_path).poll(NOW)
    with pytest.raises(ValueError, match="configuration"):
        agent(tmp_path, point_value=0.2)


@pytest.mark.parametrize("text,status", [
    ("Test Star has been ejected", "ejected"),
    ("Test Star fouled out", "fouled_out"),
    ("Test Star will not return", "injury_out"),
    ("Test Star left the game with an ankle injury", "left_injured"),
    ("Star is questionable to return", "questionable_return"),
    ("Test Star has returned to the game", "returned"),
    ("Test Star's ejection was rescinded", "rescinded"),
    ("Test Star was not ejected", "review"),
    ("Test Star wasn't ejected", "review"),
    ("Test Star will return to the game shortly", "review"),
    ("Test Star was ejected in pregame warmup", "review"),
    ("Test Star may not return", "review"),
    ("Test Star was ejected last night", "review"),
    ("Test Star substituted out", "review"),
    ("Test Star ejected Other Player from a video game", "review"),
])
def test_only_explicit_in_game_statuses_are_actionable(text, status):
    _, people, game = setup()
    rows = text_events(text, game, people, {7, 8}, "x:celtics", NOW, NOW, "https://x.com/1", "1")
    assert rows and all(r["status"] == status for r in rows)


def test_table_provider_never_reads_future_observations():
    _, people, game = setup()
    f = TableInPlay(pd.DataFrame([score(), score(NOW + pd.Timedelta(seconds=1), home_score=80)]),
                   pd.DataFrame([event(observed_at=NOW + pd.Timedelta(seconds=1))]))
    b = f.fetch(game, people, {7, 8}, NOW)
    assert b.score["home_score"] == 50 and b.events == []


def nba_payload():
    game = {"gameId": "0022500001", "gameTimeUTC": TIP2.isoformat(), "gameStatus": 2,
            "period": 2, "gameClock": "PT06M00.00S",
            "homeTeam": {"teamTricode": "BOS", "score": 50, "players": [
                {"personId": 70007, "name": "Test Star", "status": "ACTIVE",
                 "statistics": {"minutes": "PT20M00S", "foulsPersonal": 2}}]},
            "awayTeam": {"teamTricode": "LAL", "score": 49, "players": [
                {"personId": 80008, "name": "Other Player", "status": "ACTIVE",
                 "statistics": {"minutes": "PT18M00S", "foulsPersonal": 6}}]}}
    pbp = {"game": {"gameId": "0022500001", "actions": [
        {"personId": 70007, "actionNumber": 40, "actionType": "ejection", "description": "Star ejected",
         "timeActual": NOW.isoformat(), "edited": NOW.isoformat()}]}}
    return {"game": game}, pbp


def test_nba_ids_are_mapped_by_name_and_six_fouls_are_terminal():
    _, people, game = setup()
    box, pbp = nba_payload()
    b = nba_batch(box, pbp, game, people, {7, 8}, NOW, NOW.isoformat())
    assert {(e["player_id"], e["status"]) for e in b.events} == {(7, "ejected"), (8, "fouled_out")}
    assert b.score["player_minutes"] == {"7": 20, "8": 18}
    box["game"]["homeTeam"]["teamTricode"] = "NYK"
    with pytest.raises(ValueError, match="teams"):
        nba_batch(box, pbp, game, people, {7, 8}, NOW)


def espn_payload():
    return {"header": {"id": "g2", "competitions": [{"date": TIP2.isoformat(),
        "status": {"type": {"state": "in"}, "period": 2, "clock": 360},
        "competitors": [{"homeAway": "home", "score": "50", "team": {"abbreviation": "BOS"}},
                        {"homeAway": "away", "score": "49", "team": {"abbreviation": "LAL"}}]}]},
        "boxscore": {"players": [{"team": {"abbreviation": "BOS"}, "statistics": [{
            "keys": ["minutes", "fouls"], "athletes": [{"athlete": {"id": "999", "displayName": "Test Star"},
                                                         "stats": ["20", "6"]}]}]}]},
        "plays": [{"id": "p1", "text": "Test Star has been ejected", "wallclock": NOW.isoformat(),
                   "participants": [{"athlete": {"id": "999"}}]}]}


def test_espn_clock_and_player_mapping_are_explicit():
    _, people, game = setup()
    data = espn_payload()
    b = espn_batch(data, game, people, {7, 8}, NOW)
    assert b.score["period"] == 2 and b.score["clock_seconds"] == 360
    assert all(e["player_id"] == 7 for e in b.events)
    del data["header"]["competitions"][0]["status"]["clock"]
    with pytest.raises(ValueError, match="clock"):
        espn_batch(data, game, people, {7, 8}, NOW)


class Response:
    def __init__(self, data=None, status=200, headers=None):
        self.data, self.status_code, self.headers = data or {}, status, headers or {}
    def json(self):
        return self.data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("fixture HTTP error")


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)
    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def test_source_timestamps_do_not_guess_naive_timezone():
    r = Response(headers={"Last-Modified": "Mon, 02 Feb 2026 01:30:00 GMT"})
    assert source_time(r, {})[0] == NOW.isoformat()
    assert source_time(Response(), {"meta": {"time": "2026-02-02 01:30:00"}})[0] is None
    assert source_time(Response(), {"meta": {"lastUpdatedAt": NOW.isoformat()}})[0] == NOW.isoformat()


def test_nba_failure_is_explicit_even_when_espn_fallback_works():
    _, people, game = setup()
    data = espn_payload()
    competition = data["header"]["competitions"][0]
    board = {"events": [{"id": "g2", "competitions": [competition]}]}
    session = Session([Response(status=403), Response(data), Response(board)])
    client = LiveScore("0022500001", session=session, clock=lambda: NOW)
    batch = client.fetch(game, people, {7, 8}, NOW)
    assert batch.score["source"] == "espn_live" and batch.errors[0]["source"] == "nba_live"
    assert client.retry_nba_at and batch.coverage[-1]["fallback"]


def post(handle="celtics", text="Test Star will not return", at=NOW, post_id="1"):
    return {"data": {"id": post_id, "author_id": "11", "created_at": at.isoformat(), "text": text},
            "includes": {"users": [{"id": "11", "username": handle}]}}


def test_x_missing_inplay_key_does_not_use_pregame_key(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "pregame-secret")
    _, people, game = setup()
    client = InPlayX(NewsRegistry.load(), token="")
    b = client.fetch(game, people, {7, 8}, NOW)
    assert b.errors and b.coverage[0]["state"] == "unconfigured"
    assert "pregame-secret" not in json.dumps(client.dump_state())


def test_x_first_observation_author_allowlist_and_pre_tip_filter():
    _, people, game = setup()
    client = InPlayX(NewsRegistry.load(), token="")
    client.accept(post(), NOW)
    client.accept(post(handle="Untrusted"), NOW)
    client.accept(post(at=TIP2 - pd.Timedelta(seconds=1)), NOW)
    b = client.fetch(game, people, {7, 8}, NOW + pd.Timedelta(seconds=5))
    assert len(b.events) == 1 and b.events[0]["observed_at"] == NOW.isoformat()
    assert b.events[0]["source"] == "x:celtics"


def test_x_rule_addition_preserves_other_rules_and_only_targets_matchup():
    _, _, game = setup()
    session = Session([Response({"data": [{"id": "other", "tag": "someone-else", "value": "cat"}]}),
                       Response({"data": [{"id": "own"}]}, status=201)])
    client = InPlayX(NewsRegistry.load(), token="secret", rest_session=session)
    tag = client.ensure_rule(game)
    assert tag.startswith("nba-agent-inplay:")
    added = session.calls[1][2]["json"]
    assert "delete" not in added and "from:celtics" in added["add"][0]["value"]
    assert "from:Lakers" in added["add"][0]["value"] and "from:NYPost_Lewis" not in added["add"][0]["value"]


def test_x_recovery_preserves_pagination_and_rate_limit_state():
    _, _, game = setup()
    p = post()
    first = {"data": [p["data"]], "includes": p["includes"], "meta": {"next_token": "a"}}
    second = {"data": [], "meta": {"next_token": "b"}}
    session = Session([Response(first), Response(second), Response(status=429)])
    client = InPlayX(NewsRegistry.load(), token="secret", rest_session=session)
    errors = client.recover(game, NOW)
    assert errors and client.pending["next_token"] == "b" and client.cursor is None
    saved = client.dump_state()
    assert "secret" not in json.dumps(saved)
    resumed = InPlayX(NewsRegistry.load(), token="secret", rest_session=session)
    resumed.load_state(saved)
    resumed.recover(game, NOW + pd.Timedelta(seconds=5))
    assert session.calls[-1][2]["params"]["next_token"] == "b"
    assert resumed.retry_at and resumed.pending


def test_stream_queue_overflow_is_visible():
    _, people, game = setup()
    client = InPlayX(NewsRegistry.load(), token="", queue_limit=1)
    client.accept(post(post_id="1"), NOW)
    client.accept(post(post_id="2"), NOW)
    b = client.fetch(game, people, {7, 8}, NOW)
    assert b.coverage[0]["dropped"] == 1 and len(b.evidence) == 1


@pytest.mark.parametrize("raw,seconds", [("PT12M00.00S", 720), ("PT00M30.5S", 30.5), ("1:20", 80)])
def test_clock_formats(raw, seconds):
    assert clock_seconds(raw) == seconds


def test_replay_loop_closes_provider_and_stops_at_final(tmp_path):
    f = Feed(current=score(phase="final", home_score=100, away_score=90))
    a = agent(tmp_path, f)
    run_loop(a, replay_times=[NOW, NOW + pd.Timedelta(seconds=1)])
    assert len(f.calls) == 1 and f.closed


def test_live_model_monotonicity_and_overtime():
    model = InPlayWinModel()
    assert model.predict(score(), 0.62) > model.predict(score(home_score=40), 0.62)
    assert model.predict(score(), 0.62, -3) < model.predict(score(), 0.62)
    assert model.predict(score(period=1, clock_seconds=720, home_score=0, away_score=0), 0.62) == pytest.approx(0.62)
    assert remaining_seconds(score(period=5, clock_seconds=180)) == 180


def test_nba_retry_deadline_is_not_extended_by_every_poll():
    _, people, game = setup()
    box, pbp = nba_payload()
    session = Session([Response(status=403), Response(box), Response(pbp)])
    client = LiveScore("0022500001", espn_fallback=False, session=session, clock=lambda: NOW + pd.Timedelta(seconds=60))
    assert client.fetch(game, people, {7, 8}, NOW).score is None
    deadline = client.retry_nba_at
    assert client.fetch(game, people, {7, 8}, NOW + pd.Timedelta(seconds=5)).score is None
    assert client.retry_nba_at == deadline and len(session.calls) == 1
    assert client.fetch(game, people, {7, 8}, NOW + pd.Timedelta(seconds=60)).score["source"] == "nba_live"
    assert client.retry_nba_at is None


def test_current_box_roster_maps_transferred_player_and_placeholder_fouls():
    _, people, game = setup()
    data = espn_payload()
    # Player is now in this game's box score, but absent from old team history.
    data["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["stats"] = ["20", "--"]
    batch = espn_batch(data, game, people, {8}, NOW)
    assert batch.score["roster"] == [7]
    assert batch.score["player_teams"]["7"] == game.home_team_id
    assert [e["status"] for e in batch.events] == ["ejected"]


def test_espn_utah_alias_is_local_to_inplay():
    _, people, game = setup()
    data = espn_payload()
    game.home_team = "UTA"
    data["header"]["competitions"][0]["competitors"][0]["team"]["abbreviation"] = "UTAH"
    data["boxscore"]["players"][0]["team"]["abbreviation"] = "UTAH"
    assert espn_batch(data, game, people, {7, 8}, NOW).score["player_teams"]["7"] == game.home_team_id


def test_espn_authority_overrides_conflicting_official_x_report(tmp_path):
    f = Feed([event(source="x:celtics"), event("returned", source="espn_live")])
    snap = agent(tmp_path, f).poll(NOW)["snapshot"]
    assert snap["factors"][0]["source"] == "espn_live"
    assert snap["news_margin"] == 0 and snap["conflicts"]


def test_x_expired_retry_deadline_is_replaced_after_another_failure():
    _, _, game = setup()
    session = Session([Response(status=500)])
    client = InPlayX(NewsRegistry.load(), token="secret", rest_session=session)
    client.retry_at = (NOW - pd.Timedelta(seconds=1)).isoformat()
    assert client.recover(game, NOW)
    assert pd.Timestamp(client.retry_at) == NOW + pd.Timedelta(seconds=30)
    client.recover(game, NOW + pd.Timedelta(seconds=1))
    assert len(session.calls) == 1


def test_x_recovery_receipt_after_fetch_start_is_not_dropped():
    _, people, game = setup()
    p = post()
    session = Session([Response({"data": [p["data"]], "includes": p["includes"]})])
    received = NOW + pd.Timedelta(seconds=2)
    client = InPlayX(NewsRegistry.load(), token="secret", rest_session=session, clock=lambda: received)
    client.start = lambda game: None  # Offline fixture; no background HTTP.
    batch = client.fetch(game, people, {7, 8}, NOW)
    assert len(batch.events) == 1 and batch.events[0]["observed_at"] == received.isoformat()


def test_replay_selects_latest_score_across_timezones():
    _, people, game = setup()
    older = score(observed_at=(NOW - pd.Timedelta(seconds=10)).tz_convert("Asia/Shanghai").isoformat(), home_score=40)
    newer = score(observed_at=NOW.isoformat(), home_score=50)
    batch = TableInPlay(pd.DataFrame([older, newer])).fetch(game, people, {7, 8}, NOW)
    assert batch.score["home_score"] == 50


def test_trained_model_uses_game_weights_and_preserves_score_direction():
    from forecast.inplay import features
    rows = []
    for i in range(24):
        for margin in (-15, 15):
            rows.append({"game_id": str(i) + str(margin), "home_win": int(margin > 0),
                         **features(score(home_score=50 + margin), 0.5)})
    model = InPlayWinModel().fit(pd.DataFrame(rows))
    assert model.predict(score(home_score=65), 0.5) > 0.5
    assert model.predict(score(home_score=35), 0.5) < 0.5
    assert model.validation is None  # Fitting alone does not certify calibration.


def training_table():
    from forecast.inplay import features
    rows = []
    for i, (day, final_day, win) in enumerate([(20, 20, 0), (21, 21, 1), (22, 22, 0),
                                              (31, 33, 1), (34, 34, 0), (35, 35, 1)]):
        tip = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=day - 1)
        final = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=final_day - 1, hours=3)
        for minute in range(8):
            rows.append({"game_id": str(i), "tip_time": tip, "final_at": final,
                         "as_of": tip + pd.Timedelta(minutes=minute), "home_win": win,
                         **features(score(home_score=60 if win else 40), 0.5)})
    return pd.DataFrame(rows)


def test_training_cli_excludes_games_not_finished_at_cutoff(tmp_path, monkeypatch):
    import pickle
    import forecast.inplay as module
    data, output = tmp_path / "train.parquet", tmp_path / "model.pkl"
    training_table().to_parquet(data)
    monkeypatch.setattr("sys.argv", ["inplay", "--training-data", str(data), "--split", "2026-02-01T00:00:00Z", "--output", str(output)])
    module.main()
    model = pickle.loads(output.read_bytes())
    assert model.validation["train_games"] == 3  # Jan 31 game finishes after cutoff.
    assert model.validation["test_games"] == 2
    assert 0 <= model.validation["brier"] <= 1


def test_training_cli_rejects_snapshots_after_the_outcome(tmp_path, monkeypatch):
    import forecast.inplay as module
    data = tmp_path / "leaking.parquet"
    frame = training_table()
    frame.loc[0, "as_of"] = frame.loc[0, "final_at"]
    frame.to_parquet(data)
    monkeypatch.setattr("sys.argv", ["inplay", "--training-data", str(data), "--split", "2026-02-01", "--output", str(tmp_path / "model.pkl")])
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path / "model.pkl").exists()
