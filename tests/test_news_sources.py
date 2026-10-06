"""Source routing, provenance, social ingestion and conflict policy; no network."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.graph import record_forecaster
from agents.pregame import PregameAgent
from data_sources.live_news import LiveNews, NewsBatch, post_rows, rss_rows
from data_sources.news_registry import NewsRegistry, TEAM_CODES
from data_sources.x_news import XNews
from test_agent_graph import make_tables, NEWS_AT, TIP2


@pytest.fixture
def registry():
    return NewsRegistry.load()


def game():
    g = next(make_tables()["games"].iloc[[1]].itertuples(index=False))
    return SimpleNamespace(**{**g._asdict(), "home_team": "BOS", "away_team": "LAL"})


def players():
    return pd.DataFrame({"player_id": [7, 8], "player_name": ["Test Star", "Other Player"]})


def article(source, status, at, news_id=None):
    return {"news_id": news_id or source + status, "game_id": "g2", "player_id": 7,
            "published_at": at, "observed_at": at, "source": source,
            "status": status, "url": "https://example.com/news", "text": status}


class Feed:
    def __init__(self, rows=(), items=()):
        self.rows, self.items = list(rows), list(items)
    def fetch(self, *args):
        return NewsBatch(rows=self.rows, items=self.items, coverage=[{"source": "espn_rss", "state": "ok"}])


def agent(tmp_path, feed):
    return PregameAgent(make_tables(), players(), game(), record_forecaster, feed, tmp_path)


def test_all_thirty_teams_have_official_and_reporter_accounts(registry):
    assert set(registry.data["teams"]) == TEAM_CODES
    for team in TEAM_CODES:
        plan = registry.for_game(SimpleNamespace(home_team=team, away_team="BOS"))
        selected = [a for a in plan["x_accounts"] if a["team"] == team]
        assert any(a["role"] == "team_official" for a in selected)
        assert any(a["role"] == "team_reporter" and a["evidence_urls"] for a in selected)
    assert registry.data["teams"]["ATL"]["official"] == "ATLHawks"
    assert registry.data["teams"]["OKC"]["reporter"]["handle"] == "CAlmanza1007"


def test_per_game_plan_does_not_query_other_teams(registry):
    plan = registry.for_game(game())
    assert {a["handle"] for a in plan["x_accounts"]} == {
        "ShamsCharania", "celtics", "SouichiTerada", "Lakers", "DanWoikeSports"}
    assert {m["source"] for m in plan["media"]} == {"espn_rss", "cbs_rss"}
    assert all(a["team"] in {None, "BOS", "LAL"} for a in plan["x_accounts"])
    assert len(registry.for_game(SimpleNamespace(home_team="GS", away_team="NY"))["missing_teams"]) == 0


def test_bad_registry_cannot_inject_query_operators(registry):
    raw = json.loads(json.dumps(registry.data))
    raw["insider"]["handle"] = "x OR from:evil"
    with pytest.raises(ValueError, match="invalid X handle"):
        NewsRegistry(raw)


def test_newer_social_conflict_cannot_override_authority(tmp_path):
    feed = Feed([article("espn_rss", "available", NEWS_AT - pd.Timedelta(minutes=20)),
                 article("x:shamscharania", "out", NEWS_AT - pd.Timedelta(minutes=5)),
                 article("x:souichiterada", "out", NEWS_AT)])
    a = agent(tmp_path, feed)
    snapshot = a.poll(NEWS_AT)["snapshot"]
    assert snapshot["expected_lost_share"] == {"7": 0.0}
    assert snapshot["factors"][0]["confirmation"] == "primary"
    conflict = snapshot["conflicts"][0]
    assert conflict["selected_source"] == "espn_rss" and conflict["newer_secondary_pending"]
    assert len(conflict["alternatives"]) == 2
    # No new articles next poll: losing evidence must survive a process restart.
    feed.rows = []
    resumed = agent(tmp_path, feed).poll(NEWS_AT + pd.Timedelta(minutes=1))["snapshot"]
    assert resumed["conflicts"] == snapshot["conflicts"]
    assert resumed["p_home"] == snapshot["p_home"]


def test_authority_correction_and_same_tier_recency(tmp_path):
    feed = Feed([article("espn_rss", "out", NEWS_AT),
                 article("cbs_rss", "available", NEWS_AT + pd.Timedelta(minutes=1))])
    a = agent(tmp_path, feed)
    assert a.poll(NEWS_AT)["snapshot"]["expected_lost_share"] == {"7": 1.0}
    s = a.poll(NEWS_AT + pd.Timedelta(minutes=1))["snapshot"]
    assert s["expected_lost_share"] == {"7": 0.0}
    # The newer correction from ESPN replaces only ESPN's older evidence.
    feed.rows.append(article("espn_rss", "available", NEWS_AT + pd.Timedelta(minutes=2), "correction"))
    s = a.poll(NEWS_AT + pd.Timedelta(minutes=2))["snapshot"]
    assert s["conflicts"] == []


def test_insider_used_provisionally_then_direct_team_announcement_wins(tmp_path):
    feed = Feed([article("x:souichiterada", "available", NEWS_AT),
                 article("x:shamscharania", "out", NEWS_AT - pd.Timedelta(minutes=1))])
    a = agent(tmp_path, feed)
    s = a.poll(NEWS_AT)["snapshot"]
    assert s["expected_lost_share"] == {"7": 1.0}
    assert s["factors"][0]["confirmation"] == "secondary_only"
    feed.rows.append(article("x:celtics", "available", NEWS_AT + pd.Timedelta(minutes=1)))
    s = a.poll(NEWS_AT + pd.Timedelta(minutes=1))["snapshot"]
    assert s["expected_lost_share"] == {"7": 0.0}
    assert s["factors"][0]["source_role"] == "team_official"


def test_unknown_feed_cannot_claim_authority_from_metadata(tmp_path):
    feed = Feed([article("espn_rss", "available", NEWS_AT),
                 {**article("rss:random.example", "out", NEWS_AT), "priority": 999}])
    assert agent(tmp_path, feed).poll(NEWS_AT)["snapshot"]["expected_lost_share"] == {"7": 0.0}


def test_unparsed_evidence_preserved_and_deduplicated(tmp_path):
    item = {"item_id": "lineup-image", "game_id": "g2", "published_at": NEWS_AT,
            "observed_at": NEWS_AT, "source": "x:celtics", "url": "https://x.com/celtics/status/1",
            "text": "Tonight's starting lineup", "review_required": True}
    feed = Feed(items=[item])
    a = agent(tmp_path, feed)
    first = a.poll(NEWS_AT)["snapshot"]
    assert first["new_evidence_count"] == 1 and first["factors"] == []
    second = a.poll(NEWS_AT + pd.Timedelta(minutes=1))["snapshot"]
    assert second["new_evidence_count"] == 0
    assert json.loads((tmp_path / "evidence.jsonl").read_text())["review_required"]


@pytest.mark.parametrize("text,statuses", [
    ("OUT: Test Star, Other Player", ["out", "out"]),
    ("Star (ankle): OUT", ["out"]),
    ("Test Star (ankle) will not play tonight.", ["out"]),
    ("Test Star is out; Other Player will play tonight", ["out", "available"]),
    ("Test Star could be out tonight", ["review"]),
    ("Test Star is out tomorrow", ["review"]),
    ("Test Star is OUT for the rest of the game", ["review"]),
    ("OUT: Test Star QUESTIONABLE: Other Player", ["review", "review"]),
    ("Test Star is out of form tonight", ["review"]),
    ("Test Star: out of form tonight", ["review"]),
    ("Test Star is available for trade", ["review"]),
    ("Test Star will play next game", ["review"]),
])
def test_short_social_updates(text, statuses):
    rows = post_rows(text, game(), players(), {7, 8}, "x:celtics", NEWS_AT, NEWS_AT,
                     "https://x.com/celtics/status/1", "post-1")
    assert [r["status"] for r in rows] == statuses


def test_ambiguous_surname_is_not_attached_to_a_player():
    p = pd.DataFrame({"player_id": [7, 8], "player_name": ["First Smith", "Second Smith"]})
    assert post_rows("Smith is out tonight", game(), p, {7, 8}, "x:celtics", NEWS_AT, NEWS_AT, "https://x.com", "1") == []


def test_feed_domains_are_checked_and_team_articles_are_retained():
    xml = "<rss><channel><item><title>Celtics change their starting lineup tonight</title>" \
          "<link>https://www.espn.com/news/1</link><pubDate>Sun, 01 Feb 2026 22:00:00 GMT</pubDate>" \
          "</item></channel></rss>"
    items = []
    assert rss_rows(xml, game(), players(), {7, 8}, NEWS_AT - pd.Timedelta(days=1), NEWS_AT,
                    "espn_rss", items, ["espn.com"]) == []
    assert len(items) == 1
    items = []
    rss_rows(xml.replace("www.espn.com", "fake.example"), game(), players(), {7, 8},
             NEWS_AT - pd.Timedelta(days=1), NEWS_AT, "espn_rss", items, ["espn.com"])
    assert items == []


class Response:
    def __init__(self, data=None, status=200, headers=None):
        self.data, self.status_code, self.headers = data or {}, status, headers or {}
    def json(self):
        return self.data


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def payload(handle="ShamsCharania", text="Test Star ruled out tonight", post_id="1", next_token=None):
    return {"data": [{"id": post_id, "author_id": "11", "text": text,
                      "created_at": (NEWS_AT - pd.Timedelta(minutes=2)).isoformat()}],
            "includes": {"users": [{"id": "11", "username": handle}]},
            "meta": {"next_token": next_token} if next_token else {}}


def test_x_fetch_uses_only_selected_authors_and_preserves_provenance(registry):
    session = Session([Response(payload())])
    client = XNews(registry, bearer_token="test-secret", clock=lambda: NEWS_AT, session=session)
    batch = client.fetch(game(), players(), {7, 8}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert batch.rows[0]["status"] == "out" and batch.rows[0]["source"] == "x:shamscharania"
    assert batch.items[0]["url"] == "https://x.com/ShamsCharania/status/1"
    params = session.calls[0][1]["params"]
    assert "from:SouichiTerada" in params["query"] and "from:DanWoikeSports" in params["query"]
    assert "from:NYPost_Lewis" not in params["query"] and "-is:retweet" in params["query"]
    assert "test-secret" not in json.dumps(client.dump_state())
    assert len(batch.coverage) == 5 and all(c["state"] == "ok" for c in batch.coverage)


@pytest.mark.parametrize("long_field", ["note_tweet", "note_post"])
def test_x_long_post_uses_complete_text(registry, long_field):
    data = payload(text="Injury update follows...")
    data["data"][0][long_field] = {"text": "Test Star ruled out tonight"}
    client = XNews(registry, bearer_token="secret", clock=lambda: NEWS_AT, session=Session([Response(data)]))
    batch = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert batch.rows[0]["status"] == "out" and batch.items[0]["text"] == "Test Star ruled out tonight"


def test_x_missing_credentials_are_explicit_and_make_no_request(registry):
    session = Session([])
    batch = XNews(registry, bearer_token="", session=session).fetch(
        game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert batch.errors and all(c["state"] == "unconfigured" for c in batch.coverage)
    assert session.calls == []


def test_x_pagination_backlog_resumes_after_restart_without_skipping(registry):
    session = Session([Response(payload(next_token="next")), Response(payload(post_id="2"))])
    client = XNews(registry, bearer_token="secret", max_pages=1, clock=lambda: NEWS_AT, session=session)
    first = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert first.coverage[0]["state"] == "backlog" and client.cursor == {}
    saved = client.dump_state()
    resumed = XNews(registry, bearer_token="secret", max_pages=1, clock=lambda: NEWS_AT, session=session)
    resumed.load_state(saved)
    second = resumed.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT + pd.Timedelta(minutes=1))
    assert second.rows and second.errors == []
    assert session.calls[1][1]["params"]["next_token"] == "next"
    assert session.calls[1][1]["params"]["end_time"] == session.calls[0][1]["params"]["end_time"]
    assert resumed.pending == {} and resumed.cursor["g2"]


def test_x_rate_limit_defers_retry_without_sleeping_or_losing_window(registry):
    reset = int((NEWS_AT + pd.Timedelta(minutes=2)).timestamp())
    session = Session([Response(status=429, headers={"x-rate-limit-reset": str(reset)})])
    client = XNews(registry, bearer_token="secret", clock=lambda: NEWS_AT, session=session)
    first = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    second = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT + pd.Timedelta(minutes=1))
    assert first.coverage[0]["state"] == second.coverage[0]["state"] == "rate_limited"
    assert len(session.calls) == 1 and client.cursor == {} and "g2" in client.pending


def test_x_403_and_foreign_author_are_not_valid_news(registry):
    session = Session([Response(status=403), Response(payload(handle="FakeInsider"))])
    client = XNews(registry, bearer_token="secret", clock=lambda: NEWS_AT, session=session)
    first = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert first.errors and first.rows == [] and "secret" not in json.dumps(first.errors)
    second = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert second.errors and second.rows == []
    assert second.coverage[0]["state"] == "failed" and client.cursor == {} and "g2" in client.pending


def test_x_post_tip_does_not_request(registry):
    session = Session([])
    batch = XNews(registry, bearer_token="secret", clock=lambda: TIP2, session=session).fetch(
        game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert session.calls == [] and batch.coverage[0]["state"] == "stopped_at_tip"


def test_image_only_official_post_is_retained_for_review(registry):
    data = payload(handle="celtics", text="Tonight's starting lineup")
    data["data"][0]["attachments"] = {"media_keys": ["m1"]}
    data["includes"]["media"] = [{"media_key": "m1", "type": "photo", "url": "https://images.example/lineup.jpg"}]
    client = XNews(registry, bearer_token="secret", clock=lambda: NEWS_AT, session=Session([Response(data)]))
    batch = client.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert batch.rows == [] and batch.items[0]["review_required"]
    assert batch.items[0]["attachments"][0]["url"].endswith("lineup.jpg")


def test_live_missing_x_does_not_disable_authority_sources(registry, monkeypatch):
    from data_sources.x_news import XNews
    provider = LiveNews(rss_urls=[registry.media["espn_rss"]["url"]], official=False, registry=registry,
                        x_client=XNews(registry, bearer_token="", clock=lambda: NEWS_AT), clock=lambda: NEWS_AT)
    xml = ("<rss><channel><item><title>Test Star ruled out tonight</title>"
           "<link>https://www.espn.com/news/1</link><pubDate>" + NEWS_AT.strftime("%a, %d %b %Y %H:%M:%S GMT")
           + "</pubDate></item></channel></rss>")
    monkeypatch.setattr(provider, "_get", lambda *a: SimpleNamespace(content=xml, raise_for_status=lambda: None))
    batch = provider.fetch(game(), players(), {7}, NEWS_AT - pd.Timedelta(hours=1), NEWS_AT)
    assert batch.rows[0]["source"] == "espn_rss" and batch.rows[0]["status"] == "out"
    assert batch.coverage[0]["state"] == "ok" and len(batch.coverage) == 6 and len(batch.errors) == 1
