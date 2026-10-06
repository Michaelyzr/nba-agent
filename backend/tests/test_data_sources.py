"""Parsers and table builders in data_sources, on hand-made inputs (no network)."""
from datetime import date

import pandas as pd

from data_sources import et_to_utc, name_key
from data_sources.kalshi import parse_ticker, settlement_rows
from data_sources.nba_stats import build_games, build_player_games
from data_sources.news import report_names, rows_from_words, to_frames
from data_sources.polymarket import parse_slug


def test_name_key_matches_report_and_api_spellings():
    assert name_key("Russell,D'Angelo") == name_key("D'Angelo Russell") == "dangelorussell"
    assert name_key("Jokić, Nikola") == name_key("Nikola Jokic")
    assert name_key("JacksonJr.,Jaren") == name_key("Jaren Jackson Jr.")


def test_eastern_to_utc_handles_daylight_saving():
    assert et_to_utc("2026-02-07", "07:30 PM") == pd.Timestamp("2026-02-08T00:30:00Z")
    assert et_to_utc("2026-04-07", "07:30 PM") == pd.Timestamp("2026-04-07T23:30:00Z")


def test_kalshi_ticker():
    assert parse_ticker("KXNBAGAME-26FEB07WASBKN-BKN") == {
        "series": "KXNBAGAME", "date": "2026-02-07", "away_team": "WAS", "home_team": "BKN", "team": "BKN"}
    assert parse_ticker("KXNBAPTS-26JUN13NYKSAS-SASVWEMBANYAMA1-40")["team"] == "SAS"


def test_kalshi_settlements_skip_unresolved():
    rows = settlement_rows([
        {"ticker": "A", "result": "yes", "settlement_ts": "2026-02-08T03:00:00Z"},
        {"ticker": "B", "result": "no", "settlement_ts": "2026-02-08T03:00:00Z"},
        {"ticker": "C", "result": "", "settlement_ts": None}])
    assert rows.set_index("market_ticker").outcome.to_dict() == {"A": 1, "B": 0}


def test_polymarket_slug():
    assert parse_slug("nba-was-bkn-2026-02-07") == {"away_team": "WAS", "home_team": "BKN", "date": "2026-02-07"}


def test_injury_report_filenames():
    assert report_names(date(2026, 2, 7), 17, 0) == ["Injury-Report_2026-02-07_05_00PM.pdf",
                                                     "Injury-Report_2026-02-07_05PM.pdf"]
    assert report_names(date(2026, 2, 7), 11, 45) == ["Injury-Report_2026-02-07_11_45AM.pdf"]


def _word(text, x0, top):
    return {"text": text, "x0": x0, "top": top}


HEADER = [_word(t, x, 100) for t, x in [("GameDate", 23), ("GameTime", 119), ("Matchup", 200), ("Team", 264),
                                         ("PlayerName", 425), ("CurrentStatus", 585), ("Reason", 666)]]


def test_injury_rows_carry_game_and_team_forward_across_pages():
    page1 = HEADER + [
        _word("Injury", 20, 80),
        _word("02/07/2026", 24, 120), _word("03:00(ET)", 120, 120), _word("WAS@BKN", 201, 120),
        _word("WashingtonWizards", 265, 120), _word("Coulibaly,Bilal", 426, 120), _word("Out", 586, 120),
        _word("Injury/Illness-LowerBack;Soreness", 667, 120),
        _word("Davis,Anthony", 426, 140), _word("Out", 586, 140), _word("TradePending", 667, 140),
        _word("Page1of2", 380, 560)]
    page2 = HEADER + [
        _word("BrooklynNets", 265, 120), _word("Etienne,Tyson", 426, 120), _word("Questionable", 586, 120),
        _word("Injury/Illness-Right", 667, 120), _word("Ankle;Sprain", 667, 132),
        _word("NOT", 426, 150), _word("YET", 440, 150), _word("SUBMITTED", 455, 150)]
    rows = rows_from_words([page1, page2])
    assert [(r["PlayerName"], r["CurrentStatus"], r["Team"]) for r in rows] == [
        ("Coulibaly,Bilal", "Out", "WashingtonWizards"), ("Davis,Anthony", "Out", "WashingtonWizards"),
        ("Etienne,Tyson", "Questionable", "BrooklynNets")]
    assert rows[2]["Matchup"] == "WAS@BKN" and rows[2]["Reason"] == "Injury/Illness-Right Ankle;Sprain"


def test_news_keeps_only_status_changes_and_tip_times():
    t0 = pd.Timestamp("2026-02-07T17:15:00Z")
    raw = pd.DataFrame([
        {"GameDate": "02/07/2026", "GameTime": "07:30(ET)", "Matchup": "WAS@BKN", "Team": "BrooklynNets",
         "PlayerName": "Etienne,Tyson", "CurrentStatus": s, "Reason": "Ankle", "published_at": t0 + pd.Timedelta(hours=i),
         "url": "u"} for i, s in enumerate(["Questionable", "Questionable", "Out"])])
    games = pd.DataFrame({"game_id": ["0022500777"], "date": ["2026-02-07"], "home_team": ["BKN"], "away_team": ["WAS"]})
    players = pd.DataFrame({"player_id": [5], "player_name": ["Tyson Etienne"]})
    news, tips = to_frames(raw, games, players)
    assert news.status.tolist() == ["questionable", "out"]
    assert news.published_at.tolist() == [t0, t0 + pd.Timedelta(hours=2)]
    assert tips.tip_time.iloc[0] == pd.Timestamp("2026-02-08T00:30:00Z")


def test_build_games_pairs_home_and_away():
    team_logs = pd.DataFrame({
        "GAME_ID": ["0022500001", "0022500001"], "GAME_DATE": ["2025-10-21T00:00:00"] * 2,
        "TEAM_ID": [1610612738, 1610612752], "TEAM_ABBREVIATION": ["BOS", "NYK"],
        "MATCHUP": ["BOS vs. NYK", "NYK @ BOS"], "PTS": [110, 104]})
    tips = pd.DataFrame({"game_id": ["0022500001"], "tip_time": [pd.Timestamp("2025-10-21T23:30:00Z")]})
    g = build_games(team_logs, tips).iloc[0]
    assert (g.home_team, g.away_team, g.home_pts, g.away_pts, g.date) == ("BOS", "NYK", 110, 104, "2025-10-21")
    assert g.final_at == pd.Timestamp("2025-10-22T02:30:00Z")


def test_build_player_games_schema():
    logs = pd.DataFrame({"GAME_ID": ["0022500001"], "PLAYER_ID": [1], "TEAM_ID": [2], "MIN": ["33.5"],
                         "PTS": [20], "FGA": [15], "FTA": [4]})
    p = build_player_games(logs)
    assert list(p.columns) == ["game_id", "player_id", "team_id", "min", "pts", "fga", "fta", "usage", "started"]
    assert p["min"].iloc[0] == 33.5


def test_espn_box_rows_split_players_and_inactive_news():
    from data_sources.espn import box_rows, game_row
    team_ids = {"ORL": 1, "MEM": 2, "GSW": 3}
    keys = ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "freeThrowsMade-freeThrowsAttempted"]
    event = {"id": "401", "date": "2026-01-15T19:00Z", "season": {"type": 2},
             "competitions": [{"status": {"type": {"name": "STATUS_FINAL"}}, "competitors": [
                 {"homeAway": "home", "team": {"abbreviation": "ORL"}, "score": "118"},
                 {"homeAway": "away", "team": {"abbreviation": "GS"}, "score": "111"}]}]}
    game = game_row(event, team_ids, "2025-26")
    assert (game["home_team"], game["away_team"], game["date"]) == ("ORL", "GSW", "2026-01-15")
    athletes = [
        {"athlete": {"id": "7", "displayName": "A Starter"}, "starter": True, "didNotPlay": False,
         "stats": ["33", "30", "12-22", "3-4"]},
        {"athlete": {"id": "8", "displayName": "B Hurt"}, "didNotPlay": True, "reason": "RIGHT CALF STRAIN", "stats": []},
        {"athlete": {"id": "9", "displayName": "C Bench"}, "didNotPlay": True, "reason": "COACH'S DECISION", "stats": []},
        {"athlete": {}, "stats": []}]
    data = {"boxscore": {"players": [{"team": {"abbreviation": "ORL"},
                                      "statistics": [{"keys": keys, "athletes": athletes}]}]}}
    played, people, news = box_rows(game, data, team_ids)
    assert [(p["player_id"], p["min"], p["pts"], p["fga"], p["fta"], p["started"]) for p in played] == \
        [(7, 33, 30, 22, 4, True)]
    assert [n["player_id"] for n in news] == [8]                        # coach's decision is not news
    assert news[0]["published_at"] == game["tip_time"] - pd.Timedelta(minutes=30)
    assert len(people) == 3
