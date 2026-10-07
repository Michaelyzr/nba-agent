"""Injury-report timing: report slots, matching parsed rows to games and players, first listings, move split."""
import numpy as np
import pandas as pd

from evaluation.injury_timing import first_listing, refine_slots, slot_times, split_move, tidy

GAMES = pd.DataFrame({"game_id": ["g1"], "date": ["2026-02-11"], "home_team": ["NYK"], "away_team": ["IND"],
                      "tip_time": [pd.Timestamp("2026-02-12T00:30Z")]})
PLAYERS = pd.DataFrame({"player_id": [1, 2], "player_name": ["Tyrese Haliburton", "T.J. McConnell"]})


def parsed(slot, name, status, date="02/11/2026", matchup="IND@NYK"):
    return {"slot": pd.Timestamp(slot), "file": f"{slot}.pdf", "GameDate": date, "GameTime": "07:30(ET)",
            "Matchup": matchup, "Team": "IndianaPacers", "PlayerName": name, "CurrentStatus": status, "Reason": "x"}


def test_slots_cover_noon_the_day_before_to_game_night():
    s = slot_times(["2026-02-11"])
    assert s[0] == pd.Timestamp("2026-02-10 12:00") and s[-1] == pd.Timestamp("2026-02-11 23:00")
    assert len(s) == 36


def test_tidy_matches_report_names_dates_and_matchups():
    raw = pd.DataFrame([parsed("2026-02-10 17:00", "Haliburton,Tyrese", "Out"),
                        parsed("2026-02-10 17:00", "McConnell,T.J.", "Questionable"),
                        parsed("2026-02-10 17:00", "Nobody,Known", "Out"),
                        parsed("2026-02-10 17:00", "Haliburton,Tyrese", "Out", date="02/10/2026")])
    r = tidy(raw, GAMES, PLAYERS)
    assert sorted(r.player_id) == [1, 2] and set(r.game_id) == {"g1"}
    # 5 pm Eastern in February is 22:00 UTC; public 15 minutes later
    assert r.public_at.iloc[0] == pd.Timestamp("2026-02-10T22:15Z")


def test_first_listing_ignores_questionable_and_refines_the_hour_before():
    raw = pd.DataFrame([parsed("2026-02-11 13:00", "McConnell,T.J.", "Questionable"),
                        parsed("2026-02-11 17:00", "McConnell,T.J.", "Doubtful"),
                        parsed("2026-02-11 18:00", "McConnell,T.J.", "Out")])
    first = first_listing(tidy(raw, GAMES, PLAYERS))
    assert first.first_slot.iloc[0] == pd.Timestamp("2026-02-11 17:00")
    assert refine_slots(first) == [pd.Timestamp(f"2026-02-11 16:{m}") for m in ("15", "30", "45")]


def prices(points):
    ts = pd.DatetimeIndex([pd.Timestamp(t) for t, _ in points]).as_unit("ns").asi8
    return ts, np.array([m for _, m in points])


TIP = pd.Timestamp("2026-02-12T00:30Z")
P = prices([("2026-02-10T21:00Z", 0.50), ("2026-02-11T03:00Z", 0.45), ("2026-02-11T20:00Z", 0.40),
            ("2026-02-12T00:10Z", 0.38)])


def test_split_move_when_report_is_between_anchor_and_list():
    s = split_move(P, TIP - pd.Timedelta(hours=24), pd.Timestamp("2026-02-11T12:00Z"), TIP - pd.Timedelta(minutes=30),
                   TIP, sign=-1.0)
    assert np.isclose(s["pre_report"], 0.05) and np.isclose(s["report_to_list"], 0.05)
    assert np.isclose(s["after_list"], 0.02) and np.isclose(s["total"], 0.12)


def test_split_move_report_before_window_or_never():
    start, listed = TIP - pd.Timedelta(hours=24), TIP - pd.Timedelta(minutes=30)
    early = split_move(P, start, pd.Timestamp("2026-02-10T22:00Z"), listed, TIP, sign=-1.0)
    assert early["pre_report"] == 0.0 and np.isclose(early["report_to_list"], 0.10)
    never = split_move(P, start, pd.NaT, listed, TIP, sign=-1.0)
    assert np.isclose(never["pre_report"], 0.10) and never["report_to_list"] == 0.0
    for s in (early, never):
        assert np.isclose(s["pre_report"] + s["report_to_list"] + s["after_list"], s["total"])


def test_event_window_sums_to_total_and_places_the_jump():
    from evaluation.injury_timing import WINDOW_LABELS, event_window

    start, listed = TIP - pd.Timedelta(hours=24), TIP - pd.Timedelta(minutes=30)
    w = event_window(P, start, pd.Timestamp("2026-02-11T19:50Z"), listed, TIP, sign=-1.0)
    assert len(w) == len(WINDOW_LABELS) and np.isclose(w.sum(), 0.12)
    assert np.isclose(w[WINDOW_LABELS.index("slot → +15 min")], 0.05)     # the 20:00Z quote lands 10 min after
    assert np.isclose(w[0], 0.05) and np.isclose(w[-1], 0.02)
