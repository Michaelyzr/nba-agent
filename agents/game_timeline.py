"""Presentation of game time and same-state news effects, independent of feeds."""


def elapsed_seconds(score):
    period = int(score["period"])
    length = 720 if period <= 4 else 300
    start = (period - 1) * 720 if period <= 4 else 2880 + (period - 5) * 300
    return start + length - float(score["clock_seconds"])


def clock_label(score):
    seconds = int(score["clock_seconds"])
    period = int(score["period"])
    quarter = f"Q{period}" if period <= 4 else f"OT{period - 4}"
    return f"{quarter} {seconds // 60:02d}:{seconds % 60:02d}"


def news_effect(snapshot):
    """Compare old vs new factors at the current score/clock; never a prior poll."""
    before = snapshot.get("p_home_before_new_events")
    after = snapshot.get("p_home")
    if snapshot.get("pipeline", "pregame") == "pregame" and after is not None:
        before = after - snapshot.get("delta_home", 0)
    if snapshot.get("quote_state") == "final" or before is None or after is None:
        return None
    if not 0 < before < 1 or not 0 < after < 1:
        return None
    return {"p_home_before": before, "p_home_after": after,
            "home_delta_pp": (after - before) * 100,
            "away_delta_pp": (before - after) * 100,
            "home_odds_before": 1 / before, "home_odds_after": 1 / after,
            "home_odds_delta": 1 / after - 1 / before,
            "away_odds_before": 1 / (1 - before), "away_odds_after": 1 / (1 - after),
            "away_odds_delta": 1 / (1 - after) - 1 / (1 - before)}


def timeline_events(steps):
    events = []
    for index, step in enumerate(steps):
        snapshot, report = step["snapshot"], step["snapshot"]["report"]
        if not report["new_evidence"] or not snapshot.get("score"):
            continue
        effect = report.get("news_effect")
        if effect is None:
            continue
        score = snapshot["score"]
        events.append({"number": len(events) + 1, "index": index, "step": step["step"],
                       "elapsed_minutes": elapsed_seconds(score) / 60, "clock": clock_label(score),
                       "label": step["label"], "score": f"{score['home_score']}–{score['away_score']}",
                       "time_basis": "synthetic_occurrence" if report["synthetic"] else "first_observed_game_clock",
                       "players": ", ".join(e["player"] for e in report["new_evidence"]), **effect})
    return events
