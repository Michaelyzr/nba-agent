"""Coach agent: planner choices, pass/caution-only nudges, no look-ahead, seeded simulation, stubbed LLM mode."""
import numpy as np
import pandas as pd
import pytest

from agents import coach_agent
from agents.coach import NOTICE, banned_words, snapshot
from agents.coach_agent import ENCOURAGE, NUDGES, advise, habits, trace_table, validate
from evaluation import coach_sim
from replay import AsOf, Replay
from tests.test_agent_graph import NEWS_AT, make_tables
from tests.test_coach import FakeForecaster

T = pd.Timestamp("2026-02-01 23:00", tz="UTC")


def side(team, ask, p, move):
    fee = 0.07 * ask * (1 - ask)
    mid = ask - 0.005
    return {"team": team, "ticker": team, "bid": ask - 0.01, "ask": ask, "mid": mid, "fee": fee,
            "breakeven": ask + fee, "anchor": mid - move, "move": move, "p_model": p, "p": p, "gap": p - ask - fee}


def snap(ask=0.55, p=0.60, move=0.0, shift=0.0, news=()):
    sides = {"HOM": side("HOM", ask, p, move), "AWY": side("AWY", round(1 - ask + 0.01, 2), 1 - p, -move)}
    return {"game_id": "g", "home": "HOM", "away": "AWY", "as_of": T, "hours_to_tip": 2.0, "news": list(news),
            "out": [], "p_home_before": p - shift, "p_home_after": p, "shift": shift, "sides": sides,
            "best": max(sides.values(), key=lambda s: s["gap"]), "pick": None}


def checks(a):
    return [s["check"] for s in a["steps"]]


def test_planner_runs_the_priced_in_check_when_the_market_moved():
    moved = advise(snap(move=0.03), "HOM")
    assert checks(moved)[:2] == ["break_even", "priced_in"]
    assert "moved +3.0 points" in moved["steps"][1]["why"] and moved["steps"][1]["flag"]
    assert moved["lesson"]["id"] == "priced_in" and "priced in" in moved["message"]
    still = advise(snap(move=0.0), "HOM")
    assert "priced_in" not in checks(still) and "priced_in" in still["skipped"]
    chaser_history = [{"choice": "trade", "team": "HOM", "move": 0.03, "gap": 0.0, "price": 0.6}] * 3
    assert "priced_in" in checks(advise(snap(move=0.0), "HOM", chaser_history))      # habit triggers the check
    assert list(trace_table(moved).columns) == ["step", "check", "why", "finding"]


def test_planner_flags_long_shots_and_negative_edges():
    cheap = advise(snap(ask=0.75, p=0.72), "AWY")           # AWY ask 0.26, estimate 0.28: long shot, thin edge
    assert "long_shot" in checks(cheap) and cheap["lesson"]["id"] == "long_shot" and cheap["nudge"] in ("pass", "caution")
    bad = advise(snap(ask=0.62, p=0.58), "HOM")
    assert bad["nudge"] == "pass" and bad["message"].startswith("Consider passing")
    good = advise(snap(ask=0.50, p=0.62), "HOM")
    assert good["nudge"] == "none" and "mostly luck" in good["message"]


def test_nudge_only_ever_says_pass_or_caution_never_bet_more():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(300):
        ask = round(float(rng.uniform(0.05, 0.95)), 2)
        s = snap(ask=ask, p=float(np.clip(ask + rng.normal(0, 0.08), 0.02, 0.98)), move=float(rng.normal(0, 0.03)),
                 shift=float(rng.normal(0, 0.03)),
                 news=[{"player": "X", "status": "Out",
                        "published_at": T - pd.Timedelta(minutes=int(rng.integers(1, 200)))}])
        hist = [{"choice": "trade", "team": "HOM", "move": 0.03, "gap": -0.01, "price": 0.2}] * int(rng.integers(0, 6))
        for team in (*s["sides"], None):
            a = advise(s, team, hist)
            seen.add(a["nudge"])
            assert a["nudge"] in NUDGES
            assert ENCOURAGE.search(a["message"]) is None and banned_words([a["message"]]) == []
            assert a["message"].endswith(NOTICE)
            assert not any(w in a["message"].lower() for w in ("stake more", "bet more", "increase"))
    assert seen == set(NUDGES)
    assert validate("You should bet more on this one.", {}) and validate("It's a lock.", {})


def test_advice_has_no_look_ahead():
    tables = make_tables()
    rp = Replay(tables, lambda *a: [])
    game = next(g for g in tables["games"].itertuples() if g.game_id == "g2")
    early = snapshot(FakeForecaster(), AsOf(tables, NEWS_AT - pd.Timedelta(minutes=30), rp.price_index), game, {7: "P7"})
    a = advise(early, "AWY")
    assert "news_freshness" not in checks(a) and "m4_shift" not in checks(a)       # the later news is invisible
    late = snapshot(FakeForecaster(), AsOf(tables, NEWS_AT, rp.price_index), game, {7: "P7"})
    assert "m4_shift" in checks(advise(late, "HOM"))
    picks = [{"choice": "trade", "team": "HOM", "move": 0.03, "gap": 0.01, "price": 0.6, "long_shot": False,
              "chased": True}] * 3
    with_results = [{**p, "pnl": 50.0, "clv": 0.2, "won": True, "close_price": 0.9} for p in picks]
    assert advise(late, "AWY", picks) == advise(late, "AWY", with_results)        # outcomes of earlier picks unused
    assert habits(picks) == habits(with_results)


def tiny_slates():
    tables = make_tables()
    rp = Replay(tables, lambda *a: [])
    game = next(g for g in tables["games"].itertuples() if g.game_id == "g2")
    snaps = [snapshot(FakeForecaster(), AsOf(tables, t, rp.price_index), game, {7: "P7"})
             for t in (NEWS_AT - pd.Timedelta(minutes=30), NEWS_AT)]
    return rp, [{"name": "s0", "snapshots": snaps}, {"name": "s1", "snapshots": snaps[::-1]}]


def test_simulation_is_deterministic_and_paired_with_a_seed():
    rp, slates = tiny_slates()
    a, b = coach_sim.run_all(rp, slates), coach_sim.run_all(rp, slates)
    pd.testing.assert_frame_equal(a, b)
    assert coach_sim.draw("overtrader", 1, 2) == coach_sim.draw("overtrader", 1, 2) != coach_sim.draw("overtrader", 1, 3)
    o = a[a.persona == "overtrader"].set_index(["arm", "slate", "index"])
    half, full = o.loc["with c=0.5"].complied, o.loc["with c=1"].complied
    assert not (half & ~full).any()                                                # c=1 complies whenever c=0.5 does
    assert (o.loc["without"].complied == False).all()                              # noqa: E712
    nudged = o.loc["with c=1"]
    assert (nudged[nudged.nudge == "pass"].team.isna()).all()
    assert set(a.persona) == set(coach_sim.PERSONAS)
    t = coach_sim.slate_table(a[(a.persona == "overtrader") & (a.arm == "without")], 2)
    d = coach_sim.slate_paired(t, t)
    assert all(v == (0.0, 0.0, 0.0) for v in d.values())


def test_personas_follow_their_rules():
    s = snap(ask=0.70, p=0.60, move=0.03)
    assert coach_sim.chaser(s) == "HOM"
    assert coach_sim.long_shot_lover(s) == "AWY"
    assert coach_sim.overtrader(s) == "AWY"                   # larger raw gap after fees
    assert coach_sim.cautious(s) is None and coach_sim.cautious(snap(ask=0.40, p=0.60)) == "HOM"


def test_llm_mode_uses_the_tool_numbers_and_falls_back_when_unsafe(tmp_path, monkeypatch):
    monkeypatch.setattr(coach_agent, "CACHE", tmp_path)
    s = snap(ask=0.62, p=0.58, move=0.03)
    calls = []

    def good(prompt):
        calls.append(prompt)
        return "HOM breaks even at 63.6% after the fee, above our 58% estimate, and the price already moved."

    a = advise(s, "HOM", llm=True, ask=good)
    assert a["mode"] == "llm" and a["message"].endswith(NOTICE) and a["nudge"] == "pass"
    assert "breakeven" in calls[0] and len(list(tmp_path.glob("coach_*.json"))) == 1
    advise(s, "HOM", llm=True, ask=good)
    assert len(calls) == 1                                                         # second call served from the cache
    monkeypatch.setattr(coach_agent, "CACHE", tmp_path / "bad")
    bad = advise(s, "HOM", llm=True, ask=lambda p: "Bet more: HOM wins 91% of the time, a sure thing.")
    assert bad["mode"].startswith("rules (llm rejected") and bad["message"].startswith("Consider passing")
    monkeypatch.setattr(coach_agent, "CACHE", tmp_path / "error")
    assert advise(s, "HOM", llm=True, ask=lambda p: 1 / 0)["mode"].startswith("rules")
    assert advise(s, "HOM")["mode"] == "rules"
