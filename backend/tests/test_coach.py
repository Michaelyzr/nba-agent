"""Coach on the tiny slate: numbers match the as-of view, no future news, paper fills like the agent, safe wording."""
import pandas as pd
import pytest

from agents.coach import NOTICE, banned_words, explain, feedback, lessons, paper_trade, snapshot
from replay import AsOf, Replay
from evaluation.fixtures import NEWS_AT, TIP2, make_tables


class FakeForecaster:
    def win(self, view, game, out=()):
        return {"p_home": 0.52 if 7 in set(out) else 0.62, "missing": {}}


@pytest.fixture
def slate():
    tables = make_tables()
    rp = Replay(tables, lambda *a: [])
    game = next(g for g in tables["games"].itertuples() if g.game_id == "g2")
    return tables, rp, game


def snap(slate, now):
    tables, rp, game = slate
    return snapshot(FakeForecaster(), AsOf(tables, now, rp.price_index), game, {7: "Player Seven"})


def test_snapshot_anchors_to_market_and_picks_the_side_the_news_favours(slate):
    s = snap(slate, NEWS_AT)
    assert s["out"] == ["Player Seven"] and s["shift"] == pytest.approx(-0.10)
    home, away = s["sides"]["HOM"], s["sides"]["AWY"]
    assert home["anchor"] == pytest.approx(0.61) and home["p"] == pytest.approx(0.51)
    assert away["p"] == pytest.approx(0.49) and away["breakeven"] == pytest.approx(0.40 + 0.07 * 0.4 * 0.6)
    assert s["pick"] == "AWY"


def test_snapshot_never_sees_news_published_later(slate):
    s = snap(slate, NEWS_AT - pd.Timedelta(minutes=30))
    assert s["news"] == [] and s["shift"] == 0 and s["pick"] is None


def test_explanation_and_lessons_use_the_snapshot_numbers(slate):
    s = snap(slate, NEWS_AT)
    text = " ".join(explain(s))
    assert "Player Seven is out" in text and "52% chance" in text and "-10.0 points" in text
    assert "Backing AWY" in text
    ids = [c["id"] for c in lessons(s)]
    assert {"price_is_probability", "fees_and_spread", "injury_shift"} <= set(ids)
    assert "passing" not in ids
    quiet = snap(slate, NEWS_AT - pd.Timedelta(minutes=30))
    assert "passing" in [c["id"] for c in lessons(quiet)] and "Passing is the disciplined call" in " ".join(explain(quiet))


def test_paper_trade_fills_at_the_ask_with_fee_and_settles(slate):
    tables, rp, game = slate
    s = snap(slate, NEWS_AT)
    t = paper_trade(rp, game, NEWS_AT, s["sides"]["AWY"], 20, s)
    assert t["filled"] and t["price"] == pytest.approx(0.40) and t["contracts"] == 50
    assert t["clv"] == pytest.approx(-0.01) and t["won"] is False
    assert t["pnl"] == pytest.approx(-20 - t["fee"])
    late = paper_trade(rp, game, TIP2, s["sides"]["AWY"], 20, s)
    assert not late["filled"] and late["why"] == "post_tip"


def test_feedback_flags_long_shots_and_chasing():
    trade = {"filled": True, "pnl": -10.0, "clv": -0.02, "clv_dollars": -1.0, "long_shot": True, "chased": True,
             "gap": -0.01}
    f = feedback([trade] * 3)
    tips = " ".join(f["tips"])
    assert f["trades"] == 3 and "long shots" in tips and "chasing" in tips and "negative edge" in tips
    assert feedback([])["trades"] == 0


def test_coach_copy_never_promises_wins(slate):
    s = snap(slate, NEWS_AT)
    texts = explain(s) + [c["body"] for c in lessons(s)] + [NOTICE]
    assert banned_words(texts) == []
    assert banned_words(["This is a lock"]) == ["lock"]
