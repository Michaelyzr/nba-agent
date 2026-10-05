"""M6 market-impact model: as-of features only, ordered quantile outputs, and the agent trading on the impact signal."""
import numpy as np
import pandas as pd
import pytest
import torch

from agents.graph import MarketAgent
from forecast.history import History
from forecast.impact import (QUANTILES, SEQ_COLS, SEQ_LEN, STATIC, ImpactModel, ImpactNet, build_dataset,
                             build_row, load_impact, price_arrays)
from replay import Replay
from tests.test_agent_graph import AWAY, HOME, NEWS_AT, TIP2, make_tables


class FakeWin:
    """Stands in for M4: P(home) falls with the home side's missing minutes."""
    def predict(self, rows):
        return (0.6 - 0.005 * rows["missing_min_diff"]).to_numpy()


def as_of_tables(tables, now):
    """Everything after `now` deleted: later quotes, later news, and box scores of games not yet final."""
    t = {k: v.copy() for k, v in tables.items()}
    t["prices"] = t["prices"][t["prices"].ts <= now]
    t["news"] = t["news"][t["news"].published_at <= now]
    done = t["games"].loc[t["games"].final_at <= now, "game_id"]
    t["player_games"] = t["player_games"][t["player_games"].game_id.isin(done)]
    return t


def row_at(tables, now):
    game = next(tables["games"][tables["games"].game_id == "g2"].itertuples())
    a = price_arrays(tables["prices"][tables["prices"].market_ticker == HOME])
    h = History(tables["player_games"], tables["games"])
    return build_row(a, tables["news"][tables["news"].game_id == "g2"], h, FakeWin(), game, now)


def moving_tables():
    """The home price drifts every minute, so any peek at later quotes would change the features."""
    t = make_tables()
    home = t["prices"].market_ticker == HOME
    drift = np.linspace(0, 0.15, home.sum())
    t["prices"].loc[home, "bid"] = (0.55 + drift).round(3)
    t["prices"].loc[home, "ask"] = (0.57 + drift).round(3)
    return t


@pytest.mark.parametrize("now", [NEWS_AT - pd.Timedelta(minutes=30), NEWS_AT, NEWS_AT + pd.Timedelta(minutes=40)])
def test_features_use_nothing_after_the_decision_time(now):
    full = moving_tables()
    seq, static = row_at(full, now)
    seq_cut, static_cut = row_at(as_of_tables(full, now), now)
    assert np.array_equal(seq, seq_cut) and static == pytest.approx(static_cut)

    poisoned = moving_tables()
    later = poisoned["prices"].ts > now
    poisoned["prices"].loc[later, ["bid", "ask"]] = (0.01, 0.02)
    poisoned["news"] = pd.concat([poisoned["news"], poisoned["news"].assign(
        news_id="n_future", player_id=8, published_at=now + pd.Timedelta(minutes=1))])
    seq_p, static_p = row_at(poisoned, now)
    assert np.array_equal(seq, seq_p) and static == pytest.approx(static_p)
    assert static["has_news"] == float(now >= NEWS_AT)


def test_dataset_rows_match_as_of_rebuild_and_label_is_move_to_tip():
    tables = moving_tables()
    rows, seq = build_dataset(tables, FakeWin())
    assert len(rows) and seq.shape == (len(rows), SEQ_LEN, len(SEQ_COLS))
    assert set(rows.columns) >= set(STATIC) | {"move", "is_decision"}
    assert rows.is_decision.sum() == 2                                   # the news and tip - LEAD
    assert (rows.as_of < TIP2).all()
    for i, r in rows.iterrows():
        s, st = row_at(as_of_tables(tables, r.as_of), r.as_of)
        assert np.array_equal(s, seq[i]) and all(st[k] == pytest.approx(r[k]) for k in STATIC)
    close = tables["prices"][(tables["prices"].market_ticker == HOME) & (tables["prices"].ts <= TIP2)].iloc[-1]
    assert rows.move.to_numpy() == pytest.approx((close.bid + close.ask) / 2 - rows["mid"].to_numpy())
    after = rows[rows.as_of >= NEWS_AT]
    assert (after.m4_shift_c < 0).all() and (after.missing_min_home > 0).all()


def test_impact_net_outputs_ordered_quantiles():
    torch.manual_seed(0)
    out = ImpactNet()(torch.randn(32, SEQ_LEN, len(SEQ_COLS)), torch.randn(32, len(STATIC)) * 5)
    assert out.shape == (32, len(QUANTILES))
    assert (out[:, 1:] >= out[:, :-1]).all()


def tiny_model(tables):
    rows, seq = build_dataset(tables, FakeWin())
    return ImpactModel(FakeWin(), epochs=2).fit(rows, seq, rows, seq, verbose=False), rows, seq


def test_trained_model_predicts_from_an_as_of_view():
    tables = moving_tables()
    model, rows, seq = tiny_model(tables)
    q = model.predict_arrays(rows, seq)
    assert q.shape == (len(rows), len(QUANTILES)) and (np.diff(q, axis=1) >= 0).all()
    agent = MarketAgent(impact=model)
    decisions, fills = Replay(tables, agent.policy).run("2026-02-01", "2026-02-01")
    assert all(t["status"] in ("brief_only", "sent") for t in agent.traces)


class FakeImpact:
    def __init__(self, move):
        self.move = move

    def predict(self, view, game, now):
        q = view.quote(HOME)
        mid = float((q.bid + q.ask) / 2)
        return {"ticker": HOME, "mid": mid, "move": self.move, "q10": self.move - 0.02, "q90": self.move + 0.02}


def test_agent_trades_the_predicted_move_to_tip():
    agent = MarketAgent(impact=FakeImpact(-0.10))
    decisions, fills = Replay(make_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    assert len(fills) == 1
    d = decisions.iloc[0]
    p_home = 0.61 - 0.10                                     # current mid plus M6's median move
    assert d.p_model == pytest.approx(p_home if d.market_ticker == HOME else 1 - p_home)
    assert (d.market_ticker, d.side) in {(HOME, "no"), (AWAY, "yes")}
    assert "M6 expects" in d.reason


def test_agent_with_small_predicted_move_stays_out():
    agent = MarketAgent(impact=FakeImpact(0.005))
    decisions, fills = Replay(make_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    assert decisions.empty and fills.empty


def test_missing_impact_model_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m forecast.impact"):
        load_impact(tmp_path / "impact.pkl")
