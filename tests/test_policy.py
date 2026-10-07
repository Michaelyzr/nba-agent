"""Learned trade/pass policy: selection inside training data, abstention, one trade per game, labels unused."""
import math

import numpy as np
import pandas as pd
import pytest

from forecast import policy
from forecast.policy import FEATURES, LABEL, LearnedPolicy, PolicyModel, choose, select


def rows_for(dates, n_games=6, signal=0.0, seed=0):
    """Synthetic option rows: net CLV = signal * gap + noise - 0.01 (costs)."""
    rng = np.random.default_rng(seed)
    out = []
    for d in dates:
        tip = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=23)
        for g in range(n_games):
            for k, lead in enumerate((3, 1)):
                as_of = tip - pd.Timedelta(hours=lead)
                for side in ("yes", "no"):
                    feats = {f: float(rng.normal()) for f in FEATURES}
                    feats["is_yes"] = float(side == "yes")
                    net = signal * feats["gap"] + float(rng.normal(0, 0.01)) - 0.01
                    out.append({"date": d, "game_id": f"{d}-{g}", "as_of": as_of, "tip_time": tip,
                                "ticker": f"{d}-{g}-HOM", "side": side, "p_yes": 0.5, "price": 0.5,
                                "contracts": 40, "clv": net + 0.005, LABEL: net, **feats})
    return pd.DataFrame(out)


TRAIN_DAYS = [str(d.date()) for d in pd.date_range(policy.FIT[0], policy.VALID[1], freq="5D")]


def test_selection_abstains_when_nothing_has_value(monkeypatch):
    monkeypatch.setattr(policy, "GRID", [("ridge", {"alpha": 1.0})])
    table, best, forced = select(rows_for(TRAIN_DAYS, signal=0.0))
    assert (table.net_dollars <= 0).all()
    assert best["family"] == "never" and math.isinf(best["tau"]) and best["trades"] == 0
    assert forced is None or forced["trades"] >= policy.MIN_FORCED_TRADES


def test_selection_trades_when_the_label_is_learnable(monkeypatch):
    monkeypatch.setattr(policy, "GRID", [("ridge", {"alpha": 1.0})])
    _, best, _ = select(rows_for(TRAIN_DAYS, signal=0.05))
    assert best["family"] == "ridge" and best["net_dollars"] > 0


def test_selection_refuses_rows_after_the_training_window():
    rows = rows_for(TRAIN_DAYS + ["2026-02-03"])
    with pytest.raises(ValueError):
        select(rows)


def test_choose_takes_the_first_time_that_clears_tau_once_per_game():
    r = rows_for(["2025-11-05"], n_games=2)
    pred = np.where(r.as_of == r.as_of.max(), 0.05, -0.05)          # only the later decision clears tau
    picked = choose(r, pred, 0.0)
    assert picked.game_id.is_unique and len(picked) == 2
    assert (picked.as_of == r.as_of.max()).all()


def test_replay_policy_trades_once_per_game_and_ignores_labels():
    r = rows_for(["2026-02-05"], n_games=1, signal=0.05)
    model = PolicyModel("ridge", {"alpha": 1.0}).fit(rows_for(TRAIN_DAYS, signal=0.05))
    scrambled = r.assign(**{LABEL: -r[LABEL], "clv": -r["clv"]})
    game = type("G", (), {"game_id": r.game_id.iloc[0]})()
    times = sorted(r.as_of.unique())
    a, b = LearnedPolicy(r, model, -1.0), LearnedPolicy(scrambled, model, -1.0)
    first = [a.policy(None, game, t) for t in times]
    assert [len(x) for x in first] == [1, 0]                         # one position per game
    assert [len(b.policy(None, game, t)) for t in times] == [1, 0]
    assert first[0][0].market_ticker == b.rows[(game.game_id, times[0])].pipe(
        lambda o: o.loc[o.pred.idxmax(), "ticker"])                   # same choice with scrambled labels
    never = LearnedPolicy(r, None, math.inf)
    assert all(never.policy(None, game, t) == [] for t in times)
