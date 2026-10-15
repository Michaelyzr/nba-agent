"""Approach C learned filter: as-of features, no label leakage, threshold tuning, statistics (synthetic, no runs/)."""
import numpy as np
import pandas as pd
import pytest

from evaluation import learned_filter as lf

T0 = pd.Timestamp("2026-02-10 23:00", tz="UTC")       # tip-off
NOW = T0 - pd.Timedelta(hours=1)


def tables():
    games = pd.DataFrame({"game_id": ["g1", "g0"], "date": ["2026-02-10", "2026-02-09"],
                          "tip_time": [T0, T0 - pd.Timedelta(days=1)], "home_team_id": [1, 1],
                          "away_team_id": [2, 3], "home_team": ["HOM", "HOM"], "away_team": ["AWY", "OTH"]})
    ts = [NOW - pd.Timedelta(hours=8), NOW - pd.Timedelta(hours=3), NOW, NOW + pd.Timedelta(minutes=30)]
    prices = pd.DataFrame({"market_ticker": ["T-HOM"] * 4, "ts": ts, "bid": [0.50, 0.55, 0.58, 0.90],
                           "ask": [0.52, 0.57, 0.60, 0.92], "volume": [10.0] * 4})
    news = pd.DataFrame({"news_id": ["n1", "n2", "n3"], "game_id": ["g1"] * 3, "player_id": [11, 21, 22],
                         "team_id": [1, 2, 2], "status": ["Out", "out", "doubtful"],
                         "published_at": [NOW - pd.Timedelta(hours=2), NOW - pd.Timedelta(minutes=10),
                                          NOW + pd.Timedelta(minutes=20)]})
    pg = pd.DataFrame({"game_id": ["g0", "g1"], "player_id": [11, 11], "team_id": [1, 1]})
    return {"games": games, "prices": prices, "news": news, "player_games": pg,
            "settlements": pd.DataFrame({"market_ticker": ["T-HOM"], "outcome": [1]})}


def rows():
    return pd.DataFrame({"game_id": ["g1", "g1"], "as_of": [NOW, NOW], "tip_time": [T0, T0],
                         "date": ["2026-02-10"] * 2, "ticker": ["T-HOM"] * 2, "team": ["HOM", "HOM"],
                         "side": ["yes", "no"], "side_price": [0.60, 0.42]})


def test_features_and_labels_are_disjoint():
    assert not set(lf.FEATURES + lf.LLM_FEATURES) & lf.LABEL_COLS
    assert {"clv", "close", "pnl", "won"} <= lf.LABEL_COLS


def test_add_features_are_side_relative_and_as_of():
    out = lf.add_features(rows(), tables())
    yes, no = out.iloc[0], out.iloc[1]
    assert (yes.home_side, no.home_side) == (1.0, 0.0)
    assert (yes.side_team, no.side_team) == ("HOM", "AWY")
    assert (yes.is_fav, no.is_fav) == (1.0, 0.0)
    assert (yes.b2b_side, yes.b2b_opp) == (1.0, 0.0)          # HOM played on the 9th, AWY did not
    assert (no.b2b_side, no.b2b_opp) == (0.0, 1.0)
    # Two AWY players ruled out by NOW count; the doubtful report published after NOW does not.
    assert (yes.n_out_side, yes.n_out_opp) == (1.0, 1.0)
    # Mid 0.51 six hours before NOW -> 0.59 at NOW; the post-NOW quote at 0.91 is ignored.
    assert yes.move_6h == pytest.approx(0.08) and no.move_6h == pytest.approx(-0.08)


def test_features_unchanged_when_the_future_is_removed():
    full = lf.add_features(rows(), tables())
    cut = lf.add_features(rows(), lf.truncated_tables(tables(), NOW))
    pd.testing.assert_frame_equal(full, cut)


def test_truncated_tables_drop_everything_after_the_decision_time():
    t = lf.truncated_tables(tables(), NOW)
    assert t["prices"].ts.max() <= NOW and t["news"].published_at.max() <= NOW
    assert set(t["player_games"].game_id) == {"g0"} and t["settlements"].empty


def test_tune_tau_maximises_kept_clv_with_a_minimum_and_prefers_higher_tau():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    c = np.array([-5.0, 3.0, 2.0, 0.0, -1.0])
    assert lf.tune_tau(s, c, min_kept=1) == 0.2           # keeps 0.2..0.5: +4.0, ties broken upward
    assert lf.tune_tau(s, c, min_kept=5) == 0.1
    assert lf.tune_tau(s, c, min_kept=6) == np.inf        # never enough kept: keep nothing


def test_out_of_fold_scores_never_use_the_scored_day(monkeypatch):
    seen = []

    def fake_fit_score(train, test, kind, feats):
        seen.append((set(train.date), set(test.date)))
        return np.zeros(len(test))
    monkeypatch.setattr(lf, "fit_score", fake_fit_score)
    days = [f"2026-01-{d:02d}" for d in range(1, 13)]
    train = pd.DataFrame({"date": np.repeat(days, 3)})
    out = lf.oof_scores(train, train.iloc[::2], "lr", [])
    assert np.isfinite(out).all() and len(seen) == lf.INNER_FOLDS
    for tr, te in seen:
        assert te and not (tr & te)


def test_filter_stats_oracle_share_and_random_percentile():
    w = pd.DataFrame({"date": ["d1", "d1", "d2", "d3"], "clv": [0.05, -0.02, 0.01, -0.04],
                      "clv_usd": [2.0, -1.0, 0.5, -2.0], "clv_usd_after_fee": [1.5, -1.5, 0.0, -2.5],
                      "pnl": [10.0, -20.0, 5.0, -20.0]})
    boot = lf.day_boot_idx(w.date, reps=200)
    oracle = lf.filter_stats(w, w.clv.to_numpy() > 0, boot, None)
    # 1 of the 6 random pairs is the oracle pair itself (a tie, half credit): percentile 100 * (5/6 + 1/12).
    assert oracle["kept"] == 2 and oracle["oracle_share"] == pytest.approx(1.0)
    assert oracle["random_pctile"] == pytest.approx(100 * (5 / 6 + 1 / 12), abs=1.5)
    worst = lf.filter_stats(w, np.array([False, True, False, True]), boot, w.clv.to_numpy() > 0)
    assert worst["random_pctile"] == pytest.approx(100 / 12, abs=1.5) and worst["diff_current"] == pytest.approx(-5.5)
    assert worst["diff_keep_all"] == pytest.approx(-3.0 - (-0.5))


def test_auc_and_day_bootstrap_shapes():
    assert lf.auc(np.array([0.9, 0.8, 0.1]), np.array([True, True, False])) == 1.0
    assert lf.auc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    b = lf.day_boot_idx(pd.Series(["a", "a", "b"]), reps=50)
    assert b.shape == (50, 3) and (b[:, 0] == b[:, 1]).all()        # rows of a day are drawn together
