"""M4-NN: sequences use only games before tip; the model fits and predicts probabilities."""
import numpy as np
import pandas as pd

from forecast.win import WIN_FEATURES
from forecast.win_nn import SEQ_LEN, NeuralWinModel, sequences


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    tips = pd.date_range("2025-10-01", periods=n, freq="D", tz="UTC")
    games = pd.DataFrame({"game_id": [str(i) for i in range(n)], "home_team_id": rng.integers(0, 4, n),
                          "away_team_id": 0, "home_pts": rng.integers(90, 120, n), "away_pts": rng.integers(90, 120, n)})
    games["away_team_id"] = (games.home_team_id + 1) % 4
    rows = pd.DataFrame({c: rng.normal(size=n) for c in WIN_FEATURES})
    rows["game_id"], rows["tip_time"] = games.game_id, tips
    rows["home_win"] = (games.home_pts > games.away_pts).astype(float)
    return rows, games


def test_sequences_are_strictly_before_tip():
    rows, games = toy()
    seq = sequences(rows, games)
    assert seq.shape == (len(rows), 2, SEQ_LEN, 8)
    assert seq[0].sum() == 0                                   # first game: no history
    changed = rows.copy()
    changed.loc[50:, "rating_diff"] = 99.0                     # future games must not leak into game 50
    assert np.allclose(sequences(changed, games)[:51], seq[:51])


def test_fit_predict_probabilities():
    rows, games = toy()
    for kind, seq in (("mlp", None), ("gru", sequences(rows, games))):
        m = NeuralWinModel(kind, epochs=3, patience=2).fit(rows, seq, val_start="2025-12-15")
        p = m.predict(rows, seq)
        assert p.shape == (len(rows),) and ((p > 0) & (p < 1)).all()
        assert m.predict_before_news(rows, seq).shape == p.shape
