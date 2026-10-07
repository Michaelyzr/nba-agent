"""M4 calibration: monotone maps into (0, 1), rolling-origin predictions that never see their own month."""
import numpy as np
import pandas as pd
import pytest

from forecast.calibrate import Calibrator, CalibratedWinModel, fit_before, month_starts, out_of_sample
from forecast.win import WIN_FEATURES, WinModel


def compressed(n=3000, seed=0):
    """Raw probabilities squeezed toward 0.5: the true probability is further from 0.5 than p."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, n)
    p = 0.5 + 0.5 * (truth - 0.5)
    return p, (rng.uniform(size=n) < truth).astype(float)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_preserves_order_and_stays_inside_unit_interval(method):
    p, y = compressed()
    cal = Calibrator(method).fit(p, y)
    grid = np.linspace(0, 1, 501)
    out = cal(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out > 0) & (out < 1))
    if method == "platt":
        assert np.all(np.diff(cal(np.linspace(0.05, 0.95, 50))) > 0)


def test_platt_stretches_a_compressed_model():
    p, y = compressed()
    cal = Calibrator("platt").fit(p, y)
    assert cal.params()["slope"] > 1.5
    assert np.mean((cal(p) - y) ** 2) < np.mean((p - y) ** 2)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        Calibrator("beta")


def rows(n=800, seed=1):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, len(WIN_FEATURES))), columns=WIN_FEATURES)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x.rating_diff))).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="12h").strftime("%Y-%m-%d")
    return x.assign(game_id=[str(i) for i in range(n)], date=dates, home_win=y)


def test_out_of_sample_uses_only_earlier_games():
    r = rows()
    oos = out_of_sample(r, month_starts("2024-01-01", "2024-12-01"), min_train=100)
    assert len(oos) and (oos.date >= oos.fit_before).all()
    first = oos.fit_before.min()
    assert (r.date < first).sum() >= 100
    m = WinModel().fit(r[r.date < first])
    month = r[(r.date >= first) & (r.date < sorted(oos.fit_before.unique())[1])]
    np.testing.assert_allclose(oos[oos.fit_before == first].p_raw.to_numpy(), m.predict(month))


def test_calibrated_win_model_is_a_drop_in_for_predict():
    r = rows()
    win = WinModel().fit(r)
    oos = out_of_sample(r, month_starts("2024-01-01", "2024-12-01"), min_train=100)
    cal = fit_before(oos, "2024-12-01", "platt")
    wrapped = CalibratedWinModel(win, cal)
    raw, out = win.predict(r), wrapped.predict(r)
    assert wrapped.name == "win-logit+platt"
    assert np.all((out > 0) & (out < 1))
    order = np.argsort(raw)
    assert np.all(np.diff(out[order]) >= -1e-12)
    assert "cal_slope" in wrapped.coefficients()
