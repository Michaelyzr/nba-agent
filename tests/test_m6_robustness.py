"""M6 robustness grid: sequence cuts and hourly resampling, and the alternative networks keep ordered quantiles."""
import numpy as np
import torch

from evaluation.m6_robustness import CNNNet, StaticMLP, net_class, sequences
import forecast.impact as impact


def test_sequences_cut_and_resample():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(5, 96, 4)).astype(np.float32)
    full[:, :, 2] = np.log1p(rng.uniform(0, 10, size=(5, 96)))
    assert np.array_equal(sequences(full, "15m24"), full[:, -24:])
    assert np.array_equal(sequences(full, "15m48"), full[:, -48:])
    h = sequences(full, "1h24")
    assert h.shape == (5, 24, 4)
    assert np.array_equal(h[:, -1, 0], full[:, -1, 0])
    np.testing.assert_allclose(h[:, -1, 2], np.log1p(np.expm1(full[:, -4:, 2]).sum(1)), rtol=1e-5)


def test_alternative_nets_output_ordered_quantiles():
    seq, static = torch.randn(7, 48, 4), torch.randn(7, 18)
    for net in (StaticMLP(4, 18), CNNNet(4, 18, 8)):
        q = net(seq, static)
        assert q.shape == (7, 3) and torch.all(q[:, 1:] >= q[:, :-1])


def test_net_class_restores_the_default():
    original = impact.ImpactNet
    with net_class(StaticMLP):
        assert impact.ImpactNet is StaticMLP
    assert impact.ImpactNet is original
