import numpy as np
import torch

from imonet.c2st import TrainConfig, features, run

L = 256


def windows(fake_scale, n_sta=40, per=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_sta * per
    real = rng.normal(0, 1, (n, L, 3)).astype(np.float32)
    fake = rng.normal(0, fake_scale, (n, L, 3)).astype(np.float32)
    return {"real": real, "fake": fake, "ok": np.ones((n, L), bool),
            "nbr": rng.normal(0, 1, (n, 4, L, 3)).astype(np.float32), "nbr_ok": np.ones((n, 4, L), bool),
            "station": np.repeat([f"S{i:02d}" for i in range(n_sta)], per), "t0": np.tile(np.arange(per) * 64, n_sta),
            "test": np.tile(np.arange(per) % 2 == 0, n_sta)}


def test_features_balance_and_channels():
    w = windows(1.0)
    x, y = features(w, np.arange(10), "marginal")
    assert x.shape == (20, 4, L) and y.sum() == 10
    x, _ = features(w, np.arange(10), "conditional")
    assert x.shape == (20, 4 + 4 * 3 + 4, L)


import pytest


@pytest.mark.parametrize("clf", ["features", "cnn"])
def test_same_distribution_is_at_chance_and_different_is_detected(clf):
    cfg = TrainConfig(classifier=clf, epochs=12, patience=4)
    dev = torch.device("cpu")
    same, _ = run(windows(1.0), "marginal", cfg, dev)
    diff, _ = run(windows(1.6), "marginal", cfg, dev)
    assert 0.35 < same["accuracy"] < 0.65 and same["p_value"] > 0.01
    assert diff["accuracy"] > 0.9 and diff["p_value"] < 1e-6


def test_paired_test_and_filtering():
    from types import SimpleNamespace
    from imonet.c2st import filter_windows
    cfg = TrainConfig(classifier="features")
    same, ex = run(windows(1.0), "marginal", cfg, torch.device("cpu"))
    assert same["windows"] == 120 and 0.35 < same["paired_accuracy"] < 0.65 and same["paired_p_value"] > 0.01
    diff, _ = run(windows(1.6), "marginal", cfg, torch.device("cpu"))
    assert diff["paired_accuracy"] > 0.9 and diff["paired_p_value"] < 1e-6
    w = windows(1.0)
    sta = np.array(sorted(set(w["station"])))
    keep = np.ones(len(sta), bool)
    keep[[0, 1]] = False
    setup = SimpleNamespace(cube=SimpleNamespace(sta=sta), keep=keep)
    out, dropped = filter_windows(w, setup)
    assert dropped == 2 and len(out["t0"]) == len(w["t0"]) - 12 and not np.isin(out["station"], sta[:2]).any()
    assert filter_windows(w, SimpleNamespace(cube=SimpleNamespace(sta=sta), keep=np.ones(len(sta), bool)))[1] == 0
