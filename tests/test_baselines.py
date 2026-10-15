import math

import numpy as np
import pytest

from imonet.baselines import IDW, IDWNoise, Zero, idw_mean, noise_psd, synthesise
from imonet.cube import HELD_OUT, NORMAL, VALIDATION
from imonet.data import component_scale
from imonet.evaluate import period_windows, predict

from test_data import synthetic_cube


def batch_of(y, ok, dist, sigma=1.0):
    k, length, _ = y.shape
    return {"y": y[None], "y_ok": ok[None], "y_sigma": np.full((1, k, length, 3), sigma),
            "geom": np.stack([np.zeros(k), np.zeros(k), dist], axis=1)[None], "valid": np.ones((1, k), bool)}


def test_idw_mean_averages_available_neighbours():
    y = np.stack([np.full((4, 3), 1.0), np.full((4, 3), 3.0)])
    ok = np.ones((2, 4), bool)
    ok[1, 2] = False
    m = idw_mean(batch_of(y, ok, np.array([10.0, 20.0])), math.inf)[0]
    assert m[0] == pytest.approx([2.0] * 3)
    assert m[2] == pytest.approx([1.0] * 3)
    ok[:, 3] = False
    assert idw_mean(batch_of(y, ok, np.array([10.0, 20.0])), math.inf)[0, 3] == pytest.approx([0.0] * 3)


def test_idw_kernel_favours_the_nearer_neighbour():
    y = np.stack([np.full((4, 3), 1.0), np.full((4, 3), 3.0)])
    m = idw_mean(batch_of(y, np.ones((2, 4), bool), np.array([10.0, 50.0])), 20.0)[0, 0, 0]
    assert 1.0 < m < 1.5


@pytest.mark.parametrize("a,b", [(1.0, 0.0), (0.5, 0.02), (0.0, 0.05)])
def test_noise_synthesis_round_trip(a, b):
    rng = np.random.default_rng(0)
    x = synthesise(np.array([a]), np.array([b]), 2048, 16, rng)
    est = np.array([noise_psd(xi, np.ones(len(xi), bool)) for xi in x])
    assert np.median(est[:, 0, 0]) == pytest.approx(a, abs=0.15 * max(a, 0.3))
    assert np.median(est[:, 1, 0]) == pytest.approx(b, abs=0.3 * max(b, 0.02))


def test_white_noise_variance():
    x = synthesise(np.array([4.0]), np.array([0.0]), 4096, 4, np.random.default_rng(1))
    assert x.var() == pytest.approx(4.0, rel=0.1)


def test_period_windows_tile_the_split():
    cube = synthetic_cube()
    w = period_windows(cube, VALIDATION, 32)           # validation: days 120–159
    assert w == [(120, 120), (128, 152)]


@pytest.mark.parametrize("method", [Zero(), IDW(10.0), IDWNoise(10.0, n_samples=4)])
def test_predict_shapes_and_coverage(method):
    cube = synthetic_cube()
    scale = component_scale(cube)
    targets = cube.stations(HELD_OUT)
    pred = predict(method, cube, scale, targets, cube.stations(NORMAL), VALIDATION, length=32, k=6)
    assert pred.samples.shape == (method.n_samples, len(targets), 40, 3)
    assert np.isfinite(pred.samples).all()
    for w in pred.windows:
        assert targets[w["ti"]] not in w["nbr"]


def test_idw_noise_spread_comes_from_the_neighbours():
    cube = synthetic_cube()
    scale = component_scale(cube)
    targets = cube.stations(HELD_OUT)
    pred = predict(IDWNoise(10.0, n_samples=8), cube, scale, targets, cube.stations(NORMAL), VALIDATION,
                   length=32, k=6)
    # Neighbours are independent N(0, σ) noise, so the departures from their
    # idw means are about as large as the series themselves.
    spread = pred.samples.std(axis=0).mean(axis=(0, 1))
    assert spread == pytest.approx([2.0, 2.0, 6.0], rel=0.5)
