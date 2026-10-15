import numpy as np
import pytest
from torch.utils.data import DataLoader

from imonet.cube import HELD_OUT, NORMAL, TEST, TRAIN, VALIDATION, Cube, pairwise_km
from imonet.data import (ThinningBank, component_scale, context, day_of_year, detrend, training_windows,
                         validation_windows)
from imonet.network import spacing_km

from test_network import grid

L, K = 32, 6
KW = dict(length=L, k=K, spacing_range=(5.0, 15.0), min_km=(1.0, 10.0))


def synthetic_cube(seed: int = 0) -> Cube:
    """A 10×10 grid of stations 5 km apart over 200 days: 120 training,
    40 validation, 40 test. Every 7th station is held out."""
    rng = np.random.default_rng(seed)
    lat, lon = grid(10, 5.0)
    s, t = len(lat), 200
    r = rng.normal(0, [2.0, 2.0, 6.0], (s, t, 3)).astype(np.float32)
    avail = rng.random((s, t)) > 0.1
    avail[3, :] = False                              # a station that is always down
    r[~avail] = np.nan
    sigma = np.where(avail[..., None], 1.0, np.nan).astype(np.float32)
    split_day = np.repeat(np.array([TRAIN, VALIDATION, TEST], dtype=np.int8), [120, 40, 40])
    split_sta = np.where(np.arange(s) % 7 == 0, HELD_OUT, NORMAL).astype(np.int8)
    exclude = np.zeros((s, t), dtype=bool)
    exclude[:, 50:60] = True
    return Cube(r=r, sigma=sigma, avail=avail, sta=np.array([f"S{i:03d}" for i in range(s)]),
                lat=lat, lon=lon, days=np.datetime64("2010-01-01") + np.arange(t),
                split_day=split_day, split_sta=split_sta, exclude=exclude)


@pytest.fixture(scope="module")
def cube():
    return synthetic_cube()


@pytest.fixture(scope="module")
def scale(cube):
    return component_scale(cube)


def test_component_scale_is_the_robust_scale(cube, scale):
    assert scale == pytest.approx([2.0, 2.0, 6.0], rel=0.05)


def test_day_of_year():
    assert day_of_year(np.datetime64("2010-01-01")) == 0
    assert day_of_year(np.datetime64("2012-12-31")) == 365


def test_context_values_and_padding(cube, scale):
    net = cube.stations(NORMAL)
    t0 = 10
    out = context(cube, scale, cube.lat[44], cube.lon[44], t0, net, cube.avail, L, k=40, max_km=12.0)
    v, nbr = out["valid"], out["nbr"]
    assert (nbr[v] >= 0).all() and (nbr[~v] == -1).all()
    assert 3 not in nbr                               # always down
    assert out["geom"][v, 2].max() <= 12.0 and np.all(np.diff(out["geom"][v, 2]) >= 0)
    assert not out["y"][~v].any() and not out["y_ok"][~v].any() and not out["geom"][~v].any()
    for slot in np.flatnonzero(v):
        s, ok = nbr[slot], out["y_ok"][slot]
        assert (ok == cube.avail[s, t0:t0 + L]).all()
        want = detrend(np.nan_to_num(cube.r[s, t0:t0 + L] / scale), ok)
        assert np.allclose(out["y"][slot], want, atol=1e-5)
        assert not out["y"][slot][~ok].any()


def test_thinning_bank_spacings(cube):
    net = cube.stations(NORMAL)
    bank = ThinningBank(cube, net, np.random.default_rng(0), (5.0, 15.0), levels=3, seeds=2)
    d = pairwise_km(cube.lat[net], cube.lon[net])
    for target, keep in zip(bank.spacing, bank.masks):
        assert spacing_km(d, keep) >= target - 1e-6
    draws = [bank.draw(np.random.default_rng(i)) for i in range(400)]
    full = np.mean([len(net) == len(bank.network) for _, net in draws])
    assert 0.12 < full < 0.28


@pytest.fixture(scope="module")
def train(cube, scale):
    return training_windows(cube, scale, n_samples=64, seed=3, **KW)


def test_training_samples(cube, scale, train):
    for i in range(len(train)):
        b = train[i]
        s, t0, v, nbr = b["target"], b["t0"], b["valid"], b["nbr"]
        assert cube.split_sta[s] == NORMAL
        assert (cube.split_day[t0:t0 + L] == TRAIN).all()
        ok = cube.trainable()[s, t0:t0 + L]
        assert (b["x_ok"] == ok).all() and ok.mean() >= 0.5
        assert np.allclose(b["x"], detrend(np.nan_to_num(cube.r[s, t0:t0 + L] / scale), ok), atol=1e-5)
        assert s not in nbr
        assert (cube.split_sta[nbr[v]] == NORMAL).all()
        assert (b["geom"][v, 2] > b["min_km"]).all()
        # Excluded cells are never input or target.
        assert not b["y_ok"][v][:, max(0, 50 - t0):max(0, 60 - t0)].any()


def test_samples_are_repeatable_and_change_with_epoch(train):
    a, b = train[5], train[5]
    assert all(np.array_equal(a[k], b[k]) for k in a)
    train.set_epoch(1)
    c = train[5]
    train.set_epoch(0)
    assert not (a["t0"] == c["t0"] and a["target"] == c["target"] and np.array_equal(a["x"], c["x"]))


def test_validation_samples(cube, scale):
    val = validation_windows(cube, scale, n_samples=16, seed=4, **KW)
    for i in range(len(val)):
        b = val[i]
        assert cube.split_sta[b["target"]] == HELD_OUT
        assert (cube.split_day[b["t0"]:b["t0"] + L] == VALIDATION).all()
        assert (cube.split_sta[b["nbr"][b["valid"]]] == NORMAL).all()


def test_collates_into_batches(train):
    batch = next(iter(DataLoader(train, batch_size=8)))
    assert batch["x"].shape == (8, L, 3)
    assert batch["y"].shape == (8, K, L, 3)
    assert batch["geom"].shape == (8, K, 3)
    assert batch["valid"].dtype.is_floating_point is False


def test_detrend_removes_mean_and_line_over_available_days():
    rng = np.random.default_rng(0)
    t = np.arange(50, dtype=float)
    noise = rng.normal(size=(50, 3))
    x = 3.0 + 0.2 * t[:, None] * np.array([1, -1, 2]) + noise
    ok = rng.random(50) > 0.3
    out = detrend(x, ok)
    assert not out[~ok].any()
    a = np.stack([np.ones(ok.sum()), t[ok]], axis=1)
    assert np.abs(a.T @ out[ok]).max() < 1e-3                 # orthogonal to mean and line
    want = noise[ok] - a @ np.linalg.lstsq(a, noise[ok], rcond=None)[0]
    assert np.allclose(out[ok], want, atol=1e-4)


def test_detrend_keeps_every_day_of_generated_series_and_batches():
    x = np.random.default_rng(1).normal(size=(4, 2, 30, 3))
    ok = np.ones((4, 2, 30), bool)
    ok[..., 10:] = False
    full = detrend(x, ok[:, :1], zero_missing=False)
    assert full.shape == x.shape and np.all(full[..., 10:, :] != 0)
    assert np.allclose(full[1, 0], detrend(x[1, 0], ok[1, 0], zero_missing=False), atol=1e-5)


def test_detrend_with_one_day_removes_the_mean_only():
    x = np.arange(12, dtype=float).reshape(4, 3)
    ok = np.array([False, True, False, False])
    out = detrend(x, ok, zero_missing=False)
    assert np.allclose(out, x - x[1])
