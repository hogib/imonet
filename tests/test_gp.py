import numpy as np
import pytest

from imonet.baselines import synthesise
from imonet.cube import HELD_OUT, NORMAL, VALIDATION
from imonet.data import component_scale, detrend
from imonet.evaluate import predict
from imonet.gp import (GM_DAYS, GP, LAGS, SpaceTime, _flicker_acf, _gm_acf, _white_acf, detrended_lag_curve,
                       empirical_lag_curves, fit_space_time)
from imonet.kriging import Covariance, fit_covariance, empirical_covariance

from test_data import synthetic_cube

L = 64
MODEL = SpaceTime(white={"cm": (0.05, 0.04, 0.06), "r": (0.01, 0.02, 0.01), "n": (0.3, 0.2, 0.4)},
                  flicker={"cm": (0.01, 0.01, 0.02), "r": (0.002, 0.004, 0.002), "n": (0.02, 0.01, 0.03)},
                  ell_km=(20.0, 10.0, 15.0),
                  gm={"cm": (0.0, 0.02, 0.0), "r": (0.0, 0.0, 0.0), "n": (0.1, 0.05, 0.2)},
                  gm_days={"cm": (10.0, 10.0, 10.0), "r": (10.0, 10.0, 10.0), "n": (12.0, 12.0, 12.0)})


def test_flicker_covariance_matches_the_synthesised_noise():
    rng = np.random.default_rng(0)
    x = synthesise(np.array([0.0]), np.array([1.0]), L, 4000, rng)[..., 0]     # m×L
    emp = np.array([(x[:, :L - u] * x[:, u:]).mean() for u in range(8)])
    assert emp == pytest.approx(_flicker_acf(L)[:8], rel=0.1)


def test_space_time_round_trip(tmp_path):
    MODEL.save(tmp_path / "m.json")
    assert SpaceTime.load(tmp_path / "m.json") == MODEL


def test_detrended_white_noise_curve():
    curve = detrended_lag_curve(_white_acf(L), (0,))
    assert curve[0] == pytest.approx((L - 2) / L)


def test_fit_recovers_known_spectra():
    lags = [u for u in LAGS if u < L]
    tau = GM_DAYS[np.argmin(np.abs(np.log(np.asarray(GM_DAYS) / 12.0)))]     # on the grid
    gm_days = {p: (tau,) * 3 for p in MODEL.gm}
    model = SpaceTime(MODEL.white, MODEL.flicker, MODEL.ell_km, MODEL.gm, gm_days)
    curve = lambda p, c: detrended_lag_curve(model.autocov(p, c, L), lags)
    spatial = Covariance((0.1,) * 3, (0.1,) * 3, MODEL.ell_km, (0.1,) * 3)
    near_km = 3.0
    from imonet.kriging import matern32
    curves = {"lags": lags, "near_km": near_km,
              "far": np.stack([curve("cm", c) for c in range(3)], 1).tolist(),
              "near": np.stack([curve("cm", c) + matern32(near_km / MODEL.ell_km[c]) * curve("r", c)
                                for c in range(3)], 1).tolist(),
              "auto": np.stack([curve("cm", c) + curve("r", c) + curve("n", c) for c in range(3)], 1).tolist()}
    fit = fit_space_time(curves, spatial, L)
    for p in ("cm", "r", "n"):
        assert fit.white[p] == pytest.approx(model.white[p], rel=1e-3, abs=1e-5)
        assert fit.flicker[p] == pytest.approx(model.flicker[p], rel=1e-3, abs=1e-5)
        assert fit.gm[p] == pytest.approx(model.gm[p], rel=1e-3, abs=1e-5)
        for c in range(3):
            if model.gm[p][c] > 0:
                assert fit.gm_days[p][c] == pytest.approx(tau)


def one_target_batch(y, ok, east, north):
    k = len(east)
    return {"y": y[None].astype(np.float32), "y_ok": ok[None], "valid": np.ones((1, k), bool),
            "geom": np.stack([east, north, np.hypot(east, north)], 1)[None].astype(np.float32),
            "nbr": np.arange(k)[None], "target": np.array([-1]), "t0": np.array([0])}


def test_gp_ignores_each_neighbours_mean_and_trend():
    rng = np.random.default_rng(1)
    east, north = rng.uniform(-50, 50, 4), rng.uniform(-50, 50, 4)
    y = rng.normal(size=(4, L, 3))
    ok = rng.random((4, L)) > 0.2
    gp = GP(MODEL, n_samples=8)
    a = gp.sample(one_target_batch(y, ok, east, north), np.random.default_rng(5))
    t = np.arange(L)[None, :, None]
    shifted = y + rng.normal(size=(4, 1, 3)) * 3 + rng.normal(size=(4, 1, 3)) * 0.1 * t
    b = gp.sample(one_target_batch(shifted, ok, east, north), np.random.default_rng(5))
    assert np.allclose(a, b, atol=1e-3)


def test_gp_follows_a_colocated_quiet_neighbour():
    rng = np.random.default_rng(2)
    quiet = SpaceTime(white={"cm": (0.5,) * 3, "r": (0.5,) * 3, "n": (1e-6,) * 3},
                      flicker={"cm": (0.0,) * 3, "r": (0.0,) * 3, "n": (0.0,) * 3}, ell_km=(20.0,) * 3)
    y = rng.normal(size=(2, L, 3))
    ok = np.ones((2, L), bool)
    s = GP(quiet, n_samples=4).sample(one_target_batch(y, ok, np.array([0.0, 200.0]), np.zeros(2)),
                                      np.random.default_rng(0))
    target = detrend(s[0], np.ones((4, L), bool), zero_missing=False)
    assert np.abs(target - detrend(y[0], ok[0])[None]).max() < 0.05


def test_gp_given_noise_level_widens_a_noisy_target():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(3, L, 3))
    ok = np.ones((3, L), bool)
    b = one_target_batch(y, ok, np.array([30.0, -40.0, 5.0]), np.array([10.0, 20.0, -60.0]))
    b |= {"target_noise": np.log(np.full((1, 3), 2.0)).astype(np.float32), "target_noise_ok": np.array([True])}
    typical = GP(MODEL, n_samples=200).sample(b, np.random.default_rng(0)).std(axis=1).mean()
    noisy = GP(MODEL, n_samples=200, target_noise="given").sample(b, np.random.default_rng(0)).std(axis=1).mean()
    assert noisy > 1.5 * typical


def test_gp_on_a_cube():
    cube = synthetic_cube()
    scale = component_scale(cube)
    spatial = fit_covariance(empirical_covariance(cube, scale, length=32))
    curves = empirical_lag_curves(cube, scale, length=32)
    assert np.isfinite(np.asarray(curves["auto"])).all()
    model = fit_space_time(curves, spatial, length=32)
    gp = GP(model, n_samples=3)
    pred = predict(gp, cube, scale, cube.stations(HELD_OUT)[:2], cube.stations(NORMAL), VALIDATION, length=32, k=6)
    assert pred.samples.shape[0] == 3 and np.isfinite(pred.samples).all()
