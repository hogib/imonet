import numpy as np
import pytest

from imonet.baselines import KrigeNoise, make
from imonet.cube import HELD_OUT, NORMAL, VALIDATION
from imonet.data import component_scale
from imonet.evaluate import predict
from imonet.kriging import Covariance, empirical_covariance, fit_covariance, krige, matern32

from test_data import synthetic_cube

COV = Covariance(s_cm=(0.3, 0.2, 0.5), s_r=(0.4, 0.3, 0.6), ell_km=(40.0, 60.0, 30.0), s_n=(0.3, 0.5, 0.4))


def batch_at(east, north, y, ok):
    """One target at the origin with neighbours at (east, north) km."""
    k = len(east)
    geom = np.stack([east, north, np.hypot(east, north)], axis=1)
    return {"y": y[None], "y_ok": ok[None], "geom": geom[None], "valid": np.ones((1, k), bool),
            "y_sigma": np.ones((1,) + y.shape)}


def gp_brute_force(east, north, y_day, cov, c):
    """R&W eqs. 2.25-2.26 for one day and component, written out."""
    pts = np.stack([east, north], axis=1)
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    k = cov.s_cm[c] + cov.s_r[c] * matern32(d / cov.ell_km[c]) + cov.s_n[c] * np.eye(len(pts))
    ks = cov.s_cm[c] + cov.s_r[c] * matern32(np.hypot(east, north) / cov.ell_km[c])
    w = np.linalg.solve(k, ks)
    return w @ y_day, cov.total_variance(c) - ks @ w


def test_krige_matches_the_gp_equations_and_handles_gaps():
    rng = np.random.default_rng(0)
    east, north = rng.uniform(-80, 80, 6), rng.uniform(-80, 80, 6)
    y = rng.normal(size=(6, 10, 3))
    ok = np.ones((6, 10), bool)
    ok[2, 3:6] = False
    ok[:, 8] = False                                     # nobody on day 8
    mean, var = krige(batch_at(east, north, y, ok), COV)
    for day in (0, 4):
        sel = ok[:, day]
        for c in range(3):
            m, v = gp_brute_force(east[sel], north[sel], y[sel, day, c], COV, c)
            assert mean[0, day, c] == pytest.approx(m, rel=1e-4, abs=1e-5)
            assert var[0, day, c] == pytest.approx(v, rel=1e-4)
    assert np.allclose(mean[0, 8], 0.0)
    assert np.allclose(var[0, 8], [COV.total_variance(c) for c in range(3)])


def test_krige_variance_shrinks_with_a_close_neighbour():
    y = np.ones((2, 4, 3))
    ok = np.ones((2, 4), bool)
    far = krige(batch_at(np.array([200.0, -200.0]), np.zeros(2), y, ok), COV)[1][0, 0]
    near = krige(batch_at(np.array([1.0, -200.0]), np.zeros(2), y, ok), COV)[1][0, 0]
    assert (near < far).all()
    # A neighbour cannot remove the target's own noise: variance stays above it.
    assert (near > np.array(COV.s_n)).all()


def test_fit_recovers_a_known_covariance():
    d = np.array([1.0, 3, 7, 12, 17, 25, 35, 45, 60, 85, 125, 175, 225, 275])
    cov = np.stack([COV.shared(d, c) for c in range(3)], axis=1)
    emp = {"d_km": d.tolist(), "cov": cov.tolist(), "pairs": [100.0] * len(d),
           "var": [COV.total_variance(c) for c in range(3)]}
    fit = fit_covariance(emp)
    for c in range(3):
        assert fit.s_cm[c] == pytest.approx(COV.s_cm[c], rel=0.05)
        assert fit.s_r[c] == pytest.approx(COV.s_r[c], rel=0.05)
        assert fit.ell_km[c] == pytest.approx(COV.ell_km[c], rel=0.1)
        assert fit.s_n[c] == pytest.approx(COV.s_n[c], rel=0.1)


def test_covariance_round_trip(tmp_path):
    COV.save(tmp_path / "c.json")
    assert Covariance.load(tmp_path / "c.json") == COV


def test_kriging_on_a_cube():
    cube = synthetic_cube()
    scale = component_scale(cube)
    emp = empirical_covariance(cube, scale, length=32)
    assert len(emp["d_km"]) > 0 and all(v > 0 for v in emp["var"])
    cov = fit_covariance(emp)
    assert all(v > 0 for v in cov.s_n)
    for method in (make("krige", covariance=cov), KrigeNoise(cov, n_samples=3)):
        pred = predict(method, cube, scale, cube.stations(HELD_OUT)[:2], cube.stations(NORMAL), VALIDATION,
                       length=32, k=6)
        assert pred.samples.shape[0] == method.n_samples and np.isfinite(pred.samples).all()
