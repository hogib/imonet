import numpy as np
import pytest

from imonet import transients as tr_mod
from imonet.data import component_scale, detrend, training_windows
from imonet.transients import Transient

from test_data import synthetic_cube

L = 32


def transient(radial=False, onset=5.0, rise=1.0, reverse=0.0, amp=10.0, radius=20.0, centre=(0.0, 0.0)):
    direction = np.array([1.0, 0.0, 0.5]) if radial else np.array([0.6, 0.8, 0.25])
    return Transient(radius, centre, radial, direction, amp, onset, rise, reverse)


@pytest.fixture(scope="module")
def cube():
    return synthetic_cube()


def windows(cube, p, **kw):
    return training_windows(cube, component_scale(cube), 40, 3, length=L, transient_p=p, **kw)


def test_history_of_a_step_a_ramp_and_a_reversal():
    f = transient(onset=5.0, rise=1.0).history(L)
    assert (f[:6] == 0).all() and (f[6:] == 1).all()
    g = transient(onset=0.0, rise=10.0).history(L)
    assert g[0] == 0 and g[5] == pytest.approx(0.5) and (g[10:] == 1).all() and (np.diff(g) >= 0).all()
    h = transient(onset=0.0, rise=5.0, reverse=10.0).history(L)
    assert h[5] == pytest.approx(1.0) and h[10] == pytest.approx(0.5) and (h[15:] == 0).all()


def test_patterns():
    t = transient(amp=10.0, radius=20.0, centre=(5.0, 0.0))
    u = t.pattern(np.array([5.0, 25.0]), np.zeros(2))
    assert u[0] == pytest.approx(10.0 * t.direction)
    assert np.linalg.norm(u[1]) == pytest.approx(10.0 * np.exp(-0.5) * np.linalg.norm(t.direction))
    r = transient(radial=True, amp=10.0, radius=20.0).pattern(np.array([0.0, 20.0, 0.0]), np.array([0.0, 0.0, -40.0]))
    assert r[0, :2] == pytest.approx(0.0) and r[0, 2] == pytest.approx(5.0)       # uplift at the centre
    assert r[1, 0] == pytest.approx(10.0) and r[1, 1] == pytest.approx(0.0)       # peak A at r = R, outward
    assert r[2, 0] == pytest.approx(0.0) and r[2, 1] < 0                           # outward to the south


def test_draws_stay_in_their_ranges():
    rng = np.random.default_rng(0)
    ts = [tr_mod.draw(rng, 256) for _ in range(2000)]
    radius = np.array([t.radius_km for t in ts])
    assert radius.min() >= 10 and radius.max() <= 300
    assert all(np.hypot(*t.centre_km) <= 1.5 * t.radius_km for t in ts)
    steps = [t for t in ts if t.rise == 1.0]
    assert 0.25 < len(steps) / len(ts) < 0.35 and max(t.amplitude_mm for t in steps) <= 50
    assert 0.45 < np.mean([t.radial for t in ts]) < 0.55
    assert all(1 <= t.amplitude_mm <= 30 for t in ts if t.rise > 1)


def test_seen_uses_the_range_over_available_days():
    s = np.zeros((3, L, 3))
    s[:, 10:, 0] = [[0.5], [2.0], [2.0]]
    ok = np.ones((3, L), bool)
    ok[2, 10:] = False                                       # down after the step: sees nothing
    assert tr_mod.seen(s, ok).tolist() == [False, True, False]


def test_p0_reproduces_the_samples_without_transients(cube):
    a, b = windows(cube, 0.0), training_windows(cube, component_scale(cube), 40, 3, length=L)
    for i in range(10):
        sa, sb = a[i], b[i]
        assert not sa["planted"]
        for k in sb:
            assert np.array_equal(sa[k], sb[k]), k


def test_planting_keeps_the_sample_and_adds_the_detrended_transient(cube):
    clean, planted = windows(cube, 0.0), windows(cube, 1.0)
    n_planted = 0
    for i in range(40):
        c, p = clean[i], planted[i]
        for k in ("target", "t0", "nbr", "y_ok", "x_ok", "geom", "spacing_km"):
            assert np.array_equal(c[k], p[k]), k
        if not p["planted"]:
            assert np.array_equal(c["x"], p["x"]) and np.array_equal(c["y"], p["y"])
            continue
        n_planted += 1
        diff = p["x"] - c["x"]                               # the planted part: itself detrended
        assert np.allclose(detrend(diff, p["x_ok"]), diff, atol=1e-5)
        assert (diff[~p["x_ok"]] == 0).all()
        dy = (p["y"] - c["y"])[p["valid"]]
        assert np.allclose(detrend(dy, p["y_ok"][p["valid"]]), dy, atol=1e-5)
        assert np.abs(dy).max() > 0
    assert n_planted > 10
    assert np.array_equal(planted[5]["y"], planted[5]["y"])  # deterministic per sample


def test_plant_is_linear_in_the_detrending(cube):
    ds = windows(cube, 0.0)
    out = ds[0]
    x0, y0 = out["x"].copy(), out["y"].copy()
    t = transient(onset=8.0, rise=6.0, amp=8.0, radius=200.0)
    ds.plant(out, t)
    assert out["planted"]
    s, t0, v = int(out["target"]), int(out["t0"]), out["valid"]
    scale = component_scale(cube)
    own = t.signal(np.zeros(1), np.zeros(1), L)[0]
    raw = np.where(out["x_ok"][:, None], cube.r[s, t0:t0 + L] / scale, 0.0)
    assert np.allclose(out["x"], detrend(raw + own / scale, out["x_ok"]), atol=1e-5)
    sig = t.signal(out["geom"][v, 0], out["geom"][v, 1], L)
    nbr = out["nbr"][v]
    raw_y = np.where(out["y_ok"][v][..., None], cube.r[nbr, t0:t0 + L] / scale, 0.0)
    assert np.allclose(out["y"][v], detrend(raw_y + sig / scale, out["y_ok"][v]), atol=1e-5)
    assert not np.allclose(out["x"], x0) and not np.allclose(out["y"], y0)


def test_an_unseen_transient_is_not_planted(cube):
    ds = windows(cube, 0.0)
    out = ds[0]
    x0 = out["x"].copy()
    ds.plant(out, transient(amp=10.0, radius=10.0, centre=(5000.0, 0.0)))   # far away
    assert not out["planted"] and np.array_equal(out["x"], x0)
