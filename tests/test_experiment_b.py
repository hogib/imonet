import numpy as np
import pandas as pd
import pytest

from imonet import events as ev_mod
from imonet.baselines import IDW, IDWNoise, Zero
from imonet.cube import HELD_OUT, VALIDATION
from imonet.evaluate import Setup, network
from imonet.experiment_b import Event, Predictor, event_signal, planted, score_event

from test_data import synthetic_cube

L = 32
STEPS = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])


def ramp_event(cube, east_mm, region, t0=125, duration=10.0):
    """A uniform eastward ramp of east_mm at the stations in `region` (S
    bool), over `duration` days from day t0: one source with unit slip."""
    tp = ev_mod.EventTemplate(name="test", slip_senses=("rlss",), length_km=1.0, top_km=0.0, bottom_km=1.0,
                              slip_m=1.0, time="ramp", duration_days=duration)
    green = np.zeros((cube.n_sta, 3, 1))
    green[region, 0, 0] = east_mm / 1000.0
    placed = ev_mod.PlacedEvent(tp, "none", np.array([0]), np.array([1.0]), 180.0, np.zeros(1), green,
                                (float(cube.lon[region].mean()), float(cube.lat[region].mean())))
    peak = np.abs(green[:, 0, 0]) * 1000
    return Event(0, placed, t0, t0 + int(duration) + 20, peak, float("nan"))


@pytest.fixture(scope="module")
def setup():
    return Setup(synthetic_cube(), STEPS)


def test_planting_adds_the_ramp_after_the_onset(setup):
    cube = setup.cube
    region = np.zeros(cube.n_sta, bool)
    region[:30] = True
    ev = ramp_event(cube, 8.0, region)
    sta, s = event_signal(cube, ev)
    assert set(sta) == set(np.flatnonzero(region))
    assert s[:, 0, 0] == pytest.approx(0.0) and s[:, -1, 0] == pytest.approx(8.0)
    pc = planted(cube, ev)
    diff = pc.r - cube.r
    assert np.nanmax(np.abs(diff[:, :ev.t0])) == 0 and np.nanmax(np.abs(diff[~region])) == 0


def test_rho_is_zero_for_zero_and_near_one_for_a_close_average(setup):
    cube = setup.cube
    region = np.ones(cube.n_sta, bool)                           # everyone moves the same
    ev = ramp_event(cube, 10.0, region)
    net = network(cube, None, 0)
    zero = pd.DataFrame(score_event(Predictor(Zero(), setup, net, length=L, k=6), cube, ev, VALIDATION))
    idw = pd.DataFrame(score_event(Predictor(IDW(10.0), setup, net, length=L, k=6), cube, ev, VALIDATION))
    seen = idw[idw.role == "seen"]
    assert len(seen) > 0 and (zero[zero.role == "seen"].rho.abs() < 1e-9).all()
    assert seen.rho.median() == pytest.approx(1.0, abs=0.02)
    assert seen["shape"].median() > 0.99


def test_pairing_removes_the_sampling_noise(setup):
    cube = setup.cube
    region = np.zeros(cube.n_sta, bool)
    region[:50] = True
    ev = ramp_event(cube, 6.0, region)
    net = network(cube, None, 0)
    a = pd.DataFrame(score_event(Predictor(IDW(10.0), setup, net, length=L, k=6), cube, ev, VALIDATION))
    b = pd.DataFrame(score_event(Predictor(IDWNoise(10.0, n_samples=4), setup, net, length=L, k=6),
                                 cube, ev, VALIDATION))
    seen_a, seen_b = a[a.role == "seen"].set_index("station"), b[b.role == "seen"].set_index("station")
    # idw+noise's mean responds as idw's, up to the change in fitted noise the event causes
    assert np.allclose(seen_a.rho, seen_b.loc[seen_a.index].rho, atol=0.15)


def test_outside_targets_are_scored_for_spurious_signal(setup):
    cube = setup.cube
    region = np.zeros(cube.n_sta, bool)
    region[:20] = True                                           # southern rows only
    ev = ramp_event(cube, 10.0, region)
    net = network(cube, None, 0)
    rows = pd.DataFrame(score_event(Predictor(IDW(1000.0), setup, net, length=L, k=6), cube, ev, VALIDATION))
    out = rows[rows.role == "outside"]
    assert len(out) > 0 and (out.spurious > 0).all()             # a wide average leaks the event
    assert set(out.station) & set(cube.sta[region]) == set()


def test_offset_of_a_step():
    x = np.zeros((40, 3))
    x[20:, 0] = 5.0
    ok = np.ones(40, bool)
    from imonet.experiment_b import offset
    assert offset(x, ok, 20)[0] == pytest.approx(5.0)
    ok[25:28] = False
    assert offset(x, ok, 20)[0] == pytest.approx(5.0)
