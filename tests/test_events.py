from pathlib import Path

import numpy as np
import pytest

from imonet import events

KM = 1000.0


def rect(x0, y0, x1, y1, z_top, z_bot, dx_bottom=0.0):
    """Two triangles of a rectangle from (x0, y0) to (x1, y1) at the top,
    shifted by dx_bottom (east) at the bottom; metres."""
    a, b = [x0, y0, z_top], [x1, y1, z_top]
    c, d = [x1 + dx_bottom, y1, z_bot], [x0 + dx_bottom, y0, z_bot]
    return np.array([[a, b, c], [a, c, d]], dtype=float)


def test_right_lateral_moves_east_side_south():
    tri = rect(0, -50 * KM, 0, 50 * KM, 0, -10 * KM)            # vertical, striking north
    obs = np.array([[5 * KM, 0, 0], [-5 * KM, 0, 0]])
    u = events.displacement(obs, tri, np.ones(2), 180.0)
    assert u[0, 1] < -0.1 and u[1, 1] > 0.1                     # east side south, west side north
    np.testing.assert_allclose(u[0, 1], -u[1, 1], rtol=1e-6)
    assert abs(u[0, 0]) < 1e-3 and abs(u[0, 2]) < 1e-3


def test_long_strike_slip_fault_matches_the_screw_dislocation():
    # slip s from the surface to depth D on a very long vertical fault:
    # u(x) = (s / π) arctan(D / x) at the surface (half-space, 2-D limit)
    D = 10 * KM
    tri = rect(0, -2000 * KM, 0, 2000 * KM, 0, -D)
    xs = np.array([2, 5, 10, 30]) * KM
    obs = np.stack([xs, np.zeros(4), np.zeros(4)], axis=1)
    u = events.displacement(obs, tri, np.ones(2), 0.0)           # left-lateral: east side north
    np.testing.assert_allclose(u[:, 1], np.arctan(D / xs) / np.pi, rtol=1e-3)


def test_reverse_slip_lifts_the_hanging_wall():
    # dips east at 45°: the hanging wall is to the east
    tri = rect(0, -30 * KM, 0, 30 * KM, -1 * KM, -11 * KM, dx_bottom=10 * KM)
    obs = np.array([[3 * KM, 0, 0], [-10 * KM, 0, 0]])
    u = events.displacement(obs, tri, np.ones(2), 90.0)
    assert u[0, 2] > 0.05                                       # uplift above the hanging wall
    assert u[0, 0] < 0                                          # hanging wall moves up-dip, i.e. west
    u_normal = events.displacement(obs, tri, np.ones(2), -90.0)
    np.testing.assert_allclose(u_normal, -u, atol=1e-9)


def test_corner_order_does_not_matter():
    tri = rect(0, -20 * KM, 3 * KM, 20 * KM, 0, -8 * KM, dx_bottom=4 * KM)
    obs = np.array([[6 * KM, 2 * KM, 0], [-7 * KM, -3 * KM, 0]])
    a = events.displacement(obs, tri, np.array([1.0, 0.5]), 150.0)
    b = events.displacement(obs, tri[:, [0, 2, 1]], np.array([1.0, 0.5]), 150.0)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_read_tsurf_merges_blocks_and_atoms(tmp_path):
    p = tmp_path / "f.ts"
    p.write_text("GOCAD TSurf 1\nHEADER {\nname:x\n}\nTFACE\n"
                 "VRTX 1 0 0 0\nVRTX 2 1000 0 0\nVRTX 3 0 0 -1000\nTRGL 1 2 3\n"
                 "TFACE\nATOM 4 2\nPVRTX 5 1000 0 -1000 7\nTRGL 4 5 3\nEND\n")
    xyz, tri = events.read_tsurf(p)
    assert xyz.shape == (4, 3)
    np.testing.assert_array_equal(tri, [[0, 1, 2], [1, 3, 2]])


CFM = Path("data/cfm/extracted/obj/preferred/1000m")


@pytest.mark.skipif(not CFM.exists(), reason="SCEC CFM not downloaded")
def test_cfm_fault_lands_in_california_and_slips():
    path = next(CFM.glob("*Superstition_Hills*"))
    fault = events.load_fault(path)
    assert -117 < fault.lon.mean() < -115 and 32 < fault.lat.mean() < 34
    # 10 cm right-lateral on the whole surface: stations 5 km either side move opposite ways
    lon0, lat0 = fault.lon.mean(), fault.lat.mean()
    u = events.station_displacement(fault, np.full(len(fault.tri), 0.1), 180.0,
                                    [lon0 + 0.06, lon0 - 0.06], [lat0 + 0.03, lat0 - 0.03])
    assert np.sign(u[0, 0]) != np.sign(u[1, 0]) or np.sign(u[0, 1]) != np.sign(u[1, 1])
    assert np.abs(u).max() < 0.1


def vertical_fault(length_km=100.0, depth_km=15.0, step_km=1.0):
    """A north-striking vertical fault at 117°W, meshed into 1 km triangles."""
    ys = np.arange(-length_km / 2, length_km / 2 + 1e-9, step_km)
    zs = np.arange(0.0, depth_km + 1e-9, step_km)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    lat = 34.0 + Y.ravel() / 111.2
    lon = np.full(lat.shape, -117.0)
    nz = len(zs)
    tri = []
    for i in range(len(ys) - 1):
        for k in range(nz - 1):
            a, b, c, d = i * nz + k, (i + 1) * nz + k, (i + 1) * nz + k + 1, i * nz + k + 1
            tri += [[a, b, c], [a, c, d]]
    return events.Fault("test", lon, lat, -Z.ravel() * KM, np.array(tri))


def template(**kw):
    base = dict(name="t", slip_senses=("rlss",), length_km=20.0, top_km=0.0, bottom_km=5.0,
                slip_m=0.05, time="ramp", duration_days=30.0)
    return events.EventTemplate(**{**base, **kw})


def test_patch_respects_length_depth_and_taper():
    f = vertical_fault()
    rng = np.random.default_rng(0)
    ev = events.place_event(template(top_km=2.0, bottom_km=8.0), f, rng, np.array([-116.9]), np.array([34.0]))
    corners = np.stack([f.lat[f.tri[ev.tri]], -f.elev[f.tri[ev.tri]] / KM], -1).mean(1)
    span_km = (corners[:, 0].max() - corners[:, 0].min()) * 111.2
    assert 17 <= span_km <= 20.5
    assert corners[:, 1].min() >= 2.0 and corners[:, 1].max() <= 8.0
    assert ev.slip_m.max() <= 0.05 + 1e-12 and ev.slip_m.min() < 0.3 * 0.05      # tapered edges
    assert ev.rake == 180.0


def test_surface_patch_is_not_tapered_at_the_top():
    f = vertical_fault()
    ev = events.place_event(template(top_km=0.0, bottom_km=4.0, taper=0.25), f, np.random.default_rng(1),
                            np.array([-116.9]), np.array([34.0]))
    depth = -f.elev[f.tri[ev.tri]].mean(1) / KM
    mid = np.abs(f.lat[f.tri[ev.tri]].mean(1) - f.lat[f.tri[ev.tri]].mean()) < 0.02
    assert ev.slip_m[mid & (depth < 1.0)].min() > 0.9 * 0.05


def test_time_functions_rise_from_zero_to_one():
    f = vertical_fault()
    sta = (np.array([-116.95, -117.05]), np.array([34.0, 34.0]))
    for kw in (dict(time="ramp"), dict(time="log", tau_days=5.0), dict(time="exp", tau_days=10.0),
               dict(time="propagating", speed_km_day=2.0, rise_days=3.0)):
        ev = events.place_event(template(**kw), f, np.random.default_rng(2), *sta)
        days = np.arange(0, int(ev.duration_days) + 5)
        g = ev.fraction(days)
        assert np.all(np.diff(g, axis=1) >= -1e-12) and g[:, 0].max() < 0.05
        np.testing.assert_allclose(g[:, -1], 1.0, atol=1e-9)
        np.testing.assert_allclose(ev.displacement_mm(days)[:, -1], ev.final_mm(), atol=1e-9)


def test_slip_front_reaches_far_triangles_later():
    f = vertical_fault()
    ev = events.place_event(template(time="propagating", speed_km_day=5.0, rise_days=2.0, length_km=40.0), f,
                            np.random.default_rng(3), np.array([-116.95]), np.array([34.0]))
    assert ev.onset_days.min() < 0.5 and 3.0 < ev.onset_days.max() <= 8.0 + 1e-9    # ≤ 40 km at 5 km/day
    assert ev.duration_days == pytest.approx(ev.onset_days.max() + 2.0)


def test_placed_event_matches_direct_displacement():
    f = vertical_fault()
    sta_lon, sta_lat = np.array([-116.97, -117.03]), np.array([34.01, 33.99])
    ev = events.place_event(template(), f, np.random.default_rng(4), sta_lon, sta_lat)
    direct = events.station_displacement(f, ev.slip_m, 180.0, sta_lon, sta_lat, select=ev.tri) * 1000
    np.testing.assert_allclose(ev.final_mm(), direct, rtol=1e-6, atol=1e-9)
    assert ev.final_mm()[0, 1] < 0 < ev.final_mm()[1, 1]             # east side south, west side north


def test_too_short_fault_is_refused():
    assert events.place_event(template(length_km=200.0), vertical_fault(), np.random.default_rng(0),
                              np.array([-117.0]), np.array([34.0])) is None


@pytest.mark.skipif(not CFM.exists(), reason="SCEC CFM not downloaded")
def test_catalogue_and_random_placement_on_real_faults():
    cat = events.fault_catalogue()
    assert len(cat) > 400 and {f.slip_sense for f in cat} <= set(events.RAKE)
    tp = template(slip_senses=("rlss", "llss"), min_dip=70.0)
    ev = events.random_event(tp, cat, np.random.default_rng(5), np.array([-117.0, -118.0]), np.array([34.0, 35.0]))
    assert ev.green.shape == (2, 3, len(ev.tri)) and np.isfinite(ev.final_mm()).all()


def test_ranges_are_drawn_per_event():
    tp = template(slip_m=(0.02, 0.05), speed_km_day=(0.4, 9.0), time="propagating", rise_days=3.0)
    rng = np.random.default_rng(0)
    draws = [tp.draw(rng) for _ in range(2000)]
    slip = np.array([d.slip_m for d in draws])
    speed = np.array([d.speed_km_day for d in draws])
    assert slip.min() >= 0.02 and slip.max() <= 0.05 and abs(np.median(slip) - 0.035) < 0.003  # uniform (×2.5)
    assert speed.min() >= 0.4 and speed.max() <= 9.0
    assert abs(np.median(np.log(speed)) - np.log(np.sqrt(0.4 * 9.0))) < 0.1                   # log-uniform
    assert all(d.rise_days == 3.0 for d in draws[:5])
    assert draws[0].drawn_values()["slip_m"] == draws[0].slip_m


def test_degenerate_triangles_are_dropped(tmp_path):
    p = tmp_path / "f.ts"
    p.write_text("GOCAD TSurf 1\nTFACE\nVRTX 1 400000 3800000 0\nVRTX 2 401000 3800000 0\n"
                 "VRTX 3 400000 3800000 -1000\nVRTX 4 402000 3800000 0\nTRGL 1 2 3\nTRGL 1 2 4\nEND\n")
    f = events.load_fault(p)
    assert len(f.tri) == 1                                   # 1-2-4 is a line: no area


def test_templates_load():
    tps = {t.name: t for t in events.load_templates()}
    assert set(tps) == {"afterslip", "shallow slow slip"}
    assert tps["afterslip"].time == "log" and tps["afterslip"].slip_m == (0.10, 0.40)
    assert tps["shallow slow slip"].time == "propagating"
