"""The NGL parsers and the cube builder, ported from gnssdl with their tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imonet import build, ngl
from imonet.cube import Cube, pairwise_km

HOLDINGS = """\
Sta  Lat(deg)   Long(deg) Hgt(m)  X(m)           Y(m)         Z(m)          Dtbeg      Dtend      Dtmod      NumSol StaOrigName
ANKR  39.8874   32.7585  976.045  4121948.0  2652187.0  4069023.0 1995-06-25 2022-01-04 2025-09-25   9000
FAR1  37.2000  217.0000  100.000  0 0 0 2010-01-01 2024-01-01 2025-09-25   5000
SHRT  39.0000   33.0000  100.000  0 0 0 2023-01-01 2023-06-01 2025-09-25    150
"""

STEPS = """\
ANKR  08MAY06  1  Antenna_and_Radome_Type_Changed
GOL2  94JAN17  2   363.078   201.965  6.7 ci3144585
"""

TENV3 = """\
site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)
ANKR 95JUN25 1995.4798 49893  807 0   32.8  -3552 -0.235481   4417023  0.323651   976  0.045141  0.0600 0.000771 0.001002 0.003364 -0.017130  0.014898 -0.175585  39.8873700200 -327.2415301033   976.04514
ANKR 95JUN26 1995.4825 49894  807 1   32.8  -3552 -0.237141   4417023  0.327936   976  0.040824  0.0600 0.000768 0.001013 0.003416 -0.005760  0.006089 -0.134956  39.8873700586 -327.2415301227   976.04082
"""


@pytest.fixture
def files(tmp_path: Path):
    (tmp_path / "h.txt").write_text(HOLDINGS)
    (tmp_path / "s.txt").write_text(STEPS)
    (tmp_path / "ANKR.tenv3").write_text(TENV3)
    return tmp_path


# --------------------------------------------------------------------------- NGL files


def test_holdings_wraps_longitude(files):
    h = ngl.read_holdings(files / "h.txt")
    assert list(h.sta) == ["ANKR", "FAR1", "SHRT"]
    assert h.loc[1, "lon"] == pytest.approx(-143.0)
    assert h.loc[0, "start"] == pd.Timestamp("1995-06-25")


def test_steps_kinds(files):
    s = ngl.read_steps(files / "s.txt")
    assert list(s.kind) == ["equipment", "quake"]
    q = s.iloc[1]
    assert q.date == pd.Timestamp("1994-01-17") and q.mag == 6.7 and q["info"] == "ci3144585"


def test_tenv3_dates_and_components(files):
    d = ngl.read_tenv3(files / "ANKR.tenv3")
    assert d.index[0] == pd.Timestamp("1995-06-25")             # MJD 49893
    assert d["e"].iloc[0] == pytest.approx(-3552.235481)
    assert d["n"].iloc[1] == pytest.approx(4417023.327936)
    assert d["u"].iloc[0] == pytest.approx(976.045141)


def test_region_selection(files):
    h = ngl.read_holdings(files / "h.txt")
    assert list(ngl.select(h, ngl.REGIONS["turkiye"]).sta) == ["ANKR"]          # SHRT has < 1 year
    assert list(ngl.select(h, ngl.Region(30, 45, -150, -140, 365)).sta) == ["FAR1"]


def test_station_id_validation():
    assert ngl.is_station_id("P595") and ngl.is_station_id("ANKR")
    assert not ngl.is_station_id("P595 P594")
    assert not ngl.is_station_id("../x")
    assert not ngl.is_station_id("ankr")
    with pytest.raises(ValueError):
        ngl.fetch_series(["../x"])


def test_download_is_cached_and_atomic(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def __init__(self):
            self.chunks = [b"abc", b"def", b""]

        def read(self, n):
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ngl.urllib.request, "urlopen", lambda req, timeout: calls.append(req) or Resp())
    dest = tmp_path / "x" / "A.tenv3"
    ngl.download("https://example.org/A", dest)
    ngl.download("https://example.org/A", dest)                                 # cached
    assert dest.read_bytes() == b"abcdef" and len(calls) == 1
    assert not dest.with_suffix(".tenv3.part").exists()


# --------------------------------------------------------------------------- stations


def _series(start="2008-01-01", end="2024-12-31", seed=0, noise_mm=1.0):
    """Synthetic station in tenv3 layout: 10 mm/yr east, 3 mm annual north."""
    days = pd.date_range(start, end, freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_mm / 1000, (len(days), 3))
    return pd.DataFrame({"decyear": t, "e": 0.010 * (t - 2008) + noise[:, 0],
                         "n": 0.003 * np.sin(2 * np.pi * t) + noise[:, 1], "u": noise[:, 2],
                         "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)


def _no_steps():
    return pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])


def _steps(rows):
    return pd.DataFrame([(s, pd.Timestamp(d), k, "", None, None, None) for s, d, k in rows],
                        columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])


def test_baseline_ignores_validation_and_test_data():
    a = _series()
    b = a.copy()
    b.loc["2022-01-01":, "e"] += 0.5                                           # 500 mm, test period only
    ra = build.process_station(a, _no_steps()).r
    rb = build.process_station(b, _no_steps()).r
    train = ra.index <= build.TRAIN_END
    np.testing.assert_allclose(ra[train].to_numpy(), rb[train].to_numpy(), atol=1e-9)
    assert (rb.loc["2022":, "e"] - ra.loc["2022":, "e"]).median() == pytest.approx(500, abs=1e-6)


def test_equipment_step_removed_quake_step_kept():
    s = _series(noise_mm=0.3)
    s.loc["2012-06-01":, "e"] += 0.008                                         # antenna change
    s.loc["2016-03-01":, "n"] += 0.006                                         # earthquake
    r = build.process_station(s, _steps([("X", "2012-06-01", "equipment"), ("X", "2016-03-01", "quake")])).r
    e_jump = r.loc["2012-06-01":"2012-08-01", "e"].mean() - r.loc["2012-04-01":"2012-05-31", "e"].mean()
    n_jump = r.loc["2016-03-01":"2016-05-01", "n"].mean() - r.loc["2016-01-01":"2016-02-29", "n"].mean()
    assert abs(e_jump) < 0.5
    assert n_jump == pytest.approx(6.0, abs=0.5)


def test_equipment_step_after_training_removed_locally():
    s = _series(noise_mm=0.3)
    s.loc["2023-05-01":, "u"] += 0.020
    r = build.process_station(s, _steps([("X", "2023-05-01", "equipment")])).r
    assert abs(r.loc["2023-05-01":"2023-06-15", "u"].median() - r.loc["2023-03-15":"2023-04-30", "u"].median()) < 1.0


def test_spike_is_screened():
    s = _series()
    s.loc["2015-07-04", "e"] += 0.050
    res = build.process_station(s, _no_steps())
    assert pd.Timestamp("2015-07-04") not in res.r.index and res.n_screened >= 1


def test_short_training_record_uses_fallback():
    assert build.process_station(_series(start="2019-06-01"), _no_steps()).fallback_fit


def test_exclusion_window_left_out_of_fit():
    a = _series()
    b = a.copy()
    b.loc["2019-07-01":"2019-12-31", "e"] += 0.3
    window = [(pd.Timestamp("2019-07-01"), pd.Timestamp("2020-06-30"))]
    ra = build.process_station(a, _no_steps(), window).r
    rb = build.process_station(b, _no_steps(), window).r
    np.testing.assert_allclose(ra.loc[:"2019-06-30"].to_numpy(), rb.loc[:"2019-06-30"].to_numpy(), atol=1e-9)


def test_unstable_site_fails_qc_but_earthquake_offset_does_not():
    wobbly = _series()
    t = wobbly["decyear"].to_numpy()
    wobbly["e"] += 0.200 * np.sin(2 * np.pi * (t - 2008) / 3.1)
    assert build.process_station(wobbly, _no_steps()).qc_scale_mm > build.QC_MAX_HORIZONTAL_SCALE_MM
    quake = _series()
    quake.loc["2014-08-24":, "n"] += 0.300
    res = build.process_station(quake, _steps([("X", "2014-08-24", "quake")]))
    assert res.qc_scale_mm < 3.0
    assert res.r.loc["2015", "n"].median() - res.r.loc["2013", "n"].median() > 250   # still kept
    assert len(res.quake_steps) == 1 and res.quake_steps[0][1][1] == pytest.approx(300, abs=1)


def test_subsidence_fails_vertical_qc():
    s = _series()
    t = s["decyear"].to_numpy()
    s["u"] -= 0.150 * np.clip(t - 2012, 0, None) ** 1.5
    res = build.process_station(s, _no_steps())
    assert res.qc_scale_mm < build.QC_MAX_HORIZONTAL_SCALE_MM < res.qc_scale_u_mm


# --------------------------------------------------------------------------- network


def _network(n=30, seed=1, lat0=35.0, lon0=-118.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sta": [f"S{i:03d}" for i in range(n)], "lat": lat0 + rng.uniform(0, 2, n),
                         "lon": lon0 + rng.uniform(0, 2, n)})


def test_twins_keep_the_longest_record():
    d = pairwise_km(np.array([35.0, 35.0001, 35.5, 36.0]), np.array([-118.0, -118.0001, -118.0, -118.0]))
    sta = np.array(["AAA1", "AAA2", "BBBB", "CCCC"])
    assert build.twin_drops(d, sta, np.array([100, 500, 300, 300])) == {0: 1}
    assert build.twin_drops(d, sta, np.array([500, 500, 300, 300])) == {1: 0}   # tie: by id


def test_heldout_selection_is_deterministic_and_spread():
    st = _network(n=200)
    d = pairwise_km(st.lat.to_numpy(), st.lon.to_numpy())
    a, b = build.farthest_point_sample(d, 20, 0), build.farthest_point_sample(d, 20, 0)
    np.testing.assert_array_equal(a, b)
    rand = np.random.default_rng(5).choice(len(st), 20, replace=False)
    assert (d[np.ix_(a, a)] + np.eye(20) * 1e9).min() > (d[np.ix_(rand, rand)] + np.eye(20) * 1e9).min()


def test_build_california_end_to_end(tmp_path):
    st = _network(n=12)
    st.loc[0, ["lat", "lon"]] = (35.77, -117.60)                               # inside the Ridgecrest zone
    st.loc[1, ["lat", "lon"]] = (40.0, -122.0)
    cube, summary = build.build_cube(
        st, lambda sta: _series(start="2020-03-01" if sta == "S005" else "2008-01-01", seed=int(sta[1:])),
        _no_steps(), build.REGION_CONFIG["california"])
    S, T = len(cube.sta), len(cube.days)
    assert cube.r.shape == (S, T, 3) and cube.avail.sum() == summary["station_days"]
    assert cube.split_day[cube.days <= np.datetime64("2019-12-31")].max() == 0
    assert set(cube.split_day[cube.days > np.datetime64("2021-12-31")]) == {2}
    i_late = int(np.flatnonzero(cube.sta == "S005")[0])
    assert cube.fallback_fit[i_late] and cube.split_sta[i_late] == 1               # forced out
    rc = cube.exclude.any(axis=1)
    assert rc[np.flatnonzero(cube.sta == "S000")[0]] and not rc[np.flatnonzero(cube.sta == "S001")[0]]
    cube.save(tmp_path / "c.npz")
    back = Cube.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(back.r, cube.r)
    np.testing.assert_array_equal(back.fallback_fit, cube.fallback_fit)
    assert back.days.dtype == np.dtype("datetime64[D]")


def test_build_turkiye_holds_nothing_out_and_marks_kahramanmaras():
    st = _network(n=8, lat0=37.0, lon0=36.0)
    cube, summary = build.build_cube(st, lambda sta: _series(seed=int(sta[1:])), _no_steps(),
                                     build.REGION_CONFIG["turkiye"])
    assert summary["heldout_stations"] == 0 and (cube.split_sta == 0).all()
    window = (cube.days >= np.datetime64("2023-01-07")) & (cube.days <= np.datetime64("2024-02-05"))
    assert cube.exclude[:, window].all() and not cube.exclude[:, ~window].any()
    split = pd.Series(cube.split_day, index=pd.DatetimeIndex(cube.days))
    assert set(split[:"2012"]) == {0} and set(split["2013":"2015"]) == {1} and set(split["2016":]) == {2}
    fit2019, _ = build.build_cube(st, lambda sta: _series(seed=int(sta[1:])), _no_steps(),
                                  build.REGION_CONFIG["turkiye-fit2019"])
    assert not np.array_equal(fit2019.r, cube.r)                                   # another fit window
    assert (fit2019.split_day[cube.days <= np.datetime64("2019-12-31")] == 0).all()
