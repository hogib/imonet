import numpy as np
import pytest

from imonet.cube import EARTH_RADIUS_KM, distance_azimuth, offsets_km, pairwise_km
from imonet.network import nearest_km, neighbours, spacing_km, thin

KM_PER_DEG = np.pi * EARTH_RADIUS_KM / 180


def grid(n: int, step_km: float, lat0: float = 36.0, lon0: float = -118.0):
    """An n×n grid of stations about step_km apart."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lat = lat0 + i.ravel() * step_km / KM_PER_DEG
    lon = lon0 + j.ravel() * step_km / (KM_PER_DEG * np.cos(np.radians(lat0)))
    return lat, lon


def test_distance_and_bearing():
    d, az = distance_azimuth(0.0, 0.0, 0.0, 1.0)
    assert d == pytest.approx(KM_PER_DEG)
    assert az == pytest.approx(np.pi / 2)
    d, az = distance_azimuth(0.0, 0.0, -1.0, 0.0)
    assert d == pytest.approx(KM_PER_DEG)
    assert abs(az) == pytest.approx(np.pi)


def test_offsets_point_the_right_way():
    e, n = offsets_km(36.0, -118.0, 37.0, -118.0)
    assert n == pytest.approx(KM_PER_DEG) and abs(e) < 1e-9
    e, n = offsets_km(36.0, -118.0, 36.0, -117.9)
    assert e > 0 and abs(n) < 0.1           # small poleward bend of the great circle
    e, n = offsets_km(36.0, -118.0, 35.5, -118.4)
    d, _ = distance_azimuth(36.0, -118.0, 35.5, -118.4)
    assert np.hypot(e, n) == pytest.approx(d)


def test_pairwise_is_symmetric_with_zero_diagonal():
    lat, lon = grid(4, 10.0)
    d = pairwise_km(lat, lon)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)


def test_spacing_of_a_grid():
    lat, lon = grid(10, 10.0)
    d = pairwise_km(lat, lon)
    assert spacing_km(d) == pytest.approx(10.0, rel=0.01)
    assert np.isinf(nearest_km(d[:1, :1])).all()


@pytest.fixture(scope="module")
def big():
    lat, lon = grid(60, 5.0)                 # 3600 stations, 300 km square
    return pairwise_km(lat, lon)


@pytest.mark.parametrize("method,target", [("disk", 25.0), ("disk", 50.0), ("random", 25.0)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_thinning_reaches_the_target_spacing(big, method, target, seed):
    # Random thinning is not tested at 50 km: a random network that sparse
    # has about ten stations in this square, too few to reach it reliably.
    keep = thin(big, target, np.random.default_rng(seed), method)
    got = spacing_km(big, keep)
    assert got >= target
    assert got < 1.25 * target


def test_random_thinning_falls_back_to_the_sparsest_prefix():
    lat, lon = grid(3, 10.0)                 # 9 stations, 20 km square
    d = pairwise_km(lat, lon)
    keep = thin(d, 500.0, np.random.default_rng(0), "random")
    assert keep.sum() >= 2
    assert spacing_km(d, keep) == pytest.approx(
        max(spacing_km(d, k) for k in _prefixes(d, np.random.default_rng(0))))


def _prefixes(d, rng):
    order = rng.permutation(len(d))
    for m in range(2, len(d) + 1):
        k = np.zeros(len(d), dtype=bool)
        k[order[:m]] = True
        yield k


def test_disk_thinning_keeps_a_minimum_separation():
    lat, lon = grid(30, 5.0)
    d = pairwise_km(lat, lon)
    keep = thin(d, 40.0, np.random.default_rng(1), "disk")
    # Spacing ≥ target means at least half the stations are that far apart;
    # disk thinning bounds every pair from below by its separation instead.
    assert nearest_km(d, keep).min() > 20.0


def test_thinning_is_repeatable_and_a_no_op_when_already_sparse():
    lat, lon = grid(20, 5.0)
    d = pairwise_km(lat, lon)
    a = thin(d, 30.0, np.random.default_rng(7))
    b = thin(d, 30.0, np.random.default_rng(7))
    assert (a == b).all()
    assert thin(d, 3.0, np.random.default_rng(7)).all()


def test_neighbours_sorted_bounded_and_padded():
    lat, lon = grid(6, 20.0)
    lat0, lon0 = lat[14], lon[14]
    nb = neighbours(lat0, lon0, lat, lon, k=40, min_km=0.0, max_km=50.0)
    assert 14 not in nb.idx                  # the target itself is at distance 0
    v = nb.valid
    assert np.all(np.diff(nb.dist_km[v]) >= 0)
    assert np.all((nb.dist_km[v] > 0) & (nb.dist_km[v] <= 50.0))
    assert not v[-1] and np.isnan(nb.dist_km[~v]).all()
    assert np.allclose(np.hypot(nb.east_km[v], nb.north_km[v]), nb.dist_km[v])


def test_neighbours_respect_min_distance_and_mask():
    lat, lon = grid(6, 20.0)
    keep = np.ones(len(lat), dtype=bool)
    keep[::2] = False
    nb = neighbours(lat[14], lon[14], lat, lon, keep, k=8, min_km=25.0)
    v = nb.valid
    assert (nb.dist_km[v] > 25.0).all()
    assert keep[nb.idx[v]].all()
