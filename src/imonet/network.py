"""Station networks: thinning to a target density, and neighbour selection
for any coordinate (docs/00-design.md §4.1–4.2, §6.1).

Density is measured by the median nearest-neighbour distance ("spacing").
Thinning works on a pairwise distance matrix of the candidate stations, so a
caller computes it once (`imonet.cube.pairwise_km`) and slices it.

Two thinning methods:

- "disk": greedy minimum-separation thinning. Stations are visited in random
  order and kept if no kept station lies within a separation s; s is chosen
  by bisection to reach the target spacing. The result is spread evenly,
  like a network designed on a grid.
- "random": a random subset of the stations, its size chosen by bisection.
  It keeps the clustering of the original network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imonet.cube import distance_azimuth, offsets_km

K_NEIGHBOURS = 16
MAX_NEIGHBOUR_KM = 300.0
BISECTION_STEPS = 24


def nearest_km(d: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Distance from each kept station to its nearest other kept station
    (inf for a lone station). `d` is the pairwise distance matrix of all
    candidates; `keep` a boolean mask over them (default all)."""
    if keep is not None:
        d = d[np.ix_(keep, keep)]
    if len(d) < 2:
        return np.full(len(d), np.inf)
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def spacing_km(d: np.ndarray, keep: np.ndarray | None = None) -> float:
    """Median nearest-neighbour distance of the kept stations (inf if fewer
    than two)."""
    nn = nearest_km(d, keep)
    return float(np.median(nn)) if len(nn) >= 2 else float("inf")


def _disk(d: np.ndarray, order: np.ndarray, sep_km: float) -> np.ndarray:
    keep = np.zeros(len(d), dtype=bool)
    blocked = np.zeros(len(d), dtype=bool)
    for j in order:
        if not blocked[j]:
            keep[j] = True
            blocked |= d[j] <= sep_km
    return keep


def thin(d: np.ndarray, target_km: float, rng: np.random.Generator, method: str = "disk") -> np.ndarray:
    """Boolean mask of a thinned network whose spacing is close to
    `target_km`. A target at or below the current spacing keeps everything.

    The spacing reached is the closest the bisection finds; it is not exact,
    since spacing moves in jumps as stations are added or dropped. Callers
    that need the value measure it with `spacing_km`."""
    n = len(d)
    if n < 2 or spacing_km(d) >= target_km:
        return np.ones(n, dtype=bool)
    order = rng.permutation(n)

    if method == "disk":
        # Spacing of a disk thinning is at least sep, so sep ≤ target.
        lo, hi = 0.0, float(target_km)
        best = _disk(d, order, hi)
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            keep = _disk(d, order, mid)
            if spacing_km(d, keep) >= target_km:
                hi, best = mid, keep
            else:
                lo = mid
        return best

    if method == "random":
        # Spacing is not monotone in the prefix length, so measure every
        # prefix (nearest distances updated one station at a time) and keep
        # the largest that reaches the target; if none does, the sparsest.
        nn = np.full(n, np.inf)
        spacing = np.zeros(n + 1)
        for m in range(1, n):
            j, prev = order[m], order[:m]
            dj = d[j, prev]
            nn[prev] = np.minimum(nn[prev], dj)
            nn[j] = dj.min()
            spacing[m + 1] = np.median(nn[order[: m + 1]])
        hit = np.flatnonzero(spacing >= target_km)
        best = int(hit[-1]) if len(hit) else int(np.argmax(spacing))
        keep = np.zeros(n, dtype=bool)
        keep[order[:best]] = True
        return keep

    raise ValueError(f"unknown thinning method {method!r}")


@dataclass
class Neighbours:
    """The k nearest network stations to a coordinate, nearest first.
    Rows past the stations found are padding: idx −1, the rest NaN."""
    idx: np.ndarray        # k int, index into the network arrays, −1 pads
    dist_km: np.ndarray    # k float
    east_km: np.ndarray    # k float, offset of the neighbour from the target
    north_km: np.ndarray   # k float

    @property
    def valid(self) -> np.ndarray:
        return self.idx >= 0

    @property
    def count(self) -> int:
        return int(self.valid.sum())


def neighbours(
    lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray, keep: np.ndarray | None = None,
    k: int = K_NEIGHBOURS, min_km: float = 0.0, max_km: float = MAX_NEIGHBOUR_KM,
) -> Neighbours:
    """The k nearest stations to (lat0, lon0) among those in `keep` (default
    all), with min_km < distance ≤ max_km. The strict lower bound drops a
    station at the target's own coordinate even at min_km = 0."""
    d, _ = distance_azimuth(lat0, lon0, lat, lon)
    ok = (d > min_km) & (d <= max_km)
    if keep is not None:
        ok &= keep
    cand = np.flatnonzero(ok)
    cand = cand[np.argsort(d[cand], kind="stable")][:k]

    idx = np.full(k, -1, dtype=np.int64)
    dist = np.full(k, np.nan)
    east = np.full(k, np.nan)
    north = np.full(k, np.nan)
    m = len(cand)
    idx[:m] = cand
    dist[:m] = d[cand]
    east[:m], north[:m] = offsets_km(lat0, lon0, lat[cand], lon[cand])
    return Neighbours(idx, dist, east, north)
