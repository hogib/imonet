"""The data cube (daily residual series of a station network) and geometry.

The California cube is gnssdl's `data/cube/california.npz`, used as it is
(gnssdl docs/00-benchmark-contract.md §1). Türkiye cubes made by
`imonet build` hold the same arrays. See docs/00-design.md §3.

gnssdl's precomputed neighbour graphs (`nbr_*`, `radii`) are not loaded:
imonet selects neighbours itself, for any coordinate and any thinned network
(`imonet.network`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

EARTH_RADIUS_KM = 6371.0
CALIFORNIA = Path("data/cube/california.npz")

# split_day
TRAIN, VALIDATION, TEST = 0, 1, 2
# split_sta
NORMAL, HELD_OUT = 0, 1


@dataclass
class Cube:
    r: np.ndarray            # S×T×3 float32, mm (E, N, U), NaN where missing
    sigma: np.ndarray        # S×T×3 float32, mm, formal 1σ, NaN where missing
    avail: np.ndarray        # S×T bool, observed and passed screening
    sta: np.ndarray          # S str, 4-character ids
    lat: np.ndarray          # S float64, degrees
    lon: np.ndarray          # S float64, degrees, −180..180
    days: np.ndarray         # T datetime64[D]
    split_day: np.ndarray    # T int8: TRAIN, VALIDATION, TEST
    split_sta: np.ndarray    # S int8: NORMAL, HELD_OUT
    exclude: np.ndarray      # S×T bool, never used for training
    # Earthquake offsets estimated by the trajectory fit and kept in r, one
    # row per offset (station index, day index, E/N/U amplitude in mm).
    qstep_sta: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    qstep_day: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    qstep_amp: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))

    @classmethod
    def load(cls, path: Path | str = CALIFORNIA) -> "Cube":
        with np.load(path, allow_pickle=False) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls) if f.name in z.files})

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{f.name: getattr(self, f.name) for f in fields(self)})
        tmp.replace(path)

    @property
    def n_sta(self) -> int:
        return len(self.sta)

    @property
    def n_days(self) -> int:
        return len(self.days)

    def stations(self, split: int) -> np.ndarray:
        """Indices of the stations in a station split (NORMAL or HELD_OUT)."""
        return np.flatnonzero(self.split_sta == split)

    def trainable(self) -> np.ndarray:
        """S×T cells a model may train on: normal stations, training days,
        available, not excluded (docs/00-design.md §3.1)."""
        ok = self.avail & ~self.exclude
        ok &= (self.split_day == TRAIN)[None, :]
        ok &= (self.split_sta == NORMAL)[:, None]
        return ok


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def distance_azimuth(lat0, lon0, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance (km) and initial bearing (radians clockwise from
    north) from (lat0, lon0) to (lat, lon). Degrees in; inputs broadcast."""
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p, l = np.radians(lat), np.radians(lon)
    dp, dl = p - p0, l - l0
    a = np.sin(dp / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    az = np.arctan2(np.sin(dl) * np.cos(p), np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl))
    return d, az


def offsets_km(lat0, lon0, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """East and north offset (km) of (lat, lon) from (lat0, lon0), in the
    azimuthal equidistant projection centred on (lat0, lon0): the offset's
    length is the great-circle distance and its direction the bearing."""
    d, az = distance_azimuth(lat0, lon0, lat, lon)
    return d * np.sin(az), d * np.cos(az)


def pairwise_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """S×S great-circle distances (km) between stations."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    return distance_azimuth(lat[:, None], lon[:, None], lat[None, :], lon[None, :])[0]
