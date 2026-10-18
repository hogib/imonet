"""Country of a station, from Natural Earth's 1:10m country polygons.

Natural Earth (public domain; version 5.1.2, pinned) gives the polygons;
`imonet fetch borders` downloads them. A station inside a polygon gets that
country's ISO 3166-1 alpha-3 code. A coastal station that falls just outside
every polygon, because the coastline is generalised, gets the country whose
boundary is nearest, if it lies within NEAREST_KM; otherwise "".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

from imonet.cube import distance_azimuth

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
       "ne_10m_admin_0_countries.geojson")
BORDERS = Path("data/meta/ne_10m_admin_0_countries.geojson")
NEAREST_KM = 20.0                # e.g. Sivriada and Yassıada (Marmara), islets missing from the polygons


def _rings(geometry: dict) -> list[np.ndarray]:
    """Outer rings (lon, lat) of a Polygon or MultiPolygon. Holes are
    ignored: a station in a lake or an enclave is not expected."""
    if geometry["type"] == "Polygon":
        return [np.asarray(geometry["coordinates"][0])]
    return [np.asarray(p[0]) for p in geometry["coordinates"]]


def load(path: Path | str = BORDERS) -> list[tuple[str, np.ndarray]]:
    """(ISO alpha-3 code, outer ring) for every ring of every country."""
    out = []
    for f in json.loads(Path(path).read_text())["features"]:
        code = f["properties"]["ADM0_A3"]
        out += [(code, ring) for ring in _rings(f["geometry"])]
    return out


def countries(lat, lon, borders: list[tuple[str, np.ndarray]] | None = None) -> np.ndarray:
    """ISO alpha-3 code per point ("" when none is within NEAREST_KM)."""
    borders = load() if borders is None else borders
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    pts = np.column_stack([lon, lat])
    lo_lon, hi_lon, lo_lat, hi_lat = lon.min() - 1, lon.max() + 1, lat.min() - 1, lat.max() + 1
    rings = [(c, r) for c, r in borders
             if r[:, 0].max() >= lo_lon and r[:, 0].min() <= hi_lon
             and r[:, 1].max() >= lo_lat and r[:, 1].min() <= hi_lat]

    out = np.full(len(lat), "", dtype="<U3")
    for code, ring in rings:
        out[(out == "") & Polygon(ring).contains_points(pts)] = code

    todo = np.flatnonzero(out == "")
    if len(todo):
        best = np.full(len(todo), NEAREST_KM)
        for code, ring in rings:
            d = distance_azimuth(lat[todo, None], lon[todo, None], ring[None, :, 1], ring[None, :, 0])[0].min(axis=1)
            closer = d < best
            best[closer] = d[closer]
            out[todo[closer]] = code
    return out
