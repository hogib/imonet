"""Event library for experiment B (docs/00-design.md §6.2): fault surfaces,
the displacement slip on them causes, and realistic transients placed on
real faults. Ported from gnssdl (gnssdl.bench.events, same author); the
templates are in imonet/event_templates/*.toml with their sources.

Fault geometry comes from the SCEC Community Fault Model 7.0 (Plesch et al.
2007; Zenodo 10.5281/zenodo.13685611), whose triangulated surfaces (GOCAD
TSurf) are in UTM zone 11, NAD27, metres, elevation positive up. Station
displacements are computed with triangular dislocations in a homogeneous
elastic half-space (Nikkhoo & Walter 2015, via `cutde`), Poisson's ratio
0.25.

Geometry is handled in a local transverse Mercator frame centred on each
event, so no part of the state is far from its projection's meridian.
The half-space surface is z = 0: fault vertices above sea level are clamped
to it, a simplification of the topography.

Slip convention (Aki & Richards): for every triangle the normal is taken to
point up, into the hanging wall; strike is ẑ × n̂ (the fault dips to the right
of strike) and up-dip is n̂ × strike. Rake 0 is left-lateral, 180
right-lateral, 90 reverse, −90 normal; slip is the motion of the hanging
wall relative to the footwall. For a vertical triangle either normal gives
the same physical motion.
"""

from __future__ import annotations

import struct
import tomllib
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

CFM_CRS = "EPSG:26711"            # NAD27 / UTM zone 11N, as stated in the CFM trace files
POISSON = 0.25


@dataclass
class Fault:
    """A triangulated fault surface in geographic coordinates."""
    name: str
    lon: np.ndarray               # V vertex longitudes
    lat: np.ndarray               # V vertex latitudes
    elev: np.ndarray              # V vertex elevations, m (positive up)
    tri: np.ndarray               # M×3 vertex indices


def read_tsurf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V×3: x, y, z) and triangles (M×3, 0-based) of a GOCAD TSurf
    file. Several TFACE blocks are merged into one surface."""
    ids, xyz, tris = {}, [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("VRTX", "PVRTX"):
            ids[int(parts[1])] = len(xyz)
            xyz.append([float(v) for v in parts[2:5]])
        elif parts[0] == "ATOM":                      # a vertex that repeats an earlier one
            ids[int(parts[1])] = ids[int(parts[2])]
        elif parts[0] == "TRGL":
            tris.append([ids[int(v)] for v in parts[1:4]])
    if not tris:
        raise ValueError(f"{path}: no triangles")
    return np.asarray(xyz, dtype=np.float64), np.asarray(tris, dtype=np.int64)


@lru_cache(maxsize=None)
def _to_lonlat():
    from pyproj import Transformer
    return Transformer.from_crs(CFM_CRS, "EPSG:4326", always_xy=True)


MIN_TRIANGLE_AREA_M2 = 1.0


def load_fault(path: Path) -> Fault:
    """Read one CFM surface and convert its vertices to longitude/latitude.
    Degenerate (zero-area) triangles, which some surfaces contain, are
    dropped: they have no normal and carry no slip."""
    xyz, tri = read_tsurf(path)
    c = xyz[tri]
    area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
    tri = tri[area >= MIN_TRIANGLE_AREA_M2]
    lon, lat = _to_lonlat().transform(xyz[:, 0], xyz[:, 1])
    name = Path(path).stem
    return Fault(name=name, lon=np.asarray(lon), lat=np.asarray(lat), elev=xyz[:, 2], tri=tri)


class LocalFrame:
    """Transverse Mercator centred on (lon0, lat0): x east, y north, metres."""

    def __init__(self, lon0: float, lat0: float):
        from pyproj import Transformer
        proj = f"+proj=tmerc +lon_0={lon0} +lat_0={lat0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    def xy(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)


def triangle_frames(tri_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per triangle (M×3×3 corners): unit normal pointing up (into the
    hanging wall), strike and up-dip vectors (each M×3), and the corner
    order that makes the normal point up (M×3×3)."""
    n = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    flip = n[:, 2] < 0
    tri_xyz = tri_xyz.copy()
    tri_xyz[flip] = tri_xyz[flip][:, [0, 2, 1]]
    n[flip] *= -1
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ez = np.array([0.0, 0.0, 1.0])
    strike = np.cross(ez, n)
    flat = np.linalg.norm(strike, axis=1) < 1e-9                 # horizontal triangle: any strike
    strike[flat] = [1.0, 0.0, 0.0]
    strike /= np.linalg.norm(strike, axis=1, keepdims=True)
    updip = np.cross(n, strike)
    return n, strike, updip, tri_xyz


def displacement(obs_xyz: np.ndarray, tri_xyz: np.ndarray, slip_m: np.ndarray,
                 rake_deg: float | np.ndarray) -> np.ndarray:
    """East, north, up displacement (m) at `obs_xyz` (N×3, local metres, z ≤ 0)
    from slip `slip_m` (M, metres) with rake `rake_deg` on triangles
    `tri_xyz` (M×3×3, local metres, z ≤ 0)."""
    import cutde.halfspace as hs
    _, strike, updip, tri_xyz = triangle_frames(np.asarray(tri_xyz, dtype=np.float64))
    rake = np.deg2rad(np.broadcast_to(rake_deg, (len(tri_xyz),)))
    slip_vec = (np.cos(rake)[:, None] * strike + np.sin(rake)[:, None] * updip) * np.asarray(slip_m)[:, None]
    # cutde wants slip in each triangle's own (strike, dip, tensile) frame,
    # built from the corner order exactly as `triangle_frames` builds it
    comp = np.stack([np.einsum("mk,mk->m", slip_vec, strike),
                     np.einsum("mk,mk->m", slip_vec, updip),        # cutde's dip axis points up-dip (tested)
                     np.zeros(len(tri_xyz))], axis=1)
    mat = hs.disp_matrix(np.asarray(obs_xyz, dtype=np.float64), tri_xyz, POISSON)   # N×3×M×3
    return np.einsum("nimj,mj->ni", mat, comp)


def fault_triangles(fault: Fault, frame: LocalFrame, select: np.ndarray | None = None) -> np.ndarray:
    """The fault's triangles (or those in `select`) as M×3×3 local
    coordinates, with vertices above sea level clamped to the surface."""
    x, y = frame.xy(fault.lon, fault.lat)
    z = np.minimum(fault.elev, 0.0)
    v = np.stack([x, y, z], axis=1)
    tri = fault.tri if select is None else fault.tri[select]
    return v[tri]


def station_displacement(fault: Fault, slip_m: np.ndarray, rake_deg, sta_lon, sta_lat,
                         select: np.ndarray | None = None) -> np.ndarray:
    """E, N, U displacement (m) at stations (lon, lat) from slip on
    `fault` (one value per selected triangle), in a frame centred on the
    slipping triangles."""
    tri_idx = np.arange(len(fault.tri)) if select is None else np.flatnonzero(select) if \
        np.asarray(select).dtype == bool else np.asarray(select)
    used = np.unique(fault.tri[tri_idx])
    frame = LocalFrame(float(np.mean(fault.lon[used])), float(np.mean(fault.lat[used])))
    tris = fault_triangles(fault, frame, tri_idx)
    sx, sy = frame.xy(sta_lon, sta_lat)
    obs = np.stack([sx, sy, np.zeros_like(sx)], axis=1)
    return displacement(obs, tris, slip_m, rake_deg)


# ---------------------------------------------------------------------------
# Fault catalogue
# ---------------------------------------------------------------------------

CFM_DIR = Path("data/cfm/extracted/obj/preferred")
# CFM slip-sense codes (CFM 7.0 README) -> rake in degrees
RAKE = {"rlss": 180.0, "llss": 0.0, "r": 90.0, "n": -90.0,
        "orlr": 135.0, "ollr": 45.0, "orln": -135.0, "olln": -45.0}


@dataclass
class FaultInfo:
    """One CFM fault object, its metadata and where its surface is."""
    name: str
    slip_sense: str
    dip: float                    # weighted average dip, degrees
    strike: float                 # weighted average strike, degrees
    exposure: str                 # "surface" or "blind"
    path: Path


def _read_dbf(path: Path) -> list[dict[str, str]]:
    """Records of a dBASE table (the attribute table of a shapefile)."""
    b = Path(path).read_bytes()
    n, header_len, record_len = struct.unpack("<IHH", b[4:12])
    fields = [(b[i:i + 11].split(b"\0")[0].decode(), b[i + 16]) for i in range(32, header_len - 1, 32)]
    out, pos = [], header_len
    for _ in range(n):
        rec, pos, o, row = b[pos + 1:pos + record_len], pos + record_len, 0, {}
        for name, length in fields:
            row[name] = rec[o:o + length].decode("latin1").strip()
            o += length
        out.append(row)
    return out


def fault_catalogue(cfm_dir: Path = CFM_DIR, resolution: str = "1000m") -> list[FaultInfo]:
    """Every preferred CFM fault that has a surface at `resolution` and a
    slip sense we can turn into a rake."""
    rows: dict[str, dict] = {}
    for kind in ("traces", "blind"):
        for r in _read_dbf(cfm_dir / "traces" / "shp" / f"CFM7.0_{kind}.dbf"):
            rows.setdefault(r["ObjectName"], r)          # multi-segment traces repeat a fault
    out = []
    for name, r in sorted(rows.items()):
        path = cfm_dir / resolution / f"{name}_{resolution}.ts"
        if r["SlipSense"] in RAKE and path.exists():
            out.append(FaultInfo(name, r["SlipSense"], float(r["wAvgDip"]), float(r["wAvgStrike"]),
                                 r["Exposure"], path))
    return out


@lru_cache(maxsize=64)
def _cached_fault(path: str) -> Fault:
    return load_fault(Path(path))


def fault_surface(info: FaultInfo) -> Fault:
    """The (cached) triangulated surface of a catalogue fault."""
    return _cached_fault(str(info.path))


# ---------------------------------------------------------------------------
# Templates and placement
# ---------------------------------------------------------------------------

@dataclass
class EventTemplate:
    """A kind of transient, from a published event (values in data/events/).

    The patch is `length_km` along strike and from `top_km` to `bottom_km`
    depth, with peak slip `slip_m` tapered to zero over the outer `taper`
    fraction of the patch on every side. Time function `time`:
    - "propagating": a slip front starts at a random point of the patch and
      runs along strike in both directions at `speed_km_day`; each part of
      the patch then slips over `rise_days` (raised cosine);
    - "log": every part slips as log(1 + t/τ) / log(1 + T/τ) up to T =
      `duration_days`, τ = `tau_days`;
    - "exp": (1 − exp(−t/τ)) / (1 − exp(−T/τ)) up to T;
    - "ramp": a raised-cosine ramp over `duration_days`.
    Slip is permanent: after the event the patch stays slipped.

    Any numeric field in RANGED may instead be a (low, high) pair; `draw`
    samples it per event, log-uniformly when high/low > 3 (so the low end of
    a wide range is not swamped), uniformly otherwise."""
    name: str
    slip_senses: tuple[str, ...]
    length_km: float
    top_km: float
    bottom_km: float
    slip_m: float
    time: str
    duration_days: float
    min_dip: float = 0.0
    max_dip: float = 90.0
    tau_days: float | None = None
    speed_km_day: float | None = None
    rise_days: float | None = None
    taper: float = 0.2
    reference: str = ""

    def draw(self, rng: np.random.Generator) -> "EventTemplate":
        """A copy with every ranged value replaced by a sample."""
        out = {}
        for k in RANGED:
            v = getattr(self, k)
            if isinstance(v, (tuple, list)):
                lo, hi = float(v[0]), float(v[1])
                if lo > 0 and hi / lo > 3:
                    out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                else:
                    out[k] = float(rng.uniform(lo, hi))
        return replace(self, **out)

    def drawn_values(self) -> dict[str, float]:
        """The numeric parameters of a drawn template, for result tables."""
        return {k: getattr(self, k) for k in RANGED if getattr(self, k) is not None}


RANGED = ("length_km", "top_km", "bottom_km", "slip_m", "duration_days", "tau_days",
          "speed_km_day", "rise_days")


@dataclass
class PlacedEvent:
    """A template placed on a fault: per-triangle slip and timing, and the
    station Green's functions (displacement per metre of slip)."""
    template: EventTemplate
    fault: str
    tri: np.ndarray               # M triangle indices on the fault
    slip_m: np.ndarray            # M final slip
    rake: float
    onset_days: np.ndarray        # M time the slip front reaches each triangle
    green: np.ndarray             # N×3×M, m of station motion per m of slip
    centre: tuple[float, float]   # lon, lat of the patch centre
    meta: dict = field(default_factory=dict)

    def fraction(self, days: np.ndarray) -> np.ndarray:
        """M×T fraction of each triangle's final slip reached at `days`
        after the event starts."""
        t = np.asarray(days, dtype=float)[None, :] - self.onset_days[:, None]
        tp = self.template
        if tp.time == "propagating":
            x = np.clip(t / tp.rise_days, 0.0, 1.0)
            return 0.5 - 0.5 * np.cos(np.pi * x)
        T = tp.duration_days
        tt = np.clip(t, 0.0, T)
        if tp.time == "log":
            return np.log1p(tt / tp.tau_days) / np.log1p(T / tp.tau_days)
        if tp.time == "exp":
            return -np.expm1(-tt / tp.tau_days) / -np.expm1(-T / tp.tau_days)
        if tp.time == "ramp":
            return 0.5 - 0.5 * np.cos(np.pi * tt / T)
        raise ValueError(f"unknown time function {tp.time}")

    def displacement_mm(self, days: np.ndarray) -> np.ndarray:
        """N×T×3 station displacement in mm, `days` after the start."""
        slip = self.slip_m[:, None] * self.fraction(days)                 # M×T
        return np.einsum("nim,mt->nti", self.green, slip) * 1000.0

    def final_mm(self) -> np.ndarray:
        """N×3 displacement in mm once the event is over."""
        return np.einsum("nim,m->ni", self.green, self.slip_m) * 1000.0

    @property
    def duration_days(self) -> float:
        tp = self.template
        if tp.time == "propagating":
            return float(self.onset_days.max() + tp.rise_days)
        return float(tp.duration_days)


def _taper(u: np.ndarray, frac: float) -> np.ndarray:
    """1 in the middle of [0, 1], falling to 0 at both ends over `frac` by a
    raised cosine."""
    if frac <= 0:
        return np.ones_like(u)
    edge = np.clip(np.minimum(u, 1 - u) / frac, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * edge)


def candidate_faults(template: EventTemplate, catalogue: list[FaultInfo]) -> list[FaultInfo]:
    return [f for f in catalogue
            if f.slip_sense in template.slip_senses and template.min_dip <= f.dip <= template.max_dip]


def place_event(template: EventTemplate, fault_info: FaultInfo | Fault, rng: np.random.Generator,
                sta_lon: np.ndarray, sta_lat: np.ndarray) -> PlacedEvent | None:
    """Put the template's patch at a random along-strike position on the
    fault, within its depth range. Returns None if the fault is too short
    or has no surface in that depth range."""
    if isinstance(fault_info, Fault):
        fault, sense, name = fault_info, None, fault_info.name
    else:
        fault, sense, name = _cached_fault(str(fault_info.path)), fault_info.slip_sense, fault_info.name
    rake = RAKE[sense] if sense else RAKE[template.slip_senses[0]]
    frame = LocalFrame(float(np.mean(fault.lon)), float(np.mean(fault.lat)))
    corners = fault_triangles(fault, frame)                               # M×3×3
    centroid = corners.mean(axis=1)
    depth_km = -centroid[:, 2] / 1000.0
    # along-strike coordinate: position along the fault's main horizontal axis
    xy = centroid[:, :2]
    axis = np.linalg.svd(xy - xy.mean(0), full_matrices=False)[2][0]
    along_km = (xy - xy.mean(0)) @ axis / 1000.0
    in_depth = (depth_km >= template.top_km) & (depth_km <= template.bottom_km)
    if not in_depth.any():
        return None
    lo, hi = along_km[in_depth].min(), along_km[in_depth].max()
    half = template.length_km / 2
    if hi - lo < template.length_km:
        return None
    s0 = rng.uniform(lo + half, hi - half)
    sel = in_depth & (np.abs(along_km - s0) <= half)
    if sel.sum() < 2:
        return None
    u_along = (along_km[sel] - (s0 - half)) / template.length_km
    u_depth = (depth_km[sel] - template.top_km) / max(template.bottom_km - template.top_km, 1e-9)
    # a patch reaching the surface is not tapered at the top (shallow creep breaks the surface)
    depth_taper = _taper(u_depth, template.taper) if template.top_km > 0.05 else \
        _taper(0.5 + 0.5 * u_depth, template.taper)
    slip = template.slip_m * _taper(u_along, template.taper) * depth_taper
    if template.time == "propagating":
        start = rng.uniform(s0 - half, s0 + half)
        onset = np.abs(along_km[sel] - start) / template.speed_km_day
    else:
        onset = np.zeros(int(sel.sum()))
    tri_idx = np.flatnonzero(sel)
    used = np.unique(fault.tri[tri_idx])
    lon0, lat0 = float(np.mean(fault.lon[used])), float(np.mean(fault.lat[used]))
    ev_frame = LocalFrame(lon0, lat0)
    tris = fault_triangles(fault, ev_frame, tri_idx)
    sx, sy = ev_frame.xy(sta_lon, sta_lat)
    obs = np.stack([sx, sy, np.zeros_like(sx)], axis=1)
    # Green's functions per triangle: unit slip with the event's rake
    import cutde.halfspace as hs
    _, strike, updip, tris = triangle_frames(tris)
    r = np.deg2rad(rake)
    comp = np.stack([np.full(len(tris), np.cos(r)), np.full(len(tris), np.sin(r)), np.zeros(len(tris))], 1)
    mat = hs.disp_matrix(obs, tris, POISSON)                              # N×3×M×3
    green = np.einsum("nimj,mj->nim", mat, comp)
    return PlacedEvent(template, name, tri_idx, slip, rake, onset, green, (lon0, lat0),
                       meta={"along_km": float(s0), "patch_triangles": int(sel.sum()), "fault": fault})


def random_event(template: EventTemplate, catalogue: list[FaultInfo], rng: np.random.Generator,
                 sta_lon: np.ndarray, sta_lat: np.ndarray, tries: int = 50) -> PlacedEvent:
    """Place the template on a random suitable fault (weighted by fault
    area would favour big faults; uniform over faults keeps variety)."""
    cands = candidate_faults(template, catalogue)
    if not cands:
        raise ValueError(f"{template.name}: no CFM fault matches {template.slip_senses}, "
                         f"dip {template.min_dip}-{template.max_dip}")
    for _ in range(tries):
        ev = place_event(template.draw(rng), cands[rng.integers(len(cands))], rng, sta_lon, sta_lat)
        if ev is not None:
            return ev
    raise RuntimeError(f"{template.name}: no fault long enough after {tries} tries")


EVENT_DIR = Path(__file__).parent / "event_templates"


def load_templates(directory: Path = EVENT_DIR) -> list[EventTemplate]:
    """Every *.toml template in `directory`. Keys are EventTemplate's fields;
    `slip_senses` is a list of CFM codes."""
    names = {f.name for f in fields(EventTemplate)}
    out = []
    for path in sorted(Path(directory).glob("*.toml")):
        raw = tomllib.loads(path.read_text())
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        raw["slip_senses"] = tuple(raw["slip_senses"])
        for k in RANGED:
            if isinstance(raw.get(k), list):
                if len(raw[k]) != 2 or raw[k][0] > raw[k][1]:
                    raise ValueError(f"{path}: {k} must be a number or [low, high]")
                raw[k] = tuple(raw[k])
        out.append(EventTemplate(**raw))
    return out
