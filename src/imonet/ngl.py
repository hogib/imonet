"""Nevada Geodetic Laboratory (NGL) daily GNSS series: download and parse.

Ported from gnssdl (gnssdl.ngl, same author), so that imonet fetches and
builds its own data. NGL processes about 20,000 stations worldwide with one
strategy (GipsyX PPP, IGS20 frame) and publishes three kinds of file:

- `DataHoldings.txt`: one row per station, its position and the span of
  its solutions.
- `steps.txt`: the days a station's series may be offset: equipment changes
  (type 1) and earthquakes (type 2, with the event's influence radius, the
  station's distance from it, the magnitude and the event id).
- `{STA}.tenv3`: the daily east, north, up positions of one station.

`fetch_cfm` also fetches the SCEC Community Fault Model, which experiment B
places its events on.

`fetch_series` downloads with a small worker pool (NGL is a university
server, not a CDN), caches each file, and writes atomically, so that an
interrupted run never leaves a truncated file that looks complete.
"""

from __future__ import annotations

import io
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from imonet.progress import progress_line

BASE = "https://geodesy.unr.edu"
HOLDINGS_URL = f"{BASE}/NGLStationPages/DataHoldings.txt"
STEPS_URL = f"{BASE}/NGLStationPages/steps.txt"
TENV3_URL = f"{BASE}/gps_timeseries/IGS20/tenv3/IGS20/{{sta}}.tenv3"
USER_AGENT = "imonet/0.1 (research; daily tenv3 fetcher)"
MJD_EPOCH = pd.Timestamp("1858-11-17")

DATA = Path("data")
HOLDINGS = DATA / "meta" / "DataHoldings.txt"
STEPS = DATA / "meta" / "steps.txt"
TENV3 = DATA / "tenv3"
MANIFESTS = DATA / "manifests"
WORKERS = 4

_STATION_ID = re.compile(r"[A-Z0-9]{4}")


@dataclass(frozen=True)
class Region:
    """Stations inside a latitude/longitude box with at least `min_days`
    daily solutions."""
    south: float
    north: float
    west: float
    east: float
    min_days: int


REGIONS = {
    # gnssdl's California selection (fetch --bbox 32.4 42.1 -124.6 -114.0 --min-days 1095)
    "california": Region(32.4, 42.1, -124.6, -114.0, 1095),
    # Türkiye and its surroundings (Greek islands, Bulgaria, Georgia, Armenia,
    # northern Iraq); at least a year of data, so that short records still
    # serve as neighbours
    "turkiye": Region(35.8, 42.2, 25.6, 44.9, 365),
}
# Regions sharing another's stations (and manifest), built with other settings.
REGIONS["turkiye-fit2019"] = REGIONS["turkiye"]
STATIONS_OF = {"turkiye-fit2019": "turkiye"}


def is_station_id(s: str) -> bool:
    """NGL ids are four upper-case alphanumerics; anything else would be
    pasted into a URL path."""
    return bool(_STATION_ID.fullmatch(s))


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #


def download(url: str, dest: Path, *, refresh: bool = False, retries: int = 3, timeout: float = 300) -> Path:
    """Fetch `url` to `dest`, unless cached. Writes to a .part file and
    renames it when complete."""
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
                while chunk := resp.read(1 << 16):
                    fh.write(chunk)
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


# SCEC Community Fault Model 7.0 (Plesch et al. 2024; Zenodo 10.5281/zenodo.13685611),
# the fault surfaces of experiment B's planted events (imonet.events)
CFM_URL = "https://zenodo.org/api/records/13685611/files/CFM7.0_release_2024.zip/content"
CFM_MD5 = "861ce170dec50e681b9b164d7cc0d1e5"


def fetch_cfm(data: Path = DATA, *, refresh: bool = False) -> Path:
    """Download the fault model to data/cfm/download/, check its MD5 against
    Zenodo's, and extract it to data/cfm/extracted/."""
    import hashlib
    import zipfile
    zpath = download(CFM_URL, data / "cfm" / "download" / "CFM7.0_release_2024.zip", refresh=refresh)
    md5 = hashlib.md5()
    with open(zpath, "rb") as fh:
        while chunk := fh.read(1 << 20):
            md5.update(chunk)
    if md5.hexdigest() != CFM_MD5:
        raise ValueError(f"{zpath}: MD5 {md5.hexdigest()}, Zenodo lists {CFM_MD5}; delete it and fetch again")
    out = (data / "cfm" / "extracted").resolve()
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():                     # never write outside the target folder
            if not (out / name).resolve().is_relative_to(out):
                raise ValueError(f"{zpath}: unsafe path {name!r}")
        z.extractall(out)
    return out


def fetch_metadata(data: Path = DATA, *, refresh: bool = False) -> tuple[Path, Path]:
    """The station table and the step catalogue, in data/meta/."""
    meta = data / "meta"
    return (download(HOLDINGS_URL, meta / "DataHoldings.txt", refresh=refresh),
            download(STEPS_URL, meta / "steps.txt", refresh=refresh))


@dataclass
class FetchResult:
    station: str
    path: Path | None
    error: str | None = None


def fetch_series(stations: list[str], data: Path = DATA, *, refresh: bool = False, workers: int = WORKERS,
                 progress: Callable[[str], None] | None = None) -> list[FetchResult]:
    """Download the daily series of `stations` into data/tenv3/. A failure
    is reported per station and never stops the batch."""
    bad = [s for s in stations if not is_station_id(s)]
    if bad:
        raise ValueError(f"not NGL station ids: {bad}")
    out = data / "tenv3"

    def one(sta: str) -> FetchResult:
        try:
            return FetchResult(sta, download(TENV3_URL.format(sta=sta), out / f"{sta}.tenv3", refresh=refresh))
        except Exception as exc:
            return FetchResult(sta, None, f"{type(exc).__name__}: {exc}")

    results, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s) for s in stations]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r.station] = r
            if progress:
                status = "ok" if r.error is None else f"FAILED ({r.error})"
                progress(progress_line(i, len(stations), t0, f"stations, {r.station} {status}"))
    return [results[s] for s in stations]


def manifest(selection: pd.DataFrame, results: list[FetchResult]) -> pd.DataFrame:
    """The selected stations with their fetch status, as gnssdl's manifests:
    sta, ok, error, lat, lon, height, start, end, n_sol."""
    status = pd.DataFrame({"sta": [r.station for r in results], "ok": [r.error is None for r in results],
                           "error": [r.error or "" for r in results]})
    cols = ["sta", "lat", "lon", "height", "start", "end", "n_sol"]
    out = status.merge(selection[cols], on="sta", how="left")
    for c in ("start", "end"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    return out


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def read_holdings(path: Path | str = HOLDINGS) -> pd.DataFrame:
    """Station table: sta, lat, lon (−180…180), height, start, end, n_sol."""
    rows = []
    with open(path) as fh:
        next(fh)                                         # header
        for line in fh:
            p = line.split()
            if len(p) < 11:
                continue
            rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), p[7], p[8], int(p[10])))
    df = pd.DataFrame(rows, columns=["sta", "lat", "lon", "height", "start", "end", "n_sol"])
    df["lon"] = (df["lon"] + 180.0) % 360.0 - 180.0
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def select(holdings: pd.DataFrame, region: Region) -> pd.DataFrame:
    """The stations of `region`, sorted by id."""
    m = (holdings["lat"].between(region.south, region.north) & holdings["lon"].between(region.west, region.east)
         & (holdings["n_sol"] >= region.min_days))
    return holdings[m].sort_values("sta").reset_index(drop=True)


def read_steps(path: Path | str = STEPS) -> pd.DataFrame:
    """One row per listed step. Columns: sta, date, kind ('equipment' or
    'quake'), info (the change description, or the event id), and for
    quakes radius_km, dist_km, mag."""
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            if p[2] == "1":
                rows.append((p[0], p[1], "equipment", " ".join(p[3:]), None, None, None))
            elif p[2] == "2" and len(p) >= 7:
                rows.append((p[0], p[1], "quake", p[6], float(p[3]), float(p[4]), float(p[5])))
    df = pd.DataFrame(rows, columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    df["date"] = pd.to_datetime(df["date"], format="%y%b%d")
    return df


TENV3_COLS = [
    "sta", "yymmmdd", "decyear", "mjd", "gps_week", "gps_dow", "reflon",
    "e0", "east", "n0", "north", "u0", "up", "ant",
    "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "lat", "lon", "height",
]


def read_tenv3(path: Path | str) -> pd.DataFrame:
    """Daily positions indexed by date. e, n, u are full metres (integer part
    plus fraction) in a local frame; subtract a fit before use."""
    text = Path(path).read_text()
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=1, header=None, names=TENV3_COLS)
    df["date"] = MJD_EPOCH + pd.to_timedelta(df["mjd"], unit="D")
    df["e"] = df["e0"] + df["east"]
    df["n"] = df["n0"] + df["north"]
    df["u"] = df["u0"] + df["up"]
    return df.set_index("date")[
        ["decyear", "e", "n", "u", "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "ant"]
    ]
