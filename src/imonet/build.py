"""Building a data cube from downloaded NGL series (`imonet build`).

Ported from gnssdl (gnssdl.dataset, same author), whose California cube
imonet used until it built its own; built from the same files, imonet's
California cube is identical to gnssdl's in every array imonet uses
(tests/test_build.py, docs/00-design.md §3). Per station:

1. From 2008 on, fit constant, rate, annual and semi-annual terms and steps
   to the days up to the region's fit end (2019-12-31; 2012-12-31 for
   Türkiye) and outside its training exclusions. A station with less than a year of such days is fitted on
   the start of its record instead (the first 730 days or 365 epochs,
   whichever reaches further) and flagged (`fallback_fit`); in California
   it is held out, so that its fit never touches days it is trained on.
2. Earthquake offsets listed by NGL are estimated and kept in the residual;
   equipment steps are removed, after the fit window by a local jump.
3. Days more than 5 robust σ from a 61-day running median are removed.
4. Stations with a horizontal robust scale above 10 mm or a vertical one
   above 30 mm in the fit window are dropped, as is the station with fewer
   days of any cluster closer than 1 km.

Region choices (`REGION_CONFIG`):

- California: 10% of the stations held out, spread by farthest-point
  sampling, plus every fallback-fit station; stations within 150 km of
  Ridgecrest excluded from training from 2019-07-01 to 2020-06-30.
- Türkiye: test only, so no station is held out and nothing is trained on;
  `exclude` marks the Kahramanmaraş window (30 days before to 365 days
  after 2023-02-06) on every station, kept out of the main scores and
  studied separately (docs/00-design.md §3.2, §6.3). The dense national
  network records 2009–2015 only, so the fit ends with 2012 and 2013–2015
  are the main scored years (split_day 1), 2016 on the sparse case (2).
  `turkiye-fit2019` fits to 2019 as California, for a separate experiment
  scored inside the fit years.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from imonet import trajectory
from imonet.cube import Cube, pairwise_km
from imonet.progress import progress_line

START = pd.Timestamp("2008-01-01")
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END = pd.Timestamp("2021-12-31")
MIN_TRAIN_EPOCHS = 365
FALLBACK_FIT_DAYS = 730
MIN_STEP_SIDE_EPOCHS = 30
QC_MAX_HORIZONTAL_SCALE_MM = 10.0
QC_MAX_VERTICAL_SCALE_MM = 30.0
TWIN_DISTANCE_KM = 1.0
SCREEN_WINDOW_DAYS = 61
SCREEN_K = 5.0
COMPONENTS = ("e", "n", "u")
SIGMA_COLS = ("sig_e", "sig_n", "sig_u")


@dataclass(frozen=True)
class Zone:
    """Stations within `radius_km` of (lat, lon), from `start` to `end`."""
    lat: float
    lon: float
    radius_km: float
    start: str
    end: str


@dataclass(frozen=True)
class RegionConfig:
    fit_end: str = "2019-12-31"                   # last day of the trajectory fit; split_day 0 up to it
    val_end: str = "2021-12-31"                   # split_day 1 up to it, 2 after
    heldout_fraction: float = 0.0
    heldout_seed: int = 0
    fallback_held_out: bool = False
    train_exclusions: tuple[Zone, ...] = ()       # left out of the trajectory fit and of training
    score_exclusions: tuple[tuple[str, str], ...] = ()   # day windows marked in `exclude` for every station


REGION_CONFIG = {
    "california": RegionConfig(
        heldout_fraction=0.10, heldout_seed=0, fallback_held_out=True,
        train_exclusions=(Zone(35.77, -117.60, 150.0, "2019-07-01", "2020-06-30"),)),   # Ridgecrest
    # Türkiye's dense network records 2009–2015 only (about 140 stations a
    # day, about 20 later): the fit ends with 2012, so that 2013–2015 are
    # scored outside it (split_day 1) and 2016 on is the sparse case (2).
    "turkiye": RegionConfig(
        fit_end="2012-12-31", val_end="2015-12-31",
        score_exclusions=(("2023-01-07", "2024-02-05"),)),                               # Kahramanmaraş
    # The same stations fitted to 2019 as California, for scoring 2009–2015
    # inside the fit years (a separate experiment).
    "turkiye-fit2019": RegionConfig(
        score_exclusions=(("2023-01-07", "2024-02-05"),)),
}


# --------------------------------------------------------------------------- #
# Per station
# --------------------------------------------------------------------------- #


@dataclass
class StationResult:
    r: pd.DataFrame              # mm, indexed by date, columns e n u
    sigma: pd.DataFrame          # mm
    fallback_fit: bool
    n_screened: int
    qc_scale_mm: float           # horizontal robust scale of the QC fit on fit epochs
    qc_scale_u_mm: float
    quake_steps: list = field(default_factory=list)   # (date, [E, N, U] mm) kept in r


def screen_outliers(r: pd.DataFrame) -> pd.Series:
    """True for days where any component is more than SCREEN_K robust σ from
    its running median (scale: running median of absolute deviations)."""
    daily = r.asfreq("D")
    bad = pd.Series(False, index=daily.index)
    for c in COMPONENTS:
        x = daily[c]
        med = x.rolling(SCREEN_WINDOW_DAYS, center=True, min_periods=15).median()
        dev = (x - med).abs()
        mad = dev.rolling(SCREEN_WINDOW_DAYS, center=True, min_periods=15).median()
        bad |= (dev > SCREEN_K * 1.4826 * mad).fillna(False)
    return bad.reindex(r.index, fill_value=False)


def _robust_scale(x: np.ndarray) -> float:
    return float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))


def process_station(series: pd.DataFrame, steps: pd.DataFrame,
                    exclusions: list[tuple[pd.Timestamp, pd.Timestamp]] = (),
                    fit_end: pd.Timestamp = TRAIN_END) -> StationResult | None:
    """Residuals of one station on its own dates from START on; None when it
    has too little data for a fit at all. `steps` holds its rows of NGL's
    steps file, `exclusions` date windows left out of the fit, which covers
    the days up to `fit_end`."""
    series = series.loc[START:]
    if len(series) < MIN_TRAIN_EPOCHS:
        return None
    idx = series.index
    fit_mask = idx <= fit_end
    for a, b in exclusions:
        fit_mask &= ~((idx >= a) & (idx <= b))

    fallback = fit_mask.sum() < MIN_TRAIN_EPOCHS
    if fallback:
        cutoff = max(idx[0] + pd.Timedelta(days=FALLBACK_FIT_DAYS - 1), idx[MIN_TRAIN_EPOCHS - 1])
        fit_mask = idx <= cutoff

    quake = sorted(set(steps.loc[steps.kind == "quake", "date"]))
    equipment = sorted(set(steps.loc[steps.kind == "equipment", "date"]) - set(quake))
    kept: list = []
    try:
        r = trajectory.residuals(series, fit_mask, step_dates=quake + equipment, keep_step_dates=quake,
                                 equipment_dates=equipment, min_epochs_each_side=MIN_STEP_SIDE_EPOCHS,
                                 min_fit_epochs=MIN_TRAIN_EPOCHS, kept_steps_out=kept)
        qc = trajectory.residuals(series, fit_mask, step_dates=quake + equipment,
                                  min_epochs_each_side=MIN_STEP_SIDE_EPOCHS, min_fit_epochs=MIN_TRAIN_EPOCHS)
    except ValueError:
        return None
    qc_fit = qc[fit_mask]
    qc_scale = max(_robust_scale(qc_fit["e"].to_numpy()), _robust_scale(qc_fit["n"].to_numpy()))
    qc_scale_u = _robust_scale(qc_fit["u"].to_numpy())

    sigma = pd.DataFrame({c: series[s].to_numpy() * 1000.0 for c, s in zip(COMPONENTS, SIGMA_COLS)}, index=idx)
    bad = screen_outliers(r)
    keep = ~bad.to_numpy()
    to_date = {trajectory.to_decimal_year(d): d for d in quake}
    quake_steps = [(to_date[t], amp) for t, amp in kept if t in to_date]
    return StationResult(r[keep], sigma[keep], bool(fallback), int(bad.sum()), qc_scale, qc_scale_u, quake_steps)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


def twin_drops(d: np.ndarray, sta: np.ndarray, n_days: np.ndarray) -> dict[int, int]:
    """Clusters of stations closer than TWIN_DISTANCE_KM (connected
    components): {dropped index: kept index}; the kept station has the most
    station-days, ties broken by id."""
    parent = list(range(len(sta)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*np.nonzero(np.triu(d < TWIN_DISTANCE_KM, 1))):
        parent[root(a)] = root(b)
    clusters: dict[int, list[int]] = {}
    for i in range(len(sta)):
        clusters.setdefault(root(i), []).append(i)
    drops = {}
    for members in clusters.values():
        if len(members) > 1:
            keep = min(members, key=lambda i: (-n_days[i], sta[i]))
            drops.update({i: keep for i in members if i != keep})
    return drops


def farthest_point_sample(d: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n station indices spread over the network: start at a random station,
    repeatedly add the one farthest from those already chosen."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(d.shape[0]))]
    nearest = d[chosen[0]].copy()
    for _ in range(n - 1):
        j = int(np.argmax(nearest))
        chosen.append(j)
        nearest = np.minimum(nearest, d[j])
    return np.array(sorted(chosen))


def day_splits(days: pd.DatetimeIndex, fit_end: pd.Timestamp = TRAIN_END,
               val_end: pd.Timestamp = VAL_END) -> np.ndarray:
    """0 up to `fit_end` (California: training, to 2019), 1 up to `val_end`
    (validation, 2020–2021), 2 after (test, 2022 on)."""
    out = np.full(len(days), 2, dtype=np.int8)
    out[days <= val_end] = 1
    out[days <= fit_end] = 0
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_cube(stations: pd.DataFrame, load_series: Callable[[str], pd.DataFrame], steps: pd.DataFrame,
               config: RegionConfig, end: pd.Timestamp | None = None,
               progress: Callable[[str], None] | None = None,
               countries: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> tuple[Cube, dict]:
    """`stations` needs columns sta, lat, lon; `countries` maps latitudes and
    longitudes to country codes (imonet.borders.countries). Returns the cube
    and a summary."""
    stations = stations.sort_values("sta").reset_index(drop=True)
    lat_all, lon_all = stations.lat.to_numpy(float), stations.lon.to_numpy(float)
    zone_hits = []                                    # per zone: S_all bool
    for z in config.train_exclusions:
        d = pairwise_km(np.r_[z.lat, lat_all], np.r_[z.lon, lon_all])[0, 1:]
        zone_hits.append(d <= z.radius_km)

    results: dict[str, StationResult] = {}
    dropped: dict[str, list] = {"no_baseline": [], "qc": [], "twin": []}
    steps_by_sta = dict(tuple(steps.groupby("sta")))
    empty = steps.iloc[:0]
    t0 = time.time()
    for i, row in stations.iterrows():
        windows = [(pd.Timestamp(z.start), pd.Timestamp(z.end))
                   for z, hit in zip(config.train_exclusions, zone_hits) if hit[i]]
        res = process_station(load_series(row.sta), steps_by_sta.get(row.sta, empty), windows,
                              pd.Timestamp(config.fit_end))
        if res is None:
            dropped["no_baseline"].append(row.sta)
        elif res.qc_scale_mm > QC_MAX_HORIZONTAL_SCALE_MM or res.qc_scale_u_mm > QC_MAX_VERTICAL_SCALE_MM:
            dropped["qc"].append({"sta": row.sta, "lat": round(float(row.lat), 3), "lon": round(float(row.lon), 3),
                                  "horizontal_scale_mm": round(res.qc_scale_mm, 1),
                                  "vertical_scale_mm": round(res.qc_scale_u_mm, 1)})
        else:
            results[row.sta] = res
        if progress and ((i + 1) % 25 == 0 or i + 1 == len(stations)):
            progress(progress_line(i + 1, len(stations), t0, "stations"))

    cand = stations[stations.sta.isin(results)].reset_index(drop=True)
    d_c = pairwise_km(cand.lat.to_numpy(float), cand.lon.to_numpy(float))
    n_days = np.array([len(results[s].r) for s in cand.sta])
    for i, k in sorted(twin_drops(d_c, cand.sta.to_numpy(str), n_days).items()):
        dropped["twin"].append({"sta": cand.sta[i], "kept": cand.sta[k], "km": round(float(d_c[i, k]), 3)})
        del results[cand.sta[i]]

    kept_mask = stations.sta.isin(results).to_numpy()
    kept = stations[kept_mask].reset_index(drop=True)
    last = max(r.r.index.max() for r in results.values())
    days = pd.date_range(START, end or last, freq="D")
    S, T = len(kept), len(days)

    r = np.full((S, T, 3), np.nan, dtype=np.float32)
    sigma = np.full((S, T, 3), np.nan, dtype=np.float32)
    for i, sta in enumerate(kept.sta):
        res = results[sta]
        pos = days.get_indexer(res.r.index)
        ok = pos >= 0
        r[i, pos[ok]] = res.r.to_numpy(np.float32)[ok]
        sigma[i, pos[ok]] = res.sigma.to_numpy(np.float32)[ok]
    avail = np.isfinite(r).all(axis=2)

    qs, qd, qa = [], [], []
    for i, sta in enumerate(kept.sta):
        for date, amp in results[sta].quake_steps:
            pos = days.searchsorted(date)
            if pos < T:
                qs.append(i); qd.append(int(pos)); qa.append(amp)

    lat, lon = kept.lat.to_numpy(float), kept.lon.to_numpy(float)
    fallback = np.array([results[s].fallback_fit for s in kept.sta])
    split_sta = np.zeros(S, dtype=np.int8)
    if config.heldout_fraction:
        n_held = max(1, round(config.heldout_fraction * S))
        split_sta[farthest_point_sample(pairwise_km(lat, lon), n_held, config.heldout_seed)] = 1
    if config.fallback_held_out:
        split_sta[fallback] = 1

    exclude = np.zeros((S, T), dtype=bool)
    for z, hit in zip(config.train_exclusions, zone_hits):
        in_window = (days >= pd.Timestamp(z.start)) & (days <= pd.Timestamp(z.end))
        exclude[np.ix_(hit[kept_mask], in_window)] = True
    for a, b in config.score_exclusions:
        exclude[:, (days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))] = True

    cube = Cube(r=r, sigma=sigma, avail=avail, sta=kept.sta.to_numpy(str), lat=lat, lon=lon,
                days=days.to_numpy("datetime64[D]"), split_day=day_splits(days, pd.Timestamp(config.fit_end), pd.Timestamp(config.val_end)), split_sta=split_sta,
                exclude=exclude, qstep_sta=np.array(qs, dtype=np.int32), qstep_day=np.array(qd, dtype=np.int32),
                qstep_amp=np.array(qa, dtype=np.float32).reshape(-1, 3), fallback_fit=fallback,
                country=np.asarray(countries(lat, lon) if countries else np.full(S, ""), dtype="<U3"))
    summary = {
        "stations_in": int(len(stations)), "stations_kept": S, "stations_dropped": dropped,
        "days": T, "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
        "station_days": int(avail.sum()), "coverage": round(float(avail.mean()), 4),
        "heldout_stations": int(split_sta.sum()), "fallback_fit_stations": int(fallback.sum()),
        "screened_days": int(sum(x.n_screened for x in results.values())), "kept_quake_offsets": len(qs),
        "excluded_station_days": int((exclude & avail).sum()),
        "countries": {c: int(n) for c, n in zip(*np.unique(cube.country, return_counts=True))},
        "config": {"start": str(START.date()), "fit_end": config.fit_end, "val_end": config.val_end,
                   "min_train_epochs": MIN_TRAIN_EPOCHS, "fallback_fit_days": FALLBACK_FIT_DAYS,
                   "qc_max_horizontal_scale_mm": QC_MAX_HORIZONTAL_SCALE_MM,
                   "qc_max_vertical_scale_mm": QC_MAX_VERTICAL_SCALE_MM, "twin_distance_km": TWIN_DISTANCE_KM,
                   "screen_window_days": SCREEN_WINDOW_DAYS, "screen_k": SCREEN_K,
                   "region": {"heldout_fraction": config.heldout_fraction, "heldout_seed": config.heldout_seed,
                              "fallback_held_out": config.fallback_held_out,
                              "train_exclusions": [vars(z) for z in config.train_exclusions],
                              "score_exclusions": [list(w) for w in config.score_exclusions]}},
    }
    return cube, summary


def write_summary(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n")
