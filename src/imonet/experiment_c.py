"""Experiment C: Türkiye, with the models and baselines of California
unchanged (docs/00-design.md §3.2, §6.3).

Every station is a target in turn, generated from all the others (the 16
nearest within 300 km) at the network's own density: `predict` removes the
target from its network. Nothing is tuned on Türkiye: the models and their
input scaling, IDW's length scale (California's at 60 km, Türkiye's
spacing), and the kriging and space–time GP covariances all come from
California, and a station's noise level is relative to California's typical
station, as the models learned it.

Periods (`PERIODS`):

- main: 2013–2015 on `turkiye` (trajectory fitted to 2012), when the
  national network records densely (about 140 stations a day);
- sparse: 2016 on, outside the Kahramanmaraş window, on the same cube
  (about 17 stations a day);
- infit: 2009–2015 on `turkiye-fit2019` (fitted to 2019, as California),
  inside the fit years, a separate experiment.

Scoring is as in experiment A, with three changes. The scoring QC checks
the period's own days, so that a station is not dropped from 2013–2015 for
trouble a decade later. Stations whose trajectory was fitted on the start
of their record (`fallback_fit`), which overlaps the scored days, are left
out. The Kahramanmaraş window is skipped. Every summary is given for all
stations and for those in Türkiye (`country`).

Results: runs/C/<label>/<period>/{summary.json, stations.csv,
correlations.csv}.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from imonet.cube import CALIFORNIA, NORMAL, Cube
from imonet.data import component_scale
from imonet.evaluate import RESULTS, Setup, method_for, predict, range_windows, score, screened
from imonet.metrics import correlation_by_distance, station_fast_scale, summarise

CUBES = Path("data/cube")
DENSITY_KM = 60.0               # California's tuned settings used for Türkiye's spacing
COUNTRY = "TUR"


@dataclass(frozen=True)
class Period:
    cube: str
    ranges: tuple[tuple[str, str | None], ...]   # inclusive dates; None: the cube's last day


PERIODS = {
    "main": Period("turkiye", (("2013-01-01", "2015-12-31"),)),
    "sparse": Period("turkiye", (("2016-01-01", "2023-01-06"), ("2024-02-06", None))),
    "infit": Period("turkiye-fit2019", (("2009-01-01", "2015-12-31"),)),
}


def reference(root: Path = RESULTS, california: Path = CALIFORNIA) -> dict:
    """California's component scale (the models' input scaling) and typical
    station fast scale (what noise levels are relative to), computed once
    and kept in runs/C/reference.json."""
    path = root / "C" / "reference.json"
    if path.exists():
        return json.loads(path.read_text())
    cube = Cube.load(california)
    ref = {"scale": component_scale(cube).tolist(),
           "typical": np.nanmedian(station_fast_scale(cube), axis=0).tolist(), "cube": str(california)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=1))
    return ref


def period_days(cube: Cube, period: Period) -> list[tuple[int, int]]:
    """[first, stop) day indices of the period's ranges."""
    days = pd.DatetimeIndex(cube.days)
    out = []
    for a, b in period.ranges:
        first = int(days.searchsorted(pd.Timestamp(a)))
        stop = len(days) if b is None else int(days.searchsorted(pd.Timestamp(b), side="right"))
        out.append((first, stop))
    return out


def setup_for(period_name: str, steps: pd.DataFrame, root: Path = RESULTS,
              cubes: Path = CUBES) -> tuple[Setup, list[tuple[int, int]]]:
    """The Setup and windows of a period."""
    period = PERIODS[period_name]
    cube = Cube.load(cubes / f"{period.cube}.npz")
    ranges = period_days(cube, period)
    in_period = np.zeros(cube.n_days, dtype=bool)
    for a, b in ranges:
        in_period[a:b] = True
    ref = reference(root)
    setup = Setup(cube, steps, scale=np.asarray(ref["scale"]), typical=np.asarray(ref["typical"]),
                  qc_days=in_period)
    if len(cube.fallback_fit):
        setup.keep &= ~cube.fallback_fit
    setup.skip = setup.skip | cube.exclude
    windows = [w for a, b in ranges for w in range_windows(a, b)]
    return setup, windows


def run(setup: Setup, windows: list[tuple[int, int]], period_name: str, method_name: str, n_samples: int = 32,
        root: Path = RESULTS, model_dir: Path | None = None, label: str | None = None, steps: int = 32,
        screen: bool = False, target_noise: str = "unknown", progress=None) -> list[dict]:
    """One run of experiment C; writes its results under `label` (default
    the method name) and returns the summaries."""
    for name in ("krige_covariance.json", "gp_model.json", "idw_tuning.json"):
        if not (root / "A" / name).exists():
            raise FileNotFoundError(f"{root / 'A' / name}: fitted on California by experiment A, needed here")
    t = time.time()
    cube = setup.cube
    label = label or method_name
    method, extra = method_for(setup, method_name, DENSITY_KM, n_samples, root, model_dir, steps, 1.0, target_noise)
    net = cube.stations(NORMAL)                     # every station; predict leaves out the target
    pred = predict(method, cube, setup.scale, net, net, None, 0, noise_log=setup.noise_log, progress=progress,
                   windows=windows)
    variants = [(label, pred, {})]
    if screen and method.n_samples > 1:
        spred, share = screened(pred)
        variants.append((f"{label}-screened", spred, {"screened": share}))
    country = dict(zip(map(str, cube.sta), map(str, cube.country)))
    summaries = []
    for name, p, more in variants:
        per, pairs = score(cube, p, setup.station_scale, setup.skip, setup.keep, noise=setup.noise)
        in_home = per["station"].astype(str).map(country).eq(COUNTRY).to_numpy() if len(per) else np.zeros(0, bool)
        home = per[in_home]
        summary = {"method": name, "kind": method_name, "period": period_name, "cube": PERIODS[period_name].cube,
                   "network_stations": int(len(net)), "scored_stations": int(len(per)),
                   "scored_stations_turkiye": int(len(home)), "qc_dropped": int(len(setup.qc_dropped)),
                   "fallback_left_out": int(setup.cube.fallback_fit.sum()) if len(setup.cube.fallback_fit) else 0,
                   "samples": method.n_samples, **extra, **more, **summarise(per),
                   "turkiye": summarise(home), "seconds": round(time.time() - t, 1)}
        out = root / "C" / name / period_name
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        per.assign(country=per["station"].astype(str).map(country) if len(per) else []).to_csv(
            out / "stations.csv", index=False)
        if len(pairs):
            correlation_by_distance(pairs).to_csv(out / "correlations.csv")
        summaries.append(summary)
    return summaries
