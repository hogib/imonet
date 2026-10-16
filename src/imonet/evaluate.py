"""Experiment A: generate held-out stations and score them against the real
series (docs/00-design.md §6.1).

A run is (method, split, density, thinning seed). The split's days are
tiled with windows of the model length; the last window is aligned to the
period end and scores only the days no earlier window covered. Each target
is predicted from the normal stations, thinned to the density, with the
target itself always removed, and every usable day as input.

Truth and samples are compared after removing each window's mean and line,
fitted over the target's days with data (`imonet.data.detrend`), the same
for both, as the model's inputs and training targets are.

Results go to runs/A/<method>/<split>/<density>_s<seed>/:
summary.json, stations.csv, correlations.csv (neighbour correlation by
distance).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from imonet.baselines import LENGTH_SCALES_KM, Method, make
from imonet.cube import HELD_OUT, NORMAL, VALIDATION, Cube, pairwise_km
from imonet.data import WINDOW_DAYS, component_scale, context, day_of_year, detrend, target, target_noise
from imonet.kriging import Covariance, empirical_covariance, fit_covariance
from imonet.metrics import (correlation_by_distance, event_mask, neighbour_correlations, scoring_qc, screen,
                            station_noise, station_noise_log, station_scale, station_scores, summarise)
from imonet.network import K_NEIGHBOURS, thin

RESULTS = Path("runs")
DENSITIES = (None, 25.0, 60.0, 100.0)    # median nearest-neighbour km; None = full network
THIN_SEEDS = 5
TUNING_TARGETS = 200
BATCH = 64


def density_name(density: float | None) -> str:
    return "full" if density is None else f"{density:g}km"


def period_windows(cube: Cube, split: int, length: int = WINDOW_DAYS) -> list[tuple[int, int]]:
    """(first day, first scored day) of the windows tiling a split."""
    days = np.flatnonzero(cube.split_day == split)
    a, b = int(days[0]), int(days[-1]) + 1
    if len(days) != b - a:
        raise ValueError("split days are not contiguous")
    if b - a < length:
        raise ValueError("split shorter than one window")
    out = [(t, t) for t in range(a, b - length + 1, length)]
    if out[-1][0] + length < b:
        out.append((b - length, out[-1][0] + length))
    return out


def network(cube: Cube, density: float | None, seed: int) -> np.ndarray:
    """Cube indices of the normal stations, thinned to `density`."""
    normal = cube.stations(NORMAL)
    if density is None:
        return normal
    keep = thin(pairwise_km(cube.lat[normal], cube.lon[normal]), density, np.random.default_rng((1000, seed)))
    return normal[keep]


@dataclass
class Prediction:
    targets: np.ndarray         # S' cube indices
    days: np.ndarray            # T' cube day indices of the split
    truth: np.ndarray           # S'×T'×3 mm, detrended per window, NaN where missing
    samples: np.ndarray         # m×S'×T'×3 mm, detrended per window, NaN where not predicted
    windows: list[dict]         # per (target, window): ti, t0, scored, nbr, dist_km


def predict(method: Method, cube: Cube, scale: np.ndarray, targets: np.ndarray, net: np.ndarray,
            split: int, seed: int = 0, length: int = WINDOW_DAYS, k: int = K_NEIGHBOURS,
            noise_log: np.ndarray | None = None, progress=None) -> Prediction:
    """Every method sees the same contexts; with `noise_log` (S×3,
    `station_noise_log`) they include the target's noise level, which only
    a diffusion model trained with it and told to use it reads. `progress`
    is called with a line about every tenth of the windows."""
    ok = cube.avail
    method.prepare(cube, scale, net, ok)
    wins = period_windows(cube, split, length)
    days = np.arange(wins[0][0], wins[-1][0] + length)
    out = np.full((method.n_samples, len(targets), len(days), 3), np.nan, dtype=np.float32)
    truth = np.full((len(targets), len(days), 3), np.nan, dtype=np.float32)
    jobs = [(ti, t0, first) for ti, s in enumerate(targets) for t0, first in wins
            if ok[s, first:t0 + length].any()]
    records = []
    t_start, next_report = time.time(), 0.1
    for a in range(0, len(jobs), BATCH):
        if progress and a / len(jobs) >= next_report:
            progress(f"  {a}/{len(jobs)} windows, {time.time() - t_start:.0f} s")
            next_report += 0.1
        chunk = jobs[a:a + BATCH]
        ctxs, truths = [], []
        for ti, t0, _ in chunk:
            s = targets[ti]
            ctx = context(cube, scale, cube.lat[s], cube.lon[s], t0, net[net != s], ok, length, k)
            ctx |= {"target": s, "t0": t0, "doy0": np.float32(day_of_year(cube.days[t0]))}
            ctxs.append(ctx | (target_noise(noise_log, s) if noise_log is not None else {}))
            truths.append(target(cube, scale, s, t0, ok, length))   # kept from the method
        batch = {key: np.stack([np.asarray(c[key]) for c in ctxs]) for key in ctxs[0]}
        x_ok = np.stack([t["x_ok"] for t in truths])
        rng = np.random.default_rng((seed, a))
        x = method.sample(batch, rng)                                       # B×m×L×3
        x = detrend(x, x_ok[:, None], zero_missing=False) * scale
        z = np.where(x_ok[..., None], np.stack([t["x"] for t in truths]) * scale, np.nan)
        for i, (ti, t0, first) in enumerate(chunk):
            lo, cols = first - t0, slice(first - days[0], t0 + length - days[0])
            out[:, ti, cols] = x[i, :, lo:]
            truth[ti, cols] = z[i, lo:]
            v = batch["valid"][i]
            records.append({"ti": ti, "t0": t0, "scored": first, "nbr": batch["nbr"][i, v],
                            "dist_km": batch["geom"][i, v, 2]})
    return Prediction(targets, days, truth, out, records)


def screened(pred: Prediction) -> tuple[Prediction, float]:
    """The prediction with every sample screened for outliers as the cube's
    real series were (`metrics.screen`), on the days the truth exists, and
    the share of those sample values that were outliers."""
    have = np.isfinite(pred.truth)[None]
    x = np.where(have, pred.samples, np.nan)
    out = np.empty_like(pred.samples)
    bad = np.zeros(pred.samples.shape, dtype=bool)
    for ti in range(len(pred.targets)):
        xs, b = screen(np.moveaxis(x[:, ti], 0, 1))                 # days × m × 3
        out[:, ti], bad[:, ti] = np.moveaxis(xs, 1, 0), np.moveaxis(b, 1, 0)
    out = np.where(have, out, pred.samples)
    return replace(pred, samples=out.astype(pred.samples.dtype)), float(bad[have.repeat(len(out), 0)].mean())


def score(cube: Cube, pred: Prediction, scales: np.ndarray, skip: np.ndarray, keep: np.ndarray,
          length: int = WINDOW_DAYS, noise: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-station scores, neighbour correlation pairs). `noise`
    (`station_noise`) is recorded with each station's scores."""
    rows = []
    for ti, s in enumerate(pred.targets):
        if not keep[s]:
            continue
        truth = pred.truth[ti]
        samples = pred.samples[:, ti]
        cells = np.isfinite(truth).all(axis=1) & np.isfinite(samples[0]).all(axis=1)
        row = station_scores(truth, samples, cells, skip[s, pred.days], scales[s])
        if row is not None:
            rows.append({"station": str(cube.sta[s]), **({} if noise is None else {"noise": float(noise[s])}), **row})

    pairs = []
    if pred.samples.shape[0] > 1:
        for w in pred.windows:
            s = pred.targets[w["ti"]]
            if not keep[s] or not len(w["nbr"]):
                continue
            win = np.arange(w["t0"], w["t0"] + length)
            cols = win - pred.days[0]
            cells = cube.avail[s, win] & ~skip[s, win] & (win >= w["scored"])
            truth = pred.truth[w["ti"], cols]
            nbr = np.where(cube.avail[w["nbr"]][:, win, None], cube.r[w["nbr"]][:, win], np.nan)
            corr = neighbour_correlations(truth, pred.samples[:, w["ti"], cols], nbr, cells)
            for j, d in enumerate(w["dist_km"]):
                for c in range(3):
                    pairs.append((float(d), c, corr[j, c, 0], corr[j, c, 1]))
    pairs = pd.DataFrame(pairs, columns=["dist_km", "component", "corr_truth", "corr_sample"])
    return pd.DataFrame(rows), pairs.dropna()


class Setup:
    """What every run needs once per cube: scales, scoring exclusions."""

    def __init__(self, cube: Cube, steps: pd.DataFrame):
        self.cube = cube
        self.scale = component_scale(cube)
        self.station_scale = station_scale(cube)
        self.noise_log = station_noise_log(cube)
        self.noise = station_noise(log=self.noise_log)
        self.skip = event_mask(cube, steps)
        self.keep, self.qc_dropped = scoring_qc(cube, steps)


def tune_idw(setup: Setup, density: float | None, seed: int = 0) -> dict:
    """ℓ of idw minimising the fast nRMSE on validation days, with normal
    stations as targets (leave-one-out), so held-out stations stay unseen."""
    cube = setup.cube
    net = network(cube, density, seed)
    rng = np.random.default_rng(7)
    targets = np.sort(rng.choice(cube.stations(NORMAL), TUNING_TARGETS, replace=False))
    curve = {}
    for length_km in LENGTH_SCALES_KM:
        pred = predict(make("idw", length_km), cube, setup.scale, targets, net, VALIDATION)
        per, _ = score(cube, pred, setup.station_scale, setup.skip, setup.keep)
        curve[f"{length_km:g}"] = summarise(per)["nrmse_fast"]
    best = min(curve, key=curve.get)
    return {"length_km": float(best), "curve": curve}


def run_dir(method: str, split_name: str, density: float | None, seed: int, root: Path = RESULTS) -> Path:
    return root / "A" / method / split_name / f"{density_name(density)}_s{seed}"


def method_for(setup: Setup, method_name: str, density: float | None, n_samples: int, root: Path,
               model_dir: Path | None = None, steps: int = 32, guidance: float = 1.0,
               target_noise: str = "unknown") -> tuple[Method, dict]:
    """The method to run and what to record about it."""
    if method_name in ("diffusion", "regress"):
        from imonet.model.methods import DiffusionMethod, RegressMethod
        if model_dir is None:
            raise ValueError(f"{method_name} needs a trained model (--model)")
        m = DiffusionMethod(model_dir, n_samples, steps, guidance, target_noise) if method_name == "diffusion" \
            else RegressMethod(model_dir)
        if not np.allclose(m.scale, setup.scale, rtol=1e-4):
            raise ValueError("the model was trained with a different component scale")
        extra = {"model": str(model_dir)}
        if method_name == "diffusion":
            extra |= {"sampler_steps": steps, "guidance": guidance, "target_noise": target_noise}
        return m, extra
    length_km, extra = None, {}
    if method_name in ("idw", "idw+noise"):
        tuning = root / "A" / "idw_tuning.json"
        tuned = json.loads(tuning.read_text()) if tuning.exists() else {}
        key = density_name(density)
        if key not in tuned:
            tuned[key] = tune_idw(setup, density)
            tuning.parent.mkdir(parents=True, exist_ok=True)
            tuning.write_text(json.dumps(tuned, indent=1))
        length_km = tuned[key]["length_km"]
        extra["length_km"] = length_km
    if method_name == "gp":
        from imonet.gp import GP
        model = space_time_model(setup, root)
        return GP(model, n_samples, target_noise), {"model": asdict(model), "target_noise": target_noise}
    covariance = None
    if method_name in ("krige", "krige+noise"):
        covariance = kriging_covariance(setup, root)
        extra["covariance"] = asdict(covariance)
    return make(method_name, length_km, n_samples, covariance), extra


def space_time_model(setup: Setup, root: Path = RESULTS):
    """The space–time GP (`imonet.gp`), fitted once to the training years of
    the normal stations and kept in runs/A/gp_model.json (with the lag
    curves it was fitted to)."""
    from imonet.gp import SpaceTime, empirical_lag_curves, fit_space_time
    path = root / "A" / "gp_model.json"
    if path.exists():
        return SpaceTime.load(path)
    curves = empirical_lag_curves(setup.cube, setup.scale)
    model = fit_space_time(curves, kriging_covariance(setup, root))
    model.save(path)
    (root / "A" / "gp_lag_curves.json").write_text(json.dumps(curves, indent=1))
    return model


def kriging_covariance(setup: Setup, root: Path = RESULTS) -> Covariance:
    """The kriging covariance, fitted once to the training years of the
    normal stations and kept in runs/A/krige_covariance.json (with the
    empirical covariances it was fitted to)."""
    path = root / "A" / "krige_covariance.json"
    if path.exists():
        return Covariance.load(path)
    emp = empirical_covariance(setup.cube, setup.scale)
    cov = fit_covariance(emp)
    cov.save(path)
    (root / "A" / "krige_empirical.json").write_text(json.dumps(emp, indent=1))
    return cov


def run(setup: Setup, method_name: str, split: int, split_name: str, density: float | None, seed: int,
        n_samples: int = 32, root: Path = RESULTS, model_dir: Path | None = None, label: str | None = None,
        steps: int = 32, guidance: float = 1.0, screen: bool = False, target_noise: str = "unknown",
        progress=None) -> list[dict]:
    """One run of experiment A; writes its results under `label` (default
    the method name) and returns the summary. With `screen`, the same
    samples screened for outliers are also scored, as `<label>-screened`."""
    t = time.time()
    cube = setup.cube
    label = label or method_name
    method, extra = method_for(setup, method_name, density, n_samples, root, model_dir, steps, guidance,
                               target_noise)
    net = network(cube, density, seed)
    pred = predict(method, cube, setup.scale, cube.stations(HELD_OUT), net, split, seed, noise_log=setup.noise_log,
                   progress=progress)
    variants = [(label, pred, {})]
    if screen and method.n_samples > 1:
        spred, share = screened(pred)
        variants.append((f"{label}-screened", spred, {"screened": share}))
    summaries = []
    for name, p, more in variants:
        per, pairs = score(cube, p, setup.station_scale, setup.skip, setup.keep, noise=setup.noise)
        summary = {"method": name, "kind": method_name, "split": split_name, "density": density_name(density),
                   "seed": seed, "network_stations": int(len(net)), "samples": method.n_samples, **extra, **more,
                   **summarise(per), "seconds": round(time.time() - t, 1)}
        out = run_dir(name, split_name, density, seed, root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        per.to_csv(out / "stations.csv", index=False)
        if len(pairs):
            correlation_by_distance(pairs).to_csv(out / "correlations.csv")
        summaries.append(summary)
    return summaries


def rescore(cube: Cube, split_name: str = "validation", root: Path = RESULTS) -> int:
    """Recompute every run's summary of a split from its per-station scores
    (stations.csv), after adding each station's noise level; for when the
    summary changes and the samples are long gone. Returns the runs updated."""
    noise = dict(zip(map(str, cube.sta), station_noise(cube)))
    n = 0
    for path in sorted((root / "A").glob(f"*/{split_name}/*/stations.csv")):
        per = pd.read_csv(path)
        if per.empty:
            continue
        per["noise"] = per["station"].astype(str).map(noise)
        per.to_csv(path, index=False)
        summary_path = path.parent / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary_path.write_text(json.dumps(summary | summarise(per), indent=1))
        n += 1
    return n


def report(split_name: str = "validation", root: Path = RESULTS) -> pd.DataFrame:
    """Every run of a split: mean and spread over thinning seeds per
    (method, density)."""
    rows = [json.loads(p.read_text()) for p in sorted((root / "A").glob(f"*/{split_name}/*/summary.json"))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["nrmse_fast", "nrmse_slow", "nrmse_total", "crps_fast", "crps_total", "energy_total",
            "cover50_fast", "cover90_fast", "spread_skill", "spread_skill_pooled", "cover90_fast_pooled",
            "spread_skill_q1", "spread_skill_q4", "scale_log2", "tail_log2"]
    cols = [c for c in cols if c in df]
    order = {density_name(d): i for i, d in enumerate(DENSITIES)}
    g = df.groupby(["method", "density"])[cols].mean()
    g["seeds"] = df.groupby(["method", "density"]).size()
    return g.sort_index(key=lambda s: s.map(order) if s.name == "density" else s)
