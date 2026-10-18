"""Scores of generated series against the real ones (docs/00-design.md §6.1).

Series are on the cube's daily grid, NaN where missing. Every score is
computed per station, on its scored days, in units of the station's robust
training-years scale (`station_scale`), then summarised by the median over
stations, so a few stations with large earthquake offsets cannot dominate
(as gnssdl's benchmark, contract §4).

Fast and slow parts: the slow part of a series is its 61-day running median
over the scored days, the fast part the rest. Point error (nRMSE) splits the
error of the sample mean, as gnssdl does. Probabilistic scores split the
truth and each sample separately, since a median of differences is not a
difference of medians.

A method that returns one sample is a point forecast: its CRPS is its mean
absolute error, and coverage, spread and realism scores are undefined (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imonet.cube import TRAIN, Cube

COMPONENTS = ("e", "n", "u")
SLOW_DAYS = 61
MIN_CELLS = 20
LEVELS = (0.5, 0.9)
SEGMENT_DAYS = 256
SEGMENT_MIN_FRACTION = 0.9
BANDS_DAYS = ((2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128))
DIST_BINS_KM = (0, 10, 25, 50, 100, 200, 300)

# Scoring exclusions, as gnssdl (contract §4).
EVENT_MIN_MAG = 6.0
EVENT_HALF_WINDOW_DAYS = 30
QC_SCATTER_RATIO = 3.0
QC_JUMP_MM = (50.0, 50.0, 100.0)
QC_JUMP_STEP_TOLERANCE_DAYS = 45

# Outlier screening of the cube (gnssdl dataset.screen_outliers), applied
# to generated series so that they are screened as the real ones were.
SCREEN_DAYS = 61
SCREEN_K = 5.0
SCREEN_MIN_DAYS = 15
TAIL_QUANTILE = 0.99
NOISE_GROUPS = 4             # calibration by station noise level, in quartiles


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def slow_part(x: np.ndarray) -> np.ndarray:
    """61-day centred running median along axis 0 of a daily series (NaN
    where missing; needs 3 values). Further axes are independent series."""
    flat = x.reshape(len(x), -1)
    slow = pd.DataFrame(flat).rolling(SLOW_DAYS, center=True, min_periods=3).median().to_numpy()
    return slow.reshape(x.shape)


def fast_part(x: np.ndarray) -> np.ndarray:
    return x - slow_part(x)


def screen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cube's outlier screen on daily series (axis 0, NaN where
    missing; further axes independent): a value more than SCREEN_K robust σ
    from its 61-day running median is an outlier, with σ from a running
    median of absolute deviations. The cube removed such days from the
    record; a generated series instead gets its running median there, so
    that every sample still has a value on every day. Returns (screened x,
    outlier mask)."""
    flat = pd.DataFrame(x.reshape(len(x), -1))
    roll = dict(window=SCREEN_DAYS, center=True, min_periods=SCREEN_MIN_DAYS)
    med = flat.rolling(**roll).median()
    dev = (flat - med).abs()
    mad = dev.rolling(**roll).median()
    bad = (dev > SCREEN_K * 1.4826 * mad).to_numpy()
    out = np.where(bad, med.to_numpy(), flat.to_numpy())
    return out.reshape(x.shape), bad.reshape(x.shape)


def robust_scale(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def station_scale(cube: Cube) -> np.ndarray:
    """S×3 robust scale of each station's training-years residuals (available,
    not excluded); from all its available days if it has fewer than 365 of
    them. Only a normaliser: it never feeds a prediction."""
    train = (cube.split_day == TRAIN)[None, :] & cube.avail & ~cube.exclude
    out = np.full((cube.n_sta, 3), np.nan)
    for i in range(cube.n_sta):
        cells = train[i] if train[i].sum() >= 365 else cube.avail[i]
        if cells.any():
            out[i] = robust_scale(cube.r[i, cells].astype(np.float64))
    return out


def station_fast_scale(cube: Cube) -> np.ndarray:
    """S×3 robust scale of each station's fast part in its training years
    (as `station_scale`); NaN without data."""
    train = (cube.split_day == TRAIN)[None, :] & cube.avail & ~cube.exclude
    fast = np.full((cube.n_sta, 3), np.nan)
    for i in range(cube.n_sta):
        cells = train[i] if train[i].sum() >= 365 else cube.avail[i]
        if cells.any():
            x = np.where(cells[:, None], cube.r[i], np.nan).astype(np.float64)
            fast[i] = robust_scale(fast_part(x)[cells])
    return fast


def station_noise_log(cube: Cube, typical: np.ndarray | None = None) -> np.ndarray:
    """S×3: each station's day-to-day noise level per component relative to
    a typical station, as a natural log: the log of the robust scale of its
    fast part in its training years over `typical` (3), by default the
    median of that scale over the cube's stations. 0 is a typical station;
    NaN without data. Also an optional input of the diffusion model
    (`imonet.model.net`); a cube from another region passes California's
    typical scale, which the model learned the levels against."""
    fast = station_fast_scale(cube)
    return np.log(fast / (np.nanmedian(fast, axis=0) if typical is None else typical))


def station_noise(cube: Cube | None = None, log: np.ndarray | None = None) -> np.ndarray:
    """S: a station's noise level relative to a typical station, the
    geometric mean over E, N, U of `station_noise_log` (given as `log`, or
    computed from `cube`). 1 is a typical station."""
    log = station_noise_log(cube) if log is None else log
    return np.exp(log.mean(axis=1))


def crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble (axis 0) for observations y. With m > 1 members
    the fair estimator, unbiased for the CRPS of the distribution the
    members are drawn from; with one member the absolute error."""
    m = len(samples)
    mae = np.mean(np.abs(samples - y[None]), axis=0)
    if m == 1:
        return mae
    xs = np.sort(samples, axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (samples.ndim - 1))
    pair_sum = 2 * np.sum(w * xs, axis=0)                  # Σ_i Σ_j |x_i − x_j|
    return mae - pair_sum / (2 * m * (m - 1))


def energy_score(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Energy score of an ensemble of vectors (m×…×3) for y (…×3), fair
    estimator; the multivariate CRPS."""
    m = len(samples)
    first = np.mean(np.linalg.norm(samples - y[None], axis=-1), axis=0)
    if m == 1:
        return first
    pair = sum(np.linalg.norm(samples - samples[i][None], axis=-1).sum(axis=0) for i in range(m))
    return first - pair / (2 * m * (m - 1))


def inside(samples: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """Whether y lies in the ensemble's central interval at `level`.

    Quantiles use Weibull plotting positions (the i-th of m sorted members
    at i/(m+1)), so that a draw from the members' own distribution falls
    inside with probability `level`. Numpy's default positions make the
    interval too narrow: about 85% coverage for a 90% interval of 32
    members."""
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0, method="weibull")
    return (y >= lo) & (y <= hi)


def band_psd(x: np.ndarray) -> np.ndarray | None:
    """Mean periodogram of each column of x (L×C, NaN where missing) in each
    band of BANDS_DAYS, after filling gaps linearly, removing a line and a
    Hann taper. None if under SEGMENT_MIN_FRACTION of the days exist."""
    ok = np.isfinite(x).all(axis=1)
    if ok.mean() < SEGMENT_MIN_FRACTION:
        return None
    t = np.arange(len(x))
    filled = np.stack([np.interp(t, t[ok], x[ok, c]) for c in range(x.shape[1])], axis=1)
    filled -= _line(t, filled)
    w = np.hanning(len(x))[:, None]
    p = np.abs(np.fft.rfft(filled * w, axis=0)) ** 2 / np.sum(w ** 2)
    f = np.fft.rfftfreq(len(x))
    out = np.empty((x.shape[1], len(BANDS_DAYS)))
    for b, (lo, hi) in enumerate(BANDS_DAYS):
        sel = (f > 1 / hi) & (f <= 1 / lo)
        out[:, b] = p[sel].mean(axis=0)
    return out


def _line(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.stack([np.ones_like(t), t], axis=1).astype(float)
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return a @ coef


# ---------------------------------------------------------------------------
# Scoring exclusions
# ---------------------------------------------------------------------------

def event_mask(cube: Cube, steps: pd.DataFrame) -> np.ndarray:
    """S×T: days within ±30 of an M ≥ 6 earthquake that NGL lists as
    possibly offsetting that station. Left out of the fast and slow parts."""
    days = pd.DatetimeIndex(cube.days)
    out = np.zeros((cube.n_sta, cube.n_days), dtype=bool)
    pos = {str(s): i for i, s in enumerate(cube.sta)}
    q = steps[(steps.kind == "quake") & (steps.mag >= EVENT_MIN_MAG)
              & (steps.date >= days[0]) & (steps.date <= days[-1])]
    half = pd.Timedelta(days=EVENT_HALF_WINDOW_DAYS)
    for sta, date in zip(q.sta, q.date):
        if sta in pos:
            out[pos[sta], (days >= date - half) & (days <= date + half)] = True
    return out


def scoring_qc(cube: Cube, steps: pd.DataFrame, later_days: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Stations left out of scoring for site trouble after the training
    years, as gnssdl: day-to-day scatter more than 3× the training years',
    or a jump of more than 50/50/100 mm between consecutive 30-day medians
    with no listed step within 45 days. Uses only each station's own data and
    the step list, so it is the same for every method. `later_days` (T
    bool) are the days checked, by default every day after the training
    years. Returns (S bool keep, table of the dropped)."""
    days = pd.DatetimeIndex(cube.days)
    train = (cube.split_day == TRAIN)[None, :] & cube.avail & ~cube.exclude
    later_days = cube.split_day > TRAIN if later_days is None else later_days
    later = later_days[None, :] & cube.avail & ~cube.exclude
    keep = np.ones(cube.n_sta, dtype=bool)
    dropped = []
    known = {s: pd.DatetimeIndex(g.date) for s, g in steps.groupby("sta")}
    tol = pd.Timedelta(days=QC_JUMP_STEP_TOLERANCE_DAYS)
    for i in range(cube.n_sta):
        sta = str(cube.sta[i])
        cols = np.flatnonzero(later[i])
        if len(cols) >= 60:
            monthly = pd.DataFrame(cube.r[i, cols], index=days[cols]).resample("30D").median()
            big = (monthly.diff().abs() > np.array(QC_JUMP_MM)).any(axis=1)
            k = known.get(sta, pd.DatetimeIndex([]))
            unexplained = [t for t in monthly.index[big]
                           if not ((k >= t - tol - pd.Timedelta(days=30)) & (k <= t + tol)).any()]
            if unexplained:
                keep[i] = False
                dropped.append({"sta": sta, "reason": "unexplained jump", "first": str(unexplained[0].date())})
                continue
        if train[i].sum() < 365 or later[i].sum() < 365:
            continue
        x = np.where(cube.avail[i][:, None], cube.r[i], np.nan).astype(np.float64)
        fast = fast_part(x)
        ratio = robust_scale(fast[later[i]]) / robust_scale(fast[train[i]])
        if np.nanmax(ratio) > QC_SCATTER_RATIO:
            keep[i] = False
            dropped.append({"sta": sta, "reason": "scatter increase", "max_ratio": round(float(np.nanmax(ratio)), 2)})
    return keep, pd.DataFrame(dropped)


# ---------------------------------------------------------------------------
# Per-station scores
# ---------------------------------------------------------------------------

def station_scores(truth: np.ndarray, samples: np.ndarray, cells: np.ndarray, skip: np.ndarray,
                   scale: np.ndarray) -> dict[str, float] | None:
    """Scores of one station over a period.

    truth: T×3 mm (NaN where missing); samples: m×T×3 mm; cells: T bool, the
    scored days; skip: T bool, days left out of the fast and slow parts;
    scale: 3, the station's normaliser. None with fewer than MIN_CELLS
    scored days."""
    if cells.sum() < MIN_CELLS or not np.all(scale > 0):
        return None
    m = len(samples)
    z = np.where(cells[:, None], truth / scale, np.nan)
    s = np.where(cells[None, :, None], samples / scale, np.nan)
    row: dict[str, float] = {"cells": int(cells.sum())}

    # Point error of the sample mean.
    err = s.mean(axis=0) - z
    split = cells & ~skip
    e_split = np.where(split[:, None], err, np.nan)
    slow = slow_part(e_split)
    fast = e_split - slow
    for k, c in enumerate(COMPONENTS):
        row[f"nrmse_total_{c}"] = _rms(err[cells, k])
        if split.sum() >= MIN_CELLS:
            row[f"nrmse_fast_{c}"] = _rms(fast[split, k])
            row[f"nrmse_slow_{c}"] = _rms(slow[split, k])

    # Probabilistic scores, total and fast part.
    zs = {"total": (z[cells], s[:, cells])}
    if split.sum() >= MIN_CELLS:
        z_split = np.where(split[:, None], z, np.nan)
        s_split = np.where(split[None, :, None], s, np.nan)
        zf = fast_part(z_split)[split]
        sf = fast_part(np.moveaxis(s_split, 0, 1))[split]           # days × m × 3
        zs["fast"] = (zf, np.moveaxis(sf, 1, 0))
    for part, (zz, ss) in zs.items():
        c_ = crps(ss, zz).mean(axis=0)
        for k, c in enumerate(COMPONENTS):
            row[f"crps_{part}_{c}"] = float(c_[k])
        if m > 1:
            for lev in LEVELS:
                row[f"cover{int(lev * 100)}_{part}"] = float(inside(ss, zz, lev).mean())
    zz, ss = zs["total"]
    row["energy_total"] = float(energy_score(ss, zz).mean())
    if m > 1:
        spread = np.sqrt(ss.var(axis=0, ddof=1).mean(axis=0) * (m + 1) / m)
        for k, c in enumerate(COMPONENTS):
            row[f"spread_skill_{c}"] = float(spread[k] / row[f"nrmse_total_{c}"])

    # Realism: scale of single samples' fast parts, and spectra by band.
    if m > 1 and "fast" in zs:
        zf, sf = zs["fast"]
        ratio = np.log2(robust_scale(sf, axis=1) / robust_scale(zf, axis=0)[None])   # m×3
        for k, c in enumerate(COMPONENTS):
            row[f"scale_log2_{c}"] = float(np.median(ratio[:, k]))
        # Tails: the 99th percentile of |fast| over all samples against the truth's.
        tail = np.log2(np.quantile(np.abs(sf), TAIL_QUANTILE, axis=(0, 1))
                       / np.quantile(np.abs(zf), TAIL_QUANTILE, axis=0))
        for k, c in enumerate(COMPONENTS):
            row[f"tail_log2_{c}"] = float(tail[k])
        logr = []
        for a in range(0, len(z) - SEGMENT_DAYS + 1, SEGMENT_DAYS):
            seg = slice(a, a + SEGMENT_DAYS)
            pt = band_psd(z[seg])
            if pt is None:
                continue
            mask = np.isfinite(z[seg])
            ps = np.stack([band_psd(np.where(mask, s[j, seg], np.nan)) for j in range(m)])
            logr.append(np.log10(ps / pt[None]).mean(axis=0))             # 3×bands
        if logr:
            lr = np.mean(logr, axis=0)
            for k, c in enumerate(COMPONENTS):
                for b, (lo, hi) in enumerate(BANDS_DAYS):
                    row[f"psd_log10_{lo}_{hi}_{c}"] = float(lr[k, b])
    return row


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x ** 2)))


def neighbour_correlations(truth: np.ndarray, samples: np.ndarray, nbr: np.ndarray,
                           cells: np.ndarray) -> np.ndarray:
    """Correlation of the fast parts of the target with each neighbour over
    one window: truth and the mean over samples of each sample's.

    truth: L×3 mm, samples: m×L×3, nbr: K×L×3 neighbour series (NaN where
    missing), cells: L bool. Returns K×3×2 (truth, samples); NaN where fewer
    than MIN_CELLS common days."""
    z = fast_part(np.where(cells[:, None], truth, np.nan))
    s = fast_part(np.moveaxis(np.where(cells[None, :, None], samples, np.nan), 0, 1))   # L×m×3
    y = fast_part(np.moveaxis(nbr, 0, 1))                                               # L×K×3
    out = np.full((nbr.shape[0], 3, 2), np.nan)
    for j in range(nbr.shape[0]):
        ok = np.isfinite(z).all(axis=1) & np.isfinite(y[:, j]).all(axis=1)
        if ok.sum() < MIN_CELLS:
            continue
        for k in range(3):
            out[j, k, 0] = _corr(z[ok, k], y[ok, j, k])
            out[j, k, 1] = np.nanmean([_corr(s[ok, i, k], y[ok, j, k]) for i in range(s.shape[1])])
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise(per_station: pd.DataFrame) -> dict[str, float]:
    """Median over stations of every score column, plus the mean over E, N,
    U of the component medians (e.g. `nrmse_fast`); and calibration pooled
    over stations and by station noise level (`calibration`)."""
    out: dict[str, float] = {"stations": int(len(per_station))}
    if per_station.empty:
        return out
    out["cells"] = int(per_station["cells"].sum())
    groups: dict[str, list[float]] = {}
    for col in per_station.columns:
        if col in ("station", "cells", "noise"):
            continue
        med = per_station[col].median()
        out[col] = float(med)
        base, _, comp = col.rpartition("_")
        if comp in COMPONENTS:
            groups.setdefault(base, []).append(float(med))
    for base, meds in groups.items():
        out[base] = float(np.nanmean(meds))
    return out | calibration(per_station)


def calibration(per_station: pd.DataFrame) -> dict[str, float]:
    """Calibration that a median over stations hides.

    Pooled: spread/error over all scored days of all stations together,
    √(Σ cells·spread² / Σ cells·error²) per component, then the mean over
    components; and the 90% coverage of the fast part weighted by cells.
    By noise: the median spread/error, 90% coverage and fast CRPS of the
    stations in each quartile of `noise` (`station_noise`), quietest first.
    An ensemble that cannot tell how noisy a station is will be too wide
    for quiet stations and too narrow for noisy ones, yet right pooled."""
    out: dict[str, float] = {}
    if "spread_skill_e" not in per_station:
        return out
    w = per_station["cells"].to_numpy(float)
    ratios = []
    for c in COMPONENTS:
        err = per_station[f"nrmse_total_{c}"].to_numpy(float)
        spread = per_station[f"spread_skill_{c}"].to_numpy(float) * err
        ok = np.isfinite(spread) & np.isfinite(err)
        ratios.append(np.sqrt((w[ok] * spread[ok] ** 2).sum() / (w[ok] * err[ok] ** 2).sum()))
    out["spread_skill_pooled"] = float(np.mean(ratios))
    if "cover90_fast" in per_station:
        cov = per_station["cover90_fast"].to_numpy(float)
        ok = np.isfinite(cov)
        out["cover90_fast_pooled"] = float((w[ok] * cov[ok]).sum() / w[ok].sum())
    if "noise" in per_station and per_station["noise"].notna().sum() >= NOISE_GROUPS:
        group = pd.qcut(per_station["noise"], NOISE_GROUPS, labels=False)
        spread = per_station[[f"spread_skill_{c}" for c in COMPONENTS]].mean(axis=1)
        crps_fast = per_station[[f"crps_fast_{c}" for c in COMPONENTS]].mean(axis=1)
        for q in range(NOISE_GROUPS):
            sel = group == q
            out[f"noise_q{q + 1}"] = float(per_station.loc[sel, "noise"].median())
            out[f"spread_skill_q{q + 1}"] = float(spread[sel].median())
            out[f"crps_fast_q{q + 1}"] = float(crps_fast[sel].median())
            if "cover90_fast" in per_station:
                out[f"cover90_fast_q{q + 1}"] = float(per_station.loc[sel, "cover90_fast"].median())
    return out


def correlation_by_distance(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean neighbour correlation of truth and samples per distance bin,
    over all target–neighbour–component pairs (`pairs`: columns dist_km,
    corr_truth, corr_sample)."""
    bins = pd.cut(pairs["dist_km"], DIST_BINS_KM)
    g = pairs.groupby(bins, observed=True)[["corr_truth", "corr_sample"]]
    out = g.mean()
    out["pairs"] = g.size()
    out["difference"] = out["corr_sample"] - out["corr_truth"]
    return out
