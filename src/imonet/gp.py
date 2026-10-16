"""A space–time Gaussian process for the target over a whole window
(docs/00-design.md §5): the `gp` baseline.

Each component is the sum of three independent parts, each separable in
space and time:

    common mode   spatially constant           temporal covariance T_cm(u)
    regional      Matérn 3/2 in distance (ℓ)   T_r(u)
    station       its own, no spatial share    λ² T_n(u)

each T(u) the sum of white noise, flicker noise (spectrum 1/f) and a
first-order Gauss–Markov process (correlation exp(−u/τ)): white + flicker
is the usual noise model of GNSS position series (Williams et al. 2004), and
the Gauss–Markov term carries the excess power at periods of one to a few
weeks that white + flicker leaves out (lags 6–16 days of the station part).
λ is a station's noise level relative to a typical
station (`metrics.station_noise_log`; 1 for the target unless given). A sum
of separable valid covariances is valid; space–time interaction (Gneiting
2002, β > 0) would act only on the regional part, which is small in
California (`imonet.kriging`), and is left out.

Fit (`fit_space_time`): ℓ comes from the kriging fit. The temporal spectra
(A, B, G for white, flicker and Gauss–Markov; τ by grid search) are fitted
to lag curves of the training years, measured on 256-day windows
detrended as everywhere else: the cross-covariance of station pairs 100–300
km apart (common mode), of pairs within 10 km minus that (regional, divided
by its Matérn factor), and each station's autocovariance minus both (station
part). The model's curves are computed exactly for a detrended window
(`detrended_lag_curve`), and are linear in A, B and G, so each fit is a small
non-negative least-squares problem for every τ.

Sampling (`GP.sample`): the target's series over the window is drawn from
its conditional distribution given every available neighbour value, with a
flat prior on each neighbour's mean and trend (Rasmussen & Williams 2006,
eq. 2.42): conditioning on the detrended neighbours as the data present
them. Samples are detrended afterwards like every method's.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from imonet.baselines import Method
from imonet.cube import TRAIN, Cube, pairwise_km
from imonet.data import MIN_TARGET_FRACTION, WINDOW_DAYS, detrend
from imonet.kriging import Covariance, matern32, neighbour_distances, step_windows

LAGS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
GM_DAYS = tuple(np.geomspace(2.0, 200.0, 25))
LAG0_WEIGHT = 5.0            # the variance must match; the shape of the curve follows
NEAR_KM = 10.0
FAR_KM = (100.0, 300.0)
PARTS = ("cm", "r", "n")
JITTER = 1e-6


@dataclass
class SpaceTime:
    """White (A) and flicker (B) levels of each part, per component, in
    scaled units² (the periodogram convention of `baselines.noise_psd`), the
    variance G and time constant τ (days) of its Gauss–Markov term, and the
    regional length ℓ in km."""
    white: dict[str, tuple[float, float, float]]
    flicker: dict[str, tuple[float, float, float]]
    ell_km: tuple[float, float, float]
    gm: dict[str, tuple[float, float, float]] | None = None
    gm_days: dict[str, tuple[float, float, float]] | None = None

    def autocov(self, part: str, c: int, length: int) -> np.ndarray:
        """Lags 0..length−1 of a part's temporal covariance (undetrended)."""
        acf = self.white[part][c] * _white_acf(length) + self.flicker[part][c] * _flicker_acf(length)
        if self.gm:
            acf = acf + self.gm[part][c] * _gm_acf(length, self.gm_days[part][c])
        return acf

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "SpaceTime":
        d = json.loads(path.read_text())
        parts = lambda key: {k: tuple(v) for k, v in d[key].items()} if d.get(key) else None
        return cls(parts("white"), parts("flicker"), tuple(d["ell_km"]), parts("gm"), parts("gm_days"))


def _white_acf(length: int) -> np.ndarray:
    out = np.zeros(length)
    out[0] = 1.0
    return out


def _gm_acf(length: int, tau_days: float) -> np.ndarray:
    return np.exp(-np.arange(length) / tau_days)


def _flicker_acf(length: int) -> np.ndarray:
    """Autocovariance of a process with spectrum 1/f on a circle of twice the
    window (circulant embedding, as `baselines.synthesise` draws it), so it
    is a valid covariance and matches the synthesised noise."""
    n = 2 * length
    f = np.fft.rfftfreq(n)
    p = np.where(f > 0, 1 / np.where(f > 0, f, 1.0), 0.0)
    return np.fft.irfft(p, n)[:length]


def toeplitz(acf: np.ndarray) -> np.ndarray:
    idx = np.abs(np.arange(len(acf))[:, None] - np.arange(len(acf))[None])
    return acf[idx]


def detrend_projector(length: int) -> np.ndarray:
    t = np.arange(length) - (length - 1) / 2
    h = np.stack([np.ones(length), t], axis=1)
    return np.eye(length) - h @ np.linalg.solve(h.T @ h, h.T)


def detrended_lag_curve(acf: np.ndarray, lags=LAGS) -> np.ndarray:
    """Mean over the window of the covariance at each lag, after each series
    has its mean and line removed over a full window."""
    p = detrend_projector(len(acf))
    d = p @ toeplitz(acf) @ p
    return np.array([np.diagonal(d, u).mean() for u in lags])


def empirical_lag_curves(cube: Cube, scale: np.ndarray, length: int = WINDOW_DAYS, stride: int | None = None,
                         lags=LAGS, near_km: float = NEAR_KM, far_km=FAR_KM) -> dict:
    """Mean covariance at each lag, per component, over training-year windows
    of the normal stations: each station with itself ("auto"), pairs within
    near_km ("near", with their mean distance) and pairs far_km apart
    ("far"). Station-windows with a listed step over one robust scale are
    left out, as for the kriging fit."""
    stride = stride or length // 2
    lags = [u for u in lags if u < length]
    ok_all = cube.trainable()
    steps = step_windows(cube, scale, length)
    days = np.flatnonzero(cube.split_day == TRAIN)
    dist = pairwise_km(cube.lat, cube.lon)
    sums = {k: np.zeros((len(lags), 3)) for k in ("auto", "near", "far")}
    counts = {k: np.zeros(len(lags)) for k in ("auto", "near", "far")}
    near_d = []
    for t0 in range(int(days[0]), int(days[-1]) - length + 2, stride):
        ok = ok_all[:, t0:t0 + length]
        use = ok.mean(axis=1) >= MIN_TARGET_FRACTION
        for s, ts in steps.items():
            if use[s] and any(t0 < t < t0 + length for t in ts):
                use[s] = False
        idx = np.flatnonzero(use)
        if len(idx) < 2:
            continue
        o = ok[idx].astype(np.float64)
        x = detrend(np.where(ok[idx][..., None], cube.r[idx, t0:t0 + length] / scale, 0.0), ok[idx]).astype(np.float64)
        d = dist[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        near = (d[iu] > 0) & (d[iu] <= near_km)
        far = (d[iu] > far_km[0]) & (d[iu] <= far_km[1])
        near_d.extend(d[iu][near])
        for li, u in enumerate(lags):
            n = o[:, :length - u] @ o[:, u:].T                           # common days at lag u
            have = n >= MIN_TARGET_FRACTION * (length - u)
            for c in range(3):
                prod = x[:, :length - u, c] @ x[:, u:, c].T
                cov = np.where(have, prod / np.maximum(n, 1), np.nan)
                sym = 0.5 * (cov + cov.T)                                # lag +u and −u
                diag = np.diag(cov)
                sums["auto"][li, c] += np.nansum(diag)
                for key, sel in (("near", near), ("far", far)):
                    v = sym[iu][sel]
                    sums[key][li, c] += np.nansum(v)
            counts["auto"][li] += np.isfinite(np.diag(np.where(have, 1.0, np.nan))).sum()
            hv = have[iu] & have.T[iu]
            counts["near"][li] += (hv & near).sum()
            counts["far"][li] += (hv & far).sum()
    out = {"lags": list(lags), "near_km": float(np.mean(near_d)) if near_d else near_km}
    for k in sums:
        out[k] = (sums[k] / np.maximum(counts[k], 1)[:, None]).tolist()
        out[f"{k}_n"] = counts[k].tolist()
    return out


def nnls_small(a: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Non-negative least squares for a handful of coefficients, by trying
    every subset of them free (the best feasible one is the solution)."""
    sw = np.sqrt(np.ones(len(y)) if w is None else w)
    aw, yw = a * sw[:, None], y * sw
    n = a.shape[1]
    best = (np.inf, np.zeros(n))
    for mask in range(1, 2 ** n):
        cols = [k for k in range(n) if mask >> k & 1]
        coef, *_ = np.linalg.lstsq(aw[:, cols], yw, rcond=None)
        if (coef < 0).any():
            continue
        full = np.zeros(n)
        full[cols] = coef
        err = ((aw @ full - yw) ** 2).sum()
        if err < best[0]:
            best = (err, full)
    return best[1]


def _fit_temporal(curve: np.ndarray, length: int, lags) -> tuple[float, float, float, float]:
    """(A, B, G, τ) with A·white + B·flicker + G·GM(τ) ≈ curve, all on
    detrended lag curves; τ by grid search, weights favouring lag 0."""
    w_curve = detrended_lag_curve(_white_acf(length), lags)
    f_curve = detrended_lag_curve(_flicker_acf(length), lags)
    weights = np.where(np.asarray(lags) == 0, LAG0_WEIGHT, 1.0)
    best = None
    for tau in GM_DAYS:
        g_curve = detrended_lag_curve(_gm_acf(length, tau), lags)
        a = np.stack([w_curve, f_curve, g_curve], axis=1)
        coef = nnls_small(a, curve, weights)
        err = (weights * (a @ coef - curve) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, coef, tau)
    _, (a_, b_, g_), tau = best
    return float(a_), float(b_), float(g_), float(tau)


def fit_space_time(curves: dict, spatial: Covariance, length: int = WINDOW_DAYS) -> SpaceTime:
    lags = curves["lags"]
    auto, near, far = (np.asarray(curves[k]) for k in ("auto", "near", "far"))
    fits = {p: [] for p in PARTS}
    for c in range(3):
        m_near = matern32(curves["near_km"] / spatial.ell_km[c])
        regional = (near[:, c] - far[:, c]) / max(m_near, 1e-3)
        parts = {"cm": far[:, c], "r": regional, "n": auto[:, c] - far[:, c] - regional}
        for p in PARTS:
            fits[p].append(_fit_temporal(parts[p], length, lags))
    pick = lambda k: {p: tuple(f[k] for f in fits[p]) for p in PARTS}
    return SpaceTime(pick(0), pick(1), spatial.ell_km, pick(2), pick(3))


class GP(Method):
    """Samples of the target over the window from the space–time GP,
    conditioned on every available neighbour value. target_noise "given"
    scales the target's station part by its noise level (`target_noise` in
    the batch); "unknown" uses a typical station's."""

    name = "gp"

    def __init__(self, model: SpaceTime, n_samples: int = 32, target_noise: str = "unknown"):
        self.model, self.n_samples, self.target_noise = model, n_samples, target_noise

    def prepare(self, cube, scale, network, ok):
        from imonet.metrics import station_noise_log
        self.noise_log = station_noise_log(cube)

    def sample(self, batch, rng):
        b, _, length, _ = batch["y"].shape
        out = np.zeros((b, self.n_samples, length, 3), dtype=np.float32)
        acf = {(p, c): self.model.autocov(p, c, length) for p in PARTS for c in range(3)}
        tpl = {key: toeplitz(v) for key, v in acf.items()}
        for i in range(b):
            slots = np.flatnonzero(batch["valid"][i])
            lam_n = np.ones((len(slots), 3))
            if hasattr(self, "noise_log"):
                lev = self.noise_log[batch["nbr"][i, slots]]
                lam_n = np.where(np.isfinite(lev), np.exp(lev), 1.0)
            lam_t = np.ones(3)
            if self.target_noise == "given" and batch.get("target_noise_ok", np.zeros(b, bool))[i]:
                lam_t = np.exp(batch["target_noise"][i].astype(np.float64))
            for c in range(3):
                out[i, :, :, c] = self._component(batch, i, slots, c, lam_n[:, c], lam_t[c], tpl, length, rng)
        return out

    def _component(self, batch, i, slots, c, lam_n, lam_t, tpl, length, rng) -> np.ndarray:
        m = self.model
        t_cm, t_r, t_n = tpl[("cm", c)], tpl[("r", c)], tpl[("n", c)]
        prior_tt = t_cm + t_r + lam_t ** 2 * t_n
        if not len(slots):
            return _draw(np.zeros(length), prior_tt, self.n_samples, rng)
        ok = batch["y_ok"][i, slots]                                      # K'×L
        kk, tt = np.nonzero(ok)                                           # observed (neighbour, day)
        if not len(kk):
            return _draw(np.zeros(length), prior_tt, self.n_samples, rng)
        y = batch["y"][i, slots][kk, tt, c].astype(np.float64)
        dnn = neighbour_distances(batch["geom"][i, slots])
        dtn = batch["geom"][i, slots, 2].astype(np.float64)
        r_nn = matern32(dnn / m.ell_km[c])
        r_tn = matern32(dtn / m.ell_km[c])
        k_oo = _lagged(t_cm, tt, tt) + r_nn[np.ix_(kk, kk)] * _lagged(t_r, tt, tt)
        k_oo += np.where(kk[:, None] == kk[None], (lam_n[kk] ** 2)[:, None] * _lagged(t_n, tt, tt), 0.0)
        k_to = _lagged(t_cm, np.arange(length), tt) + r_tn[kk][None] * _lagged(t_r, np.arange(length), tt)
        n = len(y)
        k_oo[np.diag_indices(n)] += JITTER * np.trace(k_oo) / n
        # Flat prior on each neighbour's mean and trend (R&W eq. 2.42).
        tc = (tt - (length - 1) / 2) / length
        h = np.zeros((2 * len(slots), n))
        h[2 * kk, np.arange(n)] = 1.0
        h[2 * kk + 1, np.arange(n)] = tc
        h = h[np.abs(h).sum(1) > 0]
        # One Cholesky factorisation for every right-hand side (y, H, K_to).
        rhs = np.concatenate([y[:, None], h.T, k_to.T], axis=1)
        chol = torch.linalg.cholesky(torch.from_numpy(k_oo))
        sol = torch.cholesky_solve(torch.from_numpy(rhs), chol).numpy()
        p_ = len(h)
        ki_y, ki_h, ki_kot = sol[:, 0], sol[:, 1:1 + p_], sol[:, 1 + p_:]   # n, n×p, n×L
        a = h @ ki_h                                                      # p×p
        beta = np.linalg.solve(a, h @ ki_y)
        mean = k_to @ (ki_y - ki_h @ beta)
        r = -(h @ ki_kot)                                                 # p×L (target has no basis)
        cov = prior_tt - k_to @ ki_kot + r.T @ np.linalg.solve(a, r)
        return _draw(mean, cov, self.n_samples, rng)


def _lagged(tpl: np.ndarray, ta: np.ndarray, tb: np.ndarray) -> np.ndarray:
    return tpl[np.ix_(ta, tb)]


def _draw(mean: np.ndarray, cov: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    return (mean[None] + rng.normal(size=(m, len(mean))) @ root.T).astype(np.float32)
