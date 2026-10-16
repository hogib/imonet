"""Kriging of the daily residual field (docs/00-design.md §5).

Each component is a Gaussian process over the stations on every day, with
the stationary, isotropic covariance

    C(d) = s_cm + s_r · M(d / ℓ) + s_n · [same station],

M the Matérn ν = 3/2 correlation (Rasmussen & Williams 2006, eq. 4.17):
a common mode shared by the whole network (correlation still about 0.45 at
300 km), a regional part decaying with distance, and a station's own noise.
The common-mode term is a random constant per day with a fitted variance,
the unknown mean of ordinary kriging with a Gaussian prior (R&W eq. 2.40).

The covariance is fitted once, to the training years of the normal stations,
and serves every network density: it describes the field, not the network.
`empirical_covariance` averages, over 256-day windows detrended like the
model's inputs, the covariance of every station pair within 300 km by
distance class and the variance of every station; `fit_covariance` fits the
model to them by weighted least squares (weights: pair count over the
squared covariance, so every distance class counts by its relative error). Station-windows with a listed earthquake step of more
than one robust scale are left out, so that a few coseismic jumps do not set
the covariance of a Gaussian model; scoring leaves those days out too.

`krige` predicts each day from the neighbours with data that day
(R&W eqs. 2.25-2.26, Cholesky as their Algorithm 2.1); days with the same
neighbours share one factorisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from imonet.cube import TRAIN, Cube, pairwise_km
from imonet.data import WINDOW_DAYS, MIN_TARGET_FRACTION, detrend
from imonet.network import MAX_NEIGHBOUR_KM

DIST_BINS_KM = (0.0, 2, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300)
LENGTH_GRID_KM = np.geomspace(5.0, 2000.0, 80)
MIN_PAIR_SHARE = 0.5           # days in common, share of the window, for a pair to count


@dataclass
class Covariance:
    """Per component (E, N, U), in scaled units squared; ℓ in km."""
    s_cm: tuple[float, float, float]
    s_r: tuple[float, float, float]
    ell_km: tuple[float, float, float]
    s_n: tuple[float, float, float]

    def shared(self, d_km: np.ndarray, c: int) -> np.ndarray:
        """Covariance of the shared part (common mode + regional) at distance d."""
        return self.s_cm[c] + self.s_r[c] * matern32(np.asarray(d_km, dtype=np.float64) / self.ell_km[c])

    def total_variance(self, c: int) -> float:
        return self.s_cm[c] + self.s_r[c] + self.s_n[c]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Covariance":
        return cls(**{k: tuple(v) for k, v in json.loads(path.read_text()).items()})


def matern32(r: np.ndarray) -> np.ndarray:
    a = np.sqrt(3.0) * np.abs(r)
    return (1 + a) * np.exp(-a)


def step_windows(cube: Cube, scale: np.ndarray, length: int) -> dict[int, list[int]]:
    """Station → days of listed earthquake steps over one robust scale."""
    out: dict[int, list[int]] = {}
    for s, t, a in zip(cube.qstep_sta, cube.qstep_day, cube.qstep_amp):
        if (np.abs(a) / scale).max() >= 1:
            out.setdefault(int(s), []).append(int(t))
    return out


def empirical_covariance(cube: Cube, scale: np.ndarray, length: int = WINDOW_DAYS, stride: int | None = None,
                         max_km: float = MAX_NEIGHBOUR_KM, bins=DIST_BINS_KM) -> dict:
    """Mean pair covariance per distance class and mean station variance, per
    component, over windows of the training years (normal stations, cells
    `cube.trainable()`). {"d_km": n_bins, "cov": n_bins×3, "pairs": n_bins,
    "var": 3, "stations": int}."""
    stride = stride or length // 2
    ok_all = cube.trainable()
    steps = step_windows(cube, scale, length)
    days = np.flatnonzero(cube.split_day == TRAIN)
    dist = pairwise_km(cube.lat, cube.lon)
    edges = np.asarray(bins, dtype=np.float64)
    nb = len(edges) - 1
    sum_cov, sum_d, n_pairs = np.zeros((nb, 3)), np.zeros(nb), np.zeros(nb)
    sum_var, n_var = np.zeros(3), 0
    for t0 in range(int(days[0]), int(days[-1]) - length + 2, stride):
        ok = ok_all[:, t0:t0 + length]
        use = ok.mean(axis=1) >= MIN_TARGET_FRACTION
        for s, ts in steps.items():
            if use[s] and any(t0 < t < t0 + length for t in ts):
                use[s] = False
        idx = np.flatnonzero(use)
        if len(idx) < 2:
            continue
        o = ok[idx]
        x = detrend(np.where(o[..., None], cube.r[idx, t0:t0 + length] / scale, 0.0), o).astype(np.float64)
        of = o.astype(np.float64)
        n = of @ of.T
        d = dist[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        pair_ok = (n[iu] >= MIN_PAIR_SHARE * length) & (d[iu] > edges[0]) & (d[iu] <= min(edges[-1], max_km))
        b = np.digitize(d[iu][pair_ok], edges[1:-1], right=True)
        np.add.at(n_pairs, b, 1)
        np.add.at(sum_d, b, d[iu][pair_ok])
        for c in range(3):
            prod = x[..., c] @ x[..., c].T
            cov = prod[iu][pair_ok] / n[iu][pair_ok]
            np.add.at(sum_cov[:, c], b, cov)
            sum_var[c] += (np.diag(prod) / np.diag(n)).sum()
        n_var += len(idx)
    have = n_pairs > 0
    return {"d_km": (sum_d[have] / n_pairs[have]).tolist(), "cov": (sum_cov[have] / n_pairs[have, None]).tolist(),
            "pairs": n_pairs[have].tolist(), "var": (sum_var / max(n_var, 1)).tolist(), "station_windows": n_var}


def fit_covariance(emp: dict) -> Covariance:
    """Weighted least squares of C(d) = s_cm + s_r·M(d/ℓ) to the pair
    covariances (non-negative amplitudes, ℓ by grid search), weights
    pairs / C(d)²; the nugget s_n is the rest of the station variance."""
    d = np.asarray(emp["d_km"])
    cov, pairs, var = np.asarray(emp["cov"]), np.asarray(emp["pairs"]), np.asarray(emp["var"])
    params = {"s_cm": [], "s_r": [], "ell_km": [], "s_n": []}
    for c in range(3):
        y = cov[:, c]
        w = pairs / np.maximum(np.abs(y), 0.05 * var[c]) ** 2
        best = None
        for ell in LENGTH_GRID_KM:
            design = np.stack([np.ones_like(d), matern32(d / ell)], axis=1)
            coef = _nnls2(design, y, w)
            sse = (w * (design @ coef - y) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, coef, ell)
        _, (s_cm, s_r), ell = best
        params["s_cm"].append(float(s_cm))
        params["s_r"].append(float(s_r))
        params["ell_km"].append(float(ell))
        params["s_n"].append(float(max(var[c] - s_cm - s_r, 1e-6)))
    return Covariance(**{k: tuple(v) for k, v in params.items()})


def _nnls2(a: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares for two non-negative coefficients."""
    sw = np.sqrt(w)
    aw, yw = a * sw[:, None], y * sw
    coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    if (coef >= 0).all():
        return coef
    single = []
    for k in range(2):
        ck = max((aw[:, k] @ yw) / (aw[:, k] @ aw[:, k]), 0.0)
        c = np.zeros(2)
        c[k] = ck
        single.append(((((aw @ c) - yw) ** 2).sum(), c))
    return min(single, key=lambda t: t[0])[1]


def neighbour_distances(geom: np.ndarray) -> np.ndarray:
    """K×K km between neighbours, from their east and north offsets from the
    target (a local plane: within 300 km the error is well under 1%)."""
    en = geom[:, :2].astype(np.float64)
    return np.sqrt(((en[:, None] - en[None]) ** 2).sum(-1))


def krige(batch: dict, cov: Covariance) -> tuple[np.ndarray, np.ndarray]:
    """(mean, variance) of the target on every day, B×L×3 each, predicting
    the station's own value (its noise included). With no neighbour on a
    day: mean 0 and the full variance."""
    y, ok, valid, geom = batch["y"], batch["y_ok"], batch["valid"], batch["geom"]
    b, _, length, _ = y.shape
    mean = np.zeros((b, length, 3))
    var = np.empty((b, length, 3))
    for c in range(3):
        var[..., c] = cov.total_variance(c)
    for i in range(b):
        slots = np.flatnonzero(valid[i])
        if not len(slots):
            continue
        dnn = neighbour_distances(geom[i, slots])
        dtn = geom[i, slots, 2].astype(np.float64)
        avail = ok[i, slots]                                            # K'×L
        patterns, inverse = np.unique(avail.T, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for p, pattern in enumerate(patterns):
            if not pattern.any():
                continue
            days = np.flatnonzero(inverse == p)
            sel = np.flatnonzero(pattern)
            for c in range(3):
                k_nn = cov.shared(dnn[np.ix_(sel, sel)], c) + cov.s_n[c] * np.eye(len(sel))
                k_tn = cov.shared(dtn[sel], c)
                chol = np.linalg.cholesky(k_nn)
                v = np.linalg.solve(chol, k_tn)
                weights = np.linalg.solve(chol.T, v)                    # K_nn⁻¹ k_tn
                mean[i, days, c] = y[i, slots[sel]][:, days, c].T @ weights
                var[i, days, c] = cov.total_variance(c) - v @ v
    return mean.astype(np.float32), var.astype(np.float32)
