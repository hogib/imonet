"""Baselines (docs/00-design.md §5).

A method turns a batch of contexts (`imonet.data.context`, stacked on a
leading batch axis) into samples of the target, in scaled units:
`sample(batch, rng) -> B×m×L×3`. Methods that need more than the context
get the cube, scale and network through `prepare` before predicting.

- `zero`: predict zero.
- `idw`: weighted mean of the neighbours on each day,
  w = exp(−d²/2ℓ²) / σ̄², σ̄ the neighbour's median formal σ over the window
  (gnssdl's M1). ℓ is tuned on validation per density.
- `idw+noise`: `idw` plus white + flicker noise, independent per component,
  with amplitudes fitted to the neighbours' departures from their own `idw`
  predictions in the same window and network.
- `krige`: the kriging mean of each day (`imonet.kriging`), with a covariance
  fitted once to the training years.
- `krige+noise`: `krige` plus noise fitted as for `idw+noise`, to the
  neighbours' departures from their own kriging predictions.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from imonet.cube import Cube
from imonet.data import context
from imonet.kriging import Covariance, krige

LENGTH_SCALES_KM = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0, math.inf)
FIT_FREQ = (1 / 64, 0.5)    # cycles/day: band of the white + flicker fit


class Method:
    name = ""
    n_samples = 1

    def prepare(self, cube: Cube, scale: np.ndarray, network: np.ndarray, ok: np.ndarray) -> None:
        pass

    def sample(self, batch: dict, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class Zero(Method):
    name = "zero"

    def sample(self, batch, rng):
        b, _, length, _ = batch["y"].shape
        return np.zeros((b, 1, length, 3), dtype=np.float32)


def idw_mean(batch: dict, length_km: float) -> np.ndarray:
    """B×L×3 weighted neighbour mean; 0 on days with no neighbour."""
    y, ok, valid = batch["y"], batch["y_ok"], batch["valid"]
    d = batch["geom"][..., 2].astype(np.float64)                          # B×K
    kern = np.ones_like(d) if math.isinf(length_km) else np.exp(-d ** 2 / (2 * length_km ** 2))
    kern = np.where(valid, kern, 0.0)
    sig = np.where(ok[..., None], batch["y_sigma"], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # padded slots: all NaN
        sbar = np.nanmedian(sig, axis=2)                                  # B×K×3
    w = np.nan_to_num(kern[..., None] / sbar ** 2)                        # B×K×3
    wt = w[:, :, None, :] * ok[..., None]                                 # B×K×L×3
    num = (wt * y).sum(axis=1)
    den = wt.sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).astype(np.float32)


class IDW(Method):
    name = "idw"

    def __init__(self, length_km: float):
        self.length_km = length_km

    def sample(self, batch, rng):
        return idw_mean(batch, self.length_km)[:, None]


def noise_psd(e: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """White and flicker levels (A, B) of each column of e (L×C, used where
    ok) for the model P(f) = A + B/f of the periodogram |FFT|²/L.

    A and B are the least-squares fit of the periodogram on [1, 1/f] over
    FIT_FREQ, held non-negative (a negative term is dropped and the other
    refitted). Fitting both at once matters: flicker still has power at high
    frequencies, so reading A off the high end alone overstates it. Raw
    periodogram values are exponentially distributed around the model, so
    least squares (a mean) is unbiased where a median would not be. Gaps are
    zero-filled after removing the mean, and the periodogram is divided by
    the share of days present."""
    n = len(e)
    frac = max(ok.mean(), 1e-9)
    x = np.where(ok[:, None], e - (e * ok[:, None]).sum(0) / max(ok.sum(), 1), 0.0)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2 / n / frac
    f = np.fft.rfftfreq(n)
    sel = (f >= FIT_FREQ[0]) & (f <= FIT_FREQ[1])
    p, g = p[sel], 1 / f[sel]
    design = np.stack([np.ones_like(g), g], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, p, rcond=None)
    only_b = (p * g[:, None]).sum(0) / (g ** 2).sum()
    only_a = p.mean(axis=0)
    a, b = np.where(a < 0, 0.0, a), np.where(a < 0, only_b, b)
    a, b = np.where(b < 0, only_a, a), np.where(b < 0, 0.0, b)
    return np.maximum(a, 0.0), np.maximum(b, 0.0)


def synthesise(a: np.ndarray, b: np.ndarray, length: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """m×L×C noise with periodogram expectation A + B/f per column, made on
    twice the length and cut, so the window does not wrap around."""
    n = 2 * length
    f = np.fft.rfftfreq(n)
    p = a[None, :] + b[None, :] / np.where(f > 0, f, np.inf)[:, None]      # F×C
    p[0] = 0.0
    z = rng.normal(size=(m, len(f), len(a))) + 1j * rng.normal(size=(m, len(f), len(a)))
    x = np.fft.irfft(z * np.sqrt(p * n / 2)[None], n=n, axis=1)
    return x[:, :length].astype(np.float32)


class MeanPlusNoise(Method):
    """A point predictor's mean plus white + flicker noise per component,
    with amplitudes fitted to the neighbours' departures from their own
    predictions by the same predictor, in the same window and network."""

    def __init__(self, n_samples: int = 32):
        self.n_samples = n_samples

    def mean(self, batch: dict) -> np.ndarray:
        raise NotImplementedError

    def prepare(self, cube, scale, network, ok):
        self.cube, self.scale, self.network, self.ok = cube, scale, network, ok

    def departures(self, batch: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Each valid neighbour's departure from its own prediction in
        sample i's window: (K'×L×3, K'×L ok)."""
        t0, length = int(batch["t0"][i]), batch["y"].shape[2]
        es, oks = [], []
        for slot in np.flatnonzero(batch["valid"][i]):
            j = int(batch["nbr"][i, slot])
            net = self.network[(self.network != j) & (self.network != batch["target"][i])]
            ctx = context(self.cube, self.scale, self.cube.lat[j], self.cube.lon[j], t0, net, self.ok,
                          length, batch["y"].shape[1])
            pred = self.mean({k: v[None] for k, v in ctx.items()})[0]
            es.append(batch["y"][i, slot] - pred)
            oks.append(batch["y_ok"][i, slot])
        return np.array(es), np.array(oks)

    def sample(self, batch, rng):
        mean = self.mean(batch)
        b, length = mean.shape[0], mean.shape[1]
        out = np.repeat(mean[:, None], self.n_samples, axis=1)
        for i in range(b):
            es, oks = self.departures(batch, i)
            if not len(es):
                continue
            ab = [noise_psd(e, ok) for e, ok in zip(es, oks) if ok.mean() > 0.5]
            if not ab:
                continue
            a = np.median([x[0] for x in ab], axis=0)
            bb = np.median([x[1] for x in ab], axis=0)
            out[i] += synthesise(a, bb, length, self.n_samples, rng)
        return out


class IDWNoise(MeanPlusNoise):
    name = "idw+noise"

    def __init__(self, length_km: float, n_samples: int = 32):
        super().__init__(n_samples)
        self.length_km = length_km

    def mean(self, batch):
        return idw_mean(batch, self.length_km)


class Krige(Method):
    name = "krige"

    def __init__(self, covariance: Covariance):
        self.covariance = covariance

    def sample(self, batch, rng):
        return krige(batch, self.covariance)[0][:, None]


class KrigeNoise(MeanPlusNoise):
    name = "krige+noise"

    def __init__(self, covariance: Covariance, n_samples: int = 32):
        super().__init__(n_samples)
        self.covariance = covariance

    def mean(self, batch):
        return krige(batch, self.covariance)[0]


def make(name: str, length_km: float | None = None, n_samples: int = 32,
         covariance: Covariance | None = None) -> Method:
    if name == "zero":
        return Zero()
    if name == "idw":
        return IDW(length_km)
    if name == "idw+noise":
        return IDWNoise(length_km, n_samples)
    if name == "krige":
        return Krige(covariance)
    if name == "krige+noise":
        return KrigeNoise(covariance, n_samples)
    raise ValueError(f"unknown method {name!r}")
