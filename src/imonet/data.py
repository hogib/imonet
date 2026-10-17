"""Samples: a target over a window of days, with the nearest stations of a
network as context (docs/00-design.md §4.1–4.2).

All series are in scaled units: residuals and σ divided by a fixed
per-component scale (`component_scale`), missing values set to 0 with a
mask beside them.

Every series in a window, target and neighbours, has its own least-squares
mean and line over its available days removed (`detrend`). In the
validation and test years the residuals carry an offset and trend that the
training years do not: the trajectory was fitted on the training years, so
later residuals include its extrapolation error. That is trajectory, which
imonet does not generate (docs/00-design.md §2, §4.1).

A sample is a dict of numpy arrays, ready for torch's default collate:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x` | L×3 float32 | target series (training and validation only) |
| `x_ok` | L bool | target days in the loss |
| `y` | K×L×3 float32 | neighbour series |
| `y_ok` | K×L bool | neighbour days available |
| `y_sigma` | K×L×3 float32 | neighbour formal σ |
| `geom` | K×3 float32 | neighbour east, north offset and distance from the target, km |
| `valid` | K bool | slot holds a neighbour (the rest are padding, all 0) |
| `nbr` | K int64 | neighbour station index, −1 pads |
| `target`, `t0` | int | target station index (−1 for a coordinate), first day index |
| `doy0` | float32 | day of year of the first day |
| `target_noise` | 3 float32 | target station's log noise level per component (`station_noise_log`), 0 if unknown |
| `target_noise_ok` | bool | the target's noise level is known |
| `spacing_km`, `min_km` | float32 | network spacing and minimum target distance drawn |
| `planted` | bool | a synthetic transient was planted (`transients`, training only) |
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from imonet import transients
from imonet.cube import HELD_OUT, NORMAL, TRAIN, VALIDATION, Cube, pairwise_km
from imonet.metrics import station_noise_log
from imonet.network import K_NEIGHBOURS, MAX_NEIGHBOUR_KM, neighbours, spacing_km, thin

WINDOW_DAYS = 256
MIN_TARGET_FRACTION = 0.5       # target days with data, to use a window
MIN_NEIGHBOUR_FRACTION = 0.25   # neighbour days with data, to count as a neighbour
SPACING_KM = (10.0, 100.0)      # thinned network spacing, log-uniform
FULL_NETWORK_P = 0.2            # share of samples on the unthinned network
MIN_KM = (1.0, 100.0)           # minimum target–neighbour distance, log-uniform
BANK_LEVELS = 64                # spacings in a thinning bank
BANK_SEEDS = 4                  # thinnings per spacing


def component_scale(cube: Cube, ok: np.ndarray | None = None) -> np.ndarray:
    """Robust scale (1.4826 × MAD) of E, N, U over the cells in `ok`
    (default: the trainable cells). The residuals are heavy-tailed (kept
    earthquake offsets), so the standard deviation would be several times
    larger and set by a few events."""
    ok = cube.trainable() if ok is None else ok
    out = np.empty(3)
    for c in range(3):
        x = cube.r[..., c][ok]
        out[c] = 1.4826 * np.median(np.abs(x - np.median(x)))
    return out.astype(np.float32)


def context(
    cube: Cube, scale: np.ndarray, lat0: float, lon0: float, t0: int, network: np.ndarray, ok: np.ndarray,
    length: int = WINDOW_DAYS, k: int = K_NEIGHBOURS, min_km: float = 0.0, max_km: float = MAX_NEIGHBOUR_KM,
    min_fraction: float = MIN_NEIGHBOUR_FRACTION,
) -> dict[str, np.ndarray]:
    """Neighbour inputs for a target at (lat0, lon0) over days [t0, t0 + length).

    `network` holds the cube indices of the stations that may be neighbours
    and `ok` (S×T) the cells usable as input. A station counts only if at
    least `min_fraction` of the window is usable, so a slot is not spent on
    a station that is down."""
    win = slice(t0, t0 + length)
    net = network[ok[network, win].mean(axis=1) >= min_fraction]
    nb = neighbours(lat0, lon0, cube.lat[net], cube.lon[net], k=k, min_km=min_km, max_km=max_km)
    v = nb.valid
    sta = net[nb.idx[v]]
    m = len(sta)

    y = np.zeros((k, length, 3), dtype=np.float32)
    y_sigma = np.zeros((k, length, 3), dtype=np.float32)
    y_ok = np.zeros((k, length), dtype=bool)
    geom = np.zeros((k, 3), dtype=np.float32)
    y_ok[:m] = ok[sta, win]
    y[:m] = detrend(np.where(y_ok[:m, :, None], cube.r[sta, win] / scale, 0.0), y_ok[:m])
    y_sigma[:m] = np.where(y_ok[:m, :, None], cube.sigma[sta, win] / scale, 0.0)
    geom[:m] = np.stack([nb.east_km[v], nb.north_km[v], nb.dist_km[v]], axis=1)
    nbr = np.full(k, -1, dtype=np.int64)
    nbr[:m] = sta
    return {"y": y, "y_ok": y_ok, "y_sigma": y_sigma, "geom": geom, "valid": v, "nbr": nbr}


def target(cube: Cube, scale: np.ndarray, s: int, t0: int, ok: np.ndarray,
           length: int = WINDOW_DAYS) -> dict[str, np.ndarray]:
    """Station s's own series over the window, and the days in the loss."""
    win = slice(t0, t0 + length)
    x_ok = ok[s, win]
    x = detrend(np.where(x_ok[:, None], cube.r[s, win] / scale, 0.0), x_ok)
    return {"x": x, "x_ok": x_ok}


def detrend(x: np.ndarray, ok: np.ndarray, zero_missing: bool = True) -> np.ndarray:
    """x (…×L×C) minus its least-squares mean and line over the days in ok
    (…×L), fitted per series and component and removed on every day; then
    set to 0 where not ok, unless `zero_missing` is off (generated series,
    which have every day). With fewer than two days only the mean is
    removed. float32."""
    length = x.shape[-2]
    t = (np.arange(length) - (length - 1) / 2) / length                   # centred, O(1)
    w = ok.astype(np.float64)[..., None]                                    # …×L×1
    tt = t[:, None]
    n, st, stt = w.sum(-2), (w * tt).sum(-2), (w * tt ** 2).sum(-2)          # …×1
    sx, stx = (w * x).sum(-2), (w * tt * x).sum(-2)                         # …×C
    det = n * stt - st ** 2
    line = det > 1e-12 * np.maximum(n, 1) ** 2
    b = np.where(line, (n * stx - st * sx) / np.where(line, det, 1.0), 0.0)
    a = (sx - b * st) / np.maximum(n, 1)
    out = x - a[..., None, :] - b[..., None, :] * tt
    if zero_missing:
        out = np.where(ok[..., None], out, 0.0)
    return out.astype(np.float32)


def target_noise(noise_log: np.ndarray, s: int) -> dict[str, np.ndarray]:
    """The `target_noise` inputs of station s (S×3 `station_noise_log`)."""
    level = noise_log[s]
    ok = bool(np.isfinite(level).all())
    return {"target_noise": np.where(ok, level, 0.0).astype(np.float32), "target_noise_ok": np.bool_(ok)}


def day_of_year(day: np.datetime64) -> float:
    year = day.astype("datetime64[Y]")
    return float((day - year.astype("datetime64[D]")).astype(int))


class ThinningBank:
    """Precomputed disk thinnings of a network at spacings spread
    log-uniformly over a range, plus the network itself. Thinning takes up
    to ~0.1 s at small spacings, too slow to repeat for every sample.

    The unthinned network is drawn with probability `full_p`. Even the
    lightest disk thinning removes close pairs, and without these draws the
    model would never see the close neighbours of a full-density network
    (13% of held-out California stations have one within 5 km)."""

    def __init__(self, cube: Cube, network: np.ndarray, rng: np.random.Generator,
                 spacing_range: tuple[float, float] = SPACING_KM, levels: int = BANK_LEVELS,
                 seeds: int = BANK_SEEDS, full_p: float = FULL_NETWORK_P):
        d = pairwise_km(cube.lat[network], cube.lon[network])
        self.network, self.full_p = network, full_p
        self.full_spacing = spacing_km(d)
        self.spacing = np.repeat(np.geomspace(*spacing_range, levels), seeds).astype(np.float32)
        self.masks = np.stack([thin(d, float(s), rng) for s in self.spacing])

    def __len__(self) -> int:
        return len(self.spacing)

    def draw(self, rng: np.random.Generator) -> tuple[float, np.ndarray]:
        """(spacing, cube indices of the kept stations) of a random draw."""
        if rng.random() < self.full_p:
            return self.full_spacing, self.network
        i = int(rng.integers(len(self)))
        return float(self.spacing[i]), self.network[self.masks[i]]


class Windows(Dataset):
    """Random samples: a target station from `targets`, a window inside the
    days of `split` where the target has at least `min_target_fraction` of
    its days in `ok`, a thinned copy of `network` (without the target), and
    a minimum target distance (none on the unthinned network). With
    probability `transient_p` the sample gets a synthetic transient
    (`transients`), drawn after the sample itself.

    Sample i of epoch e is drawn from a generator seeded with (seed, e, i),
    so a sample does not depend on worker count or order.

    Training: targets and network the normal stations, split TRAIN,
    ok = cube.trainable(). Validation: targets the held-out stations,
    network the normal stations, split VALIDATION, ok = cube.avail."""

    def __init__(
        self, cube: Cube, scale: np.ndarray, targets: np.ndarray, network: np.ndarray, split: int,
        ok: np.ndarray, n_samples: int, seed: int, length: int = WINDOW_DAYS, k: int = K_NEIGHBOURS,
        spacing_range: tuple[float, float] = SPACING_KM, min_km: tuple[float, float] = MIN_KM,
        min_target_fraction: float = MIN_TARGET_FRACTION, bank: ThinningBank | None = None,
        noise_log: np.ndarray | None = None, transient_p: float = 0.0,
    ):
        self.cube, self.scale, self.ok = cube, np.asarray(scale, dtype=np.float32), ok
        self.noise_log = station_noise_log(cube) if noise_log is None else noise_log
        self.n_samples, self.seed, self.epoch = n_samples, seed, 0
        self.transient_p = transient_p
        self.length, self.k, self.min_km = length, k, min_km
        self.bank = bank or ThinningBank(cube, network, np.random.default_rng((seed, 1)), spacing_range)

        # Valid (station, first day) pairs: window inside the split, enough target data.
        in_split = (cube.split_day == split).astype(np.int32)
        whole = np.convolve(in_split, np.ones(length, dtype=np.int32), "valid") == length
        cs = np.zeros((len(targets), cube.n_days + 1), dtype=np.int32)
        np.cumsum(ok[targets], axis=1, out=cs[:, 1:])
        enough = (cs[:, length:] - cs[:, :-length]) >= min_target_fraction * length
        start = enough & whole[None, :]
        self.starts = np.flatnonzero(start).astype(np.int64)    # into targets × first day
        self.targets, self.n_first = targets, start.shape[1]
        if not len(self.starts):
            raise ValueError("no window meets the requirements")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        flat = self.starts[rng.integers(len(self.starts))]
        s, t0 = int(self.targets[flat // self.n_first]), int(flat % self.n_first)
        spacing, net = self.bank.draw(rng)
        full = len(net) == len(self.bank.network)
        net = net[net != s]
        # The full network mirrors full-density evaluation, which keeps
        # every neighbour; thinned networks also open a gap at the target.
        lo, hi = self.min_km
        min_km = 0.0 if full else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

        c = self.cube
        out = context(c, self.scale, c.lat[s], c.lon[s], t0, net, self.ok, self.length, self.k, min_km)
        out |= target(c, self.scale, s, t0, self.ok, self.length)
        out |= {"target": s, "t0": t0, "doy0": np.float32(day_of_year(c.days[t0])),
                "spacing_km": np.float32(spacing), "min_km": np.float32(min_km)}
        out |= target_noise(self.noise_log, s) | {"planted": np.bool_(False)}
        if self.transient_p and rng.random() < self.transient_p:
            self.plant(out, transients.draw(rng, self.length))
        return out

    def plant(self, out: dict, tr: transients.Transient) -> None:
        """Add transient tr to sample `out` in place, if enough neighbours
        see it. Detrending is linear, so the detrended transient is added
        to the detrended series."""
        v, length = out["valid"], self.length
        sig = tr.signal(out["geom"][v, 0], out["geom"][v, 1], length)       # m×L×3 mm
        if transients.seen(sig, out["y_ok"][v]).sum() < transients.MIN_SEEN:
            return
        out["y"][v] += detrend(sig / self.scale, out["y_ok"][v])
        own = tr.signal(np.zeros(1), np.zeros(1), length)[0]
        out["x"] += detrend(own / self.scale, out["x_ok"])
        out["planted"] = np.bool_(True)


def training_windows(cube: Cube, scale: np.ndarray, n_samples: int, seed: int, **kw) -> Windows:
    normal = cube.stations(NORMAL)
    return Windows(cube, scale, normal, normal, TRAIN, cube.trainable(), n_samples, seed, **kw)


def validation_windows(cube: Cube, scale: np.ndarray, n_samples: int, seed: int, **kw) -> Windows:
    """Fixed validation samples (epoch 0): held-out targets on validation
    days, predicted from the normal stations."""
    return Windows(cube, scale, cube.stations(HELD_OUT), cube.stations(NORMAL), VALIDATION, cube.avail,
                   n_samples, seed, **kw)
