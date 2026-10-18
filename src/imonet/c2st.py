"""Classifier two-sample test (C2ST; Lopez-Paz & Oquab 2017) of generated
stations against real ones (docs/00-design.md §6.1, "indistinguishability").

Following the paper (§2): pool real and generated examples labelled 0 and
1, split them into disjoint training and test sets, train a binary
classifier on the first, and take its accuracy t̂ on the second. Under
H0 (real and generated equal in distribution) n_te·t̂ is Binomial(n_te,
1/2), which gives an exact p-value; an accuracy near 0.5 means the
classifier cannot tell them apart. Its predictions show where they differ.

Here:

- Examples are 256-day windows of the held-out California stations in the
  validation years, each window once real and once generated (one sample),
  so the classes are balanced and paired. Both are detrended per window and
  divided by the component scale, as the model's inputs; the generated
  window keeps only the real one's days with data, and is screened for
  outliers as the cube's series were, so that neither gaps nor spikes the
  cube removed give it away.
- Training windows start every STRIDE days; test windows do not overlap
  (one per WINDOW_DAYS), so that test examples are close to independent.
- The split is by station, never by window: a station's windows are all in
  training or all in test, so the classifier cannot recognise a station.
  Two folds swap the roles (cross-fitting), so every station is tested once;
  t̂ pools both folds' test predictions. A fifth of each training fold's
  stations is kept for early stopping.
- Variants: "marginal" sees the window only (does it look like a GNSS
  series?); "conditional" also sees the N_NEIGHBOURS nearest neighbours'
  windows (is it consistent with its surroundings?).

Results: runs/C2ST/<label>/<density>/<variant>/{summary.json,
examples.csv}; the generated windows are kept in runs/C2ST/<label>/<density>/
windows.npz, so that both variants and reruns use the same samples.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from imonet.cube import HELD_OUT, VALIDATION
from imonet.data import WINDOW_DAYS, detrend
from imonet.evaluate import RESULTS, Setup, method_for, network, predict, screened
from imonet.metrics import fast_part, slow_part
from imonet.progress import progress_line
from imonet.significance import binomial_two_sided

STRIDE = 64
N_NEIGHBOURS = 4
VARIANTS = ("marginal", "conditional")


@dataclass
class TrainConfig:
    classifier: str = "features"   # "features" (logistic regression on summary features) or "cnn"
    epochs: int = 60
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    width: int = 16                # CNN channels in the first block
    dropout: float = 0.3
    crop: int = 192                # CNN training crops (days); tested on the whole window
    patience: int = 8              # epochs without validation improvement before stopping
    val_share: float = 0.2         # of the training fold's stations (all their windows), for early stopping
    seed: int = 0


# --------------------------------------------------------------------------- data


def make_windows(setup: Setup, method_name: str, density: float | None, seed: int = 0, root: Path = RESULTS,
                 model_dir: Path | None = None, target_noise: str = "unknown", progress=None) -> dict:
    """Real and generated windows of the held-out stations in the validation
    years: dict of arrays real, fake (N×L×3, scaled, 0 where missing), ok
    (N×L), nbr (N×K×L×3), nbr_ok (N×K×L), station (N), t0 (N), test (N bool:
    on the non-overlapping grid)."""
    cube = setup.cube
    method, _ = method_for(setup, method_name, density, 1, root, model_dir, target_noise=target_noise)
    net = network(cube, density, seed)
    targets = cube.stations(HELD_OUT)
    days = np.flatnonzero(cube.split_day == VALIDATION)
    a, b = int(days[0]), int(days[-1]) + 1
    out = {k: [] for k in ("real", "fake", "ok", "nbr", "nbr_ok", "station", "t0", "test")}
    for off in range(0, WINDOW_DAYS, STRIDE):
        wins = [(t, t) for t in range(a + off, b - WINDOW_DAYS + 1, WINDOW_DAYS)]
        if not wins:
            continue
        pred = predict(method, cube, setup.scale, targets, net, None, seed=seed + off,
                       noise_log=setup.noise_log, windows=wins)
        pred, _ = screened(pred)                      # as the cube's real series were
        for w in pred.windows:
            s, t0 = int(pred.targets[w["ti"]]), int(w["t0"])
            if not setup.keep[s]:
                continue
            cols = slice(t0 - pred.days[0], t0 - pred.days[0] + WINDOW_DAYS)
            real = pred.truth[w["ti"], cols] / setup.scale
            ok = np.isfinite(real).all(axis=1)
            if ok.sum() < WINDOW_DAYS // 2:
                continue
            fake = pred.samples[0, w["ti"], cols] / setup.scale
            nb = np.asarray(w["nbr"][:N_NEIGHBOURS], dtype=int)
            nbr = np.zeros((N_NEIGHBOURS, WINDOW_DAYS, 3), dtype=np.float32)
            nbr_ok = np.zeros((N_NEIGHBOURS, WINDOW_DAYS), dtype=bool)
            if len(nb):
                win = slice(t0, t0 + WINDOW_DAYS)
                nok = cube.avail[nb, win]
                nbr[:len(nb)] = detrend(np.where(nok[..., None], cube.r[nb, win] / setup.scale, 0.0), nok)
                nbr_ok[:len(nb)] = nok
            out["real"].append(np.where(ok[:, None], real, 0.0).astype(np.float32))
            out["fake"].append(np.where(ok[:, None], fake, 0.0).astype(np.float32))
            out["ok"].append(ok)
            out["nbr"].append(nbr)
            out["nbr_ok"].append(nbr_ok)
            out["station"].append(str(cube.sta[s]))
            out["t0"].append(t0)
            out["test"].append(off == 0)
        if progress:
            progress(f"  windows starting {off} days into the period: {len(out['t0'])} so far")
    return {k: np.asarray(v) for k, v in out.items()}


def features(w: dict, idx: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Classifier inputs and labels for windows idx: each window twice, real
    (label 0) and generated (1). Channels: the series (3) and its mask (1);
    the conditional variant adds each neighbour's series and mask."""
    def stack(series):
        ch = [np.moveaxis(series[idx], -1, 1), w["ok"][idx][:, None].astype(np.float32)]
        if variant == "conditional":
            nb = w["nbr"][idx]                                             # n×K×L×3
            ch += [np.moveaxis(nb, -1, 2).reshape(len(idx), -1, WINDOW_DAYS),
                   w["nbr_ok"][idx].astype(np.float32)]
        return np.concatenate(ch, axis=1).astype(np.float32)
    x = np.concatenate([stack(w["real"]), stack(w["fake"])])
    y = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))]).astype(np.float32)
    return x, y


# --------------------------------------------------------------------------- classifier


class Classifier(nn.Module):
    """A small 1D CNN: four stride-2 convolution blocks, global average and
    max pooling, one logit."""

    def __init__(self, cin: int, width: int = 16, dropout: float = 0.3):
        super().__init__()
        chans = [cin, width, 2 * width, 2 * width, 4 * width]
        layers = []
        for c0, c1 in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv1d(c0, c1, 7, stride=2, padding=3), nn.GroupNorm(8, c1), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * chans[-1], 1))

    def forward(self, x):
        h = self.body(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1)).squeeze(1)


def augment(x: torch.Tensor, crop: int, g: torch.Generator) -> torch.Tensor:
    """The same kind of random change for real and generated examples: a
    random crop of `crop` days, and per example a sign flip of each component
    and a swap of east and north, applied to the target and its neighbours
    alike."""
    b, c, length = x.shape
    start = int(torch.randint(0, length - crop + 1, (1,), generator=g))
    x = x[..., start:start + crop]
    sign = (torch.randint(0, 2, (b, 3), generator=g) * 2 - 1).to(x.device, x.dtype)
    swap = (torch.rand(b, generator=g) < 0.5).to(x.device)
    out = x.clone()
    for lo, hi in _series_slices(c):
        part = x[:, lo:hi] * sign[:, :, None]
        out[:, lo:hi] = torch.where(swap[:, None, None], part[:, [1, 0, 2]], part)
    return out


def _series_slices(c: int) -> list[tuple[int, int]]:
    """Channel ranges of the (E, N, U) series in a feature stack: the target's
    first three channels, then (conditional) N_NEIGHBOURS × 3 neighbour
    channels after the target's mask."""
    out = [(0, 3)]
    if c > 4:
        out += [(4 + 3 * k, 4 + 3 * k + 3) for k in range(N_NEIGHBOURS)]
    return out


def train_classifier(x_tr, y_tr, x_va, y_va, cfg: TrainConfig, device, log=None) -> nn.Module:
    torch.manual_seed(cfg.seed)
    model = Classifier(x_tr.shape[1], cfg.width, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x_tr).to(device), torch.from_numpy(y_tr).to(device)
    xv, yv = torch.from_numpy(x_va).to(device), torch.from_numpy(y_va).to(device)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    best, best_state, since, t0 = math.inf, None, 0, time.time()
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(xt), generator=g).to(device)
        total = 0.0
        for i in range(0, len(order), cfg.batch):
            j = order[i:i + cfg.batch]
            loss = loss_fn(model(augment(xt[j], cfg.crop, g)), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            lv = model(xv)
            val = loss_fn(lv, yv).item()
            acc = ((lv > 0).float() == yv).float().mean().item()
        if log:
            log(progress_line(epoch + 1, cfg.epochs, t0, "epochs")
                + f", train loss {total / len(xt):.4f}, val loss {val:.4f}, val acc {acc:.3f}")
        if val < best - 1e-4:
            best, best_state, since = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


# --------------------------------------------------------------------------- feature classifier

FEATURE_BANDS = ((2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128))   # periods, days


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation along axis 1 over finite pairs (n×L×3 → n×3)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    n = ok.sum(axis=1).clip(min=1)
    ma, mb = a.sum(1) / n, b.sum(1) / n
    a, b = np.where(ok, a - ma[:, None], 0.0), np.where(ok, b - mb[:, None], 0.0)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def series_features(x: np.ndarray, nbr: np.ndarray | None = None) -> np.ndarray:
    """Summary features of windows x (n×L×3, NaN where missing): per
    component the log robust scale of the fast and of the slow part, the
    log ratio of the 99th percentile of |fast| to its scale, the lag-1 and
    lag-7 autocorrelation of the fast part, and log power in FEATURE_BANDS.
    With `nbr` (n×L×3, the neighbours' mean, NaN where none) also the
    correlation of fast and slow parts with it and the log ratio of fast
    scales."""
    n, length, _ = x.shape
    flat = np.moveaxis(x, 1, 0).reshape(length, -1)                     # L × (n·3)
    slow = np.moveaxis(slow_part(flat).reshape(length, n, 3), 0, 1)
    fast = x - slow
    def rscale(v):
        med = np.nanmedian(v, axis=1, keepdims=True)
        return 1.4826 * np.nanmedian(np.abs(v - med), axis=1) + 1e-6
    sf = rscale(fast)
    out = [np.log(sf), np.log(rscale(slow)),
           np.log(np.nanquantile(np.abs(fast), 0.99, axis=1) / sf + 1e-6),
           _corr(fast[:, 1:], fast[:, :-1]), _corr(fast[:, 7:], fast[:, :-7])]
    spec = np.abs(np.fft.rfft(np.nan_to_num(x - np.nanmean(x, axis=1, keepdims=True)), axis=1)) ** 2
    freq = np.fft.rfftfreq(length)
    for lo, hi in FEATURE_BANDS:
        band = (freq > 1 / hi) & (freq <= 1 / lo)
        out.append(np.log(spec[:, band].mean(axis=1) + 1e-9))
    if nbr is not None:
        nflat = np.moveaxis(nbr, 1, 0).reshape(length, -1)
        nslow = np.moveaxis(slow_part(nflat).reshape(length, n, 3), 0, 1)
        nfast = nbr - nslow
        out += [_corr(fast, nfast), _corr(slow, nslow), np.log(sf / rscale(nfast))]
    return np.concatenate([np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in out], axis=1)


def window_features(w: dict, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """(real, generated) feature matrices, one row per window."""
    ok = w["ok"][..., None]
    nbr = None
    if variant == "conditional":
        nb = np.where(w["nbr_ok"][..., None], w["nbr"], np.nan)
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            nbr = np.nanmean(nb, axis=1)
    real = series_features(np.where(ok, w["real"], np.nan), nbr)
    fake = series_features(np.where(ok, w["fake"], np.nan), nbr)
    return real, fake


def fit_logistic(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression (standardised inputs), by L-BFGS."""
    xt, yt = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    wgt = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([wgt, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(xt @ wgt + b, yt) + lam * (wgt ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return wgt.detach().numpy(), float(b.detach())


def run(w: dict, variant: str, cfg: TrainConfig = TrainConfig(), device=None, log=None) -> tuple[dict, pd.DataFrame]:
    """Two-fold cross-fitted C2ST on windows w: (summary, per-example test
    predictions)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)
    stations = np.array(sorted(set(w["station"])))
    fold = dict(zip(stations, rng.permutation(len(stations)) % 2))
    st_fold = np.array([fold[s] for s in w["station"]])
    rows, feats_cache = [], None
    for k in (0, 1):
        train_st = stations[[fold[s] != k for s in stations]]
        val_st = set(rng.choice(train_st, max(1, int(round(cfg.val_share * len(train_st)))), replace=False))
        tr = np.flatnonzero((st_fold != k) & ~np.isin(w["station"], list(val_st)))
        va = np.flatnonzero(np.isin(w["station"], list(val_st)))
        te = np.flatnonzero((st_fold == k) & w["test"])
        if log:
            log(f"  fold {k}: {len(tr)} training, {len(va)} early-stopping and {len(te)} test windows")
        if cfg.classifier == "features":
            fr, ff = feats_cache if feats_cache is not None else window_features(w, variant)
            feats_cache = (fr, ff)
            stack = lambda idx: (np.concatenate([fr[idx], ff[idx]]),
                                 np.concatenate([np.zeros(len(idx)), np.ones(len(idx))]))
            (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = stack(tr), stack(va), stack(te)
            mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
            z = lambda v: (v - mu) / sd
            best = None
            for lam in (1e-4, 1e-3, 1e-2, 1e-1):          # chosen on the early-stopping stations
                wgt, b0 = fit_logistic(z(x_tr), y_tr, lam)
                pv = 1 / (1 + np.exp(-(z(x_va) @ wgt + b0)))
                lv = -np.mean(y_va * np.log(pv + 1e-12) + (1 - y_va) * np.log(1 - pv + 1e-12))
                if best is None or lv < best[0]:
                    best = (lv, lam, wgt, b0)
            _, lam, wgt, b0 = best
            if log:
                log(f"  fold {k}: logistic regression, λ = {lam:g}, early-stopping loss {best[0]:.4f}")
            p = 1 / (1 + np.exp(-(z(x_te) @ wgt + b0)))
        else:
            model = train_classifier(*features(w, tr, variant), *features(w, va, variant), cfg, device, log)
            x_te, y_te = features(w, te, variant)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(x_te).to(device))).cpu().numpy()
        for i, (prob, lab) in enumerate(zip(p, y_te)):
            j = te[i % len(te)]
            rows.append({"station": w["station"][j], "t0": int(w["t0"][j]), "fold": k, "generated": int(lab),
                         "p_generated": float(prob)})
    ex = pd.DataFrame(rows)
    correct = ((ex.p_generated > 0.5).astype(int) == ex.generated)
    n, c = len(ex), int(correct.sum())
    # station-block bootstrap of the accuracy (windows of a station are not independent)
    by_station = correct.groupby(ex.station).agg(["sum", "count"])
    boot = []
    for _ in range(2000):
        pick = by_station.iloc[rng.integers(len(by_station), size=len(by_station))]
        boot.append(pick["sum"].sum() / pick["count"].sum())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    summary = {"variant": variant, "classifier": cfg.classifier, "test_examples": n, "accuracy": c / n, "ci_low": float(lo), "ci_high": float(hi),
               "p_value": binomial_two_sided(c, n), "stations": int(len(stations)),
               "accuracy_real": float(correct[ex.generated == 0].mean()),
               "accuracy_generated": float(correct[ex.generated == 1].mean())}
    return summary, ex
