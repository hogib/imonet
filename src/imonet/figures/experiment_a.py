"""Experiment A figures and tables, read from runs/A (docs/00-design.md §6.1).

- scores_by_density: point, probabilistic and calibration scores against
  network density, one line per method (mean over thinning seeds, range as
  a band).
- correlation_by_distance: neighbour correlation of truth and of samples
  against distance, per generative method, at one density.
- spectra: log ratio of sample to truth power by period band, per
  generative method, at one density.
- example: one held-out station window, truth against samples.
- table: the main scores as a LaTeX table (booktabs).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imonet.cube import HELD_OUT, VALIDATION, Cube
from imonet.data import component_scale
from imonet.evaluate import DENSITIES, RESULTS, density_name, network, predict
from imonet.baselines import make
from imonet.figures.style import (BASELINE, COMPONENTS, INK, INK_2, METHODS, STYLE, SURFACE, WIDTH, method_style,
                                  save)
from imonet.metrics import BANDS_DAYS

DENSITY_ORDER = [density_name(d) for d in DENSITIES]


def load_runs(split: str = "validation", root: Path = RESULTS) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((root / "A").glob(f"*/{split}/*/summary.json"))]
    return pd.DataFrame(rows)


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    """The main methods present, in their fixed order. Variants (screened,
    guided) are compared in `variants` only."""
    present = set(df["method"])
    return [m for m in METHODS if m in present]


def _x(densities) -> np.ndarray:
    return np.array([DENSITY_ORDER.index(d) for d in densities])


PANELS = (
    ("nrmse_fast", "nRMSE, fast part", None),
    ("crps_fast", "CRPS, fast part", None),
    ("cover90_fast", "90% interval coverage, fast", 0.9),
    ("spread_skill", "Spread / error, total", 1.0),
)


def scores_by_density(df: pd.DataFrame, out: Path, name: str = "scores_by_density") -> Path:
    """`zero` does not depend on density and sits far above the other
    methods; drawn as a line it would flatten the differences that matter,
    so its value is written in the panel instead."""
    ticks = [d.removesuffix("km") for d in DENSITY_ORDER]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(PANELS), figsize=(WIDTH, 2.4))
        for ax, (col, title, ideal) in zip(axes, PANELS):
            for m in _ordered_methods(df):
                d = df[df.method == m]
                if col not in d or d[col].isna().all():
                    continue
                if m == "zero":
                    ax.text(0.03, 0.97, f"zero: {d[col].mean():.3f}", transform=ax.transAxes, va="top",
                            fontsize=7, color=INK_2)
                    continue
                g = d.groupby("density")[col].agg(["mean", "min", "max"]).reindex(DENSITY_ORDER).dropna()
                x, st = _x(g.index), method_style(m)
                ax.fill_between(x, g["min"], g["max"], color=st["color"], alpha=0.15, lw=0)
                ax.plot(x, g["mean"], **st)
            if ideal is not None:
                ax.axhline(ideal, color=INK_2, lw=0.8, ls=(0, (1, 2)))
            ax.set_title(title, fontsize=8)
            ax.set_xticks(range(len(DENSITY_ORDER)), ticks, fontsize=7)
            ax.set_xlim(-0.3, len(DENSITY_ORDER) - 0.7)
            ax.margins(y=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
        fig.supxlabel("Network density: full, or median nearest-neighbour distance (km)", fontsize=8, y=0.08)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return save(fig, out, name)


def _generative(df: pd.DataFrame) -> list[str]:
    return [m for m in _ordered_methods(df) if (df[df.method == m]["samples"] > 1).any()]


def correlation_by_distance(split: str, density: str, out: Path, root: Path = RESULTS,
                            seed: int = 0) -> Path | None:
    files = {m.parent.parent.parent.name: m
             for m in (root / "A").glob(f"*/{split}/{density}_s{seed}/correlations.csv")}
    if not files:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.6))
        truth = None
        for m in [k for k in METHODS if k in files]:
            c = pd.read_csv(files[m], index_col=0)
            x = np.arange(len(c))
            truth = (x, c["corr_truth"].to_numpy(), list(c.index))
            ax.plot(x, c["corr_sample"], **method_style(m))
        x, y, labels = truth
        ax.plot(x, y, color=INK, lw=2, label="truth")
        ax.set_xticks(x, [lab.strip("(]").replace(", ", "–") for lab in labels], fontsize=7)
        ax.set_xlabel("Neighbour distance (km)")
        ax.set_ylabel("Correlation of fast parts")
        ax.set_title(f"Neighbour correlation, {density}", fontsize=8)
        ax.legend()
        return save(fig, out, f"correlation_{density}")


def spectra(split: str, density: str, out: Path, root: Path = RESULTS, seed: int = 0) -> Path | None:
    runs = {p.parent.parent.parent.name: json.loads(p.read_text())
            for p in (root / "A").glob(f"*/{split}/{density}_s{seed}/summary.json")}
    runs = {m: s for m, s in runs.items() if s.get("samples", 1) > 1}
    if not runs:
        return None
    bands = [f"{lo}–{hi}" for lo, hi in BANDS_DAYS]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharey=True)
        for k, (ax, comp) in enumerate(zip(axes, "enu")):
            for m in [k_ for k_ in METHODS if k_ in runs]:
                y = [runs[m].get(f"psd_log10_{lo}_{hi}_{comp}", np.nan) for lo, hi in BANDS_DAYS]
                ax.plot(range(len(bands)), y, **method_style(m))
            ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (1, 2)))
            ax.set_xticks(range(len(bands)), bands, fontsize=6.5, rotation=35, ha="right")
            ax.set_title(COMPONENTS[k], fontsize=8)
            ax.set_xlabel("Period band (days)")
        axes[0].set_ylabel("log10 power, sample / truth")
        axes[0].legend()
        return save(fig, out, f"spectra_{density}")


def example(cube: Cube, out: Path, method: str = "idw+noise", density: float | None = 60.0,
            station: str | None = None, n_show: int = 8, root: Path = RESULTS,
            model_dir: Path | None = None) -> Path | None:
    """One held-out station in its first validation window: the truth, a
    few samples and the sample mean, per component."""
    if method == "diffusion":
        if model_dir is None or not (model_dir / "checkpoint.pt").exists():
            return None
        from imonet.model.methods import DiffusionMethod
        m = DiffusionMethod(model_dir, n_samples=32)
    else:
        tuning = root / "A" / "idw_tuning.json"
        if method in ("idw", "idw+noise") and not tuning.exists():
            return None
        length_km = json.loads(tuning.read_text())[density_name(density)]["length_km"] \
            if method in ("idw", "idw+noise") else None
        m = make(method, length_km, 32)
    held = cube.stations(HELD_OUT)
    s = held[0] if station is None else int(np.flatnonzero(cube.sta == station)[0])
    pred = predict(m, cube, component_scale(cube), np.array([s]), network(cube, density, 0), VALIDATION)
    w = pred.windows[0]
    cols = np.arange(w["t0"], w["t0"] + 256) - pred.days[0]
    days = cube.days[pred.days[cols]]
    truth, samples = pred.truth[0, cols], pred.samples[:, 0, cols]
    st = method_style(method)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.6), sharex=True)
        for k, ax in enumerate(axes):
            for j in range(min(n_show, len(samples))):
                ax.plot(days, samples[j, :, k], color=st["color"], lw=0.6, alpha=0.35,
                        label="samples" if j == 0 else None)
            ax.plot(days, samples[:, :, k].mean(0), color=st["color"], lw=1.4, label="sample mean")
            ax.plot(days, truth[:, k], color=INK, lw=0.9, label="truth")
            ax.set_ylabel(f"{COMPONENTS[k]} (mm)")
        axes[0].legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
        axes[0].set_title(f"{cube.sta[s]}, {st['label']}, network {density_name(density)}", fontsize=8, loc="left")
        return save(fig, out, f"example_{method.replace('+', '_')}")


TABLE_COLS = (("nrmse_fast", "nRMSE$_f$"), ("crps_fast", "CRPS$_f$"), ("crps_total", "CRPS$_t$"),
              ("cover90_fast", "Cov$_{90,f}$"), ("spread_skill", "Spread/err."),
              ("scale_log2", "log$_2$ scale"))      # f: fast part, t: total; defined in the caption


def table(df: pd.DataFrame, out: Path, name: str = "experiment_a_table") -> Path:
    """Mean over thinning seeds per method and density, as a LaTeX tabular."""
    cols = [(c, h) for c, h in TABLE_COLS if c in df]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "Density & Method & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for i, dens in enumerate(DENSITY_ORDER):
        sub = df[df.density == dens]
        if sub.empty:
            continue
        if i and len(lines) > 4:
            lines.append(r"\addlinespace")
        for j, m in enumerate(_ordered_methods(sub)):
            g = sub[sub.method == m][[c for c, _ in cols]].mean()
            vals = ["--" if not np.isfinite(g[c]) else f"{g[c]:.3f}" for c, _ in cols]
            lines.append(f"{dens if j == 0 else ''} & {method_style(m)['label']} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


NOISE_PANELS = (
    ("spread_skill", "Spread / error, total", 1.0),
    ("cover90_fast", "90% interval coverage, fast", 0.9),
)


def calibration_by_noise(df: pd.DataFrame, out: Path, density: str = "60km") -> Path | None:
    """Calibration of the generative methods by the held-out station's own
    noise level (quartiles of `station_noise`, quietest first), and pooled
    over all stations. Mean over thinning seeds, with their range."""
    d = df[(df.density == density) & df.method.isin(_generative(df))]
    if d.empty or "spread_skill_q1" not in d:
        return None
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    noise = d[[f"noise_q{q}" for q in range(1, 5)]].iloc[0].to_numpy()
    ticks = [f"{q}\n{n:.2f}" for q, n in zip(quarters, noise)] + ["all\npooled"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(NOISE_PANELS), figsize=(WIDTH, 2.5))
        for ax, (col, title, ideal) in zip(axes, NOISE_PANELS):
            for m in _generative(d):
                g = d[d.method == m]
                st = method_style(m)
                q = g[[f"{col}_q{i}" for i in range(1, 5)]]
                ax.fill_between(range(4), q.min(), q.max(), color=st["color"], alpha=0.15, lw=0)
                ax.plot(range(4), q.mean(), **st)
                pooled = g[f"{col}_pooled"]
                ax.errorbar([4.6], [pooled.mean()], yerr=[[pooled.mean() - pooled.min()], [pooled.max() - pooled.mean()]],
                            color=st["color"], marker=st["marker"], markerfacecolor=SURFACE, lw=1, capsize=0)
            ax.axhline(ideal, color=INK_2, lw=0.8, ls=(0, (1, 2)))
            ax.axvline(3.8, color=BASELINE, lw=0.6)
            ax.set_xticks([0, 1, 2, 3, 4.6], ticks, fontsize=7)
            ax.set_xlim(-0.3, 4.9)
            ax.set_title(title, fontsize=8)
            ax.margins(y=0.2)
        axes[0].legend(loc="upper right")
        fig.supxlabel("Held-out station's noise quartile (median noise relative to a typical station)",
                      fontsize=8, y=0.03)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return save(fig, out, f"calibration_by_noise_{density}")


VARIANT_COLS = (("crps_fast", "CRPS$_f$"), ("cover90_fast", "Cov.$_{90,f}$"), ("spread_skill", "S/E"),
                ("spread_skill_pooled", "S/E$_\\mathrm{pool}$"), ("tail_log2", "Tail"), ("corr_error", "$|\\Delta r|$"),
                ("screened", "Scr.\\,\\%"))


def variants(split: str, out: Path, root: Path = RESULTS, density: str = "60km", seed: int = 0,
             name: str = "variants_table") -> Path | None:
    """Sampling variants of the generative methods on one thinned network:
    screened samples and classifier-free guidance weights, as a LaTeX
    tabular. |Δr| is the mean over distance bins of the absolute error of
    the neighbour correlation."""
    rows = []
    for p in sorted((root / "A").glob(f"*/{split}/{density}_s{seed}/summary.json")):
        s = json.loads(p.read_text())
        if s.get("samples", 1) <= 1:
            continue
        corr = p.parent / "correlations.csv"
        s["corr_error"] = pd.read_csv(corr)["difference"].abs().mean() if corr.exists() else np.nan
        s["screened"] = 100 * s["screened"] if "screened" in s else np.nan
        rows.append(s)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["base"] = df["kind"].fillna(df["method"]) if "kind" in df else df["method"]
    df["w"] = df["guidance"].fillna(1.0) if "guidance" in df else 1.0
    df["scr"] = df["method"].str.endswith("-screened")
    df = df.sort_values(["base", "w", "scr"], key=lambda c: c.map(list(METHODS).index) if c.name == "base" else c)
    lines = [r"\begin{tabular}{lrl" + "r" * len(VARIANT_COLS) + "}", r"\toprule",
             "Method & $w$ & Screened & " + " & ".join(h for _, h in VARIANT_COLS) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        vals = ["--" if c not in r or not np.isfinite(r[c]) else
                (f"{r[c]:.2f}" if c == "screened" else f"{r[c]:.3f}") for c, _ in VARIANT_COLS]
        w = f"{r['w']:g}" if r["base"] == "diffusion" else "--"
        lines.append(f"{method_style(r['base'])['label']} & {w} & {'yes' if r['scr'] else 'no'} & "
                     + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path
