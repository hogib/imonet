"""Figures and tables of experiment B (runs/B/, `imonet eval-b`).

- `rho_by_ratio`: the recovered fraction ρ against the event's footprint
  divided by the distance from the target to its nearest network station,
  the design's main figure; one line per method (median over target-events
  on the networks every method has run on), and the number of
  target-events per class.
- `rho_table`: ρ pooled by signal energy, per template, method and density.
- `ridgecrest_table`: the real 2019 Ridgecrest offsets, per method and
  density.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imonet.evaluate import RESULTS
from imonet.figures.style import INK_2, METHODS, STYLE, WIDTH, method_style, save

RATIO_BINS = (0, 0.25, 0.5, 1, 2, 4, 100)
DENSITY_ORDER = ("full", "25km", "60km", "100km")
# Model versions drawn in their family's colour: (family, label, style changes).
# v1/v2 dashed with open markers, the models trained with planted transients
# (diffusion-v3, regress-v2) solid.
_OLD = {"linestyle": (0, (4, 2)), "markerfacecolor": "white"}
B_VARIANTS = {
    "diffusion": ("diffusion", "diffusion v1", {"linestyle": (0, (1, 1.5)), "markerfacecolor": "white"}),
    "diffusion-v2": ("diffusion", "diffusion v2", _OLD),
    "diffusion-v2-given": ("diffusion", "diffusion v2, level given", _OLD),
    "diffusion-v3": ("diffusion", "diffusion v3 (transients)", {}),
    "diffusion-v3-given": ("diffusion", "diffusion v3, level given", {}),
    "regress": ("regress", "regress v1", _OLD),
    "regress-v2": ("regress", "regress v2 (transients)", {}),
}
B_METHODS = {k: v[0] for k, v in B_VARIANTS.items() if k != v[0]}
SINGLE = ("idw", "krige", "regress", "regress-v2")      # one series, no ensemble range


def _style(name: str) -> dict:
    if name not in B_VARIANTS:
        return method_style(name)
    family, label, extra = B_VARIANTS[name]
    return method_style(family) | {"label": label} | extra


def _tex(label: str) -> str:
    return label.replace("\u2013", "--")


def _order(names) -> list[str]:
    def key(m):
        family = B_METHODS.get(m, m)
        return (list(METHODS).index(family) if family in METHODS else 99, list(B_VARIANTS).index(m) if m in B_VARIANTS else -1)
    return sorted(names, key=key)


def load_targets(split: str = "validation", root: Path = RESULTS) -> pd.DataFrame:
    frames = []
    for p in sorted((root / "B").glob(f"*/{split}/*/targets.csv")):
        d = pd.read_csv(p)
        d["method"] = p.parent.parent.parent.name
        d["density"], d["seed"] = p.parent.name.rsplit("_s", 1)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rho_by_ratio(out: Path, split: str = "validation", root: Path = RESULTS) -> Path | None:
    df = load_targets(split, root)
    if df.empty:
        return None
    seen = df[(df.role == "seen") & (df.method != "zero")].copy()
    # Only the networks every method has run on, so that the lines compare like with like.
    nets = seen.groupby("method").apply(lambda g: set(zip(g.density, g.seed)), include_groups=False)
    common = set.intersection(*nets)
    seen = seen[[(d, s) in common for d, s in zip(seen.density, seen.seed)]]
    seen["bin"] = pd.cut(seen.footprint_km / seen.nearest_km, RATIO_BINS)
    labels = [f"{a:g}–{b:g}" if b < 100 else f"> {a:g}" for a, b in zip(RATIO_BINS[:-1], RATIO_BINS[1:])]
    with plt.rc_context(STYLE):
        fig, (ax, axn) = plt.subplots(2, 1, figsize=(WIDTH * 0.62, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        for m in _order(seen.method.unique()):
            g = seen[seen.method == m].groupby("bin", observed=False).rho.median().to_numpy()
            ax.plot(range(len(labels)), g, **_style(m))
        ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (1, 2)))
        ax.axhline(1, color=INK_2, lw=0.8, ls=(0, (1, 2)))
        ax.set_ylabel("Recovered fraction ρ (median)")
        ax.legend(loc="upper left", fontsize=7)
        counts = seen[seen.method == seen.method.iloc[0]].groupby("bin", observed=False).size().to_numpy()
        axn.bar(range(len(labels)), counts, color="#c3c2b7", width=0.6)
        axn.set_ylabel("Pairs", fontsize=7)
        axn.set_xticks(range(len(labels)), labels, fontsize=7)
        axn.set_xlabel("Event footprint / distance to the nearest network station")
        fig.tight_layout()
        return save(fig, out, "b_rho_by_ratio")


def rho_table(out: Path, split: str = "validation", root: Path = RESULTS, name: str = "b_rho_table") -> Path | None:
    df = load_targets(split, root)
    if df.empty:
        return None
    seen = df[(df.role == "seen") & (df.method != "zero")]
    pooled = seen.groupby(["template", "method", "density"]).apply(
        lambda g: g.projection.sum() / g.signal_energy.sum(), include_groups=False).unstack("density")
    cols = [d for d in DENSITY_ORDER if d in pooled]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "Event & Method & " + " & ".join(c.replace("km", "~km") for c in cols) + r" \\", r"\midrule"]
    for i, tp in enumerate(sorted(pooled.index.get_level_values(0).unique())):
        if i:
            lines.append(r"\addlinespace")
        for j, m in enumerate(_order(pooled.loc[tp].index)):
            vals = [f"{pooled.loc[(tp, m), c]:.3f}" if np.isfinite(pooled.loc[(tp, m), c]) else "--" for c in cols]
            lines.append(f"{tp if j == 0 else ''} & {_tex(_style(m)['label'])} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def ridgecrest_table(out: Path, root: Path = RESULTS, name: str = "b_ridgecrest_table") -> Path | None:
    rows = []
    for p in sorted((root / "B").glob("*/ridgecrest/*/summary.json")):
        s = json.loads(p.read_text())
        rows.append({"method": p.parent.parent.parent.name, "density": s["density"], **s})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cols = (("recovered_h_median_5mm", "Recovered"), ("error_h_mm_median_5mm", "Error (mm)"),
            ("inside90_h_5mm", "In 90\\%"))
    dens = [d for d in DENSITY_ORDER if d in set(df.density)]
    lines = [r"\begin{tabular}{l" + "rrr" * len(dens) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{d.replace('km', '~km')}}}" for d in dens) + r" \\",
             "Method & " + " & ".join(h for _ in dens for _, h in cols) + r" \\", r"\midrule"]
    for m in _order(df.method.unique()):
        vals = []
        for d in dens:
            r = df[(df.method == m) & (df.density == d)]
            for c, _ in cols:
                v = r[c].iloc[0] if len(r) and r[c].iloc[0] is not None else np.nan
                single = c.startswith("inside") and m in SINGLE
                vals.append("--" if single or not np.isfinite(v) else (f"{v:.1f}" if "error" in c else f"{v:.2f}"))
        lines.append(f"{_tex(_style(m)['label'])} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path
