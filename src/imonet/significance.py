"""Paired comparison of two methods in experiment A (`imonet compare`).

Two methods scored on the same held-out stations and the same thinned
networks are compared station by station. The score is a per-station value
(`stations.csv`): the mean over E, N, U of a component score such as
`crps_fast`, or a score defined on all components such as `energy_total`.

- The headline difference is that of the summaries the tables report: the
  median over stations, per component, then the mean over components, as
  `metrics.summarise`, averaged over the thinning seeds.
- Its 95% interval comes from a bootstrap over stations (10,000 resamples).
  A station keeps all its thinning seeds together, so repeated networks do
  not count as independent evidence.
- The sign test counts the stations where the first method is better, with
  each station's scores averaged over the seeds first; the p-value is exact
  (two-sided binomial with p = 1/2), ties left out.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from imonet.evaluate import RESULTS
from imonet.metrics import COMPONENTS

N_BOOT = 10_000


def per_station(method: str, split: str, density: str, root: Path = RESULTS) -> pd.DataFrame:
    """Every run of a method at one density: stations.csv with a `seed` column."""
    frames = []
    for p in sorted((root / "A" / method / split).glob(f"{density}_s*/stations.csv")):
        df = pd.read_csv(p)
        df["seed"] = int(p.parent.name.rsplit("_s", 1)[1])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no runs of {method} at {density} ({split})")
    return pd.concat(frames, ignore_index=True)


def _columns(df: pd.DataFrame, score: str) -> list[str]:
    comp = [f"{score}_{c}" for c in COMPONENTS]
    if all(c in df for c in comp):
        return comp
    if score in df:
        return [score]
    raise KeyError(f"no score {score!r}")


def headline(values: np.ndarray) -> float:
    """values: stations × seeds × columns → mean over columns and seeds of
    the median over stations (as the tables)."""
    return float(np.nanmean(np.nanmedian(values, axis=0)))


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of k successes in n trials with p = 1/2."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a: str, b: str, score: str = "crps_fast", split: str = "validation", density: str = "60km",
            root: Path = RESULTS, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Paired comparison of methods a and b; negative differences mean a is
    better (lower)."""
    da, db = per_station(a, split, density, root), per_station(b, split, density, root)
    cols = _columns(da, score)
    seeds = sorted(set(da["seed"]) & set(db["seed"]))
    stations = sorted(set(da["station"]) & set(db["station"]))
    if not seeds or not stations:
        raise ValueError(f"{a} and {b} share no runs at {density}")

    def cube(df):                                  # stations × seeds × columns
        x = df.set_index(["station", "seed"])[cols]
        idx = pd.MultiIndex.from_product([stations, seeds], names=["station", "seed"])
        return x.reindex(idx).to_numpy().reshape(len(stations), len(seeds), len(cols))

    xa, xb = cube(da), cube(db)
    diff = headline(xa) - headline(xb)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(len(stations), size=len(stations))
        boot[i] = headline(xa[pick]) - headline(xb[pick])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    sa, sb = np.nanmean(xa, axis=(1, 2)), np.nanmean(xb, axis=(1, 2))   # per station
    ok = np.isfinite(sa) & np.isfinite(sb) & (sa != sb)
    wins = int((sa[ok] < sb[ok]).sum())
    n = int(ok.sum())
    return {"a": a, "b": b, "score": score, "density": density, "stations": len(stations), "seeds": len(seeds),
            "a_value": headline(xa), "b_value": headline(xb), "difference": diff,
            "relative": diff / headline(xb), "ci_low": float(lo), "ci_high": float(hi),
            "a_better_stations": wins, "compared_stations": n, "sign_p": binomial_two_sided(wins, n)}


DENSITY_ORDER = ("full", "25km", "60km", "100km")
TABLE_PAIRS = (("diffusion", "idw+noise"), ("diffusion", "krige+noise"), ("diffusion", "gp"),
               ("diffusion-v2-given", "gp-given"), ("diffusion-v3-unknown", "gp"), ("diffusion-v3-given", "gp-given"),
               ("diffusion-v3-given", "diffusion-v2-given"), ("diffusion-v3-unknown", "diffusion-v2-unknown"))
LABELS = {"diffusion": "diffusion v1", "idw+noise": "IDW + noise", "krige+noise": "kriging + noise",
          "gp": "GP", "diffusion-v2-given": "diffusion v2 (given)", "gp-given": "GP (given)",
          "diffusion-v3-given": "diffusion v3 (given)", "diffusion-v2-unknown": "diffusion v2 (unknown)",
          "diffusion-v3-unknown": "diffusion v3 (unknown)"}


def table(out: Path, split: str = "validation", root: Path = RESULTS, score: str = "crps_fast",
          name: str = "significance_table") -> Path | None:
    """Paired comparisons as a LaTeX tabular, one row per comparison and
    density: relative difference of the summaries (%), its 95% bootstrap
    interval, and the stations where the first method is better."""
    lines = [r"\begin{tabular}{llrcr}", r"\toprule",
             r"Comparison & Density & Difference (\%) & 95\% interval & Stations \\", r"\midrule"]
    any_row = False
    for k, (a, b) in enumerate(TABLE_PAIRS):
        first, row = True, 0
        for d in DENSITY_ORDER:
            try:
                r = compare(a, b, score, split, d, root)
            except (FileNotFoundError, ValueError):
                continue
            if first and any_row:
                lines.append(r"\addlinespace")
            base = r["b_value"]
            # the comparison on two rows, "a" then "vs b", to keep the table narrow
            label = LABELS.get(a, a) if row == 0 else f"vs {LABELS.get(b, b)}" if row == 1 else ""
            lines.append(f"{label} & {d.replace('km', '~km')} & ${100 * r['relative']:+.1f}$ & "
                         f"[${100 * r['ci_low'] / base:+.1f}$, ${100 * r['ci_high'] / base:+.1f}$] & "
                         f"{r['a_better_stations']}/{r['compared_stations']} \\\\")
            first, any_row, row = False, True, row + 1
    if not any_row:
        return None
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path
