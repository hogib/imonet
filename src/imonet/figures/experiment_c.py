"""Tables of experiment C (runs/C/, `imonet eval-c`).

- `scores_table`: the main scores of every method per period, for all
  stations and for those in Türkiye.
- `significance_table`: paired comparisons per period (as experiment A).
- `distance_table`: fast CRPS by the distance from the target to its
  nearest station that records in the period, for the generative methods.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from imonet.cube import Cube, pairwise_km
from imonet.evaluate import RESULTS
from imonet.experiment_c import CUBES, PERIODS, period_days
from imonet.significance import paired

METHODS = {   # name: label
    "zero": "zero",
    "idw": "IDW",
    "idw+noise": "IDW + noise",
    "krige+noise": "kriging + noise",
    "gp": "space--time GP",
    "gp-given": "space--time GP, level given",
    "regress-v2": "regress v2",
    "diffusion-v3-unknown": "diffusion v3",
    "diffusion-v3-given": "diffusion v3, level given",
}
PERIOD_LABELS = {"main": "Main: 2013--2015", "sparse": "Sparse: 2016 on", "infit": "In-fit: 2009--2015"}
PAIRS = (("diffusion-v3-unknown", "gp"), ("diffusion-v3-given", "gp-given"), ("diffusion-v3-unknown", "idw+noise"))
COLS = (("crps_fast", "CRPS$_f$"), ("nrmse_fast", "nRMSE$_f$"), ("energy_total", "ES"),
        ("cover90_fast", "Cov.$_{90,f}$"), ("spread_skill", "S/E"), ("spread_skill_pooled", "S/E$_\\mathrm{pool}$"))
DISTANCE_BINS_KM = (0, 25, 50, 75, 1000)   # Türkiye: median nearest-station distance ~60 km
DISTANCE_METHODS = ("idw+noise", "gp", "diffusion-v3-unknown")
MIN_DAYS = 90


def _summary(method: str, period: str, root: Path) -> dict | None:
    p = root / "C" / method / period / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def _write(lines: list[str], out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def scores_table(out: Path, root: Path = RESULTS, name: str = "c_scores_table") -> Path | None:
    """One block per period, its name as a heading row."""
    rows = [(p, m) for p in PERIODS for m in METHODS if _summary(m, p, root)]
    if not rows:
        return None
    n = len(COLS) + 2
    lines = [r"\begin{tabular}{l" + "r" * (len(COLS) + 1) + "}", r"\toprule",
             "Method & " + " & ".join(h for _, h in COLS) + r" & CRPS$_f$ TR \\", r"\midrule"]
    last = None
    for p, m in rows:
        if p != last:
            if last is not None:
                lines.append(r"\addlinespace")
            lines.append(f"\\multicolumn{{{n}}}{{l}}{{\\emph{{{PERIOD_LABELS[p]}}}}} \\\\")
            last = p
        s_ = _summary(m, p, root)
        vals = [f"{s_[c]:.3f}" if isinstance(s_.get(c), (int, float)) and np.isfinite(s_[c]) else "--" for c, _ in COLS]
        tur = s_.get("turkiye", {}).get("crps_fast")
        lines.append(f"{METHODS[m]} & " + " & ".join(vals) + (f" & {tur:.3f}" if tur is not None else " & --") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(lines, out, name)


def per_station(method: str, period: str, root: Path = RESULTS) -> pd.DataFrame | None:
    p = root / "C" / method / period / "stations.csv"
    return pd.read_csv(p).set_index("station") if p.exists() else None


def compare(a: str, b: str, period: str, root: Path = RESULTS, score: str = "crps_fast",
            turkiye_only: bool = False) -> dict | None:
    da, db = per_station(a, period, root), per_station(b, period, root)
    if da is None or db is None:
        return None
    st = da.index.intersection(db.index)
    if turkiye_only:
        st = st[da.loc[st, "country"].eq("TUR").to_numpy()]
    cols = [f"{score}_{c}" for c in "enu"]
    xa = da.loc[st, cols].to_numpy()[:, None, :]
    xb = db.loc[st, cols].to_numpy()[:, None, :]
    return {"a": a, "b": b, "period": period, **paired(xa, xb)}


def significance_table(out: Path, root: Path = RESULTS, name: str = "c_significance_table") -> Path | None:
    lines = [r"\begin{tabular}{llrcr}", r"\toprule",
             r"Comparison & Period & Difference (\%) & 95\% interval & Stations \\", r"\midrule"]
    any_row = False
    for a, b in PAIRS:
        row = 0
        for p in PERIODS:
            r = compare(a, b, p, root)
            if r is None:
                continue
            if row == 0 and any_row:
                lines.append(r"\addlinespace")
            base = r["b_value"]
            label = METHODS[a] if row == 0 else f"vs {METHODS[b]}" if row == 1 else ""
            lines.append(f"{label} & {PERIOD_LABELS[p].split(':')[0]} & ${100 * r['relative']:+.1f}$ & "
                         f"[${100 * r['ci_low'] / base:+.1f}$, ${100 * r['ci_high'] / base:+.1f}$] & "
                         f"{r['a_better_stations']}/{r['compared_stations']} \\\\")
            row, any_row = row + 1, True
        if row == 1:                                   # one period only: put the "vs" on its own line
            lines.append(f"vs {METHODS[b]} & & & & \\\\")
    if not any_row:
        return None
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(lines, out, name)


def nearest_km(period: str, cubes: Path = CUBES) -> pd.Series:
    """Per station: distance (km) to the nearest other station with at
    least MIN_DAYS available days in the period."""
    spec = PERIODS[period]
    cube = Cube.load(cubes / f"{spec.cube}.npz")
    days = np.zeros(cube.n_days, dtype=bool)
    for a, b in period_days(cube, spec):
        days[a:b] = True
    recording = (cube.avail[:, days] & ~cube.exclude[:, days]).sum(axis=1) >= MIN_DAYS
    d = pairwise_km(cube.lat, cube.lon)
    np.fill_diagonal(d, np.inf)
    d[:, ~recording] = np.inf
    return pd.Series(d.min(axis=1), index=cube.sta.astype(str))


def distance_table(out: Path, root: Path = RESULTS, name: str = "c_distance_table",
                   cubes: Path = CUBES) -> Path | None:
    """Median fast CRPS by nearest-station distance, per period, for
    DISTANCE_METHODS, with diffusion's paired difference to the GP."""
    edges = DISTANCE_BINS_KM
    labels = [f"{a}--{b}~km" if b < 1000 else f"$\\geq${a}~km" for a, b in zip(edges[:-1], edges[1:])]
    lines = [r"\begin{tabular}{ll" + "r" * (len(DISTANCE_METHODS) + 2) + "}", r"\toprule",
             "Period & Nearest station & Stations & " + " & ".join(METHODS[m] for m in DISTANCE_METHODS)
             + r" & Diff.\ to GP (\%) \\", r"\midrule"]
    any_row = False
    for p in PERIODS:
        frames = {m: per_station(m, p, root) for m in DISTANCE_METHODS}
        if any(f is None for f in frames.values()):
            continue
        st = frames[DISTANCE_METHODS[0]].index
        for f in frames.values():
            st = st.intersection(f.index)
        near = nearest_km(p, cubes).reindex(st)
        bins = pd.cut(near, edges, labels=labels, right=False)
        if any_row:
            lines.append(r"\addlinespace")
        for j, lab in enumerate(labels):
            sel = st[(bins == lab).to_numpy()]
            if not len(sel):
                continue
            crps = {m: frames[m].loc[sel, [f"crps_fast_{c}" for c in "enu"]].median().mean() for m in DISTANCE_METHODS}
            d, g = DISTANCE_METHODS[-1], "gp"
            rel = 100 * (crps[d] / crps[g] - 1)
            lines.append(f"{PERIOD_LABELS[p].split(':')[0] if j == 0 else ''} & {lab} & {len(sel)} & "
                         + " & ".join(f"{crps[m]:.3f}" for m in DISTANCE_METHODS) + f" & ${rel:+.1f}$ \\\\")
        any_row = True
    if not any_row:
        return None
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(lines, out, name)
