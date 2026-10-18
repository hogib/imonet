"""Trajectory model of daily GNSS positions, and its residuals.

Ported from gnssdl (gnssdl.model, same author).

    x(t) = a + v·t + Σ seasonal(1/yr, 2/yr) + Σ H(t − t_k)·b_k

is fitted by least squares on a reference window. Extrapolated beyond the
window and subtracted, it leaves what the window did not contain: a
coseismic offset, postseismic relaxation, a change of rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("e", "n", "u")


def design_matrix(t: np.ndarray, steps: list[float], t0: float) -> np.ndarray:
    """`t`, `steps` and the velocity reference epoch `t0` in decimal years.
    Centring on `t0` keeps the velocity column well conditioned."""
    w = 2 * np.pi * t
    cols = [np.ones_like(t), t - t0, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    cols += [(t >= s).astype(float) for s in steps]
    return np.column_stack(cols)


def to_decimal_year(d: pd.Timestamp) -> float:
    return d.year + (d.dayofyear - 0.5) / (366 if d.is_leap_year else 365)


def residuals(
    series: pd.DataFrame,
    fit_mask: np.ndarray,
    step_dates: list[pd.Timestamp] = (),
    keep_step_dates: list[pd.Timestamp] = (),
    equipment_dates: list[pd.Timestamp] = (),
    equipment_window_days: int = 30,
    min_epochs_each_side: int = 1,
    min_fit_epochs: int = 365,
    kept_steps_out: list | None = None,
) -> pd.DataFrame:
    """Residuals (mm) of every epoch against a model fitted on the rows
    `fit_mask` selects.

    A step is estimated when the fit set has at least `min_epochs_each_side`
    epochs on both sides of it; others are left in the residual. Steps in
    `keep_step_dates` (earthquakes) are estimated, so that they do not bias
    the trend, and added back: they stay in the residual as signal.
    Equipment changes after the last fit epoch are not signal, and are
    removed with a local jump estimate (median of the residual
    `equipment_window_days` after minus before).

    If `kept_steps_out` is a list, one (decimal-year epoch, [E, N, U] mm)
    tuple is appended per earthquake step that was estimated and kept.
    """
    fit_mask = np.asarray(fit_mask, dtype=bool)
    t_all = series["decyear"].to_numpy()
    t_fit = t_all[fit_mask]
    if len(t_fit) < min_fit_epochs:
        raise ValueError(f"only {len(t_fit)} epochs in the fit window")

    keep = {to_decimal_year(d) for d in keep_step_dates}
    steps = sorted(
        s for s in {to_decimal_year(d) for d in step_dates}
        if (t_fit < s).sum() >= min_epochs_each_side and (t_fit >= s).sum() >= min_epochs_each_side
    )
    kept_cols = [6 + k for k, s in enumerate(steps) if s in keep]

    t0 = t_fit.mean()
    g_fit = design_matrix(t_fit, steps, t0)
    g_all = design_matrix(t_all, steps, t0)

    out = pd.DataFrame(index=series.index)
    kept_amp = np.zeros((len(kept_cols), 3))
    for ci, c in enumerate(COMPONENTS):
        m, *_ = np.linalg.lstsq(g_fit, series[c].to_numpy()[fit_mask], rcond=None)
        model = g_all @ m - g_all[:, kept_cols] @ m[kept_cols]
        out[c] = (series[c].to_numpy() - model) * 1000.0
        kept_amp[:, ci] = m[kept_cols] * 1000.0
    if kept_steps_out is not None:
        for col, amp in zip(kept_cols, kept_amp):
            kept_steps_out.append((steps[col - 6], amp))

    last_fit = series.index[fit_mask].max()
    w = pd.Timedelta(days=equipment_window_days)
    for d in sorted(equipment_dates):
        if d <= last_fit:
            continue
        before, after = out.loc[d - w: d - pd.Timedelta(days=1)], out.loc[d: d + w]
        if len(before) < 5 or len(after) < 5:
            continue                                   # too little data to estimate: leave it visible
        out.loc[d:] -= after.median() - before.median()
    return out
