"""Training curves of the trained models (runs/models/<name>/).

Two panels, one per kind of model, since the diffusion (EDM) loss and the
regression (squared-error) loss are on different scales: every model of a
kind on one axis, so versions and training seeds can be compared directly.
For each model the validation loss at every evaluation (markers) and the
5,000-step running mean of the training loss (thin). The raw training loss
is not drawn: it is noisy because the targets are heavy-tailed, and with
several models it hides the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from imonet.evaluate import RESULTS
from imonet.figures.style import STYLE, WIDTH, method_style, save

KINDS = (("diffusion", "Diffusion models: EDM loss"), ("regress", "Regression models: squared error"))
# Line style per model version, in the family colour; unknown names get the last.
VERSION_STYLES = ((":", "o"), ("--", "s"), ("-", "D"), ("-.", "^"), ((0, (5, 1, 1, 1, 1, 1)), "v"))


def _models(root: Path) -> list[Path]:
    return [d for d in sorted((root / "models").glob("*"))
            if (d / "steps.csv").exists() and (d / "history.json").exists() and (d / "config.json").exists()]


def _label(name: str, kind: str) -> str:
    """'diffusion-v3-s1' -> 'v3, seed 1'."""
    rest = name.removeprefix(kind).strip("-")
    version, _, seed = rest.partition("-s")
    return f"{version}, seed {seed}" if seed else version or name


def training_curves(out: Path, root: Path = RESULTS) -> Path | None:
    models = _models(root)
    if not models:
        return None
    by_kind = {k: [d for d in models if json.loads((d / "config.json").read_text())["kind"] == k] for k, _ in KINDS}
    kinds = [(k, t) for k, t in KINDS if by_kind[k]]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(kinds), figsize=(WIDTH, 2.9), squeeze=False)
        for ax, (kind, title) in zip(axes[0], kinds):
            colour = method_style(kind)["color"]
            lo, hi = float("inf"), float("-inf")
            for i, d in enumerate(by_kind[kind]):
                ls, marker = VERSION_STYLES[min(i, len(VERSION_STYLES) - 1)]
                steps = pd.read_csv(d / "steps.csv")
                hist = pd.DataFrame(json.loads((d / "history.json").read_text()))
                train = steps["loss"].rolling(50, min_periods=1).mean()
                ax.plot(steps["step"] / 1000, train, color=colour, lw=0.7, ls=ls, alpha=0.35)
                ax.plot(hist["step"] / 1000, hist["val_loss"], color=colour, lw=1.3, ls=ls, marker=marker,
                        markersize=2.5, markevery=max(1, len(hist) // 12), label=_label(d.name, kind))
                lo = min(lo, hist["val_loss"].min(), train.quantile(0.02))
                hi = max(hi, hist["val_loss"].quantile(0.9), train.quantile(0.9))
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.15 * (hi - lo))
            ax.set_title(title, fontsize=8, loc="left")
            ax.set_xlabel("Training step (thousands)")
            ax.set_ylabel("Loss")
            ax.legend(loc="upper right", fontsize=7, title="validation", title_fontsize=7)
        fig.text(0.5, 0.005, "Lines with markers: validation loss; thin lines: training loss, 5,000-step running mean.",
                 ha="center", va="bottom", fontsize=7)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return save(fig, out, "training_curves")
