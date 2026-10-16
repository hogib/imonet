"""Training curves of the trained models (runs/models/<name>/).

One panel per model: the training loss every log_every steps (faint), its
1,000-step running mean, and the validation loss at every evaluation (dark,
markers). The training loss is noisy because the targets are heavy-tailed.
The diffusion and regress losses are on different scales, so they never
share an axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from imonet.evaluate import RESULTS
from imonet.figures.style import HALF, STYLE, WIDTH, method_style, save


def _models(root: Path) -> list[Path]:
    return [d for d in sorted((root / "models").glob("*"))
            if (d / "steps.csv").exists() and (d / "history.json").exists() and (d / "config.json").exists()]


def training_curves(out: Path, root: Path = RESULTS) -> Path | None:
    models = _models(root)
    if not models:
        return None
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(models), figsize=(WIDTH if len(models) > 1 else HALF + 0.5, 2.6),
                                 squeeze=False)
        for ax, d in zip(axes[0], models):
            kind = json.loads((d / "config.json").read_text())["kind"]
            st = method_style(kind)
            steps = pd.read_csv(d / "steps.csv")
            hist = pd.DataFrame(json.loads((d / "history.json").read_text()))
            ax.plot(steps["step"] / 1000, steps["loss"], color=st["color"], lw=0.5, alpha=0.2)
            ax.plot(steps["step"] / 1000, steps["loss"].rolling(10, min_periods=1).mean(), color=st["color"],
                    lw=1.0, alpha=0.6, label="training (1,000-step mean)")
            ax.plot(hist["step"] / 1000, hist["val_loss"], color=st["color"], lw=1.5, marker=st["marker"],
                    markersize=3, label="validation")
            ax.set_title(f"{d.name} ({st['label']})", fontsize=8, loc="left")
            ax.set_xlabel("Step (thousands)")
            ax.set_ylabel("EDM loss" if kind == "diffusion" else "Squared error")
            lo = min(hist["val_loss"].min(), steps["loss"].quantile(0.02))
            hi = max(hist["val_loss"].max(), steps["loss"].quantile(0.98))
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.legend(loc="upper right")
        fig.tight_layout()
        return save(fig, out, "training_curves")
