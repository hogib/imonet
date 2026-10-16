"""Shared look of every figure: print figures for the report and the paper.

Colour follows the method, never its rank: each method has one fixed colour
and marker in every figure. Slots are the first seven of the reference
categorical palette in its order, validated together (adjacent CVD ΔE ≥ 9.1,
normal-vision ΔE ≥ 19.6). Aqua, yellow and magenta are under 3:1 contrast on
white, so every line
also has its own marker and a legend entry, and the numbers are also given
as tables; colour never carries identity alone.
`zero` is a reference, drawn in muted grey and dashed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET = "#e87ba4", "#008300", "#6250d6"
BLUE_LIGHT = "#b7d3f6"
# Ordinal blue ramp for the evaluation densities, dense → sparse as light →
# dark (steps 250, 400, 550, 700; the lightest still clears 2:1 on white).
DENSITY_SHADES = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

METHODS = {   # name: (label, colour, marker, linestyle)
    "zero": ("zero", MUTED, "", (0, (4, 2))),
    "idw": ("idw", BLUE, "o", "-"),
    "idw+noise": ("idw + noise", ORANGE, "s", "-"),
    "diffusion": ("diffusion", AQUA, "D", "-"),
    "regress": ("regress", YELLOW, "^", "-"),
    "krige": ("krige", MAGENTA, "v", "-"),
    "krige+noise": ("krige + noise", GREEN, "P", "-"),
    "gp": ("space–time GP", VIOLET, "X", "-"),
}
COMPONENTS = ("East", "North", "Up")

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4,
    "legend.frameon": False, "legend.fontsize": 7.5, "font.family": "sans-serif",
    "pdf.fonttype": 42,
}

WIDTH = 7.2        # inches, a full text width
HALF = 3.5


def method_style(name: str) -> dict:
    label, colour, marker, ls = METHODS.get(name, (name, INK_2, "x", "-"))
    return {"label": label, "color": colour, "marker": marker, "linestyle": ls}


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

