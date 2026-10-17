"""Figures and tables for the report and the paper (`imonet figures`).

Every figure is a function writing one PDF (or a .tex table) into an output
directory, by default paper/figures/. Figures whose inputs do not exist yet
(results not run) are skipped and reported.
"""

from __future__ import annotations

from pathlib import Path

from imonet.cube import Cube
from imonet.evaluate import RESULTS

DIFFUSION_MODEL = "diffusion-v1"      # the model drawn in example figures
FIGURES = ("network_map", "thinning", "training", "scores_by_density", "calibration", "correlation", "spectra",
           "example", "table", "variants", "generators", "noise_models", "significance", "b")


def make_all(cube: Cube, out: Path, results: Path = RESULTS, split: str = "validation",
             only: tuple[str, ...] = FIGURES) -> dict[str, Path | None]:
    from imonet.figures import experiment_a as A
    from imonet.figures import network as N
    from imonet.figures import training as T

    made: dict[str, Path | None] = {}
    runs = A.load_runs(split, results)
    if "network_map" in only:
        made["network_map"] = N.network_map(cube, out)
    if "thinning" in only:
        made["thinning"] = N.thinning(cube, out)
    if "training" in only:
        made["training"] = T.training_curves(out, results)
    if runs.empty:
        for f in only:
            made.setdefault(f, None)
        return made
    if "scores_by_density" in only:
        made["scores_by_density"] = A.scores_by_density(runs, out)
    if "table" in only:
        made["table"] = A.table(runs, out)
    if "calibration" in only:
        made["calibration"] = A.calibration_by_noise(runs, out)
    if "variants" in only:
        made["variants"] = A.variants(split, out, results)
    if "generators" in only:
        made["generators"] = A.generators(split, out, results)
    if "noise_models" in only:
        made["calibration_noise_models"] = A.calibration_by_noise_models(runs, out)
        made["noise_models_table"] = A.noise_models_table(runs, out)
    if "significance" in only:
        from imonet.significance import table as significance_table
        made["significance"] = significance_table(out, split, results)
    if "b" in only:
        from imonet.figures import experiment_b as B
        made["b_rho_by_ratio"] = B.rho_by_ratio(out, split, results)
        made["b_rho_table"] = B.rho_table(out, split, results)
        made["b_ridgecrest_table"] = B.ridgecrest_table(out, results)
    for dens in ("full", "60km"):
        if "correlation" in only:
            made[f"correlation_{dens}"] = A.correlation_by_distance(split, dens, out, results)
        if "spectra" in only:
            made[f"spectra_{dens}"] = A.spectra(split, dens, out, results)
    if "example" in only:
        made["example"] = A.example(cube, out, root=results)
        model = results / "models" / DIFFUSION_MODEL
        made["example_diffusion"] = A.example(cube, out, "diffusion", root=results, model_dir=model)
    return made
