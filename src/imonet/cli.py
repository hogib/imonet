"""The imonet command."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imonet.cube import CALIFORNIA, TEST, VALIDATION, Cube
from imonet.evaluate import DENSITIES, RESULTS, THIN_SEEDS, Setup, density_name, report, run
from imonet.figures import FIGURES
from imonet.ngl import STEPS, read_steps
from imonet.train import TrainConfig

SPLITS = {"validation": VALIDATION, "test": TEST}


def _density(s: str) -> float | None:
    return None if s == "full" else float(s.removesuffix("km"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="imonet", description="Virtual GNSS stations from their neighbours.")
    p.add_argument("--cube", type=Path, default=CALIFORNIA)
    p.add_argument("--steps-file", type=Path, default=STEPS, help="NGL steps.txt")
    p.add_argument("--results", type=Path, default=RESULTS)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="experiment A: generate held-out stations and score them")
    e.add_argument("method", choices=["zero", "idw", "idw+noise", "diffusion", "regress"])
    e.add_argument("--split", choices=list(SPLITS), default="validation")
    e.add_argument("--density", nargs="+", default=[density_name(d) for d in DENSITIES],
                   help="full, or median nearest-neighbour spacing in km (default: full 25 60 100)")
    e.add_argument("--seeds", type=int, default=THIN_SEEDS, help="thinning seeds per thinned density")
    e.add_argument("--samples", type=int, default=32)
    e.add_argument("--model", type=Path, help="trained model directory (diffusion, regress)")
    e.add_argument("--label", help="name of the results (default: the method)")
    e.add_argument("--sampler-steps", type=int, default=32)
    e.add_argument("--guidance", type=float, default=1.0)

    t = sub.add_parser("train", help="train the diffusion model or regress")
    t.add_argument("kind", choices=["diffusion", "regress"])
    t.add_argument("--name", required=True, help="model directory under runs/models/")
    t.add_argument("--steps", type=int, default=TrainConfig.steps)
    t.add_argument("--batch", type=int, default=TrainConfig.batch)
    t.add_argument("--lr", type=float, default=TrainConfig.lr)
    t.add_argument("--val-every", type=int, default=TrainConfig.val_every)
    t.add_argument("--workers", type=int, default=TrainConfig.workers)
    t.add_argument("--seed", type=int, default=TrainConfig.seed)

    r = sub.add_parser("report", help="print the experiment A table")
    r.add_argument("--split", choices=list(SPLITS), default="validation")

    f = sub.add_parser("figures", help="figures and tables for the report and the paper")
    f.add_argument("--out", type=Path, default=Path("paper/figures"))
    f.add_argument("--split", choices=list(SPLITS), default="validation")
    f.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))

    a = p.parse_args(argv)
    if a.cmd == "report":
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
            print(report(a.split, a.results))
        return
    if a.cmd == "figures":
        from imonet.figures import make_all
        made = make_all(Cube.load(a.cube), a.out, a.results, a.split, tuple(a.only))
        for name, path in made.items():
            print(f"{name:24s} {path if path else 'skipped (no results yet)'}")
        return

    if a.cmd == "train":
        from imonet.train import train
        cfg = TrainConfig(kind=a.kind, steps=a.steps, batch=a.batch, lr=a.lr, val_every=a.val_every,
                          workers=a.workers, seed=a.seed)
        root = a.results / "models"
        log_path = root / a.name / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def log(msg: str) -> None:
            print(msg, flush=True)
            with open(log_path, "a") as fh:
                fh.write(msg + "\n")
        train(a.name, cfg, Cube.load(a.cube), root, log)
        return

    setup = Setup(Cube.load(a.cube), read_steps(a.steps_file))
    print(f"scoring QC drops {len(setup.qc_dropped)} stations")
    for d in map(_density, a.density):
        for seed in ([0] if d is None else range(a.seeds)):
            s = run(setup, a.method, SPLITS[a.split], a.split, d, seed, a.samples, a.results, a.model, a.label,
                    a.sampler_steps, a.guidance)
            print(f"{a.method} {s['density']} s{seed}: nrmse_fast {s.get('nrmse_fast', float('nan')):.3f} "
                  f"crps_fast {s.get('crps_fast', float('nan')):.3f} ({s['stations']} stations, "
                  f"{s['seconds']} s)", flush=True)
