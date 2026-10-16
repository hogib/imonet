"""Training of the diffusion model and of `regress` (docs/00-design.md §4).

A model lives in runs/models/<name>/:
  config.json      what was trained, with the component scale it uses
  checkpoint.pt    latest weights, EMA weights, optimiser, step (resumable)
  history.json     training and validation loss per evaluation
  steps.csv        training loss, gradient norm, learning rate and speed every log_every steps
  train.log        progress: a line every log_every steps, and one per evaluation

The EMA weights are the model: they are what evaluation samples from. The
validation loss uses fixed samples (held-out stations, validation days)
and, for the diffusion model, fixed noise at fixed noise levels, so it is
comparable across steps.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from imonet.cube import Cube
from imonet.data import WINDOW_DAYS, component_scale, training_windows, validation_windows
from imonet.evaluate import RESULTS
from imonet.model.edm import EDM, EDMConfig, Regress, drop_condition, n_params
from imonet.model.net import Net, NetConfig

MODELS = RESULTS / "models"
VAL_SIGMAS = (0.02, 0.1, 0.3, 1.0, 3.0, 10.0)
BOOL_KEYS = ("x_ok", "y_ok", "valid")
FLOAT_KEYS = ("x", "y", "y_sigma", "geom", "doy0")


@dataclass
class TrainConfig:
    kind: str = "diffusion"            # or "regress"
    steps: int = 100_000
    batch: int = 64
    lr: float = 2e-4
    warmup: int = 2_000
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    log_every: int = 100
    val_every: int = 2_000
    val_samples: int = 1_024
    samples_per_epoch: int = 64_000
    workers: int = 6
    seed: int = 0
    length: int = WINDOW_DAYS          # days per window
    compile: bool = True               # torch.compile on GPU: about 2× faster (memory-bound cross-attention)
    net: NetConfig = field(default_factory=NetConfig)
    edm: EDMConfig = field(default_factory=EDMConfig)


def config_from_json(d: dict) -> TrainConfig:
    d = dict(d)
    d["net"] = NetConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["net"].items()})
    d["edm"] = EDMConfig(**d["edm"])
    return TrainConfig(**{k: v for k, v in d.items() if k in TrainConfig.__dataclass_fields__})


def to_torch(batch: dict, device) -> dict:
    out = {}
    for k in BOOL_KEYS:
        if k in batch:
            out[k] = torch.as_tensor(batch[k]).to(device, non_blocking=True).bool()
    for k in FLOAT_KEYS:
        if k in batch:
            out[k] = torch.as_tensor(batch[k]).to(device, non_blocking=True).float()
    return out


def wrapper(kind: str, net: Net, edm_cfg: EDMConfig):
    return EDM(net, edm_cfg) if kind == "diffusion" else Regress(net)


@torch.no_grad()
def validation_loss(model, loader, device, kind: str) -> float:
    total, n = 0.0, 0
    g = torch.Generator(device=device)
    for i, batch in enumerate(loader):
        b = to_torch(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if kind == "diffusion":
                g.manual_seed(i)
                bs = b["x"].shape[0]
                noise = torch.randn(b["x"].shape, device=device, generator=g)
                vals = [model.loss(b, sigma=torch.full((bs,), s, device=device), noise=noise) for s in VAL_SIGMAS]
                loss = sum(v.item() for v in vals) / len(vals)
            else:
                loss = model.loss(b).item()
        total += loss * b["x"].shape[0]
        n += b["x"].shape[0]
    return total / n


def train(name: str, cfg: TrainConfig, cube: Cube, root: Path = MODELS, log=print) -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    scale = component_scale(cube)

    net = Net(cfg.net).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    fast = torch.compile(net) if cfg.compile and device.type == "cuda" else net
    model = wrapper(cfg.kind, fast, cfg.edm)
    ema_model = wrapper(cfg.kind, ema, cfg.edm)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.0)
    step, history = 0, []

    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        history = json.loads((out / "history.json").read_text()) if (out / "history.json").exists() else []
        log(f"resumed {name} at step {step}")
    else:
        (out / "config.json").write_text(json.dumps(
            {**asdict(cfg), "scale": scale.tolist()}, indent=1))
    log(f"{name}: {cfg.kind}, {n_params(net) / 1e6:.1f} M parameters, device {device}")

    train_ds = training_windows(cube, scale, cfg.samples_per_epoch, cfg.seed, length=cfg.length)
    val_ds = validation_windows(cube, scale, cfg.val_samples, cfg.seed + 1000, length=cfg.length)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, num_workers=cfg.workers)
    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed * 7919 + step)

    def lr_at(s: int) -> float:
        warm = min(1.0, (s + 1) / cfg.warmup)
        return cfg.lr * warm * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps)))

    steps_csv = out / "steps.csv"
    if not steps_csv.exists():
        steps_csv.write_text("step,epoch,loss,grad_norm,lr,steps_per_s,elapsed_s\n")
    per_epoch = cfg.samples_per_epoch // cfg.batch
    t0, run_loss, run_n = time.time(), 0.0, 0
    t_log, log_loss, log_grad, log_n, t_start = time.time(), 0.0, 0.0, 0, time.time()
    while step < cfg.steps:
        train_ds.set_epoch(step // (cfg.samples_per_epoch // cfg.batch))
        loader = DataLoader(train_ds, batch_size=cfg.batch, num_workers=cfg.workers, shuffle=False,
                            drop_last=True, pin_memory=device.type == "cuda")
        skip = step % (cfg.samples_per_epoch // cfg.batch)
        for i, batch in enumerate(loader):
            if i < skip:
                continue                                  # resumed mid-epoch
            if step >= cfg.steps:
                break
            b = to_torch(batch, device)
            b = drop_condition(b, cfg.edm.cond_drop, g) if cfg.kind == "diffusion" else b
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model.loss(b, generator=g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip).item()
            opt.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1 - cfg.ema_decay)
            run_loss += loss.item()
            run_n += 1
            log_loss += loss.item()
            log_grad += grad
            log_n += 1
            step += 1

            if step % cfg.log_every == 0:
                rate = log_n / (time.time() - t_log)
                eta = (cfg.steps - step) / max(rate, 1e-9)
                epoch = step / per_epoch
                log(f"step {step:7d}  epoch {epoch:6.2f}  loss {log_loss / log_n:.4f}  "
                    f"grad {log_grad / log_n:.3f}  lr {lr_at(step):.2e}  {rate:.1f} steps/s  "
                    f"eta {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m")
                with open(steps_csv, "a") as fh:
                    fh.write(f"{step},{epoch:.4f},{log_loss / log_n:.6f},{log_grad / log_n:.6f},"
                             f"{lr_at(step):.6e},{rate:.3f},{time.time() - t_start:.1f}\n")
                t_log, log_loss, log_grad, log_n = time.time(), 0.0, 0.0, 0

            if step % cfg.val_every == 0 or step == cfg.steps:
                rate = run_n / (time.time() - t0)                 # training only, before validating
                val = validation_loss(ema_model, val_loader, device, cfg.kind)
                row = {"step": step, "train_loss": run_loss / max(run_n, 1), "val_loss": val,
                       "lr": lr_at(step), "steps_per_s": round(rate, 2)}
                history.append(row)
                log(f"VAL  step {step:7d}  epoch {step / per_epoch:6.2f}  train {row['train_loss']:.4f}  "
                    f"val {val:.4f}  lr {row['lr']:.2e}  {rate:.1f} steps/s")
                torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                            "step": step}, ckpt)
                (out / "history.json").write_text(json.dumps(history, indent=1))
                t0, run_loss, run_n = time.time(), 0.0, 0
                t_log = time.time()                               # validation time is not training speed
    return out


def load(model_dir: Path, device=None) -> tuple[str, Net, TrainConfig, np.ndarray]:
    """(kind, EMA network in eval mode, config, component scale) of a trained model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((model_dir / "config.json").read_text())
    cfg = config_from_json(meta)
    net = Net(cfg.net).to(device)
    state = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=False)
    net.load_state_dict(state["ema"])
    return cfg.kind, net.eval(), cfg, np.asarray(meta["scale"], dtype=np.float32)
