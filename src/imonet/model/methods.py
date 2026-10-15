"""Trained models as evaluation methods (`imonet.baselines.Method`).

`sample(batch, rng)` takes a numpy batch of contexts and returns B×m×L×3 in
scaled units. The diffusion model encodes the neighbours once, repeats the
batch m times and runs the sampler; its initial noise comes from a torch
generator seeded from `rng`, so a run is repeatable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from imonet.baselines import Method
from imonet.model.edm import EDM, Regress
from imonet.train import load, to_torch

SAMPLE_CHUNK = 512           # sequences per sampler call (8 GB GPU)


class DiffusionMethod(Method):
    name = "diffusion"

    def __init__(self, model_dir: Path, n_samples: int = 32, steps: int = 32, guidance: float = 1.0):
        kind, net, cfg, self.scale = load(model_dir)
        if kind != "diffusion":
            raise ValueError(f"{model_dir} is a {kind} model")
        self.edm, self.n_samples, self.steps, self.guidance = EDM(net, cfg.edm), n_samples, steps, guidance
        self.device = next(net.parameters()).device

    def sample(self, batch, rng):
        b, _, length, _ = batch["y"].shape
        cond = to_torch(batch, self.device)
        rep = {k: v.repeat_interleave(self.n_samples, dim=0) for k, v in cond.items()}   # B·m rows
        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng.integers(2 ** 62)))
        out = []
        for a in range(0, b * self.n_samples, SAMPLE_CHUNK):
            part = {k: v[a:a + SAMPLE_CHUNK] for k, v in rep.items()}
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out.append(self.edm.sample(part, length, g, self.steps, self.guidance).float())
        x = torch.cat(out).reshape(b, self.n_samples, length, 3)
        return x.cpu().numpy()


class RegressMethod(Method):
    name = "regress"

    def __init__(self, model_dir: Path):
        kind, net, _, self.scale = load(model_dir)
        if kind != "regress":
            raise ValueError(f"{model_dir} is a {kind} model")
        self.model = Regress(net)
        self.device = next(net.parameters()).device

    @torch.no_grad()
    def sample(self, batch, rng):
        cond = to_torch(batch, self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            x = self.model.predict(cond, batch["y"].shape[2])
        return x.float().cpu().numpy()[:, None]
