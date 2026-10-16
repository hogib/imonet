"""EDM preconditioning, loss and sampler (Karras et al. 2022), and the
squared-error twin `regress` (docs/00-design.md §4.4, §5).

D(x; σ) = c_skip(σ) x + c_out(σ) F(c_in(σ) x, c_noise(σ); cond), with
c_skip = σd²/(σ² + σd²), c_out = σ σd/√(σ² + σd²), c_in = 1/√(σ² + σd²),
c_noise = ln(σ)/4. Training draws ln σ ~ N(P_mean, P_std²) and weights the
squared error by (σ² + σd²)/(σ σd)². Target days without data are left out
of the loss.

Sampling solves the probability-flow ODE with Heun's method on the σ
schedule σ_i = (σ_max^{1/ρ} + i/(N−1) (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ. It is
deterministic given the initial noise, which comes from a seeded generator,
so paired runs (experiment B) are exactly repeatable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from imonet.model.net import Net

COND_KEYS = ("y", "y_ok", "y_sigma", "geom", "valid", "doy0", "target_noise", "target_noise_ok")


@dataclass
class EDMConfig:
    sigma_data: float = 1.0      # std of the detrended, scaled targets is about 0.9
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    cond_drop: float = 0.1       # share of training samples with every neighbour dropped


def drop_condition(cond: dict, p: float, generator: torch.Generator | None = None) -> dict:
    """Mark every neighbour of a random share p of the samples invalid, so
    the model also learns p(x) for classifier-free guidance."""
    if p <= 0:
        return cond
    b = cond["valid"].shape[0]
    keep = torch.rand(b, device=cond["valid"].device, generator=generator) >= p
    return cond | {"valid": cond["valid"] & keep[:, None]}


def drop_station_noise(cond: dict, p: float, generator: torch.Generator | None = None) -> dict:
    """Mark the target's noise level unknown for a random share p of the
    samples, so the model also learns to generate without it."""
    if p <= 0 or "target_noise_ok" not in cond:
        return cond
    ok = cond["target_noise_ok"]
    keep = torch.rand(ok.shape[0], device=ok.device, generator=generator) >= p
    return cond | {"target_noise_ok": ok & keep}


def unconditional(cond: dict) -> dict:
    return cond | {"valid": torch.zeros_like(cond["valid"])}


class EDM:
    def __init__(self, net: Net, cfg: EDMConfig = EDMConfig()):
        self.net, self.cfg = net, cfg

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, cond: dict,
                feats: list[torch.Tensor] | None = None) -> torch.Tensor:
        sd = self.cfg.sigma_data
        s = sigma.reshape(-1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / torch.sqrt(s ** 2 + sd ** 2)
        c_in = 1 / torch.sqrt(s ** 2 + sd ** 2)
        f = self.net(c_in * x, torch.log(sigma) / 4, cond, feats)
        return c_skip * x + c_out * f.float()

    def loss(self, batch: dict, sigma: torch.Tensor | None = None, noise: torch.Tensor | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """Weighted denoising loss, averaged over target days with data.
        `sigma` and `noise` fix the draw (validation)."""
        x, ok = batch["x"], batch["x_ok"].float()
        b = x.shape[0]
        if sigma is None:
            sigma = torch.exp(torch.randn(b, device=x.device, generator=generator) * self.cfg.p_std
                              + self.cfg.p_mean)
        if noise is None:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
        sd = self.cfg.sigma_data
        weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
        d = self.denoise(x + noise * sigma.reshape(-1, 1, 1), sigma, batch)
        err = ((d - x) ** 2 * ok[..., None]).sum(dim=(1, 2)) / (3 * ok.sum(dim=1)).clamp(min=1)
        return (weight * err).mean()

    def schedule(self, steps: int, device) -> torch.Tensor:
        c = self.cfg
        i = torch.arange(steps, device=device, dtype=torch.float64)
        s = (c.sigma_max ** (1 / c.rho) + i / (steps - 1) * (c.sigma_min ** (1 / c.rho) - c.sigma_max ** (1 / c.rho))) ** c.rho
        return torch.cat([s, torch.zeros(1, device=device, dtype=torch.float64)]).float()

    @torch.no_grad()
    def sample(self, cond: dict, length: int, generator: torch.Generator | None, steps: int = 32,
               guidance: float = 1.0, noise: torch.Tensor | None = None) -> torch.Tensor:
        """One sample per row of `cond`: B×L×3. guidance w > 1 uses
        D = D_uncond + w (D_cond − D_uncond). The sampler is deterministic
        given its initial noise: unit Gaussian `noise` (B×L×3) if given,
        else drawn from `generator`."""
        b = cond["valid"].shape[0]
        device = cond["valid"].device
        feats = self.net.encode(cond)
        if guidance != 1.0:
            ucond = unconditional(cond)
            ufeats = self.net.encode(ucond)

        def d(x, s):
            sig = torch.full((b,), float(s), device=device)
            out = self.denoise(x, sig, cond, feats)
            if guidance != 1.0:
                u = self.denoise(x, sig, ucond, ufeats)
                out = u + guidance * (out - u)
            return out

        sig = self.schedule(steps, device)
        if noise is None:
            noise = torch.randn((b, length, 3), device=device, generator=generator)
        x = noise.to(device) * sig[0]
        for i in range(steps):
            s0, s1 = sig[i], sig[i + 1]
            dx = (x - d(x, s0)) / s0
            x1 = x + (s1 - s0) * dx
            if s1 > 0:
                dx1 = (x1 - d(x1, s1)) / s1
                x1 = x + (s1 - s0) * 0.5 * (dx + dx1)
            x = x1
        return x


class Regress:
    """The same network as a point predictor: F(0, 0; cond), trained with
    squared error on the target days with data."""

    def __init__(self, net: Net):
        self.net = net

    def predict(self, cond: dict, length: int) -> torch.Tensor:
        b = cond["valid"].shape[0]
        device = cond["valid"].device
        zeros = torch.zeros((b, length, 3), device=device)
        return self.net(zeros, torch.zeros(b, device=device), cond).float()

    def loss(self, batch: dict, **_) -> torch.Tensor:
        x, ok = batch["x"], batch["x_ok"].float()
        err = ((self.predict(batch, x.shape[1]) - x) ** 2 * ok[..., None]).sum(dim=(1, 2))
        return (err / (3 * ok.sum(dim=1)).clamp(min=1)).mean()


def n_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
