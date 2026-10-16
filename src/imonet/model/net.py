"""The network: a 1D U-Net over time for the target, conditioned on the
neighbours by cross-attention (docs/00-design.md §4.4).

The neighbour encoder runs once per sample, independent of the noise level,
and returns features at every U-Net resolution: for each neighbour a temporal
convolution stack over its series, availability and σ, plus an embedding of
its offset and distance from the target. At every resolution each time step
of the target attends to the neighbours at the same time step. A learned
null token is always among the keys, so a sample with no neighbours (padding,
or conditioning dropped for classifier-free guidance) attends to it alone.

Batch tensors, as `imonet.data` makes them (B batch, K neighbours, L days):
y B×K×L×3, y_ok B×K×L, y_sigma B×K×L×3, geom B×K×3 (east, north, distance
in km), valid B×K, doy0 B.

Optionally (`NetConfig.station_noise`), the target station's own noise level
conditions every block alongside the diffusion noise level: target_noise B×3,
the log of its day-to-day scale per component relative to a typical station
(`metrics.station_noise_log`), and target_noise_ok B, whether it is given.
Without it, or where it is not given, the model hedges over the stations that
could stand at the location; with it, the user chooses how noisy the virtual
station is. Its embedding starts at zero, so training starts from the
unconditioned network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GEOM_SCALES_KM = (10.0, 30.0, 100.0, 300.0)


@dataclass
class NetConfig:
    channels: tuple[int, ...] = (64, 128, 192, 256)
    blocks: int = 2                    # residual blocks per level
    self_attention_levels: int = 2     # coarsest levels with temporal self-attention
    heads: int = 4
    emb: int = 256                     # noise-level embedding width
    dropout: float = 0.0
    station_noise: bool = False        # optional input: the target station's own noise level


def _groups(c: int) -> int:
    return math.gcd(32, c)


class ResBlock(nn.Module):
    """GroupNorm–SiLU–conv twice, with a scale and shift from the embedding
    (FiLM) when one is given."""

    def __init__(self, cin: int, cout: int, emb: int | None, dropout: float = 0.0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(_groups(cin), cin), nn.Conv1d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(_groups(cout), cout), nn.Conv1d(cout, cout, 3, padding=1)
        self.film = nn.Linear(emb, 2 * cout) if emb else None
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.film is not None:
            scale, shift = self.film(emb)[..., None].chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class SelfAttention(nn.Module):
    """Attention over time steps."""

    def __init__(self, c: int, heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(c), c)
        self.qkv, self.out, self.heads = nn.Linear(c, 3 * c), nn.Linear(c, c), heads
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        q, k, v = self.qkv(self.norm(x).transpose(1, 2)).chunk(3, dim=-1)       # B×T×C each
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, c)
        return x + self.out(h).transpose(1, 2)


class CrossAttention(nn.Module):
    """Each target time step attends to the K neighbours (and a null token)
    at the same time step.

    Computed in the neighbour features' own layout (B×K×C×T): keys and
    values by a 1×1 projection, scores and weighted sums elementwise. With
    one query against K + 1 keys per time step, rearranging the tensors for
    a fused attention kernel costs more than the attention itself. The
    neighbour features arrive normalised (`NeighbourEncoder`)."""

    def __init__(self, c: int, heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(c), c)
        self.q = nn.Conv1d(c, c, 1)
        self.kv = nn.Linear(c, 2 * c)
        self.out = nn.Conv1d(c, c, 1)
        self.null = nn.Parameter(torch.zeros(c))
        self.heads = heads
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, nbr: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x B×C×T, nbr B×K×C×T, valid B×K."""
        b, c, t = x.shape
        k_ = nbr.shape[1]
        h_, d = self.heads, c // self.heads
        q = self.q(self.norm(x)).view(b, 1, h_, d, t)
        kv = torch.einsum("bkct,oc->bkot", nbr, self.kv.weight.to(nbr.dtype)) \
            + self.kv.bias.to(nbr.dtype)[None, None, :, None]                               # B×K×2C×T
        null = self.kv(self.null).to(kv.dtype)[None, None, :, None].expand(b, 1, 2 * c, t)
        kv = torch.cat([null, kv], dim=1)
        kk, vv = kv.view(b, k_ + 1, 2, h_, d, t).unbind(2)                                  # B×(K+1)×H×d×T
        scores = (q * kk).sum(3) / math.sqrt(d)                                             # B×(K+1)×H×T
        keep = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), valid], dim=1)
        w = scores.masked_fill(~keep[:, :, None, None], float("-inf")).softmax(dim=1)
        h = (w[:, :, :, None, :] * vv).sum(1).reshape(b, c, t)
        return x + self.out(h)


def geometry_features(geom: torch.Tensor) -> torch.Tensor:
    """B×K×3 (east, north, distance km) → B×K×F: scaled offsets, log
    distance, and sines and cosines of the offsets at several scales."""
    east, north, dist = geom.unbind(-1)
    feats = [east / 100.0, north / 100.0, torch.log1p(dist) / math.log(300.0)]
    for s in GEOM_SCALES_KM:
        for v in (east, north):
            feats += [torch.sin(2 * math.pi * v / s), torch.cos(2 * math.pi * v / s)]
    return torch.stack(feats, dim=-1)


class NeighbourEncoder(nn.Module):
    """Neighbour features at every U-Net resolution: level i is K×C_i×(L/2^i),
    normalised once here rather than in every cross-attention layer."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        ch = cfg.channels
        self.inp = nn.Conv1d(7, ch[0], 3, padding=1)
        nf = 3 + 4 * len(GEOM_SCALES_KM)
        self.geom = nn.Sequential(nn.Linear(nf, cfg.emb), nn.SiLU(), nn.Linear(cfg.emb, cfg.emb))
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, c in enumerate(ch):
            self.blocks.append(ResBlock(c, c, cfg.emb))
            self.norms.append(nn.GroupNorm(_groups(c), c))
            if i + 1 < len(ch):
                self.downs.append(nn.Conv1d(c, ch[i + 1], 4, stride=2, padding=1))

    def forward(self, cond: dict) -> list[torch.Tensor]:
        y, ok, sig, geom = cond["y"], cond["y_ok"], cond["y_sigma"], cond["geom"]
        b, k, length, _ = y.shape
        x = torch.cat([y, ok[..., None].to(y.dtype), sig], dim=-1)                  # B×K×L×7
        x = x.reshape(b * k, length, 7).transpose(1, 2)
        g = self.geom(geometry_features(geom)).reshape(b * k, -1)                    # BK×E
        h = self.inp(x)
        out = []
        for i, block in enumerate(self.blocks):
            h = block(h, g)                                                          # geometry as FiLM
            out.append(self.norms[i](h).reshape(b, k, h.shape[1], h.shape[2]))
            if i < len(self.downs):
                h = self.downs[i](h)
        return out


def noise_embedding(c_noise: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=c_noise.device) / half)
    a = c_noise[:, None].float() * freqs[None] * 100.0
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def station_input(cond: dict, b: int, device) -> torch.Tensor:
    """B×4: the target's log noise level (0 where not given) and whether it
    is given; not given when the batch has no target_noise."""
    if "target_noise" not in cond:
        return torch.zeros(b, 4, device=device)
    known = cond["target_noise_ok"]
    level = torch.where(known[:, None], cond["target_noise"].float(), 0.0)
    return torch.cat([level, known[:, None].float()], dim=1)


class Net(nn.Module):
    """F(x_in, c_noise, cond): the raw network inside the EDM preconditioning
    (or, for `regress`, the predictor itself with x_in = 0)."""

    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        ch, e, nb = cfg.channels, cfg.emb, cfg.blocks
        self.encoder = NeighbourEncoder(cfg)
        self.emb = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        if cfg.station_noise:
            self.station = nn.Sequential(nn.Linear(3 + 1, e), nn.SiLU(), nn.Linear(e, e))
            nn.init.zeros_(self.station[-1].weight)
            nn.init.zeros_(self.station[-1].bias)
        self.inp = nn.Conv1d(3 + 2, ch[0], 3, padding=1)
        n_lev = len(ch)
        self.down_blocks, self.down_attn, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(ch):
            self.down_blocks.append(nn.ModuleList([ResBlock(c, c, e, cfg.dropout) for _ in range(nb)]))
            self.down_attn.append(nn.ModuleList([CrossAttention(c, cfg.heads) for _ in range(nb)]))
            if i + 1 < n_lev:
                self.downs.append(nn.Conv1d(c, ch[i + 1], 4, stride=2, padding=1))
        c = ch[-1]
        self.mid1, self.mid_self = ResBlock(c, c, e, cfg.dropout), SelfAttention(c, cfg.heads)
        self.mid_cross, self.mid2 = CrossAttention(c, cfg.heads), ResBlock(c, c, e, cfg.dropout)
        self.up_blocks, self.up_attn, self.ups = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.self_attn = nn.ModuleDict()
        for i in reversed(range(n_lev)):
            c = ch[i]
            self.up_blocks.append(nn.ModuleList([ResBlock(2 * c if j == 0 else c, c, e, cfg.dropout)
                                                 for j in range(nb)]))
            self.up_attn.append(nn.ModuleList([CrossAttention(c, cfg.heads) for _ in range(nb)]))
            if i >= n_lev - cfg.self_attention_levels:
                self.self_attn[str(i)] = SelfAttention(c, cfg.heads)
            if i > 0:
                self.ups.append(nn.Conv1d(c, ch[i - 1], 3, padding=1))
        self.out_norm = nn.GroupNorm(_groups(ch[0]), ch[0])
        self.out = nn.Conv1d(ch[0], 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode(self, cond: dict) -> list[torch.Tensor]:
        return self.encoder(cond)

    def forward(self, x_in: torch.Tensor, c_noise: torch.Tensor, cond: dict,
                feats: list[torch.Tensor] | None = None) -> torch.Tensor:
        """x_in B×L×3, c_noise B → B×L×3."""
        feats = self.encode(cond) if feats is None else feats
        valid = cond["valid"]
        b, length, _ = x_in.shape
        t = cond["doy0"][:, None] + torch.arange(length, device=x_in.device)[None]
        phase = 2 * math.pi * t / 365.25
        h = torch.cat([x_in, torch.sin(phase)[..., None], torch.cos(phase)[..., None]], dim=-1)
        h = self.inp(h.transpose(1, 2).to(x_in.dtype))
        emb = self.emb(noise_embedding(c_noise, self.cfg.emb).to(h.dtype))
        if self.cfg.station_noise:
            emb = emb + self.station(station_input(cond, b, x_in.device).to(emb.dtype))

        skips = []
        for i in range(len(self.cfg.channels)):
            for block, attn in zip(self.down_blocks[i], self.down_attn[i]):
                h = attn(block(h, emb), feats[i], valid)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.mid2(self.mid_cross(self.mid_self(self.mid1(h, emb)), feats[-1], valid), emb)
        for n, i in enumerate(reversed(range(len(self.cfg.channels)))):
            h = torch.cat([h, skips[i]], dim=1)
            for block, attn in zip(self.up_blocks[n], self.up_attn[n]):
                h = attn(block(h, emb), feats[i], valid)
            if str(i) in self.self_attn:
                h = self.self_attn[str(i)](h)
            if i > 0:
                h = self.ups[n](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(h))).transpose(1, 2)
