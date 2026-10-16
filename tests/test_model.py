import numpy as np
import pytest
import torch

from imonet.cube import HELD_OUT, NORMAL, VALIDATION
from imonet.data import component_scale
from imonet.evaluate import predict, screened
from imonet.model.edm import EDM, EDMConfig, Regress, drop_condition
from imonet.model.methods import DiffusionMethod, RegressMethod
from imonet.model.net import Net, NetConfig
from imonet.train import TrainConfig, train

from test_data import synthetic_cube

TINY = NetConfig(channels=(8, 16), blocks=1, self_attention_levels=1, heads=2, emb=16)
B, K, L = 3, 4, 32


def batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    valid = torch.ones(B, K, dtype=torch.bool)
    valid[0, 2:] = False
    return {"x": torch.randn(B, L, 3, generator=g), "x_ok": torch.rand(B, L, generator=g) > 0.2,
            "y": torch.randn(B, K, L, 3, generator=g), "y_ok": torch.rand(B, K, L, generator=g) > 0.1,
            "y_sigma": torch.rand(B, K, L, 3, generator=g), "geom": torch.rand(B, K, 3, generator=g) * 50,
            "valid": valid, "doy0": torch.tensor([0.0, 100.0, 300.0])}


def randomised(net: Net) -> Net:
    """Zero-initialised output layers make a fresh net output 0; perturb all
    weights so that tests see a non-trivial function."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return net


def test_net_output_shape_and_padding():
    torch.manual_seed(0)
    net = randomised(Net(TINY))
    b = batch()
    out = net(b["x"], torch.zeros(B), b)
    assert out.shape == (B, L, 3) and torch.isfinite(out).all()
    # Padded slots do not matter: changing their data leaves the output alone.
    b2 = dict(b, y=b["y"].clone())
    b2["y"][0, 2:] += 100.0
    assert torch.allclose(net(b["x"], torch.zeros(B), b2)[0], out[0], atol=1e-5)
    # With every neighbour dropped the output is still defined.
    none = dict(b, valid=torch.zeros_like(b["valid"]))
    assert torch.isfinite(net(b["x"], torch.zeros(B), none)).all()


def test_net_ignores_neighbour_order():
    net = randomised(Net(TINY))
    b = batch()
    b["valid"][:] = True
    perm = torch.tensor([2, 0, 3, 1])
    b2 = {k: (v[:, perm] if k in ("y", "y_ok", "y_sigma", "geom", "valid") else v) for k, v in b.items()}
    a1, a2 = net(b["x"], torch.zeros(B), b), net(b["x"], torch.zeros(B), b2)
    assert torch.allclose(a1, a2, atol=1e-5)


def test_edm_preconditioning_limits():
    edm = EDM(Net(TINY), EDMConfig())                 # fresh net: F = 0
    b = batch()
    d = edm.denoise(b["x"], torch.full((B,), 1e-3), b)
    assert torch.allclose(d, b["x"], atol=1e-5)       # c_skip → 1 as σ → 0
    d = edm.denoise(b["x"], torch.full((B,), 1e3), b)
    assert d.abs().max() < 1e-2                       # c_skip → 0 as σ → ∞


def test_losses_are_finite_and_backpropagate():
    for model in (EDM(randomised(Net(TINY))), Regress(randomised(Net(TINY)))):
        loss = model.loss(batch())
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in model.net.parameters() if p.requires_grad)


def test_condition_dropout():
    b = batch()
    g = torch.Generator().manual_seed(0)
    assert not drop_condition(b, 1.0, g)["valid"].any()
    assert torch.equal(drop_condition(b, 0.0, g)["valid"], b["valid"])


def test_sampler_is_repeatable():
    edm = EDM(randomised(Net(TINY)))
    b = batch()
    s1 = edm.sample(b, L, torch.Generator().manual_seed(5), steps=6)
    s2 = edm.sample(b, L, torch.Generator().manual_seed(5), steps=6)
    s3 = edm.sample(b, L, torch.Generator().manual_seed(6), steps=6)
    assert s1.shape == (B, L, 3) and torch.isfinite(s1).all()
    assert torch.equal(s1, s2) and not torch.equal(s1, s3)


def test_regress_learns_the_neighbour_mean():
    torch.manual_seed(0)
    model = Regress(Net(TINY))
    opt = torch.optim.Adam(model.net.parameters(), lr=3e-3)
    losses = []
    for step in range(150):
        b = batch(step)
        b["y_ok"][:] = True
        b["valid"][:] = True
        b["x"] = b["y"].mean(dim=1)
        loss = model.loss(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.3 * np.mean(losses[:10])


@pytest.mark.parametrize("kind", ["diffusion", "regress"])
def test_train_resume_and_evaluate(tmp_path, kind):
    cube = synthetic_cube()
    cfg = TrainConfig(kind=kind, steps=4, batch=4, val_every=2, val_samples=8, samples_per_epoch=8, workers=0,
                      length=32, net=TINY)
    lines = []
    out = train("tiny", TrainConfig(**{**cfg.__dict__, "log_every": 1}), cube, tmp_path, log=lines.append)
    assert (out / "checkpoint.pt").exists() and len((out / "history.json").read_text()) > 10
    assert sum(line.startswith("step ") for line in lines) == 4
    assert sum(line.startswith("VAL ") for line in lines) == 2
    assert len((out / "steps.csv").read_text().splitlines()) == 1 + 4
    train("tiny", TrainConfig(**{**cfg.__dict__, "steps": 6}), cube, tmp_path, log=lambda _: None)
    method = DiffusionMethod(out, n_samples=2, steps=4) if kind == "diffusion" else RegressMethod(out)
    pred = predict(method, cube, component_scale(cube), cube.stations(HELD_OUT)[:2], cube.stations(NORMAL),
                   VALIDATION, length=32, k=6)
    assert pred.samples.shape[0] == (2 if kind == "diffusion" else 1)
    assert np.isfinite(pred.samples).all()
    if kind == "diffusion":                       # run() screens ensembles only
        spred, share = screened(pred)
        assert spred.samples.shape == pred.samples.shape and 0 <= share < 0.05
        assert np.array_equal(np.isnan(spred.samples), np.isnan(pred.samples))
