"""Tests for the liquid (LTC) substrate: the parallel scan and the LiquidCore
multi-timescale recurrence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.liquid_core import LiquidCore
from awareliquid_physics.parallel_scan import pscan, pscan_sequential


def test_pscan_matches_sequential_reference():
    """The Blelloch parallel scan must equal the O(T) sequential recurrence, incl.
    a non-power-of-2 T and a non-zero initial state."""
    torch.manual_seed(0)
    B, T, D = 3, 37, 5           # T deliberately not a power of two
    A = torch.rand(B, T) * 0.9 + 0.05     # multipliers in (0,1)
    X = torch.randn(B, T, D)
    h0 = torch.randn(B, D)
    ref = pscan_sequential(A, X, h_init=h0)
    par = pscan(A, X, h_init=h0)
    err = (ref - par).abs().max().item()
    print(f"[1] pscan vs sequential max err {err:.2e}")
    assert err < 1e-4, f"parallel scan disagrees with sequential ref: {err:.2e}"


def test_liquid_core_shapes_and_finite():
    torch.manual_seed(1)
    core = LiquidCore(d_in=6, d_model=32, n_scales=4)
    x = torch.randn(4, 20, 6)
    h = core(x)
    assert h.shape == (4, 20, 32)
    assert torch.isfinite(h).all()
    ctx = core.encode(x)
    assert ctx.shape == (4, 32)
    print("[2] LiquidCore shapes OK (seq + encode)")


def test_liquid_core_is_causal():
    """The recurrence is causal: changing an input at time t must NOT change the
    hidden state at any time < t."""
    torch.manual_seed(2)
    core = LiquidCore(d_in=4, d_model=16, n_scales=3)
    core.eval()
    x = torch.randn(2, 12, 4)
    h = core(x)
    x2 = x.clone()
    x2[:, 8:] += 3.0                     # perturb only from t=8 onward
    h2 = core(x2)
    pre = (h[:, :8] - h2[:, :8]).abs().max().item()
    post = (h[:, 8:] - h2[:, 8:]).abs().max().item()
    print(f"[3] causal: pre-change diff {pre:.2e} (must be ~0), post {post:.2e} (must be >0)")
    assert pre < 1e-5, "recurrence is not causal (past changed by a future input)"
    assert post > 1e-4, "perturbation had no effect at all (degenerate)"


def test_liquid_core_gradient_flows():
    torch.manual_seed(3)
    core = LiquidCore(d_in=4, d_model=16, n_scales=4)
    core.train()
    x = torch.randn(3, 15, 4)
    core(x).pow(2).mean().backward()
    for name, p in core.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {name}"
    assert core.log_tau.grad.abs().sum() > 0, "timescale ladder got no gradient"
    print("[4] LiquidCore gradients finite, timescales train")


if __name__ == "__main__":
    test_pscan_matches_sequential_reference()
    test_liquid_core_shapes_and_finite()
    test_liquid_core_is_causal()
    test_liquid_core_gradient_flows()
    print("all liquid_core tests passed")
