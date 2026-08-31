"""Tests for the liquid (LTC) substrate: the parallel scan and the LiquidCore
multi-timescale recurrence."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
    assert core.tau_gate.weight.grad.abs().sum() > 0, "liquid tau gate got no gradient"
    print("[4] LiquidCore gradients finite, timescales + tau gate train")


def test_tau_is_input_dependent():
    """v0.2 liquid property: the effective time constants must CHANGE with the
    input (input-dependent tau). Also pins v0.1 compatibility: with the tau gate
    zeroed, tau reduces exactly to the static ladder."""
    torch.manual_seed(4)
    core = LiquidCore(d_in=4, d_model=16, n_scales=4)
    core.eval()
    xa = torch.randn(2, 10, 4)
    xb = torch.randn(2, 10, 4)

    def taus(x):
        with torch.no_grad():
            u = core.in_proj(x)
            g = core.tau_gate(u).permute(0, 2, 1).unsqueeze(-1)
            S, D = core.n_scales, core.d_model
            return F.softplus(core.log_tau.view(1, S, 1, D) + g) + core.tau_min

    # zero gate (the initialisation): tau must be exactly the static ladder,
    # identical for any input — this is the v0.1-compatible starting point.
    with torch.no_grad():
        core.tau_gate.weight.zero_()
        core.tau_gate.bias.zero_()
    tau_static_a, tau_static_b = taus(xa), taus(xb)
    assert (tau_static_a - tau_static_b).abs().max() < 1e-6, \
        "zeroed tau gate must recover the static (input-independent) ladder"

    # after the gate learns (non-zero weights), tau must track the input.
    with torch.no_grad():
        core.tau_gate.weight.normal_(std=0.5)
        core.tau_gate.bias.normal_(std=0.5)
    tau_a, tau_b = taus(xa), taus(xb)
    assert (tau_a - tau_b).abs().max() > 1e-4, "tau does not depend on the input"
    assert (tau_a >= core.tau_min).all() and (tau_b >= core.tau_min).all(), \
        "tau fell below tau_min (stability bound)"
    dec_a = torch.exp(-core.dt / tau_a)
    assert (dec_a > 0).all() and (dec_a < 1).all(), "decay escaped (0,1)"
    print("[5] tau is input-dependent (liquid) and bounded: "
          f"tau range [{tau_a.min():.2f}, {tau_a.max():.2f}]")


def test_liquid_core_matches_sequential_with_variable_A():
    """Full-path numerical check: with input-dependent tau the scan multiplier A
    varies along T, so the whole LiquidCore forward must equal a sequential
    hand-rolled recurrence (pscan_sequential) — pins the fold-dims trick + the
    general pscan call on a non-power-of-2 T."""
    torch.manual_seed(5)
    core = LiquidCore(d_in=3, d_model=8, n_scales=3)
    core.eval()
    x = torch.randn(2, 11, 3)          # T = 11, not a power of two
    with torch.no_grad():
        h_par = core(x)

        # Reference: replicate the forward internals with the O(T) sequential scan.
        B, T, D = x.shape[0], x.shape[1], core.d_model
        S = core.n_scales
        u = core.in_proj(x)
        gt = core.tau_gate(u).permute(0, 2, 1).unsqueeze(-1)      # (B,S,T,1)
        tau = F.softplus(core.log_tau.view(1, S, 1, D) + gt) + core.tau_min
        decay = torch.exp(-core.dt / tau)                          # (B,S,T,D)
        X = (1.0 - decay) * u.unsqueeze(1)                         # (B,S,T,D)
        X5 = X.permute(0, 1, 3, 2).unsqueeze(-1)                   # (B,S,D,T,1)
        A = decay.permute(0, 1, 3, 2)                              # (B,S,D,T)
        H5 = pscan_sequential(A, X5)                               # (B,S,D,T,1)
        H = H5.squeeze(-1).permute(0, 1, 3, 2)                     # (B,S,T,D)
        gate = torch.sigmoid(core.kappa_gate(u)).permute(0, 2, 1).unsqueeze(-1)
        w = F.softmax(core.blend, dim=-1).view(1, S, 1, 1)
        gw = w * gate
        gw = gw / gw.sum(dim=1, keepdim=True).clamp_min(1e-6)
        h_ref = core.out_proj((H * gw).sum(dim=1))
    err = (h_par - h_ref).abs().max().item()
    print(f"[6] liquid core (variable A) vs sequential max err {err:.2e}")
    assert err < 1e-4, f"parallel liquid core disagrees with sequential ref: {err:.2e}"


if __name__ == "__main__":
    test_pscan_matches_sequential_reference()
    test_liquid_core_shapes_and_finite()
    test_liquid_core_is_causal()
    test_liquid_core_gradient_flows()
    test_tau_is_input_dependent()
    test_liquid_core_matches_sequential_with_variable_A()
    print("all liquid_core tests passed")
