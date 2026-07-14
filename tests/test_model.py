"""Tests for the coupled LiquidHamiltonianModel — physics ON the liquid substrate.

These pin the integration itself: the liquid core reads a prefix into a context,
that context conditions the Hamiltonian potential, gradients flow end-to-end
through BOTH the core and the physics head, and different prefixes produce
different dynamics (system identification actually happens)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.model import GRUSeqModel, LiquidHamiltonianModel


def _obs(B, T, dim, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, T, dim, generator=g), torch.randn(B, T, dim, generator=g))


def test_forward_shapes():
    torch.manual_seed(0)
    model = LiquidHamiltonianModel(phase_dim=2, d_model=32, context_dim=8, dt=0.1)
    q_obs, p_obs = _obs(4, 10, 2, 0)
    qs, ps, ctx = model(q_obs, p_obs, k=15)
    assert qs.shape == (16, 4, 2) and ps.shape == (16, 4, 2)   # k+1 including start
    assert ctx.shape == (4, 8)
    assert torch.isfinite(qs).all() and torch.isfinite(ps).all()
    print("[1] coupled model forward shapes OK")


def test_prefix_changes_context_and_rollout():
    """Two genuinely different prefixes must yield different inferred contexts and
    therefore different rollouts — proof the liquid core's system-ID is wired into
    the physics, not ignored."""
    torch.manual_seed(1)
    model = LiquidHamiltonianModel(phase_dim=2, d_model=32, context_dim=8, dt=0.1)
    model.eval()
    qa, pa = _obs(3, 12, 2, 1)
    qb, pb = _obs(3, 12, 2, 2)
    with torch.enable_grad():
        ca = model.infer_context(qa, pa)
        cb = model.infer_context(qb, pb)
        # roll BOTH from the SAME initial state so any difference is due to context
        q0, p0 = torch.zeros(3, 2), torch.ones(3, 2)
        qra, _ = model.rollout(q0, p0, ca, 20)
        qrb, _ = model.rollout(q0, p0, cb, 20)
    assert (ca - cb).abs().max() > 1e-4, "different prefixes gave the same context"
    assert (qra - qrb).abs().max() > 1e-4, "context does not affect the rollout"
    print("[2] different prefix -> different context -> different rollout")


def test_gradient_flows_through_core_and_ham():
    """A trajectory-prediction loss must backprop into BOTH the liquid core (incl.
    its timescales) and the Hamiltonian head (T and V) — the end-to-end coupling."""
    torch.manual_seed(2)
    model = LiquidHamiltonianModel(phase_dim=1, d_model=24, context_dim=6,
                                   hidden_dim=24, dt=0.1)
    model.train()
    q_obs, p_obs = _obs(5, 8, 1, 3)
    qs, ps, _ = model(q_obs, p_obs, k=6)
    qs.pow(2).mean().backward()
    checks = {
        "core.log_tau": model.core.log_tau.grad,
        "core.in_proj": model.core.in_proj.weight.grad,
        "context_proj": model.context_proj.weight.grad,
        "ham.T": model.ham.T[0].weight.grad,
        "ham.V": model.ham.V[0].weight.grad,
    }
    for name, g in checks.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"no/!finite gradient into {name}"
    print("[3] gradient flows end-to-end through liquid core AND Hamiltonian head")


def test_gru_baseline_interface_matches():
    """The GRU control obeys the same forward(q_obs, p_obs, k) contract so the
    benchmark can swap it in."""
    torch.manual_seed(3)
    gru = GRUSeqModel(phase_dim=2, hidden=32)
    q_obs, p_obs = _obs(4, 10, 2, 4)
    qs, ps, ctx = gru(q_obs, p_obs, k=12)
    assert qs.shape == (13, 4, 2) and ps.shape == (13, 4, 2)
    assert ctx is None
    assert torch.isfinite(qs).all()
    print("[4] GRU baseline matches the model interface")


if __name__ == "__main__":
    test_forward_shapes()
    test_prefix_changes_context_and_rollout()
    test_gradient_flows_through_core_and_ham()
    test_gru_baseline_interface_matches()
    print("all model tests passed")
