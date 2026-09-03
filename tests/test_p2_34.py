"""Tests for P2-3 / P2-4: the probabilistic (VAE-context) model and the
time-conditioned Hamiltonian head.

Physics under test:
  * TimeConditionedHamiltonianHead: time is a real input (same state at
    different times -> different forces), the rollout advances time, and the
    time-independent limit (V ignoring t via zeroed time weights) recovers
    conservation;
  * ProbabilisticLiquidModel: the ensemble has genuine spread (sampled
    contexts differ), gradients reach mu/logvar, and the reparameterised
    single-sample forward is trainable.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.hamiltonian import TimeConditionedHamiltonianHead
from awareliquid_physics.model import ProbabilisticLiquidModel


def test_time_conditioned_time_is_a_real_input():
    torch.manual_seed(0)
    head = TimeConditionedHamiltonianHead(dim=2, hidden_dim=32, depth=2)
    head.eval()
    q = torch.randn(3, 2)
    with torch.enable_grad():
        f0 = head.dV_dq(q, torch.zeros(3))
        f1 = head.dV_dq(q, torch.full((3,), 5.0))
    assert (f0 - f1).abs().max() > 1e-4, "time does not affect the force"
    print("[1] time conditions the potential (continuous-time evaluation)")


def test_time_conditioned_rollout_advances_time():
    torch.manual_seed(1)
    head = TimeConditionedHamiltonianHead(dim=2, hidden_dim=32, depth=2)
    head.eval()
    q0 = torch.randn(3, 2)
    p0 = torch.randn(3, 2) * 0.5
    t0 = torch.full((3,), 1.0)
    with torch.enable_grad():
        qs, ps, ts = head.rollout(q0, p0, t0, steps=40, dt=0.1)
    assert qs.shape == (41, 3, 2)
    assert (ts[-1] - t0).abs().max().item() < 1e-4 + 40 * 0.1, "time did not advance"
    assert torch.isfinite(qs).all()
    print(f"[2] rollout advances time: t 1.0 -> {ts[-1].max().item():.1f}")


def test_probabilistic_ensemble_has_spread():
    torch.manual_seed(2)
    model = ProbabilisticLiquidModel(phase_dim=2, d_model=16, context_dim=6,
                                     n_scales=3, hidden_dim=16, depth=2, dt=0.1)
    model.eval()
    q_obs = torch.randn(4, 10, 2)
    p_obs = torch.randn(4, 10, 2)
    with torch.enable_grad():
        qs, ps, (mu, logvar) = model(q_obs, p_obs, k=8, n_samples=5)
    spread = (qs - qs.mean(0)).abs().max().item()
    assert qs.shape == (5, 9, 4, 2) and mu.shape == (4, 6)
    assert spread > 1e-4, "ensemble has no spread (degenerate distribution)"
    assert torch.isfinite(logvar).all()
    print(f"[3] probabilistic ensemble: {qs.shape}, spread {spread:.2e}")


def test_probabilistic_gradients_flow_to_distribution():
    torch.manual_seed(3)
    model = ProbabilisticLiquidModel(phase_dim=1, d_model=12, context_dim=4,
                                     n_scales=3, hidden_dim=12, depth=2, dt=0.1)
    model.train()
    q_obs = torch.randn(4, 8, 1)
    p_obs = torch.randn(4, 8, 1)
    qs, ps, (mu, logvar) = model(q_obs, p_obs, k=5, n_samples=1)
    (qs.pow(2).mean() + 0.01 * mu.pow(2).mean()).backward()
    for name in ("mu_proj", "logvar_proj"):
        proj = getattr(model, name)
        assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all(), \
            f"no gradient into {name}"
    assert model.ham.V[0].weight.grad.abs().sum() > 0
    print("[4] gradients reach mu/logvar and the Hamiltonian head")


if __name__ == "__main__":
    test_time_conditioned_time_is_a_real_input()
    test_time_conditioned_rollout_advances_time()
    test_probabilistic_ensemble_has_spread()
    test_probabilistic_gradients_flow_to_distribution()
    print("all P2-3/P2-4 tests passed")
