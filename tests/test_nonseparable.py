"""Tests for the non-separable Hamiltonian head (P2-2): implicit-midpoint
integration of H(q,p) = T(p) + V(q) + C(q,p) with a velocity-dependent
coupling term (magnetic / Coriolis forces).

Physics under test:
  * implicit midpoint is symplectic for ANY smooth H: a random untrained
    head with a live coupling C must show bounded, non-growing energy drift;
  * the coupling C actually couples: dC/dp != 0 (velocity-dependent force);
  * gradients flow into T, V and C (the coupling is trainable).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.hamiltonian import NonseparableHamiltonianHead


def test_nonseparable_conserves_energy_by_construction():
    torch.manual_seed(0)
    head = NonseparableHamiltonianHead(dim=2, hidden_dim=32, depth=2,
                                       context_dim=0, iters=4)
    head.eval()
    q0 = torch.randn(3, 2)
    p0 = torch.randn(3, 2) * 0.5
    with torch.enable_grad():
        qs, ps = head.rollout(q0, p0, steps=200, dt=0.02)
        def E(t):
            return head.energy(qs[t].detach(), ps[t].detach()).detach()
        e0, e100, e200 = E(0), E(100), E(200)
        d100 = ((e100 - e0).abs() / e0.abs().clamp_min(1e-6)).max().item()
        d200 = ((e200 - e0).abs() / e0.abs().clamp_min(1e-6)).max().item()
    print(f"[1] nonseparable head drift: 100 steps {d100:.2e}, 200 steps {d200:.2e}")
    assert d200 < 0.05, f"unbounded drift on nonseparable head: {d200:.2e}"
    assert d200 < 10 * d100 + 1e-4, "drift grows secularly"


def test_nonseparable_coupling_is_velocity_dependent():
    """C(q,p) must produce a real velocity-dependent force: dC/dp depends on p."""
    torch.manual_seed(1)
    head = NonseparableHamiltonianHead(dim=2, hidden_dim=32, depth=2)
    head.eval()
    q = torch.randn(3, 2)
    pa, pb = torch.randn(3, 2), torch.randn(3, 2)
    with torch.enable_grad():
        fa = head.dC_dp(q, pa)
        fb = head.dC_dp(q, pb)
    assert (fa - fb).abs().max() > 1e-4, "coupling force does not depend on velocity"
    print("[2] coupling C(q,p) is genuinely velocity-dependent")


def test_nonseparable_gradients_flow_to_all_terms():
    torch.manual_seed(2)
    head = NonseparableHamiltonianHead(dim=2, hidden_dim=16, depth=2)
    head.train()
    q0 = torch.randn(3, 2)
    p0 = torch.randn(3, 2)
    with torch.enable_grad():
        qs, ps = head.rollout(q0, p0, steps=5, dt=0.05)
        qs.pow(2).mean().backward()
    for name in ("T", "V", "C"):
        net = getattr(head, name)
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), \
            f"no/finite gradients into {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"{name} got no gradient"
    print("[3] gradients flow into T, V and C (all trainable)")


if __name__ == "__main__":
    test_nonseparable_conserves_energy_by_construction()
    test_nonseparable_coupling_is_velocity_dependent()
    test_nonseparable_gradients_flow_to_all_terms()
    print("all nonseparable tests passed")
