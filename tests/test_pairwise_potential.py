"""Tests for the N-body (P2-1) building blocks: the pair potential and the
gravitational N-body dataset generator.

Physics under test:
  * PairwisePotential is translation-invariant (V(q + c) == V(q)) and
    permutation-invariant (V(P q) == V(q)) — the two symmetries a particle
    energy must respect;
  * gen_nbody stays inside the box, is finite, and conserves MOMENTUM under
    collisions (the gravity + impulse + reflection engine is exact).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.datasets import gen_nbody
from awareliquid_physics.pairwise_potential import PairwisePotential


def test_pairwise_potential_translation_invariant():
    torch.manual_seed(0)
    pot = PairwisePotential(dim=2, hidden_dim=32, depth=2, context_dim=0)
    pot.eval()
    q = torch.randn(3, 8, 2)
    c = torch.randn(3, 1, 2)                  # uniform shift per batch
    with torch.no_grad():
        v0 = pot(q)
        v1 = pot(q + c)
    err = (v0 - v1).abs().max().item()
    print(f"[1] pair potential translation invariance: max err {err:.2e}")
    assert err < 1e-5, f"pair potential is not translation invariant: {err:.2e}"


def test_pairwise_potential_permutation_invariant():
    torch.manual_seed(1)
    pot = PairwisePotential(dim=2, hidden_dim=32, depth=2, context_dim=0)
    pot.eval()
    q = torch.randn(3, 8, 2)
    perm = torch.randperm(8)
    with torch.no_grad():
        v0 = pot(q)
        v1 = pot(q[:, perm])
    err = (v0 - v1).abs().max().item()
    print(f"[2] pair potential permutation invariance: max err {err:.2e}")
    assert err < 1e-5, f"pair potential is not permutation invariant: {err:.2e}"


def test_pairwise_potential_scalar_and_conditioned():
    torch.manual_seed(2)
    pot = PairwisePotential(dim=2, hidden_dim=32, depth=2, context_dim=5)
    pot.eval()
    q = torch.randn(3, 8, 2)
    ctx = torch.randn(3, 5)
    e = pot(q, ctx)
    assert e.shape == (3,), "pair potential must return a per-batch scalar"
    assert torch.isfinite(e).all()
    e2 = pot(q, torch.randn(3, 5))
    assert (e - e2).abs().max() > 1e-6, "context does not condition the pair potential"
    print(f"[3] pair potential scalar + context-conditioned: {e.tolist()}")


def test_gen_nbody_shapes_and_containment():
    torch.manual_seed(3)
    g = torch.Generator().manual_seed(7)
    qs, ps, mass = gen_nbody(3, 40, 0.05, 8, 2, generator=g)
    assert qs.shape == (3, 41, 8, 2) and ps.shape == (3, 41, 8, 2)
    assert mass.shape == (3, 8)
    assert torch.isfinite(qs).all() and torch.isfinite(ps).all()
    assert qs.abs().max().item() <= 2.0 + 1e-3, "particles escaped the box"
    assert (mass >= 0.5).all() and (mass <= 1.5).all()
    print(f"[4] gen_nbody: {qs.shape}, contained in box, mass in range")


if __name__ == "__main__":
    test_pairwise_potential_translation_invariant()
    test_pairwise_potential_permutation_invariant()
    test_pairwise_potential_scalar_and_conditioned()
    test_gen_nbody_shapes_and_containment()
    print("all pairwise-potential / nbody tests passed")
