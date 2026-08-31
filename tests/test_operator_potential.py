"""Tests for the v0.2 (M2) operator-potential stack: the spectral FNO layers,
the OperatorHamiltonianHead, and the LiquidOperatorHamiltonianModel.

The physics contract under test:
  * spectral layer == manual FFT multiplication (numerical correctness)
  * spectral convolution is translation-EQUIVARIANT (circular shifts)
  * energy conservation is ARCHITECTURAL: random untrained head must show
    bounded (non-growing) energy drift under velocity-Verlet
  * the same head runs on ANY node count N (resolution-invariant interface)
  * gradients flow into spectral weights, lift, FiLM, and through context
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.operator_potential import OperatorPotential, SpectralConv1d
from awareliquid_physics.hamiltonian import OperatorHamiltonianHead
from awareliquid_physics.model import LiquidOperatorHamiltonianModel


def test_spectral_conv_matches_manual_fft():
    torch.manual_seed(0)
    B, C, N, modes = 2, 4, 33, 8          # N odd, not aligned to modes
    layer = SpectralConv1d(modes, C, C)
    layer.eval()
    v = torch.randn(B, C, N)
    with torch.no_grad():
        out = layer(v)
        vf = torch.fft.rfft(v, dim=-1)
        w = torch.complex(layer.weight[0], layer.weight[1])
        ref = torch.einsum("bcm,mco->bom", vf[..., :modes], w)
        ref = F.pad(ref, (0, vf.shape[-1] - ref.shape[-1]))
        ref = torch.fft.irfft(ref, n=N, dim=-1)
    err = (out - ref).abs().max().item()
    print(f"[1] spectral conv vs manual FFT: max err {err:.2e}")
    assert err < 1e-5, f"spectral layer disagrees with manual FFT: {err:.2e}"


def test_spectral_conv_is_translation_equivariant():
    """A convolution is shift-equivariant: K(roll(v)) == roll(K(v)) on the node
    axis (no padding -> circular convolution). This is the symmetry the energy
    landscape inherits for interior nodes."""
    torch.manual_seed(1)
    layer = SpectralConv1d(8, 4, 4)
    layer.eval()
    v = torch.randn(2, 4, 32)
    with torch.no_grad():
        out = layer(v)
        out_shift = layer(torch.roll(v, shifts=3, dims=-1))
    err = (out_shift - torch.roll(out, shifts=3, dims=-1)).abs().max().item()
    print(f"[2] spectral conv circular shift equivariance: max err {err:.2e}")
    assert err < 1e-5, "spectral conv is not translation equivariant"


def test_operator_head_conserves_energy_by_construction():
    """The core v0.1 thesis carried into v0.2: velocity-Verlet on a separable
    Hamiltonian has bounded O(dt^2) energy error with NO secular growth — for a
    RANDOM, untrained FNO potential. Conservation is architecture, not tuning."""
    torch.manual_seed(2)
    head = OperatorHamiltonianHead(dim=1, width=16, modes=6, fno_depth=2,
                                   context_dim=0, reflect_pad=4)
    head.eval()
    B, N, dt = 3, 32, 0.01
    q0 = torch.randn(B, N, 1) * 0.5
    p0 = torch.randn(B, N, 1) * 0.5
    with torch.enable_grad():
        qs, ps = head.rollout(q0, p0, steps=300, dt=dt)
        def E(t):
            return head.energy(qs[t].detach(), ps[t].detach()).detach()
        e0, e60, e300 = E(0), E(60), E(300)
        d60 = ((e60 - e0).abs() / e0.abs().clamp_min(1e-6)).max().item()
        d300 = ((e300 - e0).abs() / e0.abs().clamp_min(1e-6)).max().item()
    print(f"[3] random-head energy drift: 60 steps {d60:.2e}, 300 steps {d300:.2e}")
    assert d300 < 0.05, f"unbounded energy drift on random head: {d300:.2e}"
    assert d300 < 10 * d60 + 1e-4, "drift grows secularly (should stay bounded)"


def test_operator_head_runs_on_any_resolution():
    """Resolution invariance at the interface level: ONE set of parameters must
    integrate states with different node counts N without any change."""
    torch.manual_seed(3)
    head = OperatorHamiltonianHead(dim=2, width=16, modes=6, fno_depth=2,
                                   context_dim=0, reflect_pad=4)
    head.eval()
    for N in (8, 64):                      # same weights, 8x resolution jump
        q0 = torch.randn(2, N, 2)
        p0 = torch.randn(2, N, 2)
        with torch.enable_grad():
            qs, ps = head.rollout(q0, p0, steps=5, dt=0.05)
        assert qs.shape == (6, 2, N, 2) and ps.shape == (6, 2, N, 2)
        assert torch.isfinite(qs).all()
    print("[4] operator head integrates N=8 and N=64 with the same weights")


def test_operator_potential_is_scalar_and_finite():
    torch.manual_seed(4)
    pot = OperatorPotential(dim=2, width=16, modes=6, depth=2,
                            context_dim=5, reflect_pad=4)
    pot.eval()
    q = torch.randn(3, 40, 2)
    ctx = torch.randn(3, 5)
    e = pot(q, ctx)
    assert e.shape == (3,), "potential must be a per-batch scalar energy"
    assert torch.isfinite(e).all()
    # context must actually reshape the landscape
    e2 = pot(q, torch.randn(3, 5))
    assert (e - e2).abs().max() > 1e-6, "context does not condition the potential"
    print(f"[5] operator potential: scalar energy {e.tolist()}")


def test_operator_head_gradient_flows_through_spectral_and_context():
    torch.manual_seed(5)
    head = OperatorHamiltonianHead(dim=1, width=16, modes=6, fno_depth=2,
                                   context_dim=4, reflect_pad=0)
    head.train()
    q0 = torch.randn(2, 16, 1)
    p0 = torch.randn(2, 16, 1)
    ctx = torch.randn(2, 4, requires_grad=True)
    with torch.enable_grad():
        qs, ps = head.rollout(q0, p0, steps=4, dt=0.05, context=ctx)
        qs.pow(2).mean().backward()
    # Parameters that only shift the energy by a CONSTANT contribute nothing to
    # the force field (dV/dq, dT/dp) — gradient None here is physically correct:
    # energy is relative. (V.films.1.bias feeds a LINEAR project, so it is a pure
    # offset; films.0.bias DOES get gradient because a nonlinear block follows it.)
    constant_offset_params = {"T.4.bias", "V.project.bias"}
    for name, p in head.named_parameters():
        if name in constant_offset_params or name.startswith("V.films.1.bias"):
            assert p.grad is None, f"{name} should receive no force gradient"
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {name}"
    assert head.V.blocks[0].spectral.weight.grad.abs().sum() > 0, \
        "spectral weights got no gradient"
    assert ctx.grad is not None and ctx.grad.abs().sum() > 0, \
        "gradient did not flow back through the FiLM context"
    print("[6] gradients flow: spectral weights, lift, FiLM context — all live")


def test_operator_model_end_to_end_and_resolution_invariant():
    """The coupled model: prefix -> liquid context -> operator Hamiltonian ->
    symplectic rollout, at two different node resolutions with ONE model."""
    torch.manual_seed(6)
    model = LiquidOperatorHamiltonianModel(
        phase_dim=1, d_model=16, context_dim=6, n_scales=3,
        modes=6, width=16, fno_depth=2, hidden_dim=16, t_depth=2,
        dt=0.05, core_dt=1.0, reflect_pad=4)
    model.eval()
    for N in (16, 64):
        q_obs = torch.randn(3, 10, N, 1)
        p_obs = torch.randn(3, 10, N, 1)
        with torch.enable_grad():
            qs, ps, ctx = model(q_obs, p_obs, k=8)
        assert qs.shape == (9, 3, N, 1) and ps.shape == (9, 3, N, 1)
        assert ctx.shape == (3, 6)
        assert torch.isfinite(qs).all()
    # different prefixes must give different contexts (system-ID is wired in)
    qa, pa = torch.randn(2, 8, 16, 1), torch.randn(2, 8, 16, 1)
    qb, pb = torch.randn(2, 8, 16, 1), torch.randn(2, 8, 16, 1)
    with torch.enable_grad():
        ca = model.infer_context(qa, pa)
        cb = model.infer_context(qb, pb)
    assert (ca - cb).abs().max() > 1e-4, "different prefixes gave the same context"
    print("[7] coupled operator model: end-to-end shapes + system-ID verified")


if __name__ == "__main__":
    test_spectral_conv_matches_manual_fft()
    test_spectral_conv_is_translation_equivariant()
    test_operator_head_conserves_energy_by_construction()
    test_operator_head_runs_on_any_resolution()
    test_operator_potential_is_scalar_and_finite()
    test_operator_head_gradient_flows_through_spectral_and_context()
    test_operator_model_end_to_end_and_resolution_invariant()
    print("all operator-potential tests passed")
