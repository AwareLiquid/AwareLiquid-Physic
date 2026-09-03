"""Tests for the 2D operator-potential stack (P2): SpectralConv2d,
OperatorPotential2d, OperatorHamiltonianHead2d, and the 2D membrane generator.

Physics contract under test:
  * spectral conv 2D == manual FFT2 multiplication;
  * spectral conv 2D is circular-shift equivariant (translation symmetry);
  * gen_wave_2d ground truth is symplectic (bounded, non-growing drift);
  * random untrained 2D head conserves energy (bounded, no secular growth);
  * one weight set integrates H=16 and H=32 grids (resolution invariance).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.datasets import gen_wave_2d
from awareliquid_physics.hamiltonian import OperatorHamiltonianHead2d
from awareliquid_physics.operator_potential import (OperatorPotential2d,
                                                    SpectralConv2d)


def _membrane_energy(q, p):
    dqx = q - torch.roll(q, 1, dims=-3)
    dqy = q - torch.roll(q, 1, dims=-2)
    return 0.5 * (p ** 2).sum() + 0.5 * (dqx ** 2 + dqy ** 2).sum()


def test_spectral_conv_2d_matches_manual_fft():
    torch.manual_seed(0)
    layer = SpectralConv2d(4, 4, 3, 5)
    layer.eval()
    v = torch.randn(2, 3, 17, 15)          # odd H, W
    with torch.no_grad():
        out = layer(v)
        vf = torch.fft.rfft2(v, dim=(-2, -1))
        mx, my = min(4, vf.shape[-2]), min(4, vf.shape[-1])
        w = torch.complex(layer.weight[0][:mx, :my], layer.weight[1][:mx, :my])
        ref = torch.einsum("bchw,hwco->bohw", vf[..., :mx, :my], w)
        ref = F.pad(ref, (0, vf.shape[-1] - ref.shape[-1],
                          0, vf.shape[-2] - ref.shape[-2]))
        ref = torch.fft.irfft2(ref, s=(17, 15), dim=(-2, -1))
    err = (out - ref).abs().max().item()
    print(f"[1] spectral conv 2d vs manual FFT2: max err {err:.2e}")
    assert err < 1e-5, f"2d spectral layer disagrees with manual FFT: {err:.2e}"


def test_spectral_conv_2d_translation_equivariant():
    torch.manual_seed(1)
    layer = SpectralConv2d(4, 4, 3, 3)
    layer.eval()
    v = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out = layer(v)
        out_shift = layer(torch.roll(v, shifts=(2, 3), dims=(-2, -1)))
    err = (out_shift - torch.roll(out, shifts=(2, 3), dims=(-2, -1))).abs().max().item()
    print(f"[2] spectral conv 2d circular shift equivariance: max err {err:.2e}")
    assert err < 1e-5, "2d spectral conv is not translation equivariant"


def test_wave_2d_ground_truth_conserves_energy():
    torch.manual_seed(2)
    g = torch.Generator().manual_seed(7)
    qs, ps = gen_wave_2d(2, 60, 0.02, 16, 16, 1.0, g)
    assert qs.shape == (2, 61, 16, 16, 1)
    e0 = _membrane_energy(qs[:, 0], ps[:, 0]).item()
    e60 = _membrane_energy(qs[:, 60], ps[:, 60]).item()
    drift = abs(e60 - e0) / abs(e0)
    print(f"[3] wave-2d ground truth drift over 60 steps: {drift:.2e}")
    assert drift < 1e-3, f"membrane ground truth drifted too much: {drift:.2e}"
    assert torch.isfinite(qs).all()


def test_operator_head_2d_conserves_energy_and_resolution():
    torch.manual_seed(3)
    head = OperatorHamiltonianHead2d(dim=1, width=16, modes_x=4, modes_y=4,
                                     fno_depth=2, reflect_pad=2)
    head.eval()
    for (H, W) in ((16, 16), (32, 32)):     # same weights, different resolution
        q0 = torch.randn(2, H, W, 1)
        p0 = torch.randn(2, H, W, 1) * 0.3
        with torch.enable_grad():
            qs, ps = head.rollout(q0, p0, steps=150, dt=0.02)
            e0 = head.energy(qs[0].detach(), ps[0].detach()).detach()
            e150 = head.energy(qs[150].detach(), ps[150].detach()).detach()
        drift = ((e150 - e0).abs() / e0.abs().clamp_min(1e-6)).max().item()
        print(f"[4] 2D head ({H}x{W}) random-weight drift: {drift:.2e}")
        assert drift < 0.05, f"unbounded energy drift on 2D head: {drift:.2e}"


def test_operator_potential_2d_scalar_and_conditioned():
    torch.manual_seed(4)
    pot = OperatorPotential2d(dim=1, width=16, modes_x=4, modes_y=4,
                              depth=2, context_dim=5, reflect_pad=2)
    pot.eval()
    q = torch.randn(3, 16, 16, 1)
    ctx = torch.randn(3, 5)
    e = pot(q, ctx)
    assert e.shape == (3,), "2D potential must return a per-batch scalar"
    assert torch.isfinite(e).all()
    e2 = pot(q, torch.randn(3, 5))
    assert (e - e2).abs().max() > 1e-6, "context does not condition the 2D potential"
    print(f"[5] operator potential 2d: scalar, context-conditioned")


if __name__ == "__main__":
    test_spectral_conv_2d_matches_manual_fft()
    test_spectral_conv_2d_translation_equivariant()
    test_wave_2d_ground_truth_conserves_energy()
    test_operator_head_2d_conserves_energy_and_resolution()
    test_operator_potential_2d_scalar_and_conditioned()
    print("all 2D operator-potential tests passed")
