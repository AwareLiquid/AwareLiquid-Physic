"""P0-2 acceptance tests — pin the round-8 gap explicitly.

The PRD pins P0-2 as: "`V(q|ctx)` 支持 FNO 核积分层，可处理可变节点数（分辨率
不变）的场/粒子集合输入；对 dim=1/2 低维系统行为与 v0.1 兼容". The round-8
gap review noted the existing operator tests only prove the INTERFACE level
(random-weight rollouts at two resolutions) — nothing showed a TRAINED model
transferring zero-shot to an unseen node/grid/particle count, and nothing
pinned the dim=1/2 v0.1-compatibility claim as a test. These four tests close
that with REAL data pipelines (datasets.gen_*) and real training:

  1. 1D field: train @ N=16  → zero-shot rollout @ N=32 and N=13 (odd, prime)
  2. 2D field: train @ 8x8   → zero-shot rollout @ 12x16 (non-square H != W)
  3. particle set: train @ 4 bodies → zero-shot rollout @ 6 bodies
  4. dim=1/2 low-dim compatibility: the v0.1 architectural invariants
     (exact time-reversibility of velocity-Verlet, bounded random-head drift,
     v0.1 rollout convention) hold EXPLICITLY at dim=1 and dim=2.

"Transfer works" is judged honestly and robustly: the trained model must be
strictly better than an untrained one on the SAME unseen-resolution data
(training improved transferability), all outputs finite, and the model's own
Hamiltonian must stay bounded along the transfer rollout (conservation is
architectural). Fixed seeds keep every threshold deterministic.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.datasets import gen_nbody, gen_wave_1d, gen_wave_2d
from awareliquid_physics.hamiltonian import HamiltonianHead
from awareliquid_physics.model import (LiquidHamiltonianModel,
                                       LiquidNBodyModel,
                                       LiquidOperatorHamiltonianModel,
                                       LiquidOperatorHamiltonianModel2d)
from awareliquid_physics.train import train_semigroup

T_OBS, K = 8, 8   # observed prefix / rollout horizon shared by all tests


def _rollout_mse(qs_pred, ps_pred, qs, ps, t_obs, k):
    # qs: (B, T, ...) with arbitrary trailing node dims, pred is (k+1, B, ...)
    tail = list(range(2, qs.dim()))
    q_true = qs[:, t_obs - 1: t_obs + k].permute(1, 0, *tail)
    p_true = ps[:, t_obs - 1: t_obs + k].permute(1, 0, *tail)
    return ((qs_pred - q_true).pow(2).mean()
            + (ps_pred - p_true).pow(2).mean()).item()


def _self_energy_drift(model, qs_pred, ps_pred, ctx):
    """Relative drift of the MODEL's OWN Hamiltonian along a (k+1, B, ...) rollout
    — conservation is architectural, so this must stay bounded even zero-shot."""
    with torch.enable_grad():
        E = model.ham.energy(qs_pred, ps_pred, ctx)          # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    return ((E - E[0]).abs() / E0).max().item()


def test_p02_1d_trained_resolution_transfer():
    """P0-2 (1D field): train on N=16 nodes, zero-shot on 32 and 13 nodes."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = LiquidOperatorHamiltonianModel(
        phase_dim=1, d_model=16, context_dim=6, n_scales=3,
        modes=6, width=16, fno_depth=2, hidden_dim=16, t_depth=2,
        dt=0.03, reflect_pad=2)
    qs, ps = gen_wave_1d(8, 50, 0.03, N=16, c=1.0, generator=g)
    train_semigroup(model, qs, ps, T_OBS, K, steps=40, lr=3e-3, batch=4, seed=0)
    model.eval()

    for N in (32, 13):                    # 2x resolution jump + odd/prime count
        gt = torch.Generator().manual_seed(100 + N)
        qs_t, ps_t = gen_wave_1d(2, 50, 0.03, N=N, c=1.0, generator=gt)
        q_obs, p_obs = qs_t[:, :T_OBS], ps_t[:, :T_OBS]
        with torch.enable_grad():
            qs_pred, ps_pred, ctx = model(q_obs, p_obs, K)
        assert qs_pred.shape == (K + 1, 2, N, 1)
        assert torch.isfinite(qs_pred).all() and torch.isfinite(ps_pred).all()
        mse = _rollout_mse(qs_pred, ps_pred, qs_t, ps_t, T_OBS, K)

        fresh = LiquidOperatorHamiltonianModel(   # untrained control, same config
            phase_dim=1, d_model=16, context_dim=6, n_scales=3,
            modes=6, width=16, fno_depth=2, hidden_dim=16, t_depth=2,
            dt=0.03, reflect_pad=2)
        fresh.eval()
        with torch.enable_grad():
            qs_u, ps_u, _ = fresh(q_obs, p_obs, K)
        mse_untrained = _rollout_mse(qs_u, ps_u, qs_t, ps_t, T_OBS, K)

        drift = _self_energy_drift(model, qs_pred, ps_pred, ctx)
        print(f"[p02-1d] N={N}: transfer mse {mse:.3e} vs untrained "
              f"{mse_untrained:.3e}, self-H drift {drift:.2e}")
        assert mse < mse_untrained, \
            f"training did not improve zero-shot transfer at N={N}"
        assert drift < 0.5, f"model H unbounded on unseen resolution N={N}"


def test_p02_2d_trained_resolution_transfer():
    """P0-2 (2D field): train on 8x8, zero-shot on a NON-SQUARE 12x16 grid —
    pins that H and W are independent free axes (the round-8 question:
    '现有 2D 是固定网格还是可变节点')."""
    torch.manual_seed(1)
    g = torch.Generator().manual_seed(1)
    model = LiquidOperatorHamiltonianModel2d(
        dim=1, d_model=16, context_dim=6, n_scales=3, modes_x=6, modes_y=6,
        width=16, fno_depth=2, hidden_dim=16, t_depth=2, dt=0.03, reflect_pad=2)
    qs, ps = gen_wave_2d(8, 50, 0.03, H=8, W=8, c=1.0, generator=g)
    train_semigroup(model, qs, ps, T_OBS, K, steps=40, lr=3e-3, batch=4, seed=1)
    model.eval()

    gt = torch.Generator().manual_seed(101)
    qs_t, ps_t = gen_wave_2d(2, 50, 0.03, H=12, W=16, c=1.0, generator=gt)
    q_obs, p_obs = qs_t[:, :T_OBS], ps_t[:, :T_OBS]
    with torch.enable_grad():
        qs_pred, ps_pred, ctx = model(q_obs, p_obs, K)
    assert qs_pred.shape == (K + 1, 2, 12, 16, 1)
    assert torch.isfinite(qs_pred).all()
    mse = _rollout_mse(qs_pred, ps_pred, qs_t, ps_t, T_OBS, K)
    fresh = LiquidOperatorHamiltonianModel2d(
        dim=1, d_model=16, context_dim=6, n_scales=3, modes_x=6, modes_y=6,
        width=16, fno_depth=2, hidden_dim=16, t_depth=2, dt=0.03, reflect_pad=2)
    fresh.eval()
    with torch.enable_grad():
        qs_u, ps_u, _ = fresh(q_obs, p_obs, K)
    mse_untrained = _rollout_mse(qs_u, ps_u, qs_t, ps_t, T_OBS, K)
    drift = _self_energy_drift(model, qs_pred, ps_pred, ctx)
    print(f"[p02-2d] 12x16: transfer mse {mse:.3e} vs untrained "
          f"{mse_untrained:.3e}, self-H drift {drift:.2e}")
    assert mse < mse_untrained, "training did not improve 2D zero-shot transfer"
    assert drift < 0.5, "model H unbounded on unseen 2D grid"


def test_p02_nbody_variable_particle_count():
    """P0-2 (particle sets): train on 4 bodies, zero-shot on 6 bodies — the
    node axis of the pair-potential stack is a free count, not a fixed mesh."""
    torch.manual_seed(2)
    g = torch.Generator().manual_seed(2)
    model = LiquidNBodyModel(dim=2, d_model=16, context_dim=6, n_scales=3,
                             hidden_dim=16, depth=2, dt=0.05)
    qs, ps, _ = gen_nbody(8, 40, 0.05, N=4, D=2, generator=g)
    train_semigroup(model, qs, ps, T_OBS, K, steps=40, lr=3e-3, batch=4, seed=2)
    model.eval()

    gt = torch.Generator().manual_seed(102)
    qs_t, ps_t, _ = gen_nbody(2, 40, 0.05, N=6, D=2, generator=gt)
    q_obs, p_obs = qs_t[:, :T_OBS], ps_t[:, :T_OBS]
    with torch.enable_grad():
        qs_pred, ps_pred, ctx = model(q_obs, p_obs, K)
    assert qs_pred.shape == (K + 1, 2, 6, 2)
    assert torch.isfinite(qs_pred).all()
    mse = _rollout_mse(qs_pred, ps_pred, qs_t, ps_t, T_OBS, K)
    fresh = LiquidNBodyModel(dim=2, d_model=16, context_dim=6, n_scales=3,
                             hidden_dim=16, depth=2, dt=0.05)
    fresh.eval()
    with torch.enable_grad():
        qs_u, ps_u, _ = fresh(q_obs, p_obs, K)
    mse_untrained = _rollout_mse(qs_u, ps_u, qs_t, ps_t, T_OBS, K)
    drift = _self_energy_drift(model, qs_pred, ps_pred, ctx)
    print(f"[p02-nbody] 6 bodies: transfer mse {mse:.3e} vs untrained "
          f"{mse_untrained:.3e}, self-H drift {drift:.2e}")
    assert mse < mse_untrained, "training did not improve zero-shot transfer to 6 bodies"
    assert drift < 0.5, "model H unbounded on unseen particle count"


def test_p02_low_dim_v01_compatibility():
    """P0-2 (low-dim compatibility): the v0.1 path must still carry the v0.1
    architectural invariants EXPLICITLY at dim=1 and dim=2 — exact time
    reversibility of the velocity-Verlet step, bounded random-head drift over
    300 steps, and the (k+1, B, dim) rollout convention."""
    for dim in (1, 2):
        torch.manual_seed(3 + dim)
        head = HamiltonianHead(dim=dim, hidden_dim=32, depth=2)
        head.eval()
        # tame-init exactly as v0.1's conservation test: a raw randn potential
        # can carry force scales dt=0.01 cannot resolve — the invariant is
        # about the INTEGRATOR, tested at a resolvable force scale.
        with torch.no_grad():
            for m in (head.T, head.V):
                m[-1].weight.mul_(0.3)
                m[-1].bias.zero_()
        q = torch.randn(4, dim) * 0.5
        p = torch.randn(4, dim) * 0.5
        with torch.enable_grad():
            q1, p1 = head.step(q, p, dt=0.1)
            q2, p2 = head.step(q1, p1, dt=-0.1)
        dq = (q2 - q).abs().max().item()
        dp = (p2 - p).abs().max().item()
        assert dq < 1e-4 and dp < 1e-4, \
            f"dim={dim}: velocity-Verlet not time-reversible (dq={dq:.1e})"

        with torch.enable_grad():
            qs, ps = head.rollout(q, p, steps=300, dt=0.01)
        assert qs.shape == (301, 4, dim)
        E = head.energy(qs.detach(), ps.detach())
        drift = ((E - E[0]).abs() / E[0].abs().clamp_min(1e-6)).max().item()
        assert drift < 0.05, f"dim={dim}: random-head drift {drift:.3e} unbounded"

        model = LiquidHamiltonianModel(dim, d_model=16, context_dim=4,
                                       n_scales=3, hidden_dim=16, depth=2, dt=0.1)
        model.eval()
        q_obs, p_obs = torch.randn(3, 10, dim), torch.randn(3, 10, dim)
        with torch.enable_grad():
            qs_r, ps_r, ctx = model(q_obs, p_obs, 8)
        assert qs_r.shape == (9, 3, dim) and ctx.shape == (3, 4)
        assert torch.isfinite(qs_r).all()
        print(f"[p02-lowdim] dim={dim}: reversible, drift bounded, v0.1 shapes OK")
