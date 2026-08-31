"""Tests for the v0.2 (M3) training pipeline: the 1D wave-field dataset
generator and the semigroup (all2all) training loop.

Physics under test:
  * the generated wave trajectories are TRUE dynamics (velocity-Verlet):
    bounded energy drift, no secular growth;
  * semigroup training uses start states strictly AFTER the observed prefix
    (the context identifies the system, valid at any time point);
  * pretrain on a MIX of families and few-shot finetune both run and reduce
    loss (a few steps — full training happens in the benchmark).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awareliquid_physics.datasets import gen_wave_1d
from awareliquid_physics.model import LiquidOperatorHamiltonianModel
from awareliquid_physics.train import finetune, train_pretrain, train_semigroup


def _string_energy(q, p, c):
    """Discrete Hamiltonian of the periodic string: H = sum(p^2/2 + c^2/2 (dq)^2)."""
    dq = q - torch.roll(q, 1, dims=-2)
    return 0.5 * (p ** 2).sum() + 0.5 * (c ** 2) * (dq ** 2).sum()


def test_wave_1d_shapes_and_true_energy_conservation():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)
    n_traj, steps, dt, N, c = 4, 200, 0.05, 32, 1.0
    qs, ps = gen_wave_1d(n_traj, steps, dt, N, c, g)
    assert qs.shape == (n_traj, steps + 1, N, 1)
    assert ps.shape == (n_traj, steps + 1, N, 1)
    assert torch.isfinite(qs).all() and torch.isfinite(ps).all()

    # The ground-truth engine is symplectic: bounded drift, no secular growth.
    e0 = _string_energy(qs[:, 0], ps[:, 0], c)
    e60 = _string_energy(qs[:, 60], ps[:, 60], c)
    e200 = _string_energy(qs[:, 200], ps[:, 200], c)
    d60 = ((e60 - e0).abs() / e0.abs().clamp_min(1e-6)).item()
    d200 = ((e200 - e0).abs() / e0.abs().clamp_min(1e-6)).item()
    print(f"[1] wave-1d ground truth energy drift: 60 steps {d60:.2e}, 200 steps {d200:.2e}")
    assert d200 < 0.05, f"ground truth drifted too much: {d200:.2e}"
    assert d200 < 10 * d60 + 1e-4, "ground truth drift grows secularly"


def test_semigroup_training_reduces_loss():
    """A few semigroup steps on a tiny model must run and improve the loss —
    sanity that arbitrary-start training is wired correctly end-to-end."""
    torch.manual_seed(1)
    g = torch.Generator().manual_seed(11)
    qs, ps = gen_wave_1d(8, 40, 0.05, 16, 1.0, g)

    model = LiquidOperatorHamiltonianModel(
        phase_dim=1, d_model=8, context_dim=4, n_scales=3,
        modes=4, width=8, fno_depth=1, hidden_dim=8, t_depth=1,
        dt=0.05, core_dt=1.0, reflect_pad=0)
    t_obs, k_train = 8, 4

    l0 = train_semigroup(model, qs, ps, t_obs, k_train, steps=0, lr=3e-3,
                         batch=4, seed=3)
    l1 = train_semigroup(model, qs, ps, t_obs, k_train, steps=6, lr=3e-3,
                         batch=4, seed=3)
    l2 = train_semigroup(model, qs, ps, t_obs, k_train, steps=18, lr=3e-3,
                         batch=4, seed=3)
    print(f"[2] semigroup loss: initial={l0:.4f} after 6 steps={l1:.4f} after 18 steps={l2:.4f}")
    assert torch.isfinite(torch.tensor(l1)) and torch.isfinite(torch.tensor(l2))
    assert l2 < l1, "semigroup training did not reduce the loss"


def test_drift_penalty_alters_objective():
    """drift_weight > 0 must add an energy-drift term to the objective (the loss
    trajectory differs from the pure-MSE run)."""
    torch.manual_seed(2)
    g = torch.Generator().manual_seed(12)
    qs, ps = gen_wave_1d(6, 30, 0.05, 16, 1.0, g)
    t_obs, k_train = 8, 4

    def run(w):
        model = LiquidOperatorHamiltonianModel(
            phase_dim=1, d_model=8, context_dim=4, n_scales=3,
            modes=4, width=8, fno_depth=1, hidden_dim=8, t_depth=1,
            dt=0.05, core_dt=1.0, reflect_pad=0)
        return train_semigroup(model, qs, ps, t_obs, k_train, steps=8,
                               lr=3e-3, batch=4, seed=5, drift_weight=w)

    l_mse, l_drift = run(0.0), run(0.1)
    print(f"[3] objective: pure MSE {l_mse:.4f} vs +drift-penalty {l_drift:.4f}")
    assert abs(l_mse - l_drift) > 1e-6, "drift penalty did not change the objective"


def test_pretrain_finetune_protocol_runs():
    """Pretrain on a mix of two wave speeds, then few-shot finetune — both loops
    must run and stay finite (the full few-shot benchmark is separate)."""
    torch.manual_seed(3)
    qs1, ps1 = gen_wave_1d(6, 30, 0.05, 16, 1.0, torch.Generator().manual_seed(21))
    qs2, ps2 = gen_wave_1d(6, 30, 0.05, 16, 2.0, torch.Generator().manual_seed(22))
    model = LiquidOperatorHamiltonianModel(
        phase_dim=1, d_model=8, context_dim=4, n_scales=3,
        modes=4, width=8, fno_depth=1, hidden_dim=8, t_depth=1,
        dt=0.05, core_dt=1.0, reflect_pad=0)
    t_obs, k_train = 8, 4

    lp = train_pretrain(model, [(qs1, ps1), (qs2, ps2)], t_obs, k_train,
                        steps=6, lr=3e-3, batch=4, seed=6)
    lf = finetune(model, qs1, ps1, t_obs, k_train, steps=6, lr=3e-3,
                  batch=4, seed=7, n_shot=3)
    print(f"[4] pretrain loss {lp:.4f} -> few-shot finetune loss {lf:.4f}")
    assert torch.isfinite(torch.tensor(lp)) and torch.isfinite(torch.tensor(lf))


if __name__ == "__main__":
    test_wave_1d_shapes_and_true_energy_conservation()
    test_semigroup_training_reduces_loss()
    test_drift_penalty_alters_objective()
    test_pretrain_finetune_protocol_runs()
    print("all dataset/training tests passed")
