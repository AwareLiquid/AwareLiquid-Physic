"""
train.py — v0.2 (M3) training pipeline: semigroup (all2all) training, an
optional drift penalty, and the pretrain/finetune protocol.

Semigroup training (Poseidon-style data efficiency)
---------------------------------------------------
The time-evolution semigroup property  Phi(t_j) = Phi(t_j - t_i) ∘ Phi(t_i)
makes EVERY (start, span) pair inside a trajectory a legal training sample.
The v0.1 loop used exactly one pair per trajectory (start = prefix end,
span = k_train). Here the start state is sampled ANYWHERE strictly after the
observed prefix — physically valid because the inferred context identifies
the SYSTEM (its parameters), not the state, so it stays correct at every
time point — and the model must propagate it for k_train steps:

    ctx   = model.infer_context(q[:t_obs], p[:t_obs])     # system ID, fixed
    q0    = q[t0],  t0 ~ Uniform[t_obs, S - k_train - 1]  # arbitrary state
    loss  = MSE(rollout(q0, p0, ctx, k_train), truth[t0 : t0 + k_train + 1])

This turns one trajectory into O(S) samples (vs 1 in v0.1). Pretraining =
the same loop on a MIX of system families; finetuning = the same loop on a
single family (few-shot). The loop itself is identical — that is the point
of the pretrain/finetune paradigm.

Drift penalty (optional, ADR: weak regulariser)
-----------------------------------------------
The architecture already conserves energy (O(dt^2) bounded drift), so the
penalty is NOT needed for conservation. It is an optional smoothness signal
that discourages learning a conserved-but-wrong energy landscape:

    loss = MSE + drift_weight * mean( (H_t - H_0)^2 )

Default drift_weight = 0 (off).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def rollout_mse_loss(qs_pred: torch.Tensor, ps_pred: torch.Tensor,
                     q_true: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
    """MSE over the full predicted trajectory (including the shared start state)."""
    return ((qs_pred - q_true).pow(2).mean() + (ps_pred - p_true).pow(2).mean())


def _energy_of(model: nn.Module, qs: torch.Tensor, ps: torch.Tensor,
               ctx: torch.Tensor) -> torch.Tensor:
    """Hamiltonian energy along a rolled-out trajectory. Works for both heads
    (MLP and operator): both expose .ham.energy(q, p, context)."""
    return model.ham.energy(qs, ps, ctx)          # (k+1, B)


def train_semigroup(model: nn.Module, qs: torch.Tensor, ps: torch.Tensor,
                    t_obs: int, k_train: int, steps: int, lr: float,
                    batch: int, seed: int, drift_weight: float = 0.0
                    ) -> float:
    """Semigroup (all2all) training loop — arbitrary start states after the
    prefix, fixed span k_train, optional drift penalty. Returns the final loss.

    qs/ps: (n_traj, S, ...) — the ... part is whatever the model consumes
    (dim for the v0.1 head, (N, dim) for the operator head).
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_traj, S = qs.shape[0], qs.shape[1]
    assert S > t_obs + k_train, "trajectory too short for prefix + rollout span"
    model.train()
    loss = torch.tensor(float("nan"))

    for _ in range(steps):
        bi = torch.randint(0, n_traj, (batch,), generator=g)
        # Arbitrary start state STRICTLY after the observed prefix: the context
        # identifies the system, so it stays valid at any time point.
        t0 = torch.randint(t_obs, S - k_train, (batch,), generator=g)

        q_obs = qs[bi, :t_obs]
        p_obs = ps[bi, :t_obs]
        with torch.enable_grad():
            ctx = model.infer_context(q_obs, p_obs)             # (B, d_ctx)
            q0 = qs[bi, t0]                                    # (B, ...)
            p0 = ps[bi, t0]
            qs_pred, ps_pred = model.rollout(q0, p0, ctx, k_train)

            q_true = qs[bi[:, None], t0[:, None] + torch.arange(k_train + 1)]
            p_true = ps[bi[:, None], t0[:, None] + torch.arange(k_train + 1)]
            q_true = q_true.permute(1, 0, *range(2, q_true.dim()))   # (k+1, B, ...)
            p_true = p_true.permute(1, 0, *range(2, p_true.dim()))

            loss = rollout_mse_loss(qs_pred, ps_pred, q_true, p_true)
            if drift_weight > 0.0:
                E = _energy_of(model, qs_pred, ps_pred, ctx)
                loss = loss + drift_weight * ((E - E[0]).pow(2).mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


def train_pretrain(model: nn.Module, families, t_obs: int, k_train: int,
                   steps: int, lr: float, batch: int, seed: int,
                   drift_weight: float = 0.0) -> float:
    """Pretrain on a MIX of system families: concatenate them along the
    trajectory axis and run the semigroup loop. `families` is a list of
    (qs, ps) pairs; the model sees a single mixed stream, which is what forces
    the liquid context to do real system identification."""
    qs = torch.cat([f[0] for f in families], dim=0)
    ps = torch.cat([f[1] for f in families], dim=0)
    return train_semigroup(model, qs, ps, t_obs, k_train, steps, lr, batch,
                           seed, drift_weight)


def finetune(model: nn.Module, qs: torch.Tensor, ps: torch.Tensor,
             t_obs: int, k_train: int, steps: int, lr: float, batch: int,
             seed: int, n_shot: Optional[int] = None,
             drift_weight: float = 0.0) -> float:
    """Few-shot finetune on ONE family. If n_shot is given, only the first
    n_shot trajectories are used (Poseidon-style few-shot evaluation)."""
    if n_shot is not None:
        qs, ps = qs[:int(n_shot)], ps[:int(n_shot)]
    return train_semigroup(model, qs, ps, t_obs, k_train, steps, lr, batch,
                           seed, drift_weight)
