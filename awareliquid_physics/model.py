"""
model.py — LiquidHamiltonianModel: physics ON the liquid substrate.

This is the actual integration the project is about, not a free-standing head.
A LiquidCore (the LTC recurrence) reads an OBSERVED trajectory prefix to infer
the system's hidden parameters (system identification — e.g. a spring's unknown
stiffness), and conditions a Hamiltonian head's POTENTIAL on that inferred
context. A symplectic (velocity-Verlet) rollout then predicts the future with
energy conserved by construction. The liquid ODE core sets the energy landscape
the conservative dynamics evolve in — "the liquid core is the physics substrate",
realised in code.

Why this beats a static Hamiltonian: one fixed learned potential cannot fit a
FAMILY of systems (different stiffness / masses per trajectory). The liquid core
turns the observed prefix into a per-trajectory context, so the SAME model adapts
its energy landscape to each system. Why it beats an unstructured seq2seq: the
symplectic Hamiltonian structure conserves energy over long rollouts, which no
plain autoregressive predictor does.

Honest scope: this is validated on CONTINUOUS-STATE trajectory prediction with
physics metrics (rollout MSE, energy drift), the one place physics-informed ML
genuinely applies. It is NOT a language model and makes no hallucination claim.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .hamiltonian import (HamiltonianHead, OperatorHamiltonianHead,
                          OperatorHamiltonianHead2d)
from .liquid_core import LiquidCore
from .pairwise_potential import NBodyHamiltonianHead


class LiquidHamiltonianModel(nn.Module):
    """Liquid-core system-ID + context-conditioned symplectic rollout.

    phase_dim : dimension of q (and of p); the phase state is (q, p) in R^{2*phase_dim}.
    """

    def __init__(self, phase_dim: int, d_model: int = 64, context_dim: int = 16,
                 n_scales: int = 4, hidden_dim: int = 64, depth: int = 2,
                 dt: float = 0.1, core_dt: float = 1.0):
        super().__init__()
        self.phase_dim = int(phase_dim)
        self.context_dim = int(context_dim)
        self.dt = float(dt)

        # Liquid core reads the (q,p) prefix sequence -> hidden state.
        self.core = LiquidCore(2 * phase_dim, d_model, n_scales=n_scales, dt=core_dt)
        self.context_proj = nn.Linear(d_model, context_dim)
        # Context-conditioned Hamiltonian: V(q | ctx) — the core sets the landscape.
        self.ham = HamiltonianHead(phase_dim, hidden_dim=hidden_dim, depth=depth,
                                   context_dim=context_dim)

    def infer_context(self, q_obs: torch.Tensor, p_obs: torch.Tensor) -> torch.Tensor:
        """(B, T_obs, phase_dim) x2 -> (B, context_dim). System identification from
        the observed prefix via the liquid recurrence."""
        x = torch.cat([q_obs, p_obs], dim=-1)         # (B, T_obs, 2*phase_dim)
        return self.context_proj(self.core.encode(x)) # (B, context_dim)

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, context: torch.Tensor,
                steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symplectic rollout of `steps` from (q0, p0) under the fixed inferred
        context. Returns (qs, ps), each (steps+1, B, phase_dim)."""
        return self.ham.rollout(q0, p0, steps, self.dt, context=context)

    def forward(self, q_obs: torch.Tensor, p_obs: torch.Tensor, k: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer context from the prefix, then predict k steps forward from the
        last observed state. Returns (qs, ps, context); qs/ps are (k+1, B, phase_dim)
        INCLUDING the initial state (the last observed one)."""
        context = self.infer_context(q_obs, p_obs)
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout(q0, p0, context, k)
        return qs, ps, context


class LiquidOperatorHamiltonianModel(nn.Module):
    """v0.2 (M2): liquid system-ID + OPERATOR-potential symplectic rollout.

    The field/particle version of LiquidHamiltonianModel. Phase states are
    (B, T, N, phase_dim) with an ARBITRARY number of nodes N — resolution
    invariance is preserved end-to-end:
      * the prefix is encoded per-node by ONE shared Linear (same weights at
        every node) and then MEAN-pooled over nodes before the liquid core,
        so no parameter depends on N;
      * the Hamiltonian head is OperatorHamiltonianHead (FNO potential).
    """

    def __init__(self, phase_dim: int, d_model: int = 64, context_dim: int = 16,
                 n_scales: int = 4, modes: int = 12, width: int = 32,
                 fno_depth: int = 4, hidden_dim: int = 64, t_depth: int = 2,
                 dt: float = 0.1, core_dt: float = 1.0, reflect_pad: int = 8):
        super().__init__()
        self.phase_dim = int(phase_dim)
        self.context_dim = int(context_dim)
        self.dt = float(dt)

        # Per-node encoder: resolution-invariant (shared Linear + mean pool).
        self.node_enc = nn.Linear(2 * phase_dim, d_model)
        self.core = LiquidCore(d_model, d_model, n_scales=n_scales, dt=core_dt)
        self.context_proj = nn.Linear(d_model, context_dim)
        # Operator-conditioned Hamiltonian: V(q | ctx) in function space.
        self.ham = OperatorHamiltonianHead(phase_dim, width=width, modes=modes,
                                           fno_depth=fno_depth,
                                           context_dim=context_dim,
                                           hidden_dim=hidden_dim,
                                           t_depth=t_depth,
                                           reflect_pad=reflect_pad)

    def infer_context(self, q_obs: torch.Tensor, p_obs: torch.Tensor) -> torch.Tensor:
        """(B, T_obs, N, phase_dim) x2 -> (B, context_dim). System identification
        from the observed prefix via the liquid recurrence."""
        x = torch.cat([q_obs, p_obs], dim=-1)          # (B, T, N, 2*phase_dim)
        x = self.node_enc(x)                           # (B, T, N, d_model)
        x = x.mean(dim=2)                              # (B, T, d_model) — pool over N
        return self.context_proj(self.core.encode(x))  # (B, context_dim)

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, context: torch.Tensor,
                steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symplectic rollout of `steps` from (q0, p0) under the fixed inferred
        context. Returns (qs, ps), each (steps+1, B, N, phase_dim)."""
        return self.ham.rollout(q0, p0, steps, self.dt, context=context)

    def forward(self, q_obs: torch.Tensor, p_obs: torch.Tensor, k: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer context from the prefix, then predict k steps forward from the
        last observed state. Returns (qs, ps, context); qs/ps are
        (k+1, B, N, phase_dim) INCLUDING the initial state."""
        context = self.infer_context(q_obs, p_obs)
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout(q0, p0, context, k)
        return qs, ps, context


class LiquidOperatorHamiltonianModel2d(nn.Module):
    """v0.2 (P2): the 2D-grid counterpart of LiquidOperatorHamiltonianModel.
    Phase states are (B, T, H, W, dim); H, W arbitrary (resolution-invariant):
    the prefix is encoded per-node by ONE shared Linear then MEAN-pooled over
    the grid before the liquid core, and the head is OperatorHamiltonianHead2d."""

    def __init__(self, dim: int, d_model: int = 64, context_dim: int = 16,
                 n_scales: int = 4, modes_x: int = 12, modes_y: int = 12,
                 width: int = 32, fno_depth: int = 4, hidden_dim: int = 64,
                 t_depth: int = 2, dt: float = 0.05, core_dt: float = 1.0,
                 reflect_pad: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.dt = float(dt)

        self.node_enc = nn.Linear(2 * dim, d_model)
        self.core = LiquidCore(d_model, d_model, n_scales=n_scales, dt=core_dt)
        self.context_proj = nn.Linear(d_model, context_dim)
        self.ham = OperatorHamiltonianHead2d(
            dim, width=width, modes_x=modes_x, modes_y=modes_y,
            fno_depth=fno_depth, context_dim=context_dim, hidden_dim=hidden_dim,
            t_depth=t_depth, reflect_pad=reflect_pad)

    def infer_context(self, q_obs: torch.Tensor,
                      p_obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q_obs, p_obs], dim=-1)          # (B, T, H, W, 2*dim)
        x = self.node_enc(x)                           # (B, T, H, W, d_model)
        x = x.mean(dim=(2, 3))                         # (B, T, d_model)
        return self.context_proj(self.core.encode(x))

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, context: torch.Tensor,
                steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ham.rollout(q0, p0, steps, self.dt, context=context)

    def forward(self, q_obs: torch.Tensor, p_obs: torch.Tensor, k: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.infer_context(q_obs, p_obs)
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout(q0, p0, context, k)
        return qs, ps, context


class LiquidNBodyModel(nn.Module):
    """v0.2 (P2-1): liquid system-ID + N-body pair-potential symplectic rollout.

    Phase state is (q, v) = (positions, velocities) of N particles (N
    arbitrary — resolution-invariant via per-node encoding + mean pooling).
    The liquid core reads the observed prefix and infers a context (the hidden
    mass distribution), which conditions the radial PairwisePotential; a
    velocity-Verlet rollout then predicts the future with energy conserved by
    construction. The irregular-node counterpart of LiquidOperatorHamiltonianModel.
    """

    def __init__(self, dim: int, d_model: int = 64, context_dim: int = 16,
                 n_scales: int = 4, hidden_dim: int = 64, depth: int = 2,
                 dt: float = 0.05, core_dt: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.dt = float(dt)

        self.node_enc = nn.Linear(2 * dim, d_model)
        self.core = LiquidCore(d_model, d_model, n_scales=n_scales, dt=core_dt)
        self.context_proj = nn.Linear(d_model, context_dim)
        self.ham = NBodyHamiltonianHead(dim, hidden_dim=hidden_dim, depth=depth,
                                        context_dim=context_dim)

    def infer_context(self, q_obs: torch.Tensor, v_obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q_obs, v_obs], dim=-1)          # (B, T, N, 2*dim)
        x = self.node_enc(x)                           # (B, T, N, d_model)
        x = x.mean(dim=2)                              # (B, T, d_model)
        return self.context_proj(self.core.encode(x))

    def rollout(self, q0: torch.Tensor, v0: torch.Tensor, context: torch.Tensor,
                steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ham.rollout(q0, v0, steps, self.dt, context=context)

    def forward(self, q_obs: torch.Tensor, v_obs: torch.Tensor, k: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.infer_context(q_obs, v_obs)
        q0, v0 = q_obs[:, -1], v_obs[:, -1]
        qs, vs = self.rollout(q0, v0, context, k)
        return qs, vs, context


class ProbabilisticLiquidModel(nn.Module):
    """VAE-style probabilistic context (P2-3): the liquid core infers a
    DISTRIBUTION over the hidden system parameters (mu, logvar) instead of a
    point estimate. Sampling the context gives an ENSEMBLE of symplectic
    rollouts — probabilistic prediction with quantified uncertainty, the
    minimal ensemble counterpart of diffusion-style forecasts (GenCast line).

    Training: reparameterised single sample (n_samples=1). Inference: n_samples
    draws, mean + std of the ensemble is the prediction + uncertainty.
    """

    def __init__(self, phase_dim: int, d_model: int = 64, context_dim: int = 16,
                 n_scales: int = 4, hidden_dim: int = 64, depth: int = 2,
                 dt: float = 0.1, core_dt: float = 1.0):
        super().__init__()
        self.phase_dim = int(phase_dim)
        self.context_dim = int(context_dim)
        self.dt = float(dt)

        self.core = LiquidCore(2 * phase_dim, d_model, n_scales=n_scales,
                               dt=core_dt)
        self.mu_proj = nn.Linear(d_model, context_dim)
        self.logvar_proj = nn.Linear(d_model, context_dim)
        self.ham = HamiltonianHead(phase_dim, hidden_dim=hidden_dim,
                                   depth=depth, context_dim=context_dim)

    def infer_dist(self, q_obs: torch.Tensor,
                   p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([q_obs, p_obs], dim=-1)
        h = self.core.encode(x)
        return self.mu_proj(h), self.logvar_proj(h)

    def sample_context(self, mu: torch.Tensor, logvar: torch.Tensor,
                       n_samples: int) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        return mu + eps * std                       # (S, B, d_ctx)

    def rollout_ensemble(self, q0: torch.Tensor, p0: torch.Tensor,
                         mu: torch.Tensor, logvar: torch.Tensor,
                         n_samples: int, steps: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctxs = self.sample_context(mu, logvar, n_samples)
        all_qs, all_ps = [], []
        for s in range(n_samples):
            qs, ps = self.ham.rollout(q0, p0, steps, self.dt, context=ctxs[s])
            all_qs.append(qs)
            all_ps.append(ps)
        return torch.stack(all_qs), torch.stack(all_ps)   # (S, k+1, B, dim)

    def forward(self, q_obs: torch.Tensor, p_obs: torch.Tensor, k: int,
                n_samples: int = 1
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mu, logvar = self.infer_dist(q_obs, p_obs)
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout_ensemble(q0, p0, mu, logvar, n_samples, k)
        return qs, ps, (mu, logvar)


class GRUSeqModel(nn.Module):
    """Unstructured autoregressive baseline: a GRU encodes the (q,p) prefix, then
    an MLP head predicts the next-state DELTA autoregressively (plain Euler-style
    integration in latent, no conservation structure). The honest 'does the
    physics structure buy anything' control at matched budget."""

    def __init__(self, phase_dim: int, hidden: int = 64):
        super().__init__()
        self.phase_dim = int(phase_dim)
        self.enc = nn.GRU(2 * phase_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + 2 * phase_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * phase_dim),
        )
        self.hidden = hidden

    def forward(self, q_obs: torch.Tensor, p_obs: torch.Tensor, k: int
                ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x = torch.cat([q_obs, p_obs], dim=-1)          # (B, T_obs, 2*ph)
        _, h = self.enc(x)                              # h: (1, B, hidden)
        ctx = h[0]                                      # (B, hidden)  system summary
        q, p = q_obs[:, -1], p_obs[:, -1]
        qs, ps = [q], [p]
        for _ in range(int(k)):
            s = torch.cat([q, p], dim=-1)
            d = self.head(torch.cat([ctx, s], dim=-1)) # predicted delta (B, 2*ph)
            q = q + d[..., :self.phase_dim]
            p = p + d[..., self.phase_dim:]
            qs.append(q); ps.append(p)
        return torch.stack(qs), torch.stack(ps), None
