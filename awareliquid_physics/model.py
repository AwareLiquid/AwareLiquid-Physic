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

from .hamiltonian import HamiltonianHead
from .liquid_core import LiquidCore


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
