"""
liquid_core.py — the continuous-time LTC "liquid" recurrence, self-contained.

This is the SHARED SUBSTRATE that makes AwareLiquid-Physic a liquid network
rather than a generic net: a multi-timescale gated linear
(liquid time-constant) recurrence, trained through the same Blelloch parallel
scan (parallel_scan.pscan). It reads a sequence and produces a hidden state that
the Hamiltonian head then uses to set the energy landscape (see model.py).

Dynamics, per learned timescale s (v0.2: LIQUID time constants — the defining
LTC property, input-dependent, cf. Hasani et al., AAAI 2021):

    tau_{s,t}    = tau_min + softplus(log_tau_s + gate_tau(u_t))     > tau_min
    decay_{s,t}  = exp(-dt / tau_{s,t})                              in (0, 1)
    h_{s,t} = decay_{s,t} * h_{s,t-1} + (1 - decay_{s,t}) * u_t     (leaky integrator)

so each scale is an exponential moving average of the input drive u_t whose time
constant adapts to the input at every step (fast scales track transients, slow
scales hold context — the "liquid" multi-timescale property). The scales are
blended by a static softmax mixed with an input-dependent gate (kappa), so the
mixture is content-selective. The recurrence is run in O(log T) depth by the
general parallel scan (A now varies along T).

Self-contained: imports only torch + .parallel_scan; no external dependency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_scan import pscan


class LiquidCore(nn.Module):
    """Multi-timescale gated linear (LTC) recurrence: (B, T, d_in) -> (B, T, d_model)."""

    def __init__(self, d_in: int, d_model: int = 64, n_scales: int = 4,
                 tau_min: float = 0.5, tau_max: float = 32.0, dt: float = 1.0):
        super().__init__()
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.n_scales = int(n_scales)
        self.tau_min = float(tau_min)
        self.dt = float(dt)

        self.in_proj = nn.Linear(d_in, d_model)

        # Per-(scale, channel) tau, initialised as a geometric ladder tau_min..tau_max
        # and stored through softplus^-1 so softplus(log_tau)+tau_min stays > tau_min.
        taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), n_scales)
        inv = torch.log(torch.expm1((taus - tau_min).clamp_min(1e-4)))     # softplus^-1
        self.log_tau = nn.Parameter(inv.view(n_scales, 1).repeat(1, d_model))  # (S, d_model)

        # v0.2: liquid time constants — input-dependent modulation of the
        # timescale ladder (tau becomes a function of the input at each step).
        # Zero-init keeps the initial dynamics identical to the static v0.1 ladder.
        self.tau_gate = nn.Linear(d_model, n_scales)
        nn.init.zeros_(self.tau_gate.weight)
        nn.init.zeros_(self.tau_gate.bias)

        # Scale blend: static logits (softmax) modulated by an input-dependent gate.
        self.blend = nn.Parameter(torch.zeros(n_scales))
        self.kappa_gate = nn.Linear(d_model, n_scales)
        nn.init.constant_(self.kappa_gate.bias, 1.0)   # start with gates open

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> h: (B, T, d_model).

        Each (scale, channel) is a leaky integrator with its OWN time constant
        tau_{s,d,t} — now INPUT-DEPENDENT (the liquid property). The scan
        multiplier A is per-(batch..,t), so we fold the channel D into the scan's
        batch dims (feature dim = 1) and pass the per-timestep decay as A.
        """
        B, T, _ = x.shape
        S, D = self.n_scales, self.d_model

        u = self.in_proj(x)                                    # (B, T, D) the drive

        # v0.2: LIQUID time constants — per-(scale,channel,time) tau modulated by
        # the input at each step.  tau >= tau_min keeps decay in (0, 1).
        gate_t = self.tau_gate(u)                              # (B, T, S)
        gate_t = gate_t.permute(0, 2, 1).unsqueeze(-1)         # (B, S, T, 1)
        log_tau = self.log_tau.view(1, S, 1, D)                # (1, S, 1, D)
        tau = F.softplus(log_tau + gate_t) + self.tau_min      # (B, S, T, D)
        decay = torch.exp(-self.dt / tau)                      # (B, S, T, D) in (0,1)

        # Per-(scale,channel) leaky-integrator inputs X_{s,t} = (1-decay_{s,t})*u_t.
        X = (1.0 - decay) * u.unsqueeze(1)                     # (B, S, T, D)

        # Fold D into the scan batch dims: X5 (B,S,D,T,1), A (B,S,D,T) so the
        # scan multiplier is per-(B,S,D,t) (per channel), feature dim = 1.
        # A varies along T (liquid), so we use the general pscan.
        X5 = X.permute(0, 1, 3, 2).unsqueeze(-1)               # (B, S, D, T, 1)
        A = decay.permute(0, 1, 3, 2)                          # (B, S, D, T)
        H5 = pscan(A, X5)                                      # (B, S, D, T, 1)
        H = H5.squeeze(-1).permute(0, 1, 3, 2)                 # (B, S, T, D)

        # Blend scales: static softmax weight * input-dependent gate, renormalised.
        gate = torch.sigmoid(self.kappa_gate(u))               # (B, T, S)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)             # (B, S, T, 1)
        w = F.softmax(self.blend, dim=-1).view(1, S, 1, 1)     # (1, S, 1, 1)
        gw = w * gate
        gw = gw / gw.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B, S, T, 1)
        h = (H * gw).sum(dim=1)                                # (B, T, D)
        return self.out_proj(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a sequence to a single context vector = the last hidden state,
        which under the causal recurrence has integrated the whole prefix. Used
        by LiquidHamiltonianModel to read a trajectory prefix into a context that
        conditions the Hamiltonian potential (system identification)."""
        return self.forward(x)[:, -1]                          # (B, d_model)
