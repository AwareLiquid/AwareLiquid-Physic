"""
hamiltonian.py — hard-constraint physics-informed head (HNN).

The way to "write physics INTO the network" that fits a continuous-time (liquid
ODE) backbone is the HARD-CONSTRAINT route — a Hamiltonian Neural Network
(Greydanus et al. 2019) integrated with a SYMPLECTIC scheme — NOT a soft PDE-
residual loss.

  * SOFT PINN puts physics in the LOSS (a PDE residual penalty); the net only
    learns to approximately obey it, and it adds nothing over an already-exact
    hand-coded engine (physics_ops.py) when the equations are known.
  * HARD (this module) bakes physics into the ARCHITECTURE: the net outputs a
    scalar energy H(q,p) = T(p) + V(q), and the state is advanced by a velocity-
    Verlet (leapfrog) symplectic integrator. Total energy is then conserved BY
    CONSTRUCTION — bounded O(dt^2) drift, exactly time-reversible — for ANY
    learned T, V, trained or not. Conservation is a property of the architecture,
    not a soft penalty.

Honest scope (do not overclaim)
-------------------------------
* Operates on a CONTINUOUS phase state (q, p) in R^dim — generalized positions
  and momenta — NOT token latents. It is a trajectory-prediction component.
* A physics prior is category-mismatched to language: it does NOT reduce LLM
  hallucination and is expected to be PPL-neutral on next-token modelling. This
  whole repository is DELIBERATELY a physics variant, validated only on physics
  metrics (k-step rollout MSE, energy drift), never perplexity. See
  benchmarks/physics_rollout_eval.py.
* The affinity that IS real: a closed-form LTC core (liquid_core.py) is a
  continuous-time ODE, so a learned-Hamiltonian vector field integrated
  symplectically is the natural continuous-state counterpart to physics_ops.py's
  hand-coded symplectic integrators. The coupling lives in model.py.

Reference: Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks", NeurIPS
2019 (arXiv:1906.01563); velocity-Verlet is the same 2nd-order symplectic scheme
physics_ops.integrate_verlet uses.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .operator_potential import FiLM, OperatorPotential


def _energy_mlp(dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    """A smooth (Tanh) MLP R^dim -> R for a scalar energy. Tanh, not ReLU: the
    field is the GRADIENT of this net, so a C^inf activation gives a smooth,
    differentiable-twice force field (ReLU's kinks make dH/dx discontinuous)."""
    layers: list[nn.Module] = []
    d = dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
        d = hidden_dim
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


class FilmEnergyNet(nn.Module):
    """A Tanh energy MLP whose HIDDEN layers are FiLM-modulated by a context:
    each hidden activation is h <- scale(ctx) * h + bias(ctx). v0.2 (ADR-4):
    conditioning the potential by FiLM (per-layer per-channel modulation) is
    more expressive than concatenating the context to the input — the M1
    benchmark upgrades its pointwise potential from concat to FiLM here."""

    def __init__(self, dim: int, hidden_dim: int, depth: int, context_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.films = nn.ModuleList(
            [FiLM(hidden_dim, context_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # FiLM modulates the PRE-activation, then Tanh compresses the result
        # into [-1, 1]: every hidden activation stays bounded regardless of the
        # learned scale (an unbounded post-activation FiLM made the force field
        # explode and long rollouts diverge — this ordering fixes that).
        h = torch.tanh(self.films[0](self.in_layer(x), context))
        for layer, film in zip(self.hidden, self.films[1:]):
            h = torch.tanh(film(layer(h), context))
        return self.out_layer(h)


class HamiltonianHead(nn.Module):
    """Separable Hamiltonian H(q,p) = T(p) + V(q) advanced by velocity-Verlet.

    q, p are (..., dim). Energy conservation is architectural: leapfrog on a
    separable H has bounded O(dt^2) energy error and is time-reversible for any
    T, V. Training (a supervised 1-step or k-step trajectory MSE) shapes T, V to
    match observed dynamics while the CONSERVATION structure is never something
    the optimizer can break.

    context_dim > 0 conditions the POTENTIAL V(q | ctx) on an external context
    vector — the hook by which the liquid core (model.py) sets the energy
    landscape. context_dim = 0 (default) is the pure autonomous Hamiltonian.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, depth: int = 2,
                 context_dim: int = 0):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.T = _energy_mlp(dim, hidden_dim, depth)                       # T(p)
        self.V = _energy_mlp(dim + self.context_dim, hidden_dim, depth)    # V(q | ctx)

    # -- energy & its gradients (the conservative vector field) ---------------

    def energy(self, q: torch.Tensor, p: torch.Tensor,
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """H(q,p) = T(p) + V(q | ctx).  (..., dim) -> (...,) scalar."""
        v_in = q if context is None else torch.cat([q, context], dim=-1)
        return self.T(p).squeeze(-1) + self.V(v_in).squeeze(-1)

    def _grad(self, net: nn.Module, x: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """d sum(net([x, ctx])) / dx — gradient of a scalar (possibly context-
        conditioned) energy w.r.t. x ONLY (context is fixed within a step).

        create_graph follows self.training so that during training the gradient
        of the loss flows through the integrator into T/V's parameters AND (via
        context) into whatever produced the context (the liquid core); at eval it
        is a plain (graph-free) field evaluation. The field IS an autograd
        gradient, so eval rollouts must run under torch.enable_grad() and detach
        afterwards.
        """
        if not x.requires_grad:      # ground-truth leaves; non-leaves already require grad
            x = x.requires_grad_(True)
        inp = x if context is None else torch.cat([x, context], dim=-1)
        (g,) = torch.autograd.grad(net(inp).sum(), x, create_graph=self.training)
        return g

    def dT_dp(self, p: torch.Tensor) -> torch.Tensor:
        """dH/dp = generalized velocity q_dot."""
        return self._grad(self.T, p)

    def dV_dq(self, q: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """dH/dq;  p_dot = -dH/dq  (the force).  Context sets the potential."""
        return self._grad(self.V, q, context)

    # -- symplectic integration ----------------------------------------------

    def step(self, q: torch.Tensor, p: torch.Tensor, dt: float,
             context: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One velocity-Verlet (leapfrog, kick-drift-kick) symplectic step:

            p_half = p - (dt/2) dV/dq(q | ctx)
            q'     = q + dt      dT/dp(p_half)
            p'     = p_half - (dt/2) dV/dq(q' | ctx)

        Second order, time-reversible, energy error bounded (no secular drift).
        `context` (if given) is held fixed across the step, so the local energy
        H(.|ctx) is conserved by the symplectic scheme exactly as in the
        autonomous case; across steps a changing context does real work (the
        liquid core drives the system), which is physically correct.
        """
        dt = float(dt)
        p_half = p - 0.5 * dt * self.dV_dq(q, context)
        q_next = q + dt * self.dT_dp(p_half)
        p_next = p_half - 0.5 * dt * self.dV_dq(q_next, context)
        return q_next, p_next

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, steps: int, dt: float,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate `steps` symplectic steps from (q0, p0) under a FIXED context.

        Returns (qs, ps), each (steps + 1, ..., dim) including the initial state.
        The field is an autograd gradient, so call under enable_grad; at eval,
        detach the returned trajectory.
        """
        qs, ps = [q0], [p0]
        q, p = q0, p0
        for _ in range(int(steps)):
            q, p = self.step(q, p, dt, context)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class FiLMHamiltonianHead(nn.Module):
    """Separable H(q,p|ctx) = T(p) + V(q|ctx) with the potential FiLM-conditioned
    (ADR-4): each hidden layer of V is modulated by scale(ctx) * h + bias(ctx).
    Same velocity-Verlet dynamics as HamiltonianHead — conservation is still
    architectural; only the conditioning mechanism is upgraded from concat.

    q, p are (..., dim). context_dim must be > 0.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, depth: int = 2,
                 context_dim: int = 0):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.T = _energy_mlp(dim, hidden_dim, depth)                        # T(p)
        self.V = FilmEnergyNet(dim, hidden_dim, depth, context_dim)         # V(q|ctx)

    def energy(self, q: torch.Tensor, p: torch.Tensor,
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert context is not None, "FiLMHamiltonianHead requires a context"
        return self.T(p).squeeze(-1) + self.V(q, context).squeeze(-1)

    def _grad(self, net, x: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not x.requires_grad:
            x = x.requires_grad_(True)
        out = net(x) if context is None else net(x, context)
        (g,) = torch.autograd.grad(out.sum(), x, create_graph=self.training)
        return g

    def dT_dp(self, p: torch.Tensor) -> torch.Tensor:
        return self._grad(self.T, p)

    def dV_dq(self, q: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._grad(self.V, q, context)

    def step(self, q: torch.Tensor, p: torch.Tensor, dt: float,
             context: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(dt)
        p_half = p - 0.5 * dt * self.dV_dq(q, context)
        q_next = q + dt * self.dT_dp(p_half)
        p_next = p_half - 0.5 * dt * self.dV_dq(q_next, context)
        return q_next, p_next

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, steps: int, dt: float,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        qs, ps = [q0], [p0]
        q, p = q0, p0
        for _ in range(int(steps)):
            q, p = self.step(q, p, dt, context)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class MLPFieldHead(nn.Module):
    """Baseline / ablation control: an UNSTRUCTURED learned vector field
    f(q,p) -> (q_dot, p_dot), advanced with plain (non-symplectic) Euler. It has
    NO conservation structure — the honest counterfactual for measuring what the
    Hamiltonian + symplectic architecture actually buys on long-horizon rollout
    (energy drift).

    Fairness note: this control differs from HamiltonianHead on TWO axes — the
    energy parametrization (structured H vs unstructured field) AND the
    integrator (symplectic velocity-Verlet vs forward Euler). The benchmark
    reports both models' param counts (HamiltonianHead runs two dim->1 energy
    MLPs, so it is ~1.5-2x the params of this single 2dim->2dim field MLP — NOT
    matched); the reported advantage is therefore "structure + integrator", and
    the decisive signal is ENERGY DRIFT, which no amount of extra capacity in an
    unstructured Euler field can fix (it has no conserved quantity by
    construction). The integrator axis is isolated separately in
    tests/test_hamiltonian.py (symplectic vs forward-Euler on the SAME field).
    """

    def __init__(self, dim: int, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        self.dim = int(dim)
        layers: list[nn.Module] = []
        d = 2 * dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers += [nn.Linear(d, 2 * dim)]
        self.f = nn.Sequential(*layers)

    def field(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.f(torch.cat([q, p], dim=-1))
        return out[..., :self.dim], out[..., self.dim:]

    def step(self, q: torch.Tensor, p: torch.Tensor, dt: float
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(dt)
        dq, dp = self.field(q, p)
        return q + dt * dq, p + dt * dp

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, steps: int, dt: float
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        qs, ps = [q0], [p0]
        q, p = q0, p0
        for _ in range(int(steps)):
            q, p = self.step(q, p, dt)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class OperatorHamiltonianHead(nn.Module):
    """v0.2 (M2): separable H(q,p|ctx) = T(p) + V_operator(q|ctx) with a
    SPECTRAL-OPERATOR potential (FNO) instead of a pointwise MLP.

    q, p are (..., N, dim) node states; N is ARBITRARY (resolution-invariant).
      * T(p)      — per-node kinetic MLP, summed over nodes: T = sum_n T_mlp(p_n).
                    Same weights at every node -> translation invariant.
      * V(q|ctx)  — OperatorPotential (spectral FNO stack + FiLM conditioning),
                    summed pointwise energy with translation + resolution
                    invariance and C^2 smoothness (Tanh).
    Dynamics: velocity-Verlet (kick-drift-kick), UNCHANGED — energy conservation
    remains an architectural property for whatever T, V the network learns.
    """

    def __init__(self, dim: int, width: int = 32, modes: int = 12,
                 fno_depth: int = 4, context_dim: int = 0,
                 hidden_dim: int = 64, t_depth: int = 2, reflect_pad: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.T = _energy_mlp(dim, hidden_dim, t_depth)                 # per-node T(p)
        self.V = OperatorPotential(dim, width=width, modes=modes,
                                   depth=fno_depth, context_dim=context_dim,
                                   reflect_pad=reflect_pad)

    # -- energy & its gradients (the conservative vector field) ---------------

    def energy(self, q: torch.Tensor, p: torch.Tensor,
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """H(q,p|ctx) = sum_n T(p_n) + V(q|ctx).  (..., N, dim) -> (...,) scalar.
        Arbitrary leading dims are flattened (e.g. (k+1, B, N, dim) trajectories);
        a (B, d) context is broadcast along the extra leading dims first."""
        lead = q.shape[:-2]
        qf = q.reshape(-1, *q.shape[-2:])
        pf = p.reshape(-1, *p.shape[-2:])
        if context is not None:
            cf = context
            for _ in range(len(lead) - (context.dim() - 1)):
                cf = cf.unsqueeze(0)
            cf = cf.expand(*lead, -1).reshape(-1, context.shape[-1])
        else:
            cf = None
        t = self.T(pf).squeeze(-1).sum(dim=-1)          # (B',)
        e = t + self.V(qf, cf)                          # (B',)
        return e.reshape(lead)

    def _grad(self, energy_fn, x: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Autograd gradient of a scalar energy w.r.t. x ONLY (context fixed).
        create_graph follows self.training so loss gradients flow through the
        integrator into T/V AND (via context) into the liquid core."""
        if not x.requires_grad:
            x = x.requires_grad_(True)
        energy = energy_fn(x) if context is None else energy_fn(x, context)
        (g,) = torch.autograd.grad(energy.sum(), x, create_graph=self.training)
        return g

    def dT_dp(self, p: torch.Tensor) -> torch.Tensor:
        """dH/dp = generalized velocity q_dot (summed over nodes -> per-node)."""
        return self._grad(lambda x: self.T(x).squeeze(-1).sum(dim=-1), p)

    def dV_dq(self, q: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """dH/dq;  p_dot = -dH/dq (the force).  Context sets the potential."""
        return self._grad(self.V, q, context)

    # -- symplectic integration (identical scheme to HamiltonianHead) --------

    def step(self, q: torch.Tensor, p: torch.Tensor, dt: float,
             context: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(dt)
        p_half = p - 0.5 * dt * self.dV_dq(q, context)
        q_next = q + dt * self.dT_dp(p_half)
        p_next = p_half - 0.5 * dt * self.dV_dq(q_next, context)
        return q_next, p_next

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, steps: int, dt: float,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate `steps` symplectic steps under a FIXED context. Returns
        (qs, ps), each (steps + 1, ..., N, dim) including the initial state."""
        qs, ps = [q0], [p0]
        q, p = q0, p0
        for _ in range(int(steps)):
            q, p = self.step(q, p, dt, context)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)
