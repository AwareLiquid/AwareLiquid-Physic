"""Liquid-core + physics coupling evaluation — does the LIQUID substrate add value
ON TOP of the Hamiltonian structure, on a FAMILY of systems with a hidden parameter?

This is the experiment the whole repo exists for: physics ON the liquid core, not a
free-standing head. Ground truth is a family of harmonic oscillators whose stiffness
omega VARIES per trajectory (a hidden parameter). A model sees a short observed
prefix and must predict the future.

Three models, same objective (prefix -> free-running k_train-step rollout MSE), same
optimizer/lr/steps:
  * liquid_ham : LiquidHamiltonianModel — the liquid core reads the prefix to INFER
                 omega (system identification) and conditions the Hamiltonian
                 potential; symplectic rollout predicts the future.
  * static_ham : a single autonomous HamiltonianHead (no context) — same physics
                 structure but ONE fixed potential for the whole family, so it cannot
                 adapt to a trajectory's omega. Isolates what the liquid core buys.
  * gru_seq    : a GRU encoder + autoregressive MLP head — unstructured control.
                 Isolates what the physics structure buys.

Metrics on held-out trajectories (unseen omegas from the same range): k-step rollout
MSE and relative ENERGY DRIFT (energy computed with each trajectory's TRUE omega).

Hypothesis (falsifiable): liquid_ham should beat static_ham (it can adapt to omega)
AND beat gru_seq on long-horizon energy drift (it conserves energy by construction).
If liquid_ham does not beat static_ham, the liquid core adds nothing here and the
coupling is not worth it — reported honestly either way. CPU-runnable.

Usage:
    python benchmarks/liquid_physics_eval.py --train_steps 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from awareliquid_physics.hamiltonian import HamiltonianHead
from awareliquid_physics.model import GRUSeqModel, LiquidHamiltonianModel


def gen_spring_family(n_traj, steps, dt, dim, omega_lo, omega_hi, g):
    """Harmonic oscillators with a per-trajectory random stiffness omega ~ U[lo,hi].
    Returns qs, ps (n, steps+1, dim) and omegas (n,). Exact analytic rollout."""
    omega = omega_lo + (omega_hi - omega_lo) * torch.rand(n_traj, generator=g)  # (n,)
    q0 = torch.randn(n_traj, dim, generator=g)
    p0 = torch.randn(n_traj, dim, generator=g)
    t = torch.arange(steps + 1, dtype=torch.float32) * dt                       # (S,)
    wt = omega.view(-1, 1) * t.view(1, -1)                                      # (n, S)
    c = torch.cos(wt).unsqueeze(-1); s = torch.sin(wt).unsqueeze(-1)            # (n,S,1)
    w = omega.view(-1, 1, 1)
    qs = q0.unsqueeze(1) * c + (p0.unsqueeze(1) / w) * s
    ps = -q0.unsqueeze(1) * w * s + p0.unsqueeze(1) * c
    return qs, ps, omega


def spring_energy(q, p, omega):
    """omega broadcastable to q's leading dims. E = 0.5|p|^2 + 0.5 w^2 |q|^2."""
    return 0.5 * (p ** 2).sum(-1) + 0.5 * omega ** 2 * (q ** 2).sum(-1)


class StaticHamWrapper(torch.nn.Module):
    """Autonomous HamiltonianHead behind the same forward(q_obs,p_obs,k) interface:
    it ignores the prefix (no system-ID) and rolls out from the last observed state
    under one shared, context-free potential."""
    def __init__(self, phase_dim, hidden, depth, dt):
        super().__init__()
        self.ham = HamiltonianHead(phase_dim, hidden_dim=hidden, depth=depth)
        self.dt = dt

    def forward(self, q_obs, p_obs, k):
        qs, ps = self.ham.rollout(q_obs[:, -1], p_obs[:, -1], k, self.dt)
        return qs, ps, None


def rollout_mse_loss(qs_pred, ps_pred, q_true, p_true):
    """qs_pred/ps_pred: (k+1, B, dim); q_true/p_true: (B, k+1, dim)."""
    qt = q_true.transpose(0, 1); pt = p_true.transpose(0, 1)     # (k+1, B, dim)
    return ((qs_pred - qt) ** 2 + (ps_pred - pt) ** 2).mean()


def train(model, qs, ps, t_obs, k_train, steps, lr, batch, seed, lr_decay=1.0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: lr_decay ** t)
    N, S = qs.shape[0], qs.shape[1]
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        bi = torch.randint(0, N, (batch,), generator=g)
        # random window: [t0, t0+t_obs) observed, next k_train predicted
        t0 = torch.randint(0, S - t_obs - k_train, (1,), generator=g).item()
        q_obs = qs[bi, t0:t0 + t_obs]; p_obs = ps[bi, t0:t0 + t_obs]
        fut = slice(t0 + t_obs - 1, t0 + t_obs + k_train)        # incl. last observed
        q_true = qs[bi, fut]; p_true = ps[bi, fut]
        qs_pred, ps_pred, _ = model(q_obs, p_obs, k_train)
        loss = rollout_mse_loss(qs_pred, ps_pred, q_true, p_true)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return loss.item()


def evaluate(model, qs, ps, omega, t_obs, eval_k, dt):
    """Free-running eval from the prefix. Rollout MSE + energy drift (true omega)."""
    model.eval()
    q_obs = qs[:, :t_obs]; p_obs = ps[:, :t_obs]
    fut = slice(t_obs - 1, t_obs + eval_k)
    q_true = qs[:, fut]; p_true = ps[:, fut]
    with torch.enable_grad():
        qs_pred, ps_pred, _ = model(q_obs, p_obs, eval_k)         # (k+1, B, dim)
    qs_pred, ps_pred = qs_pred.detach(), ps_pred.detach()
    mse = rollout_mse_loss(qs_pred, ps_pred, q_true, p_true).item()
    E = spring_energy(qs_pred, ps_pred, omega.view(1, -1))         # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    drift = ((E - E[0]).abs() / E0).mean(-1)                      # (k+1,)
    return {"rollout_mse": mse,
            "energy_drift_final": drift[-1].item(),
            "energy_drift_max": drift.max().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--dim", type=int, default=1, help="DOF per oscillator")
    ap.add_argument("--gen_steps", type=int, default=160)
    ap.add_argument("--t_obs", type=int, default=24, help="observed prefix length")
    ap.add_argument("--k_train", type=int, default=8, help="train rollout horizon")
    ap.add_argument("--eval_k", type=int, default=100, help="eval rollout horizon")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--omega_lo", type=float, default=0.7)
    ap.add_argument("--omega_hi", type=float, default=1.8)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--n_scales", type=int, default=4)
    ap.add_argument("--train_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0,
                    help="per-step exponential lr decay (1.0 = constant)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="benchmarks/physics_out")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    qs, ps, omega = gen_spring_family(args.n_train + args.n_eval, args.gen_steps,
                                      args.dt, args.dim, args.omega_lo, args.omega_hi, g)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)
    print(f"family: dim={args.dim} omega~U[{args.omega_lo},{args.omega_hi}] "
          f"t_obs={args.t_obs} k_train={args.k_train} eval_k={args.eval_k} "
          f"train={args.n_train} eval={args.n_eval}", flush=True)

    def build(name):
        torch.manual_seed(args.seed)
        if name == "liquid_ham":
            return LiquidHamiltonianModel(args.dim, d_model=args.d_model,
                                          context_dim=args.context_dim,
                                          n_scales=args.n_scales, hidden_dim=args.hidden,
                                          depth=2, dt=args.dt)
        if name == "static_ham":
            return StaticHamWrapper(args.dim, args.hidden, 2, args.dt)
        return GRUSeqModel(args.dim, hidden=args.hidden)

    results = {}
    for name in ("liquid_ham", "static_ham", "gru_seq"):
        model = build(name)
        n_par = sum(p.numel() for p in model.parameters())
        floss = train(model, qs[tr], ps[tr], args.t_obs, args.k_train,
                      args.train_steps, args.lr, args.batch, args.seed,
                      args.lr_decay)
        res = evaluate(model, qs[ev], ps[ev], omega[ev], args.t_obs, args.eval_k, args.dt)
        res.update({"params": n_par, "train_loss": round(floss, 6)})
        results[name] = res
        print(f"  [{name:11s}] params {n_par:>6,} | train_loss {floss:.4e} | "
              f"rollout_mse {res['rollout_mse']:.4e} | energy_drift(final) "
              f"{res['energy_drift_final']:.4e}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "liquid_physics.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    lh, sh, gr = results["liquid_ham"], results["static_ham"], results["gru_seq"]
    print("\n" + "=" * 70, flush=True)
    print("LIQUID + PHYSICS | does the liquid core add value over static physics?", flush=True)
    print("=" * 70, flush=True)
    print(f"  rollout MSE:   liquid {lh['rollout_mse']:.3e} | static {sh['rollout_mse']:.3e} "
          f"| gru {gr['rollout_mse']:.3e}", flush=True)
    print(f"  energy drift:  liquid {lh['energy_drift_final']:.3e} | static "
          f"{sh['energy_drift_final']:.3e} | gru {gr['energy_drift_final']:.3e}", flush=True)
    beats_static = lh["rollout_mse"] < sh["rollout_mse"]
    beats_gru_energy = lh["energy_drift_final"] < gr["energy_drift_final"]
    print(f"  liquid beats static (system-ID helps): {beats_static}", flush=True)
    print(f"  liquid beats gru on energy drift (structure helps): {beats_gru_energy}", flush=True)
    verdict = ("LIQUID+PHYSICS WINS" if beats_static and beats_gru_energy
               else "PARTIAL / see numbers" if beats_static or beats_gru_energy
               else "COUPLING ADDS NOTHING — shelve")
    print(f"  verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
