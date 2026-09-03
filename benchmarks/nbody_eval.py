"""
benchmarks/nbody_eval.py — v0.2 (P2-1): the pair-potential Hamiltonian on
gravitational N-body systems (irregular nodes).

The coupling experiment for particles: a family of N-body systems with a
hidden per-trajectory MASS field, prefix -> symplectic rollout, with a RADIAL
pair potential conditioned by the liquid context.

Models:
  liquid_nbody — liquid core system-ID -> context conditions the pair potential
  static_nbody — same pair-potential Hamiltonian, context = 0 (no system-ID)

Physics metrics only: rollout MSE and energy drift (true mass field). This is
the irregular-node counterpart of field_eval.py — the pair potential replaces
the FNO spectral layer (which assumes a periodic grid).

Usage:
  python benchmarks/nbody_eval.py --train_steps 300
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.datasets import gen_nbody
from awareliquid_physics.model import LiquidNBodyModel
from awareliquid_physics.pairwise_potential import NBodyHamiltonianHead
from awareliquid_physics.train import train_semigroup


class StaticNBodyWrapper(torch.nn.Module):
    """Pair-potential Hamiltonian with a FIXED zero context — the no-system-ID
    control for the N-body coupling experiment."""

    def __init__(self, ham: NBodyHamiltonianHead, dt: float):
        super().__init__()
        self.ham = ham
        self.dt = float(dt)

    def infer_context(self, q_obs, v_obs):
        B = q_obs.shape[0]
        return torch.zeros(B, self.ham.context_dim, device=q_obs.device)

    def rollout(self, q0, v0, ctx, steps):
        return self.ham.rollout(q0, v0, steps, self.dt, context=ctx)

    def forward(self, q_obs, v_obs, k):
        ctx = self.infer_context(q_obs, v_obs)
        q0, v0 = q_obs[:, -1], v_obs[:, -1]
        qs, vs = self.rollout(q0, v0, ctx, k)
        return qs, vs, ctx


def _true_energy(q, v, mass, G, softening):
    """Plummer-softened gravitational energy matching pairwise_gravity.
    q/v: (..., N, D); mass: (B, N) broadcast along the leading dims."""
    lead = q.shape[:-2]
    qf = q.reshape(-1, *q.shape[-2:])
    vf = v.reshape(-1, *v.shape[-2:])
    N = qf.shape[1]
    rep = qf.shape[0] // mass.shape[0]
    mf = mass.unsqueeze(0).expand(rep, -1, -1).reshape(-1, mass.shape[-1])
    diff = qf.unsqueeze(2) - qf.unsqueeze(1)
    r2 = diff.pow(2).sum(-1)                     # (B', N, N)
    i, j = torch.triu_indices(N, N, offset=1)
    pot = -G * (mf[:, i] * mf[:, j] / (r2[:, i, j] + softening ** 2).sqrt()).sum(-1)
    kin = 0.5 * (mf.unsqueeze(-1) * vf.pow(2)).sum((-1, -2))
    return (kin + pot).reshape(lead)             # (...,)


def evaluate(model, qs, vs, mass, t_obs, eval_k, dt, G, softening):
    model.eval()
    q_obs, v_obs = qs[:, :t_obs], vs[:, :t_obs]
    with torch.enable_grad():
        qs_pred, vs_pred, _ = model(q_obs, v_obs, eval_k)
    qs_pred, vs_pred = qs_pred.detach(), vs_pred.detach()
    q_true = qs[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3)
    v_true = vs[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3)
    mse = ((qs_pred - q_true).pow(2).mean() + (vs_pred - v_true).pow(2).mean()).item()

    E = _true_energy(qs_pred, vs_pred, mass, G, softening)   # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    drift = ((E - E[0]).abs() / E0).max().item()
    return mse, drift


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=80)
    ap.add_argument("--t_obs", type=int, default=16)
    ap.add_argument("--k_train", type=int, default=6)
    ap.add_argument("--eval_k", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_particles", type=int, default=8)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--mass_lo", type=float, default=0.5)
    ap.add_argument("--mass_hi", type=float, default=1.5)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--softening", type=float, default=0.05)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--n_scales", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    n = args.n_train + args.n_eval
    qs, vs, mass = gen_nbody(n, args.gen_steps, args.dt, args.n_particles,
                             args.dim, mass_lo=args.mass_lo, mass_hi=args.mass_hi,
                             G=args.G, softening=args.softening, generator=g,
                             device=args.device)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)
    print(f"nbody family: N={args.n_particles} dim={args.dim} "
          f"mass~U[{args.mass_lo},{args.mass_hi}] t_obs={args.t_obs} "
          f"k_train={args.k_train} eval_k={args.eval_k}", flush=True)

    def build(name):
        torch.manual_seed(args.seed)
        ham = NBodyHamiltonianHead(args.dim, hidden_dim=args.hidden,
                                   depth=args.depth, context_dim=args.context_dim)
        if name == "liquid_nbody":
            m = LiquidNBodyModel(args.dim, d_model=args.d_model,
                                 context_dim=args.context_dim, n_scales=args.n_scales,
                                 hidden_dim=args.hidden, depth=args.depth, dt=args.dt)
            return m.to(args.device)
        return StaticNBodyWrapper(ham.to(args.device), args.dt)

    results = {}
    for name in ("liquid_nbody", "static_nbody"):
        model = build(name)
        n_par = sum(p.numel() for p in model.parameters())
        floss = train_semigroup(model, qs[tr], vs[tr], args.t_obs, args.k_train,
                                args.train_steps, args.lr, args.batch, args.seed,
                                lr_decay=args.lr_decay)
        mse, drift = evaluate(model, qs[ev], vs[ev], mass[ev], args.t_obs,
                              args.eval_k, args.dt, args.G, args.softening)
        results[name] = {"params": n_par, "train_loss": floss,
                         "rollout_mse": mse, "energy_drift_max": drift}
        print(f"  [{name:14s}] params {n_par:>6,} | train_loss {floss:.4e} | "
              f"rollout_mse {mse:.4e} | energy_drift(max) {drift:.4e}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nbody_eval.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    lq, st = results["liquid_nbody"], results["static_nbody"]
    print("\n" + "=" * 70, flush=True)
    print("N-BODY + PAIR POTENTIAL | does liquid system-ID add value?", flush=True)
    print("=" * 70, flush=True)
    print(f"  rollout MSE:   liquid {lq['rollout_mse']:.3e} | static "
          f"{st['rollout_mse']:.3e}", flush=True)
    print(f"  energy drift:  liquid {lq['energy_drift_max']:.3e} | static "
          f"{st['energy_drift_max']:.3e}", flush=True)


if __name__ == "__main__":
    main()
