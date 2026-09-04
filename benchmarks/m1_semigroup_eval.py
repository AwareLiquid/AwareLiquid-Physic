"""
benchmarks/m1_semigroup_eval.py — M1 with SEMIGROUP training.

The v0.1 fixed-window train() was the only thing M1 ever used (21% liquid edge
on CPU data). This variant trains with train_semigroup (arbitrary start states
after the prefix — the same loop M2/M3 use) on the SAME spring family, to test
whether semigroup training improves M1 and whether it rescues the CUDA-data
regime where fixed-window liquid lost its edge.

Usage: python benchmarks/m1_semigroup_eval.py --train_steps 2000 --device cuda
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.hamiltonian import HamiltonianHead
from awareliquid_physics.model import LiquidHamiltonianModel
from awareliquid_physics.train import train_semigroup


class StaticHamWrapper(torch.nn.Module):
    def __init__(self, ham, dt):
        super().__init__()
        self.ham = ham
        self.dt = float(dt)

    def infer_context(self, q_obs, p_obs):
        return torch.zeros(q_obs.shape[0], 1, device=q_obs.device)

    def rollout(self, q0, p0, ctx, steps):
        return self.ham.rollout(q0, p0, steps, self.dt)

    def forward(self, q_obs, p_obs, k):
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout(q0, p0, None, k)
        return qs, ps, None


def gen_spring(n_traj, steps, dt, dim, omega_lo, omega_hi, g, device="cpu"):
    omega = omega_lo + (omega_hi - omega_lo) * torch.rand(n_traj, generator=g,
                                                          device=device)
    q0 = torch.randn(n_traj, dim, generator=g, device=device)
    p0 = torch.randn(n_traj, dim, generator=g, device=device)
    t = torch.arange(steps + 1, dtype=torch.float32, device=device) * dt
    wt = omega.view(-1, 1) * t.view(1, -1)
    c, s = torch.cos(wt).unsqueeze(-1), torch.sin(wt).unsqueeze(-1)
    w = omega.view(-1, 1, 1)
    qs = q0.unsqueeze(1) * c + (p0.unsqueeze(1) / w) * s
    ps = -q0.unsqueeze(1) * w * s + p0.unsqueeze(1) * c
    return qs, ps, omega


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--gen_steps", type=int, default=160)
    ap.add_argument("--t_obs", type=int, default=24)
    ap.add_argument("--k_train", type=int, default=8)
    ap.add_argument("--eval_k", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--omega_lo", type=float, default=0.7)
    ap.add_argument("--omega_hi", type=float, default=1.8)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    qs, ps, omega = gen_spring(args.n_train + args.n_eval, args.gen_steps,
                               args.dt, 1, args.omega_lo, args.omega_hi, g,
                               device=args.device)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)
    print(f"spring family (semigroup): omega~U[{args.omega_lo},{args.omega_hi}] "
          f"t_obs={args.t_obs} k_train={args.k_train} eval_k={args.eval_k}",
          flush=True)

    results = {}
    for name in ("liquid_sg", "static_sg"):
        torch.manual_seed(args.seed)
        if name == "liquid_sg":
            model = LiquidHamiltonianModel(1, d_model=args.d_model,
                                           context_dim=args.context_dim,
                                           n_scales=4, hidden_dim=args.hidden,
                                           depth=2, dt=args.dt).to(args.device)
        else:
            ham = HamiltonianHead(1, hidden_dim=args.hidden, depth=2,
                                  context_dim=0).to(args.device)
            model = StaticHamWrapper(ham, args.dt)
        n_par = sum(p.numel() for p in model.parameters())
        floss = train_semigroup(model, qs[tr], ps[tr], args.t_obs, args.k_train,
                                args.train_steps, args.lr, args.batch, args.seed,
                                lr_decay=args.lr_decay)
        model.eval()
        q_obs, p_obs = qs[ev, :args.t_obs], ps[ev, :args.t_obs]
        with torch.enable_grad():
            qs_pred, ps_pred, _ = model(q_obs, p_obs, args.eval_k)
        qs_pred, ps_pred = qs_pred.detach(), ps_pred.detach()
        q_true = qs[ev, args.t_obs - 1: args.t_obs + args.eval_k].permute(1, 0, 2)
        p_true = ps[ev, args.t_obs - 1: args.t_obs + args.eval_k].permute(1, 0, 2)
        mse = ((qs_pred - q_true).pow(2).mean() + (ps_pred - p_true).pow(2).mean()).item()
        results[name] = {"params": n_par, "train_loss": floss, "rollout_mse": mse}
        print(f"  [{name:10s}] params {n_par:>6,} | train_loss {floss:.4e} | "
              f"rollout_mse {mse:.4e}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "m1_semigroup.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    lq, st = results["liquid_sg"], results["static_sg"]
    gap = 1.0 - lq["rollout_mse"] / st["rollout_mse"]
    print(f"\nSEMIGROUP M1 | liquid {lq['rollout_mse']:.3e} vs static "
          f"{st['rollout_mse']:.3e} | liquid advantage {gap * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
