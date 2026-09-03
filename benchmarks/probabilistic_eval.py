"""
benchmarks/probabilistic_eval.py — v0.2 (P2-3): probabilistic prediction via
the VAE-context model on a family of oscillators (hidden stiffness).

ProbabilisticLiquidModel infers a DISTRIBUTION (mu, logvar) over the hidden
stiffness; sampling gives an ensemble of symplectic rollouts. Metrics:
  * ensemble-mean rollout MSE (the point prediction);
  * ensemble SPREAD (uncertainty magnitude) — should be large where the
    model is uncertain (e.g. near the true stiffness range boundary).

Usage: python benchmarks/probabilistic_eval.py --train_steps 300
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.model import ProbabilisticLiquidModel


def gen_spring(n_traj, steps, dt, omega_lo, omega_hi, g, device="cpu"):
    omega = omega_lo + (omega_hi - omega_lo) * torch.rand(n_traj, generator=g,
                                                          device=device)
    q0 = torch.randn(n_traj, 1, generator=g, device=device)
    p0 = torch.randn(n_traj, 1, generator=g, device=device)
    t = torch.arange(steps + 1, device=device, dtype=torch.float32) * dt
    wt = omega[:, None] * t[None, :]
    c, s = torch.cos(wt)[..., None], torch.sin(wt)[..., None]
    qs = q0.unsqueeze(1) * c + (p0.unsqueeze(1) / omega[:, None, None]) * s
    ps = -q0.unsqueeze(1) * omega[:, None, None] * s + p0.unsqueeze(1) * c
    return qs, ps, omega


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--gen_steps", type=int, default=100)
    ap.add_argument("--t_obs", type=int, default=24)
    ap.add_argument("--k_train", type=int, default=8)
    ap.add_argument("--eval_k", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--omega_lo", type=float, default=0.7)
    ap.add_argument("--omega_hi", type=float, default=1.8)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--train_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    qs, ps, omega = gen_spring(args.n_train + args.n_eval, args.gen_steps,
                               args.dt, args.omega_lo, args.omega_hi, g,
                               device=args.device)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)

    torch.manual_seed(args.seed)
    model = ProbabilisticLiquidModel(phase_dim=1, d_model=args.d_model,
                                     context_dim=args.context_dim, n_scales=4,
                                     hidden_dim=args.hidden, dt=args.dt).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for _ in range(args.train_steps):
        bi = torch.randint(0, args.n_train, (args.batch,), generator=g)
        t0 = torch.randint(0, args.gen_steps - args.t_obs - args.k_train,
                           (1,), generator=g).item()
        q_obs = qs[bi, t0:t0 + args.t_obs]
        p_obs = ps[bi, t0:t0 + args.t_obs]
        fut = slice(t0 + args.t_obs - 1, t0 + args.t_obs + args.k_train)
        q_true = qs[bi, fut].permute(1, 0, 2)
        p_true = ps[bi, fut].permute(1, 0, 2)
        q_pred, p_pred, _ = model(q_obs, p_obs, args.k_train, n_samples=1)
        loss = (q_pred[0] - q_true).pow(2).mean() + (p_pred[0] - p_true).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    q_obs, p_obs = qs[ev, :args.t_obs], ps[ev, :args.t_obs]
    with torch.enable_grad():
        q_pred, p_pred, (mu, logvar) = model(q_obs, p_obs, args.eval_k,
                                             n_samples=args.n_samples)
    q_pred, p_pred = q_pred.detach(), p_pred.detach()   # (S, k+1, B, 1)
    q_mean = q_pred.mean(0)
    q_true = qs[ev, args.t_obs - 1: args.t_obs + args.eval_k].permute(1, 0, 2)
    mse_mean = (q_mean - q_true).pow(2).mean().item()
    spread = (q_pred - q_mean).pow(2).mean().sqrt().item()
    logvar_mean = logvar.mean().item()

    results = {"params": sum(p.numel() for p in model.parameters()),
               "ensemble_mean_mse": mse_mean, "ensemble_spread": spread,
               "context_logvar_mean": logvar_mean}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "probabilistic_eval.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"  params {results['params']:>6,} | ensemble_mean_mse {mse_mean:.4e} "
          f"| ensemble_spread {spread:.4e} | context_logvar {logvar_mean:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
