"""
benchmarks/nonseparable_eval.py 鈥?v0.2 (P2-2): learn a NON-separable
Hamiltonian for charged particles in a uniform magnetic field.

The head NonseparableHamiltonianHead integrates H = T(p) + V(q) + C(q,p) by
IMPLICIT MIDPOINT (velocity-Verlet does not apply). Ground truth is the RK4
magnetic trajectory (gen_magnetic) 鈥?energy is conserved there (magnetic
forces do no work), and the learned head must recover that.

Metrics: k-step rollout MSE and energy drift (the true H is analytic).
Usage: python benchmarks/nonseparable_eval.py --train_steps 300
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.datasets import gen_magnetic
from awareliquid_physics.hamiltonian import NonseparableHamiltonianHead


def true_H(q, p, B):
    x, y = q[..., 0], q[..., 1]
    px, py = p[..., 0], p[..., 1]
    return 0.5 * ((px + 0.5 * B * y) ** 2 + (py - 0.5 * B * x) ** 2)


def train(head, qs, ps, steps, lr, batch, seed, lr_decay):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: lr_decay ** t)
    N, S = qs.shape[0], qs.shape[1]
    head.train()
    loss = float("nan")
    for _ in range(steps):
        bi = torch.randint(0, N, (batch,), generator=g)
        t0 = torch.randint(0, S - 2, (1,), generator=g).item()
        q, p = qs[bi, t0], ps[bi, t0]
        q_true, p_true = qs[bi, t0 + 1], ps[bi, t0 + 1]
        q_next, p_next = head.step(q, p, args_dt)
        loss = (q_next - q_true).pow(2).mean() + (p_next - p_true).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return loss.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--B", type=float, default=2.0)
    ap.add_argument("--eval_k", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()
    global args_dt
    args_dt = args.dt

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    qs, ps = gen_magnetic(args.n_train + args.n_eval, args.gen_steps, args.dt,
                          args.B, g, device=args.device)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)
    print(f"magnetic: B={args.B} dt={args.dt} train={args.n_train} "
          f"eval_k={args.eval_k}", flush=True)

    torch.manual_seed(args.seed)
    head = NonseparableHamiltonianHead(dim=2, hidden_dim=args.hidden,
                                       depth=args.depth, iters=args.iters).to(args.device)
    n_par = sum(p.numel() for p in head.parameters())
    floss = train(head, qs[tr], ps[tr], args.train_steps, args.lr, args.batch,
                  args.seed, args.lr_decay)

    head.eval()
    q_obs, p_obs = qs[ev, 0], ps[ev, 0]
    with torch.enable_grad():
        qs_pred, ps_pred = head.rollout(q_obs, p_obs, args.eval_k, args.dt)
    qs_pred, ps_pred = qs_pred.detach(), ps_pred.detach()
    q_true = qs[ev, :args.eval_k + 1].permute(1, 0, 2)
    p_true = ps[ev, :args.eval_k + 1].permute(1, 0, 2)
    mse = ((qs_pred - q_true).pow(2).mean() + (ps_pred - p_true).pow(2).mean()).item()
    E = true_H(qs_pred, ps_pred, args.B)                    # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    drift = ((E - E[0]).abs() / E0).max().item()

    results = {"params": n_par, "train_loss": floss, "rollout_mse": mse,
               "energy_drift_max": drift}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nonseparable_eval.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"  params {n_par:>6,} | train_loss {floss:.4e} | rollout_mse {mse:.4e} "
          f"| energy_drift(max) {drift:.4e}", flush=True)


if __name__ == "__main__":
    main()
