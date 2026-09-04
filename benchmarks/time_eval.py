"""
benchmarks/time_eval.py 鈥?v0.2 (P2-4): time-conditioned Hamiltonian for a
DRIVEN oscillator (time-dependent force).

TimeConditionedHamiltonianHead takes time as an explicit input to the
potential V(q, t), so the SAME model is evaluated at any time (continuous-time
evaluation). Ground truth is the RK4 driven-oscillator trajectory (gen_driven);
energy is NOT conserved there (the drive does work) 鈥?the head must model the
time dependence, not fight it.

Metric: k-step rollout MSE against the driven truth.
Usage: python benchmarks/time_eval.py --train_steps 300
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.datasets import gen_driven
from awareliquid_physics.hamiltonian import TimeConditionedHamiltonianHead


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--drive_amp", type=float, default=0.5)
    ap.add_argument("--drive_freq", type=float, default=2.0)
    ap.add_argument("--eval_k", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    qs, ps = gen_driven(args.n_train + args.n_eval, args.gen_steps, args.dt,
                        args.omega0, args.drive_amp, args.drive_freq, g,
                        device=args.device)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)
    print(f"driven oscillator: omega0={args.omega0} drive={args.drive_amp}@"
          f"{args.drive_freq} dt={args.dt}", flush=True)

    torch.manual_seed(args.seed)
    head = TimeConditionedHamiltonianHead(dim=1, hidden_dim=args.hidden,
                                          depth=args.depth).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: args.lr_decay ** t)
    head.train()
    N, S = qs.shape[0], qs.shape[1]
    g_train = torch.Generator().manual_seed(args.seed + 1000)   # CPU, index sampling
    for _ in range(args.train_steps):
        bi = torch.randint(0, N, (args.batch,), generator=g_train)
        t0 = torch.randint(0, S - 2, (1,), generator=g_train).item()
        q, p = qs[bi, t0], ps[bi, t0]
        t = torch.full((args.batch,), t0 * args.dt, device=args.device)
        q_true, p_true = qs[bi, t0 + 1], ps[bi, t0 + 1]
        q_next, p_next, _ = head.step(q, p, t, args.dt)
        loss = (q_next - q_true).pow(2).mean() + (p_next - p_true).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    head.eval()
    q_obs, p_obs = qs[ev, 0], ps[ev, 0]
    t0 = torch.zeros(args.n_eval, device=args.device)
    with torch.enable_grad():
        qs_pred, ps_pred, ts = head.rollout(q_obs, p_obs, t0, args.eval_k, args.dt)
    qs_pred = qs_pred.detach()
    q_true = qs[ev, :args.eval_k + 1].permute(1, 0, 2)
    mse = (qs_pred - q_true).pow(2).mean().item()

    results = {"params": sum(p.numel() for p in head.parameters()),
               "rollout_mse": mse}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "time_eval.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"  params {results['params']:>6,} | rollout_mse {mse:.4e}", flush=True)


if __name__ == "__main__":
    main()
