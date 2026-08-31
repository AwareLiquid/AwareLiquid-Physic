"""
benchmarks/pretrain_finetune_eval.py — v0.2 (M3): pretrain/finetune protocol
(Poseidon-style sample efficiency) on the wave-field family.

Question: does pretraining on a MIX of wave speeds let the model adapt to a
NEVER-SEEN speed with only a handful of trajectories (few-shot), compared to
training from scratch on the same handful?

  pretrain:  c in {0.8, 1.0, 1.2}  mixed semigroup training
  finetune:  c = 1.5 (unseen)      n_shot trajectories (5 or 20)
  control:   from-scratch training on the same n_shot trajectories

Metric: rollout MSE on a held-out set of the unseen speed. Target (PRD P1-2):
few-shot (<= 20 trajectories) reaches the accuracy that from-scratch training
needs far more data to achieve.

Usage:
  python benchmarks/pretrain_finetune_eval.py --pretrain_steps 300 --finetune_steps 200
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.datasets import gen_wave_1d
from awareliquid_physics.model import LiquidOperatorHamiltonianModel
from awareliquid_physics.train import finetune, train_pretrain, train_semigroup


def build_model(args, seed):
    torch.manual_seed(seed)
    return LiquidOperatorHamiltonianModel(
        phase_dim=1, d_model=args.d_model, context_dim=args.context_dim,
        n_scales=args.n_scales, modes=args.modes, width=args.width,
        fno_depth=args.fno_depth, hidden_dim=args.hidden, t_depth=2,
        dt=args.dt, core_dt=1.0, reflect_pad=args.reflect_pad)


def evaluate(model, qs, ps, t_obs, eval_k, c):
    model.eval()
    q_obs, p_obs = qs[:, :t_obs], ps[:, :t_obs]
    with torch.enable_grad():
        qs_pred, ps_pred, _ = model(q_obs, p_obs, eval_k)
    qs_pred, ps_pred = qs_pred.detach(), ps_pred.detach()
    q_true = qs[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3)
    p_true = ps[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3)
    return ((qs_pred - q_true).pow(2).mean() + (ps_pred - p_true).pow(2).mean()).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_steps", type=int, default=100)
    ap.add_argument("--t_obs", type=int, default=24)
    ap.add_argument("--k_train", type=int, default=8)
    ap.add_argument("--eval_k", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_nodes", type=int, default=32)
    ap.add_argument("--n_pretrain_per_speed", type=int, default=96)
    ap.add_argument("--n_shot", type=int, default=20)
    ap.add_argument("--n_few_eval", type=int, default=64)
    ap.add_argument("--c_target", type=float, default=1.5)   # unseen speed
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--n_scales", type=int, default=4)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--fno_depth", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--reflect_pad", type=int, default=8)
    ap.add_argument("--pretrain_steps", type=int, default=300)
    ap.add_argument("--finetune_steps", type=int, default=200)
    ap.add_argument("--fromscratch_steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    print(f"target speed c={args.c_target} (unseen in pretrain); "
          f"n_shot={args.n_shot}", flush=True)

    # Pretrain corpus: three speeds, the target deliberately excluded.
    families = []
    for i, c in enumerate((0.8, 1.0, 1.2)):
        qs, ps = gen_wave_1d(args.n_pretrain_per_speed, args.gen_steps, args.dt,
                             args.n_nodes, c, torch.Generator().manual_seed(args.seed + i))
        families.append((qs, ps))

    # Few-shot + eval sets at the UNSEEN speed.
    qs_t, ps_t = gen_wave_1d(args.n_shot + args.n_few_eval, args.gen_steps,
                             args.dt, args.n_nodes, args.c_target,
                             torch.Generator().manual_seed(args.seed + 100))
    qs_few, ps_few = qs_t[:args.n_shot], ps_t[:args.n_shot]
    qs_ev, ps_ev = qs_t[args.n_shot:], ps_t[args.n_shot:]

    results = {}

    # 1) Pretrain on the mix, then few-shot finetune to the unseen speed.
    model = build_model(args, args.seed)
    lp = train_pretrain(model, families, args.t_obs, args.k_train,
                        args.pretrain_steps, args.lr, args.batch, args.seed)
    lf = finetune(model, qs_few, ps_few, args.t_obs, args.k_train,
                  args.finetune_steps, args.lr, args.batch, args.seed + 1)
    mse_ft = evaluate(model, qs_ev, ps_ev, args.t_obs, args.eval_k, args.c_target)
    results["pretrain_then_finetune"] = {"pretrain_loss": lp, "finetune_loss": lf,
                                         "rollout_mse": mse_ft}
    print(f"  [pretrain->finetune] pre {lp:.4e} -> ft {lf:.4e} | "
          f"rollout_mse {mse_ft:.4e}", flush=True)

    # 2) Control: from scratch on the same n_shot trajectories (matched budget).
    model2 = build_model(args, args.seed)
    ls = train_semigroup(model2, qs_few, ps_few, args.t_obs, args.k_train,
                         args.fromscratch_steps, args.lr, args.batch, args.seed + 2)
    mse_fs = evaluate(model2, qs_ev, ps_ev, args.t_obs, args.eval_k, args.c_target)
    results["from_scratch"] = {"train_loss": ls, "rollout_mse": mse_fs}
    print(f"  [from scratch      ] loss {ls:.4e} | rollout_mse {mse_fs:.4e}",
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pretrain_finetune.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("PRETRAIN/FINETUNE | does pretraining buy few-shot adaptation?", flush=True)
    print("=" * 70, flush=True)
    ratio = mse_fs / max(mse_ft, 1e-12)
    print(f"  rollout MSE: pretrain->finetune {mse_ft:.3e} | from scratch "
          f"{mse_fs:.3e} (ratio {ratio:.2f}x)", flush=True)
    print(f"  few-shot advantage: {ratio:.1f}x  (target: >= 3x)", flush=True)


if __name__ == "__main__":
    main()
