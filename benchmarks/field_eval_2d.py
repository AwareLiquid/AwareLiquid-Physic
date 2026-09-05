"""
benchmarks/field_eval_2d.py 鈥?v0.2 (P2): the 2D spectral-operator potential
on a periodic membrane (2D wave equation).

The 2D counterpart of field_eval.py: a family of 2D membranes, prefix ->
symplectic rollout, with the SpectralConv2d potential conditioned by the
liquid context.

Models:
  liquid_operator2d 鈥?liquid system-ID -> FiLM conditions the 2D FNO potential
  static_operator2d 鈥?same 2D Hamiltonian, context = 0 (no system-ID)

Physics metrics (rollout MSE, energy drift) + a resolution test (H=16 -> 32).

Usage:
  python benchmarks/field_eval_2d.py --train_steps 300
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.datasets import (gen_wave_2d,
                                          gen_wave_2d_inhomogeneous)
from awareliquid_physics.hamiltonian import OperatorHamiltonianHead2d
from awareliquid_physics.model import LiquidOperatorHamiltonianModel2d
from awareliquid_physics.train import train_semigroup


class StaticOperatorWrapper2d(torch.nn.Module):
    """2D operator Hamiltonian with a FIXED zero context (no system-ID)."""

    def __init__(self, ham: OperatorHamiltonianHead2d, dt: float):
        super().__init__()
        self.ham = ham
        self.dt = float(dt)

    def infer_context(self, q_obs, p_obs):
        return torch.zeros(q_obs.shape[0], self.ham.context_dim, device=q_obs.device)

    def rollout(self, q0, p0, ctx, steps):
        return self.ham.rollout(q0, p0, steps, self.dt, context=ctx)

    def forward(self, q_obs, p_obs, k):
        ctx = self.infer_context(q_obs, p_obs)
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout(q0, p0, ctx, k)
        return qs, ps, ctx


def _membrane_energy(q, p, c):
    dqx = q - torch.roll(q, 1, dims=-3)
    dqy = q - torch.roll(q, 1, dims=-2)
    c2 = c * c
    if torch.is_tensor(c):
        c2 = c2.unsqueeze(0).unsqueeze(-1)   # (1, B, H, W, 1) broadcast
    return (0.5 * (p ** 2).sum((-1, -2, -3))
            + (0.5 * c2 * (dqx ** 2 + dqy ** 2)).sum((-1, -2, -3)))


def evaluate(model, qs, ps, c_field, t_obs, eval_k, dt):
    model.eval()
    q_obs, p_obs = qs[:, :t_obs], ps[:, :t_obs]
    with torch.enable_grad():
        qs_pred, ps_pred, _ = model(q_obs, p_obs, eval_k)
    qs_pred, ps_pred = qs_pred.detach(), ps_pred.detach()
    q_true = qs[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3, 4)
    p_true = ps[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3, 4)
    mse = ((qs_pred - q_true).pow(2).mean() + (ps_pred - p_true).pow(2).mean()).item()
    E = _membrane_energy(qs_pred, ps_pred, c_field)   # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    drift = ((E - E[0]).abs() / E0).max().item()
    return mse, drift


def resolution_test(model, args, g):
    n_traj, steps = args.n_res_eval, args.gen_steps
    qs, ps = gen_wave_2d(n_traj, steps, args.dt, args.n_res_test, args.n_res_test,
                         args.c_eval, g, device=args.device)
    model.eval()
    q_obs, p_obs = qs[:, :args.t_obs], ps[:, :args.t_obs]
    with torch.enable_grad():
        qs_pred, ps_pred, _ = model(q_obs, p_obs, args.eval_k)
    qs_pred = qs_pred.detach()
    q_true = qs[:, args.t_obs - 1: args.t_obs + args.eval_k].permute(1, 0, 2, 3, 4)
    return (qs_pred - q_true).pow(2).mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=60)
    ap.add_argument("--t_obs", type=int, default=12)
    ap.add_argument("--k_train", type=int, default=6)
    ap.add_argument("--eval_k", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--c_eval", type=float, default=1.0)
    ap.add_argument("--c_var", type=float, default=0.5,
                    help="inhomogeneous mode: amplitude of the random c(x,y) field")
    ap.add_argument("--inhomogeneous", action="store_true",
                    help="wave speed is a random smooth FIELD c(x,y) per trajectory")
    ap.add_argument("--n_nodes", type=int, default=16)   # H = W = n_nodes
    ap.add_argument("--n_res_test", type=int, default=32)
    ap.add_argument("--n_res_eval", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--n_scales", type=int, default=4)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--fno_depth", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--reflect_pad", type=int, default=4)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    n = args.n_train + args.n_eval
    if args.inhomogeneous:
        qs, ps, cfields = gen_wave_2d_inhomogeneous(
            n, args.gen_steps, args.dt, args.n_nodes, args.n_nodes,
            c_mean=args.c_eval, c_var=args.c_var, generator=g, device=args.device)
        print(f"2D membrane family (INHOMOGENEOUS): {args.n_nodes}x{args.n_nodes} "
              f"c(x,y) = {args.c_eval} +/- {args.c_var} per trajectory", flush=True)
    else:
        qs, ps = gen_wave_2d(n, args.gen_steps, args.dt, args.n_nodes, args.n_nodes,
                             args.c_eval, g, device=args.device)
        cfields = torch.full((n, args.n_nodes, args.n_nodes), args.c_eval,
                             device=args.device)
        print(f"2D membrane family: {args.n_nodes}x{args.n_nodes} c={args.c_eval} "
              f"t_obs={args.t_obs} k_train={args.k_train} eval_k={args.eval_k}", flush=True)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)

    def build(name, seed):
        torch.manual_seed(seed)
        if name == "liquid_operator2d":
            return LiquidOperatorHamiltonianModel2d(
                1, d_model=args.d_model, context_dim=args.context_dim,
                n_scales=args.n_scales, modes_x=args.modes, modes_y=args.modes,
                width=args.width, fno_depth=args.fno_depth, hidden_dim=args.hidden,
                dt=args.dt, reflect_pad=args.reflect_pad).to(args.device)
        ham = OperatorHamiltonianHead2d(
            1, width=args.width, modes_x=args.modes, modes_y=args.modes,
            fno_depth=args.fno_depth, context_dim=args.context_dim,
            hidden_dim=args.hidden, reflect_pad=args.reflect_pad).to(args.device)
        return StaticOperatorWrapper2d(ham, args.dt)

    results = {}
    liquid_trained = None
    for name in ("liquid_operator2d", "static_operator2d"):
        mses, drifts = [], []
        n_par = None
        for s in range(args.n_seeds):
            seed = args.seed + s
            model = build(name, seed)
            if name == "liquid_operator2d":
                liquid_trained = model
            if n_par is None:
                n_par = sum(p.numel() for p in model.parameters())
            train_semigroup(model, qs[tr], ps[tr], args.t_obs, args.k_train,
                            args.train_steps, args.lr, args.batch, seed,
                            lr_decay=args.lr_decay)
            mse, drift = evaluate(model, qs[ev], ps[ev], cfields[ev], args.t_obs,
                                  args.eval_k, args.dt)
            mses.append(mse)
            drifts.append(drift)
        mse_mean = sum(mses) / len(mses)
        drift_mean = sum(drifts) / len(drifts)
        mse_std = (sum((m - mse_mean) ** 2 for m in mses) / len(mses)) ** 0.5
        results[name] = {"params": n_par, "rollout_mse": mse_mean,
                         "rollout_mse_std": mse_std, "energy_drift_max": drift_mean}
        print(f"  [{name:17s}] params {n_par:>6,} | n_seeds {args.n_seeds} | "
              f"rollout_mse {mse_mean:.4e} +/- {mse_std:.2e} | "
              f"energy_drift(max) {drift_mean:.4e}", flush=True)

    g3 = torch.Generator(device=args.device).manual_seed(args.seed + 2)
    mse_hi = resolution_test(liquid_trained, args, g3)
    results["resolution"] = {"train_N": args.n_nodes, "test_N": args.n_res_test,
                             "rollout_mse": mse_hi}
    print(f"  [resolution      ] trained {args.n_nodes}x{args.n_nodes} -> eval "
          f"{args.n_res_test}x{args.n_res_test} zero-shot mse {mse_hi:.4e}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "field_eval_2d.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    lq, st = results["liquid_operator2d"], results["static_operator2d"]
    print("\n" + "=" * 70, flush=True)
    print("2D FIELD + OPERATOR POTENTIAL | does liquid system-ID add value?", flush=True)
    print("=" * 70, flush=True)
    print(f"  rollout MSE:   liquid {lq['rollout_mse']:.3e} | static "
          f"{st['rollout_mse']:.3e}", flush=True)


if __name__ == "__main__":
    main()
