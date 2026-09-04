"""
benchmarks/field_eval.py 鈥?v0.2 (M2): the OPERATOR potential on a 1D field.

The coupling experiment for fields: a family of periodic strings (hidden wave
speed c per trajectory), prefix -> symplectic rollout, with a SPECTRAL (FNO)
potential conditioned by the liquid context.

Models:
  liquid_operator  鈥?liquid core system-ID -> FiLM conditions the FNO potential
  static_operator  鈥?same FNO Hamiltonian, context = 0 (no system identification)

Physics metrics only (rollout MSE, energy drift), plus the resolution test:
train at N=32, evaluate zero-shot at N=64/128 (resolution invariance).

Usage:
  python benchmarks/field_eval.py --train_steps 300
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awareliquid_physics.datasets import gen_wave_1d, gen_wave_1d_inhomogeneous
from awareliquid_physics.hamiltonian import OperatorHamiltonianHead
from awareliquid_physics.model import LiquidOperatorHamiltonianModel
from awareliquid_physics.train import train_semigroup


class StaticOperatorWrapper(torch.nn.Module):
    """Operator Hamiltonian with a FIXED zero context 鈥?the 'no system-ID'
    control, matched to the train/rollout interface."""

    def __init__(self, ham: OperatorHamiltonianHead, dt: float):
        super().__init__()
        self.ham = ham
        self.dt = float(dt)

    def infer_context(self, q_obs, p_obs):
        B = q_obs.shape[0]
        return torch.zeros(B, self.ham.context_dim, device=q_obs.device)

    def rollout(self, q0, p0, ctx, steps):
        return self.ham.rollout(q0, p0, steps, self.dt, context=ctx)

    def forward(self, q_obs, p_obs, k):
        ctx = self.infer_context(q_obs, p_obs)
        q0, p0 = q_obs[:, -1], p_obs[:, -1]
        qs, ps = self.rollout(q0, p0, ctx, k)
        return qs, ps, ctx


def make_models(args, seed):
    torch.manual_seed(seed)
    liquid = LiquidOperatorHamiltonianModel(
        phase_dim=1, d_model=args.d_model, context_dim=args.context_dim,
        n_scales=args.n_scales, modes=args.modes, width=args.width,
        fno_depth=args.fno_depth, hidden_dim=args.hidden, t_depth=2,
        dt=args.dt, core_dt=1.0, reflect_pad=args.reflect_pad).to(args.device)
    static_ham = OperatorHamiltonianHead(
        dim=1, width=args.width, modes=args.modes, fno_depth=args.fno_depth,
        context_dim=args.context_dim, hidden_dim=args.hidden, t_depth=2,
        reflect_pad=args.reflect_pad).to(args.device)
    static = StaticOperatorWrapper(static_ham, args.dt)
    return liquid, static


def evaluate(model, qs, ps, c_field, t_obs, eval_k, dt):
    """Free-running eval from the prefix: rollout MSE + energy drift. c_field is
    the per-trajectory wave-speed field (n_traj, N) for the energy diagnostic."""
    model.eval()
    q_obs, p_obs = qs[:, :t_obs], ps[:, :t_obs]
    with torch.enable_grad():
        qs_pred, ps_pred, ctx = model(q_obs, p_obs, eval_k)
    qs_pred, ps_pred = qs_pred.detach(), ps_pred.detach()
    q_true = qs[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3)
    p_true = ps[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2, 3)
    mse = ((qs_pred - q_true).pow(2).mean() + (ps_pred - p_true).pow(2).mean()).item()

    def string_energy(q, p):
        dq = q - torch.roll(q, 1, dims=-2)
        c2 = (c_field ** 2).unsqueeze(0).unsqueeze(-1)       # (1, B, N, 1)
        return 0.5 * (p ** 2).sum((-1, -2)) + 0.5 * (c2 * dq ** 2).sum((-1, -2))
    E = string_energy(qs_pred, ps_pred)                      # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    drift = ((E - E[0]).abs() / E0).max().item()
    return mse, drift


def resolution_test(model, args, g):
    """Zero-shot super-resolution: trained at N_train, evaluated at N_test."""
    n_traj, steps = args.n_res_eval, args.gen_steps
    qs, ps = gen_wave_1d(n_traj, steps, args.dt, args.n_res_test, args.c_eval, g,
                         device=args.device)
    qs, ps = qs[:n_traj], ps[:n_traj]
    t_obs, k = args.t_obs, args.eval_k
    model.eval()
    q_obs, p_obs = qs[:, :t_obs], ps[:, :t_obs]
    with torch.enable_grad():
        qs_pred, ps_pred, _ = model(q_obs, p_obs, k)
    qs_pred = qs_pred.detach()
    q_true = qs[:, t_obs - 1: t_obs + k].permute(1, 0, 2, 3)
    mse = (qs_pred - q_true).pow(2).mean().item()
    return mse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=120)
    ap.add_argument("--t_obs", type=int, default=24)
    ap.add_argument("--k_train", type=int, default=8)
    ap.add_argument("--eval_k", type=int, default=80)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--c_lo", type=float, default=0.8)
    ap.add_argument("--c_hi", type=float, default=1.6)
    ap.add_argument("--c_eval", type=float, default=1.0)
    ap.add_argument("--c_var", type=float, default=0.5,
                    help="inhomogeneous mode: amplitude of the random c(x) field")
    ap.add_argument("--inhomogeneous", action="store_true",
                    help="wave speed is a random smooth FIELD c(x) per trajectory")
    ap.add_argument("--n_nodes", type=int, default=32)
    ap.add_argument("--n_res_test", type=int, default=64)
    ap.add_argument("--n_res_eval", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--n_scales", type=int, default=4)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--fno_depth", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--reflect_pad", type=int, default=8)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0,
                    help="per-step exponential lr decay (1.0 = constant)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=1,
                    help="repeat training across seeds, report mean +/- std")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    n = args.n_train + args.n_eval
    if args.inhomogeneous:
        # Hard version: the hidden parameter is a whole spatial FIELD c(x) per
        # trajectory 鈥?a static potential can only approximate an average medium.
        qs, ps, cfields = gen_wave_1d_inhomogeneous(
            n, args.gen_steps, args.dt, args.n_nodes,
            c_mean=args.c_eval, c_var=args.c_var, generator=g,
            device=args.device)
        print(f"field family (INHOMOGENEOUS): N={args.n_nodes} "
              f"c(x) = {args.c_eval} +/- {args.c_var} per trajectory", flush=True)
    else:
        # Family of constant wave speeds: half at c_eval, rest spread over
        # [c_lo, c_hi] 鈥?per-trajectory hidden scalar the context must identify.
        qs, ps = gen_wave_1d(n, args.gen_steps, args.dt, args.n_nodes,
                             c=args.c_eval, generator=g, device=args.device)
        g2 = torch.Generator(device=args.device).manual_seed(args.seed + 1)
        cs = torch.empty(n, device=args.device)
        cs[:n // 2] = args.c_eval
        cs[n // 2:] = args.c_lo + (args.c_hi - args.c_lo) * torch.rand(
            n - n // 2, generator=g2, device=args.device)
        for i in range(n // 2, n):
            qi, pi = gen_wave_1d(1, args.gen_steps, args.dt, args.n_nodes,
                                 c=cs[i].item(), generator=g2, device=args.device)
            qs[i], ps[i] = qi[0], pi[0]
        cfields = cs[:, None].expand(n, args.n_nodes)
        print(f"field family: N={args.n_nodes} c~{{{args.c_lo}..{args.c_hi}}} "
              f"t_obs={args.t_obs} k_train={args.k_train} eval_k={args.eval_k}",
              flush=True)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)

    results = {}
    liquid_trained = None
    for name in ("liquid_operator", "static_operator"):
        mses, drifts = [], []
        n_par = None
        for s in range(args.n_seeds):
            seed = args.seed + s
            liquid, static = make_models(args, seed)
            model = liquid if name == "liquid_operator" else static
            if name == "liquid_operator":
                liquid_trained = model
            if n_par is None:
                n_par = sum(p.numel() for p in model.parameters())
            floss = train_semigroup(model, qs[tr], ps[tr], args.t_obs, args.k_train,
                                    args.train_steps, args.lr, args.batch, seed,
                                    lr_decay=args.lr_decay)
            mse, drift = evaluate(model, qs[ev], ps[ev], cfields[ev], args.t_obs,
                                  args.eval_k, args.dt)
            mses.append(mse)
            drifts.append(drift)
        mse_mean = sum(mses) / len(mses)
        drift_mean = sum(drifts) / len(drifts)
        mse_std = (sum((m - mse_mean) ** 2 for m in mses) / len(mses)) ** 0.5
        drift_std = (sum((d - drift_mean) ** 2 for d in drifts) / len(drifts)) ** 0.5
        results[name] = {"params": n_par, "train_loss": floss,
                         "rollout_mse": mse_mean, "rollout_mse_std": mse_std,
                         "energy_drift_max": drift_mean,
                         "energy_drift_max_std": drift_std}
        print(f"  [{name:16s}] params {n_par:>6,} | n_seeds {args.n_seeds} | "
              f"rollout_mse {mse_mean:.4e} +/- {mse_std:.2e} | "
              f"energy_drift(max) {drift_mean:.4e} +/- {drift_std:.2e}", flush=True)

    # Resolution invariance: the SAME trained model at 2x the node count.
    g3 = torch.Generator(device=args.device).manual_seed(args.seed + 2)
    mse_hi = resolution_test(liquid_trained, args, g3)
    results["resolution"] = {"train_N": args.n_nodes, "test_N": args.n_res_test,
                             "rollout_mse": mse_hi}
    print(f"  [resolution      ] trained N={args.n_nodes} -> eval N="
          f"{args.n_res_test} zero-shot rollout_mse {mse_hi:.4e}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    import json
    with open(os.path.join(args.out_dir, "field_eval.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    lq, st = results["liquid_operator"], results["static_operator"]
    print("\n" + "=" * 70, flush=True)
    print("FIELD + OPERATOR POTENTIAL | does liquid system-ID add value?", flush=True)
    print("=" * 70, flush=True)
    print(f"  rollout MSE:   liquid {lq['rollout_mse']:.3e} | static "
          f"{st['rollout_mse']:.3e}", flush=True)
    print(f"  energy drift:  liquid {lq['energy_drift_max']:.3e} | static "
          f"{st['energy_drift_max']:.3e}", flush=True)
    gap = 1.0 - lq["rollout_mse"] / st["rollout_mse"]
    print(f"  liquid advantage over static: {gap * 100:.1f}% "
          f"(target >= 30%)", flush=True)


if __name__ == "__main__":
    main()
