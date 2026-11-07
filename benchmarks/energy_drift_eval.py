"""P0-3 closure artifact — 100-step energy drift under the v0.2 architecture
(new liquid context + operator-ready heads) vs the recorded v0.1 baseline.

The PRD pins P0-3 as: "新能量场 + 新 context 下，100 步能量漂移 ≤ v0.1 基线
（守恒是架构性质，指标用于防退化验证）", with the v0.1 baseline recorded as
orbit-system 100-step drift 6.2 (unstructured control 37.8) — docs/PRD.md §1/§6.
Rounds 5-7 proved the new context wins big on rollout MSE (liquid advantage
61-90%) but NEVER measured drift under that config (m1_semigroup_eval reports
no drift at all) — the round-8 gap review flagged exactly this as the only
blocker to an all-green P0. This benchmark closes it.

Protocol (IDENTICAL for every model in a system — that is the point):
  each model observes the same t_obs-step prefix of a held-out trajectory,
  free-runs eval_k=100 steps from the last observed state, and is scored with
  the v0.1 drift diagnostic: relative energy drift |E(t)-E(0)|/|E(0)| against
  the TRUE system energy (true omega for the spring family; KE + softened
  gravity PE for orbit), averaged over eval trajectories.

Models:
  liquid_v2_sg : LiquidHamiltonianModel — the v0.2 new context (input-dependent
                 tau), trained with the SEMIGROUP recipe that produced the
                 61-90% advantage (PRD §13-14, train_semigroup).
  static_ham   : HamiltonianHead without context — the v0.1 architecture whose
                 orbit drift (6.2) IS the P0-3 baseline, 1-step supervised
                 (train_head, the v0.1 protocol).
  mlp_field    : MLPFieldHead unstructured Euler control (v0.1 recorded 37.8).

Systems:
  orbit  : 2-body softened gravity (gen_orbit) — the system the 6.2 baseline
           was recorded on; the pinned P0-3 acceptance row.
  spring : harmonic family with hidden omega (gen_spring) — the M1 home task.

Honest scope: the recorded 6.2/37.8 are quoted, not re-measured here; the
same-protocol anchor re-run lives in physics_rollout_orbit.json (run
`python benchmarks/physics_rollout_eval.py --system orbit --eval_k 100`).
P0-3 gate (what `verdict.p03_pass` means): the PRD pins the drift baseline to
the ORBIT system only (§6 metric table, one row: "100 步能量漂移（轨道）≤ 6.2"),
so the gate is orbit liquid_v2_sg ≤ 6.2 recorded AND ≤ the re-measured v0.1
static anchor. spring is supplementary coverage with no recorded baseline; its
liquid-vs-static comparison is reported but does not gate. Caveat recorded for
that comparison: the true-energy diagnostic charges omega system-ID error to
drift (liquid infers omega; drift_max ≈ drift_final → bounded oscillation, not
secular loss — the learned H is conserved by construction regardless).
Within THIS file every number is same-device (P5) and averaged over --n_seeds
seeds with std/stderr attached (P6). Per-step drift curves stream to a JSONL
file (JsonlMetricWriter) so the "jsonl/表" the round-8 review asks for exists
at both resolutions. CPU-runnable; no GPU required.

Usage:
    python benchmarks/energy_drift_eval.py --n_seeds 3
    python benchmarks/energy_drift_eval.py --system spring --n_seeds 1  # quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from awareliquid_physics.hamiltonian import HamiltonianHead, MLPFieldHead
from awareliquid_physics.model import LiquidHamiltonianModel
from awareliquid_physics.observability import JsonlMetricWriter, run_metadata
from awareliquid_physics.train import train_semigroup
from benchmarks.liquid_physics_eval import spring_energy
from benchmarks.m1_semigroup_eval import gen_spring
from benchmarks.physics_rollout_eval import gen_orbit, orbit_energy, train_head

# v0.1 recorded baseline (docs/PRD.md §1/§6) — quoted for the verdict, never
# re-measured by this script (the re-measurement is physics_rollout_orbit.json).
V01_RECORDED = {
    "orbit": {"static_ham": 6.2, "mlp_field": 37.8},
    "source": "docs/PRD.md §1/§6 (v0.1 记录值，本脚本引用不重测；"
              "同协议重测见 physics_rollout_orbit.json)",
}


def gen_data(system, n_traj, steps, dt, args, g):
    """Dispatch to the EXACT generators the recorded benchmarks used."""
    if system == "spring":
        qs, ps, omega = gen_spring(n_traj, steps, dt, 1, args.omega_lo,
                                   args.omega_hi, g, device="cpu")
        return qs, ps, {"omega": omega}
    qs, ps, meta = gen_orbit(n_traj, steps, dt, g)
    return qs, ps, meta


def true_energy(system, qs, ps, meta):
    """TRUE system energy along a (k+1, B, dim) rollout — the diagnostic the
    v0.1 benchmark used (never the model's own learned H)."""
    if system == "spring":
        return spring_energy(qs, ps, meta["omega"].view(1, -1))
    return orbit_energy(qs, ps, meta)


def drift_stats(qs_pred, ps_pred, system, meta):
    """v0.1 drift diagnostic on a (k+1, B, dim) rollout: relative drift
    |E(t)-E(0)|/|E(0)| per step, mean over trajectories → (k+1,)."""
    E = true_energy(system, qs_pred, ps_pred, meta)                # (k+1, B)
    E0 = E[0].abs().clamp_min(1e-6)
    rel = (E - E[0]).abs() / E0                                    # (k+1, B)
    drift = rel.mean(-1)                                           # (k+1,)
    per_traj_final = rel[-1]                                       # (B,)
    n = max(per_traj_final.numel(), 2)
    return {"drift_final": drift[-1].item(),
            "drift_max": drift.max().item(),
            "drift_traj_stderr": per_traj_final.std(correction=1).item() / n ** 0.5,
            "curve": drift}


def eval_liquid(model, qs, ps, meta, system, t_obs, eval_k):
    model.eval()
    q_obs, p_obs = qs[:, :t_obs], ps[:, :t_obs]
    with torch.enable_grad():
        qs_pred, ps_pred, _ = model(q_obs, p_obs, eval_k)
    return drift_stats(qs_pred.detach(), ps_pred.detach(), system, meta), \
        qs_pred.detach(), ps_pred.detach()


def rollout_head(head, qs, ps, meta, system, t_obs, eval_k, dt):
    """Anchors roll from the SAME last-observed state as the liquid model."""
    q0, p0 = qs[:, t_obs - 1], ps[:, t_obs - 1]
    with torch.enable_grad():          # field/energy grads need autograd
        qs_pred, ps_pred = head.rollout(q0, p0, eval_k, dt)
    return drift_stats(qs_pred.detach(), ps_pred.detach(), system, meta), \
        qs_pred.detach(), ps_pred.detach()


def mse(qs_pred, ps_pred, qs, ps, t_obs, eval_k):
    q_true = qs[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2)
    p_true = ps[:, t_obs - 1: t_obs + eval_k].permute(1, 0, 2)
    return ((qs_pred - q_true).pow(2).mean()
            + (ps_pred - p_true).pow(2).mean()).item()


def run_one(system, seed, args):
    """Train + evaluate all three models under one seed. Returns per-model
    metrics with the per-step drift curve attached."""
    g = torch.Generator().manual_seed(seed)
    qs, ps, meta = gen_data(system, args.n_train + args.n_eval,
                            args.gen_steps, args.dt, args, g)
    tr, ev = slice(0, args.n_train), slice(args.n_train, None)
    meta_ev = {"omega": meta["omega"][ev]} if system == "spring" else meta
    phase_dim = 1 if system == "spring" else 4
    out = {}

    # -- new context, semigroup recipe (PRD §13-14, the P0-1-winning config) --
    torch.manual_seed(seed)
    liquid = LiquidHamiltonianModel(phase_dim, d_model=args.d_model,
                                    context_dim=args.context_dim,
                                    n_scales=args.n_scales,
                                    hidden_dim=args.hidden, depth=2,
                                    dt=args.dt).to(args.device)
    qs_d, ps_d = qs.to(args.device), ps.to(args.device)
    train_semigroup(liquid, qs_d[tr], ps_d[tr], args.t_obs, args.k_train,
                    args.train_steps, args.lr, args.batch, seed,
                    lr_decay=args.lr_decay)
    stats, qsp, psp = eval_liquid(liquid, qs_d[ev], ps_d[ev], meta_ev, system,
                                  args.t_obs, args.eval_k)
    stats["params"] = sum(p.numel() for p in liquid.parameters())
    stats["rollout_mse"] = mse(qsp, psp, qs_d[ev], ps_d[ev], args.t_obs, args.eval_k)
    out["liquid_v2_sg"] = stats

    # -- v0.1 anchors: 1-step supervised (train_head), same prefix protocol --
    for name, Head in (("static_ham", HamiltonianHead), ("mlp_field", MLPFieldHead)):
        torch.manual_seed(seed)
        head = Head(phase_dim, hidden_dim=args.anchor_hidden, depth=2).to(args.device)
        train_head(head, qs[tr], ps[tr], args.anchor_train_steps, args.anchor_lr,
                   args.anchor_batch, args.dt, seed)
        stats, qsp, psp = rollout_head(head, qs[ev], ps[ev], meta_ev, system,
                                       args.t_obs, args.eval_k, args.dt)
        stats["params"] = sum(p.numel() for p in head.parameters())
        stats["rollout_mse"] = mse(qsp, psp, qs[ev], ps[ev], args.t_obs, args.eval_k)
        out[name] = stats
    return out


def aggregate(per_seed, keys):
    """Across-seed mean/std/stderr for each metric key (P6: no bare single-seed
    means; per-seed values are kept as drift_final_seed{i} etc.)."""
    agg = {}
    for model, seed_vals in per_seed.items():
        m = {"params": seed_vals[0]["params"]}
        for key in keys:
            vals = [s[key] for s in seed_vals]
            mean = sum(vals) / len(vals)
            m[key] = mean
            if len(vals) > 1:
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                std = var ** 0.5
                m[f"{key}_std"] = std
                m[f"{key}_stderr"] = std / len(vals) ** 0.5
            for i, v in enumerate(vals):
                m[f"{key}_seed{i}"] = v
        agg[model] = m
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=["orbit", "spring", "both"], default="both")
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--gen_steps", type=int, default=200)
    ap.add_argument("--t_obs", type=int, default=24)
    ap.add_argument("--k_train", type=int, default=8)
    ap.add_argument("--eval_k", type=int, default=100,
                    help="drift horizon; PRD P0-3 pins 100")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--omega_lo", type=float, default=0.7)
    ap.add_argument("--omega_hi", type=float, default=1.8)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--context_dim", type=int, default=8)
    ap.add_argument("--n_scales", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--train_steps", type=int, default=2000,
                    help="semigroup steps for liquid_v2_sg (M1 recipe)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_decay", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--anchor_hidden", type=int, default=64,
                    help="v0.1 anchor width (physics_rollout_eval default)")
    ap.add_argument("--anchor_train_steps", type=int, default=800)
    ap.add_argument("--anchor_lr", type=float, default=3e-3)
    ap.add_argument("--anchor_batch", type=int, default=256)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="base seed; seed+i per run")
    ap.add_argument("--device", default="cpu", choices=["cpu"],
                    help="protocol pinned to CPU (P5 same-device; anchors are "
                         "the v0.1 CPU-only protocol)")
    ap.add_argument("--out_dir", default="benchmarks/physics_out_v02")
    args = ap.parse_args()

    systems = ["orbit", "spring"] if args.system == "both" else [args.system]
    metric_keys = ["drift_final", "drift_max", "rollout_mse", "drift_traj_stderr"]
    results, verdict = {}, {}
    os.makedirs(args.out_dir, exist_ok=True)
    curves_path = os.path.join(args.out_dir, "energy_drift_p03_curves.jsonl")
    if os.path.exists(curves_path):   # a run owns its artifact: the writer
        os.remove(curves_path)        # appends, so drop rows from prior runs
    curves = JsonlMetricWriter(
        curves_path,
        static_fields={"benchmark": "energy_drift_eval", "device": args.device,
                       "git_sha": run_metadata()["git_sha"]})

    for system in systems:
        per_seed = {m: [] for m in ("liquid_v2_sg", "static_ham", "mlp_field")}
        for i in range(args.n_seeds):
            seed = args.seed + i
            print(f"[{system}] seed {seed}: training liquid_v2_sg (semigroup "
                  f"{args.train_steps} steps) + v0.1 anchors ...", flush=True)
            run = run_one(system, seed, args)
            for name, stats in run.items():
                per_seed[name].append(stats)
                for t, d in enumerate(stats["curve"].tolist()):
                    curves.write("drift", {"system": system, "model": name,
                                           "seed": seed, "step": t, "drift": d})
            print(f"[{system}] seed {seed}: " + " | ".join(
                f"{n} drift_final {s['drift_final']:.4e}" for n, s in run.items()),
                flush=True)
        results[system] = aggregate(per_seed, metric_keys)

        base = V01_RECORDED.get(system, {})
        liq = results[system]["liquid_v2_sg"]
        stat = results[system]["static_ham"]
        print(f"\n=== P0-3 DRIFT ({system}, {args.eval_k} steps, "
              f"{args.n_seeds} seeds, device {args.device}) ===", flush=True)
        for name in ("liquid_v2_sg", "static_ham", "mlp_field"):
            r = results[system][name]
            rec = f"  (v0.1 recorded {base[name]:g})" if name in base else ""
            print(f"  [{name:12s}] drift_final {r['drift_final']:.4e}"
                  f"±{r.get('drift_final_std', 0.0):.1e}{rec} | drift_max "
                  f"{r['drift_max']:.4e} | rollout_mse {r['rollout_mse']:.4e}",
                  flush=True)
        verdict[system] = {
            "liquid_le_static_remeasured":
                bool(liq["drift_final"] <= stat["drift_final"])}
        if base:  # the recorded baseline exists ONLY for orbit (PRD §6)
            verdict[system]["liquid_le_v01_recorded"] = bool(
                liq["drift_final"] <= base["static_ham"])
            print(f"  verdict: liquid_v2_sg ≤ {base['static_ham']:g} recorded "
                  f"baseline: {verdict[system]['liquid_le_v01_recorded']} | "
                  f"≤ re-measured static anchor: "
                  f"{verdict[system]['liquid_le_static_remeasured']}", flush=True)
        else:
            print(f"  verdict: no recorded v0.1 baseline for {system}; "
                  f"liquid ≤ re-measured static anchor: "
                  f"{verdict[system]['liquid_le_static_remeasured']} "
                  f"(reported, does not gate — see docstring)", flush=True)

    curves.close()

    # The P0-3 gate: the PRD pins the drift baseline to orbit only (§6).
    gate_systems = [s for s in systems if s in V01_RECORDED]
    p03 = {s: verdict[s]["liquid_le_v01_recorded"]
           and verdict[s]["liquid_le_static_remeasured"] for s in gate_systems}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "energy_drift_p03.json"), "w") as f:
        json.dump({"args": vars(args), "v01_recorded_baseline": V01_RECORDED,
                   "meta": run_metadata({"benchmark": "energy_drift_eval",
                                         "device": args.device}),
                   "results": results,
                   "verdict": {**verdict,
                               "p03_pass": bool(gate_systems) and all(p03.values()),
                               "p03_gate_systems": gate_systems}},
                  f, indent=2)

    passed = bool(gate_systems) and all(p03.values())
    print("\n" + "=" * 70, flush=True)
    print(f"P0-3 VERDICT (gate = {gate_systems}, the systems with a recorded "
          f"baseline): {p03} → "
          f"{'PASS — conservation does not regress' if passed else 'FAIL / NOT GATED — see drift table'}",
          flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
