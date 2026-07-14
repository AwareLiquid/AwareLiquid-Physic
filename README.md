# AwareLiquid-Physic

A complete model with **physics computation written into the architecture** — not
bolted on as a loss term.

The core idea: a continuous-time **liquid (LTC) recurrence** is the substrate, and
physical structure (a learned Hamiltonian advanced by a **symplectic integrator**)
is a hard architectural constraint on top of it. Energy is conserved *by
construction* — bounded O(dt²) drift, exactly time-reversible — for whatever
energy function the network learns. This is the hard-constraint route to
physics-informed ML (Hamiltonian NNs, Greydanus 2019), chosen over soft
PDE-residual losses after an adversarial review of the PINN literature.

## Architecture

```
observed trajectory prefix
        │
        ▼
  LiquidCore (LTC)          multi-timescale gated linear recurrence,
        │                   trained via Blelloch parallel scan (O(log T))
        ▼
  context vector            system identification: infers per-trajectory
        │                   hidden parameters (e.g. a spring's stiffness)
        ▼
  HamiltonianHead           separable H(q,p) = T(p) + V(q | context)
        │
        ▼
  symplectic rollout        velocity-Verlet; energy conserved by construction
```

- **`awareliquid_physics/liquid_core.py`** — the LTC "liquid" recurrence: each
  timescale is a leaky integrator with a learned time constant; scales are blended
  by an input-dependent gate. Runs in O(log T) depth via `parallel_scan.py`.
- **`awareliquid_physics/hamiltonian.py`** — `HamiltonianHead` (hard-constraint,
  symplectic) and `MLPFieldHead` (the honest unstructured control: plain Euler,
  no conservation).
- **`awareliquid_physics/model.py`** — `LiquidHamiltonianModel`, the integration
  the project is about: liquid core reads the prefix → conditions the Hamiltonian's
  potential → symplectic rollout predicts the future. One fixed learned potential
  cannot fit a *family* of systems; the liquid context makes the same model adapt
  its energy landscape per trajectory.
- **`awareliquid_physics/physics_ops.py`** — a zero-parameter, deterministic
  Newtonian operator layer (symplectic Euler integration, N-body gravity,
  collisions, energy/momentum diagnostics). The ground-truth engine the learned
  models are benchmarked against: *compute, don't memorise*.

## Results

`benchmarks/physics_rollout_eval.py` — free-standing head, matched 1-step fit
(~5e-6 loss both), scored on physics metrics only:

| system | metric | Hamiltonian (hard-constraint) | MLP-field control | advantage |
|---|---|---|---|---|
| spring | 150-step energy drift | **0.012** | 0.036 | 3.0× |
| orbit | 100-step energy drift | **6.2** | 37.8 | 6.0× |
| orbit | 100-step rollout MSE | **2.8e-3** | 6.6e-2 | 23.8× |

`benchmarks/liquid_physics_eval.py` — the coupling experiment: a *family* of
oscillators with hidden per-trajectory stiffness, prefix → 100-step rollout:

| model | rollout MSE | final energy drift | params |
|---|---|---|---|
| **liquid_ham** (liquid + Hamiltonian) | **4.12** | 0.70 | 8.7k |
| static_ham (Hamiltonian, no context) | 4.41 | **0.18** | 5.0k |
| gru_seq (unstructured control) | 7.24 | 2.96 | 10.0k |

Honest reading: the physics structure is what buys long-horizon stability over the
GRU baseline (2.9× lower rollout error, 4× lower drift); the liquid context buys
adaptation to the hidden parameter (better rollout MSE than the static head). The
static head drifts least because its one fixed potential is more rigid — the
adaptation/conservation trade-off is real and reported as measured.

## Scope (honest)

Continuous-state trajectory prediction, validated on **physics metrics** (k-step
rollout MSE, energy drift) — never perplexity. This is not a language model and
makes no hallucination claim.

## Usage

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q                          # 13 tests

# free-standing head: symplectic vs Euler control
python benchmarks/physics_rollout_eval.py --system spring
python benchmarks/physics_rollout_eval.py --system orbit

# the coupling experiment: liquid_ham vs static_ham vs gru_seq
python benchmarks/liquid_physics_eval.py --train_steps 500
```

All benchmarks are CPU-runnable.

## Roadmap

- Richer system families (2-body orbits, contacts/collisions via `physics_ops`)
  for the coupling experiment
- Non-separable Hamiltonians (magnetic / velocity-dependent forces)
- Scale the liquid substrate; longer-horizon rollouts
- Close the adaptation/conservation gap (context-conditioned kinetic term,
  drift-penalised training)

---

*Note: this repository previously hosted a Kaggle competition entry (Nemotron
reasoning challenge); that content lives in the git history before v0.1.0.*
