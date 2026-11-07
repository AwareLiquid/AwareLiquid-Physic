"""Integration test: run a real benchmark CLI end-to-end (tiny params) and
audit its result JSON through the provenance schema.

This is the executable form of the eval-type acceptance bar in
docs/DELIVERABLE_TYPES.md: a benchmark run is only shippable if its output
carries a valid `meta` block. time_eval is the cheapest single-model benchmark
(~2 s at these settings, forced to CPU for hermeticity).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # benchmarks/ is not a package
    sys.path.insert(0, str(ROOT))

from benchmarks.audit_results import audit_one  # noqa: E402


@pytest.mark.integration
def test_time_eval_end_to_end_produces_auditable_result(tmp_path):
    out_dir = tmp_path / "integration_out"
    cmd = [
        sys.executable, "benchmarks/time_eval.py",
        "--n_train", "16", "--n_eval", "8", "--gen_steps", "60",
        "--train_steps", "10", "--eval_k", "20", "--batch", "8",
        "--hidden", "8", "--device", "cpu",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"benchmark failed:\n{proc.stderr[-2000:]}"

    result_path = out_dir / "time_eval.json"
    assert result_path.exists()
    doc = json.loads(result_path.read_text())

    # Provenance schema (the whole point of the observability port)
    report = audit_one(result_path)
    assert report["issues"] == []
    assert report["git_sha"] not in (None, "unknown")
    assert report["device"] == "cpu"

    # Per-trajectory stderr sits next to the mean; the mean itself is a number
    metrics = report["metrics"]
    assert isinstance(metrics["rollout_mse"], float)
    assert isinstance(metrics["rollout_mse_stderr"], float)
    assert metrics["params"] > 0


@pytest.mark.integration
def test_energy_drift_eval_end_to_end_produces_auditable_result(tmp_path):
    """P0-3 closure artifact (round-8 gap #1): the drift comparison JSON must
    carry the provenance schema, per-seed values next to across-seed means,
    and the v0.1 recorded baseline it is judged against."""
    out_dir = tmp_path / "integration_out"
    cmd = [
        sys.executable, "benchmarks/energy_drift_eval.py",
        "--system", "spring", "--n_seeds", "2",
        "--n_train", "8", "--n_eval", "4", "--gen_steps", "40",
        "--t_obs", "8", "--k_train", "4", "--eval_k", "10",
        "--train_steps", "3", "--anchor_train_steps", "3",
        "--d_model", "8", "--hidden", "8", "--anchor_hidden", "8",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"benchmark failed:\n{proc.stderr[-2000:]}"

    result_path = out_dir / "energy_drift_p03.json"
    assert result_path.exists()
    report = audit_one(result_path)
    assert report["issues"] == []
    assert report["device"] == "cpu"

    metrics = report["metrics"]
    doc = json.loads(result_path.read_text())
    spring = doc["results"]["spring"]
    for model in ("liquid_v2_sg", "static_ham", "mlp_field"):
        # across-seed mean + spread + per-seed values, all as audit-visible scalars
        assert isinstance(metrics[f"spring.{model}.drift_final"], float)
        assert isinstance(metrics[f"spring.{model}.drift_final_std"], float)
        assert metrics[f"spring.{model}.drift_final_seed0"] >= 0.0
        assert metrics[f"spring.{model}.drift_final_seed1"] >= 0.0
    assert doc["v01_recorded_baseline"]["orbit"]["static_ham"] == 6.2
    assert isinstance(doc["verdict"]["p03_pass"], bool)

    # per-step drift curves stream to the JSONL sibling
    curves = out_dir / "energy_drift_p03_curves.jsonl"
    rows = [json.loads(line) for line in curves.read_text().splitlines()]
    assert {r["event"] for r in rows} == {"drift"}
    assert len([r for r in rows if r["step"] == 0]) == 3 * 2  # models × seeds


@pytest.mark.integration
def test_sample_efficiency_eval_end_to_end_produces_auditable_result(tmp_path):
    """P1-1 closure artifact (round-8 gap #2): the samples-vs-MSE curve JSON
    must carry provenance, per-seed values, and the 1/5-ratio verdict fields."""
    out_dir = tmp_path / "integration_out"
    cmd = [
        sys.executable, "benchmarks/sample_efficiency_eval.py",
        "--sizes", "8,16", "--n_seeds", "2",
        "--n_eval", "4", "--gen_steps", "40", "--t_obs", "8", "--k_train", "4",
        "--eval_k", "10", "--train_steps", "3",
        "--d_model", "8", "--hidden", "8",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"benchmark failed:\n{proc.stderr[-2000:]}"

    result_path = out_dir / "sample_efficiency_p11.json"
    assert result_path.exists()
    report = audit_one(result_path)
    assert report["issues"] == []
    assert report["device"] == "cpu"

    doc = json.loads(result_path.read_text())
    metrics = report["metrics"]
    for method in ("prefix", "all2all"):
        for n in (8, 16):
            key = f"{method}_n{n}.rollout_mse"
            assert isinstance(metrics[key], float)
            assert isinstance(metrics[f"{key}_std"], float)
            assert metrics[f"{key}_seed0"] >= 0.0
            assert metrics[f"{key}_seed1"] >= 0.0
    verdict = doc["verdict"]
    assert verdict["prefix_best_n"] in (8, 16)
    # null (not Infinity) when all2all never reaches the pinned line — strict JSON
    ratio = verdict["pinned_line_raw_ratio"]
    assert ratio is None or isinstance(ratio, float)
    assert isinstance(verdict["equal_budget_mse_ratio_at_max_n"], float)
    assert isinstance(verdict["p11_pass_ge_5x"], bool)
    assert isinstance(verdict["p11_assessment"], str) and verdict["p11_assessment"]

    # one JSONL row per (method, size, seed)
    curves = out_dir / "sample_efficiency_curves.jsonl"
    rows = [json.loads(line) for line in curves.read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "mse_point"]) == 2 * 2 * 2
