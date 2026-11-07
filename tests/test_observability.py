"""Tests for awareliquid_physics.observability — provenance metadata and the
JSONL event writer (ported from the M1 repo's observability tooling)."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import torch

from awareliquid_physics.observability import (
    JsonlMetricWriter,
    git_sha,
    rollout_mse_stderr,
    run_metadata,
)


# --------------------------------------------------------------------------- #
# JsonlMetricWriter
# --------------------------------------------------------------------------- #

def test_writer_creates_parent_dirs_and_valid_json(tmp_path):
    path = tmp_path / "nested" / "dir" / "events.jsonl"
    with JsonlMetricWriter(path, static_fields={"bench": "unit"}) as w:
        w.write("token", {"step": 1, "mse": 1.5e-3})
        w.write("summary", {"n": 3})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["event"] == "token" and rows[0]["step"] == 1
    assert rows[0]["bench"] == "unit"          # static field in every event
    assert rows[1]["event"] == "summary" and rows[1]["n"] == 3
    for row in rows:                            # ts parses as ISO-8601
        datetime.fromisoformat(row["ts"])


def test_writer_appends_not_overwrites(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlMetricWriter(path) as w:
        w.write("a", {"x": 1})
    with JsonlMetricWriter(path) as w:
        w.write("b", {"x": 2})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["a", "b"]


def test_writer_closes_on_context_exit(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlMetricWriter(path) as w:
        w.write("a")
    assert w._fh.closed


def test_writer_serialises_torch_scalars(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlMetricWriter(path) as w:
        w.write("token", {"mse": torch.tensor(3.14), "ids": torch.tensor([1, 2])})
    row = json.loads(path.read_text().splitlines()[0])
    assert row["mse"] == pytest.approx(3.14, abs=1e-6)
    assert row["ids"] == [1, 2]


# --------------------------------------------------------------------------- #
# run_metadata / git_sha
# --------------------------------------------------------------------------- #

def test_run_metadata_keys_and_git_sha_matches_git():
    meta = run_metadata()
    assert {"ts", "git_sha", "git_dirty", "device", "torch", "python"} <= set(meta)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert meta["git_sha"] == sha
    assert isinstance(meta["git_dirty"], bool)
    assert meta["device"] in {"cpu", "cuda", "mps"}
    datetime.fromisoformat(meta["ts"])


def test_run_metadata_extra_overrides():
    meta = run_metadata({"device": "test", "benchmark": "unit"})
    assert meta["device"] == "test"
    assert meta["benchmark"] == "unit"


def test_git_sha_falls_back_outside_repo(tmp_path):
    meta = git_sha(tmp_path)                    # valid dir, but not a git repo
    assert meta["git_sha"] == "unknown"
    assert meta["git_dirty"] is None


# --------------------------------------------------------------------------- #
# rollout_mse_stderr
# --------------------------------------------------------------------------- #

def test_rollout_mse_stderr_zero_for_identical_trajectories():
    q = torch.randn(11, 8, 2)                   # (k+1, B, dim)
    assert rollout_mse_stderr(q, q) == 0.0


def test_rollout_mse_stderr_matches_manual_formula():
    torch.manual_seed(0)
    q_pred, q_true = torch.randn(11, 16, 2), torch.randn(11, 16, 2)
    p_pred, p_true = torch.randn(11, 16, 2), torch.randn(11, 16, 2)
    se = rollout_mse_stderr(q_pred, q_true, p_pred, p_true)
    per_traj = ((q_pred - q_true) ** 2 + (p_pred - p_true) ** 2).mean(dim=(0, 2))
    expected = float(per_traj.std(correction=1)) / per_traj.numel() ** 0.5
    assert se == expected


def test_rollout_mse_stderr_single_trajectory_is_zero():
    q_pred, q_true = torch.randn(5, 1, 2), torch.randn(5, 1, 2)
    assert rollout_mse_stderr(q_pred, q_true) == 0.0
