"""Unit tests for benchmarks/audit_results.py — the result-JSON schema auditor."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # benchmarks/ is not a package; import via namespace path
    sys.path.insert(0, str(ROOT))

from benchmarks.audit_results import audit_one, collect_files, numeric_leaves  # noqa: E402


def make_doc(**meta_extra):
    meta = {"benchmark": "unit_bench", "git_sha": "abc1234", "git_dirty": False,
            "device": "cpu", "ts": "2026-09-07T00:00:00+00:00"}
    meta.update(meta_extra)
    return {"args": {"seed": 3}, "meta": meta,
            "results": {"liquid": {"rollout_mse": 1.5, "params": 100},
                        "static": {"rollout_mse": 2.0}},
            "note": "scalar result kept as well"}


# --------------------------------------------------------------------------- #
# numeric_leaves
# --------------------------------------------------------------------------- #

def test_numeric_leaves_flattens_nested_dicts():
    leaves = numeric_leaves({"m": {"a": 1, "nested": {"b": 2.5}}, "top": 3})
    assert leaves == {"m.a": 1.0, "m.nested.b": 2.5, "top": 3.0}


def test_numeric_leaves_excludes_bools_strings_lists():
    leaves = numeric_leaves({"flag": True, "name": "liquid", "hist": [1, 2], "ok": 1})
    assert leaves == {"ok": 1.0}


def test_numeric_leaves_empty():
    assert numeric_leaves({}) == {}
    assert numeric_leaves("scalar") == {}


# --------------------------------------------------------------------------- #
# audit_one
# --------------------------------------------------------------------------- #

def test_audit_one_valid_doc(tmp_path):
    path = tmp_path / "good.json"
    path.write_text(json.dumps(make_doc()), encoding="utf-8")
    report = audit_one(path)
    assert report["issues"] == []
    assert report["benchmark"] == "unit_bench"
    assert report["git_sha"] == "abc1234"
    assert report["device"] == "cpu"
    assert report["seed"] == 3
    assert report["metrics"]["liquid.rollout_mse"] == 1.5
    assert report["metrics"]["static.rollout_mse"] == 2.0


def test_audit_one_missing_meta(tmp_path):
    path = tmp_path / "bad.json"
    doc = make_doc()
    del doc["meta"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    report = audit_one(path)
    assert any("missing top-level key: meta" in i for i in report["issues"])
    assert all(f"meta missing: {k}" in report["issues"] for k in ("git_sha", "device", "ts"))


def test_audit_one_meta_missing_fields(tmp_path):
    path = tmp_path / "bad2.json"
    path.write_text(json.dumps(make_doc(device=None)), encoding="utf-8")
    doc = json.loads(path.read_text())
    doc["meta"].pop("device")
    path.write_text(json.dumps(doc), encoding="utf-8")
    report = audit_one(path)
    assert report["issues"] == ["meta missing: device"]


def test_audit_one_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    report = audit_one(path)
    assert report["issues"] and report["issues"][0].startswith("invalid JSON")


def test_audit_one_non_object_top(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    report = audit_one(path)
    assert "top level is not a JSON object" in report["issues"]


# --------------------------------------------------------------------------- #
# collect_files
# --------------------------------------------------------------------------- #

def test_collect_files_recursive_and_missing_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "ignore.txt").write_text("x")
    files = collect_files([tmp_path])
    names = {f.name for f in files}
    assert names == {"a.json", "b.json"}
    assert collect_files([tmp_path / "nope"]) == []
