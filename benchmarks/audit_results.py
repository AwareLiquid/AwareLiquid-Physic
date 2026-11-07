"""benchmarks/audit_results.py — 审计 benchmark 结果 JSON 的可溯源 schema。

M1 仓库 bench_trace_audit.py（trace 审计）精神的移植：读 `*.json` 结果文件
（各 benchmark 的落盘格式：``{"args": ..., "meta": ..., "results": {...}}``），
校验溯源块并汇总指标。结果结构各脚本不同，这里用通用方式遍历数值叶子节点，
不写死任何指标名。

校验项（--check 模式下任一缺失 → exit 1，供提交前自检）：
  - 顶层含 ``args``、``results``、``meta``
  - ``meta`` 含 ``git_sha`` / ``device`` / ``ts``（由
    ``awareliquid_physics.observability.run_metadata`` 生成）

用法::

    python benchmarks/audit_results.py                          # 扫描默认目录
    python benchmarks/audit_results.py benchmarks/physics_out_v02
    python benchmarks/audit_results.py benchmarks --format json
    python benchmarks/audit_results.py benchmarks --check       # CI/提交前自检

局限：只做 schema 与数值汇总，不判断指标数值是否合理。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_TOP = ("args", "results", "meta")
REQUIRED_META = ("git_sha", "device", "ts")


def default_roots() -> List[Path]:
    """Result dirs live in benchmarks/*out*/ by convention; also allow stray
    top-level JSONs under benchmarks/."""
    root = Path(__file__).resolve().parent
    roots = [d for d in root.iterdir() if d.is_dir() and "out" in d.name]
    return roots if roots else [root]


def collect_files(paths: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        elif p.exists():
            files.append(p)
    return files


def numeric_leaves(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten numeric leaves: ``{model: {metric: x}}`` → ``{"model.metric": x}``.
    Numbers only (bool excluded); lists/dicts of other shapes are skipped."""
    out: Dict[str, float] = {}
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return {prefix: float(obj)}
    if isinstance(obj, dict):
        for key, val in obj.items():
            out.update(numeric_leaves(val, f"{prefix}.{key}" if prefix else str(key)))
    return out


def audit_one(path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": str(path), "issues": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        report["issues"].append(f"invalid JSON: {exc}")
        return report
    if not isinstance(doc, dict):
        report["issues"].append("top level is not a JSON object")
        return report

    for key in REQUIRED_TOP:
        if key not in doc:
            report["issues"].append(f"missing top-level key: {key}")
    meta = doc.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        report["issues"].append("meta is not an object")
    for key in REQUIRED_META:
        if key not in meta:
            report["issues"].append(f"meta missing: {key}")

    report["benchmark"] = meta.get("benchmark", path.stem)
    report["git_sha"] = meta.get("git_sha")
    report["git_dirty"] = meta.get("git_dirty")
    report["device"] = meta.get("device")
    report["ts"] = meta.get("ts")
    args = doc.get("args", {})
    if isinstance(args, dict):
        report["seed"] = args.get("seed")
    report["metrics"] = numeric_leaves(doc.get("results", {}))
    return report


def format_human(reports: List[Dict[str, Any]]) -> str:
    lines = []
    for r in reports:
        lines.append(f"\n=== {Path(r['path']).name} (benchmark: {r.get('benchmark')}) ===")
        lines.append(f"  git: {r.get('git_sha')} (dirty={r.get('git_dirty')})  "
                     f"device: {r.get('device')}  seed: {r.get('seed')}")
        lines.append(f"  ts : {r.get('ts')}")
        for metric, value in sorted((r.get("metrics") or {}).items()):
            lines.append(f"  {metric:<44s} {value:.6g}")
        for issue in r["issues"]:
            lines.append(f"  ISSUE: {issue}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="result JSON files or directories "
                                            "(default: benchmarks/*out*/)")
    ap.add_argument("--format", choices=["human", "json"], default="human")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any file fails schema validation")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths] if args.paths else default_roots()
    files = collect_files(roots)
    if not files:
        print("no result JSON files found", file=sys.stderr)
        return 1 if args.check else 0

    reports = [audit_one(p) for p in files]
    if args.format == "json":
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        print(format_human(reports))

    if args.check:
        bad = [r for r in reports if r["issues"]]
        print(f"\naudit: {len(reports) - len(bad)} passed, {len(bad)} with issues",
              file=sys.stderr)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
