"""observability.py — 轻量结果/事件记录工具。

移植自 M1 仓库 ``mt_lnn/observability.py``（砍掉 MT-LNN 专用的 v2 模块诊断）。
两类用途：

1. ``run_metadata()``：给 benchmark 结果 JSON 补齐可溯源字段
   （git_sha / device / torch 版本 / 时间戳），使任何对外数字都能溯源到
   一次具体运行——对应 M1 的"测量优先于声明"制度。
2. ``JsonlMetricWriter``：逐事件 JSONL 写入器，用于 per-step 指标流
   （能量漂移、守恒误差随步数曲线等），可离线聚合或画图。

核心模型保持框架无关；本模块除 torch 外零第三方依赖。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


def setup_logging(level: Union[int, str] = "INFO") -> logging.Logger:
    """Configure a simple stderr logger and return the project logger."""
    logger = logging.getLogger("awareliquid_physics")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    return logger


def _json_default(value: Any) -> Any:
    if hasattr(value, "dim"):  # torch tensor: 0-dim → scalar, otherwise → list
        return value.item() if value.dim() == 0 else value.tolist()
    if hasattr(value, "item"):  # numpy / torch scalar
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class JsonlMetricWriter:
    """Append one JSON object per metric event.

    Parameters
    ----------
    path:
        Destination JSONL file. Parent directories are created automatically.
    static_fields:
        Fields included in every event, for example benchmark name or device.
    """

    def __init__(
        self,
        path: Union[str, Path],
        static_fields: Optional[Mapping[str, Any]] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.static_fields = dict(static_fields or {})
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        row: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self.static_fields,
        }
        if fields:
            row.update(dict(fields))
        self._fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlMetricWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def rollout_mse_stderr(qs_pred, q_true, ps_pred=None, p_true=None, batch_dim: int = 1) -> float:
    """Standard error (std/sqrt(n)) of the per-trajectory rollout MSE.

    Inputs are prediction/truth tensors broadcastable to the same shape, with
    the trajectory dimension at ``batch_dim`` (default 1 — the ``(k+1, B, ...)``
    convention shared by every benchmark here). The reported scalar
    ``rollout_mse`` (mean over all elements) is unchanged; this only quantifies
    its spread across eval trajectories.
    """
    sq = (qs_pred - q_true).pow(2)
    if ps_pred is not None:
        sq = sq + (ps_pred - p_true).pow(2)
    dims = tuple(d for d in range(sq.dim()) if d != batch_dim)
    per_traj = sq.mean(dim=dims)
    n = per_traj.numel()
    if n < 2:
        return 0.0
    return float(per_traj.std(correction=1)) / n ** 0.5


def git_sha(repo_dir: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Return ``{"git_sha", "git_dirty"}`` for the enclosing git repository.

    Falls back to ``{"git_sha": "unknown", "git_dirty": None}`` outside a repo
    or when git is unavailable — result files must stay writable anywhere.
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=cwd, timeout=10,
        ).stdout.strip() != ""
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": None}


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_metadata(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Provenance block for a benchmark result JSON.

    Returns a dict with ``ts, git_sha, git_dirty, device, torch, python``;
    keys from ``extra`` (e.g. benchmark name, hostname) are merged last so a
    caller can override any field. Note ``device`` defaults to the *detected*
    accelerator of the host — a benchmark running on a specific device should
    override it with the device actually used (``{"device": args.device}``),
    which is what the P5 "GPU/CPU 数据不可比" audit rule needs.
    """
    meta: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **git_sha(),
        "device": _detect_device(),
        "torch": "unknown",
        "python": ".".join(map(str, sys.version_info[:3])),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except Exception:
        pass
    if extra:
        meta.update(dict(extra))
    return meta
