## {标题}

**类型**（必选其一，自检清单见 `docs/DELIVERABLE_TYPES.md`）: feature / perf / eval / infra / docs

**原则状态**：本 PR 是否改变 `docs/PRINCIPLES.md` 的原则状态？
不改变 / 改变（注明，如 `P2: UNCERTAIN → SUPPORTED，证据：第六波 7 配置`）——原则状态变化是 commit message 的一等公民。

### 变更摘要

<!-- 一段话说清改了什么、为什么 -->

### 自检清单

- [ ] `pytest tests/ -q` 全绿
- [ ] **eval 类**：结果 JSON 含 `meta` 溯源块，`python benchmarks/audit_results.py --check` 退出码 0
- [ ] **eval 类**：`docs/PRD.md` / `docs/PRINCIPLES.md` 的结果表与原则状态已更新
- [ ] **perf 类**：附 before/after 数据（墙钟时间、峰值内存）+ 等价性测试
- [ ] **feature 类**：新模块有对应 `tests/test_*.py`，`docs/architecture.md` 提及
- [ ] **infra 类**：`--help` 正常退出，文档含用途/用法/局限
- [ ] **docs 类**：交叉引用无断链，数字可溯源到结果 JSON + PRD 表

### 诚实边界（eval / feature 必填）

<!-- 这组结果/能力覆盖什么、不覆盖什么。例：单 seed、仅 CPU 数据、仅某系统族、smoke 规模。 -->
