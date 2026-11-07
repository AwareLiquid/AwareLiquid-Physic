# 交付物类型注册表（提交前自检清单）

> 每次提交必须属于且仅属于一种类型。类型决定交付物清单和自检标准。
> 本仓库无 CI / PR 流水线——本文件是**提交前人工自检清单**，不是工具用法。
> PR 描述使用 `.github/PULL_REQUEST_TEMPLATE.md`（含按类型的自检清单与"诚实边界"栏）。
> 形式参考 M1 仓库 `docs/DELIVERABLE_TYPES.md`，按单人研究仓库裁剪。

## 类型一览

| 类型 | 触发条件 | 必须包含 | 自检门槛 |
|------|---------|---------|---------|
| `feature` | 新功能/机制/模块 | 代码 + 测试 + 文档 | 测试通过 + README 或 docs 提及 |
| `perf` | 性能优化（不改语义） | 代码 + 等价性测试 + before/after 表 | 测试通过 + 墙钟/显存数据 |
| `eval` | 评估/基准/训练结果 | benchmark 脚本改动或结果 JSON + PRD/PRINCIPLES 更新 | 结果 JSON 过 `audit --check` + 结果表更新 |
| `infra` | 工具/流程/环境 | 脚本 + 使用文档 + ≥1 个示例 | `--help` 可退出 + 文档含用途/用法/局限 |
| `docs` | 文档 | 文档 + 交叉引用 | 无断链 + 代码引用准确 |

## 各类型验收标准

### feature

- [ ] `pytest tests/ -q` 全绿（含新增测试；守恒类改动必须有守恒测试）
- [ ] 新模块在 `docs/architecture.md` 有接口描述；影响原则状态时同步更新 `docs/PRINCIPLES.md`
- [ ] 实验性模块（无 kill criterion）不进 `awareliquid_physics/` 主包

### perf

- [ ] 等价性测试：优化前后输出一致（容差内）
- [ ] before/after 表：墙钟时间、峰值内存（可选 FLOPs）
- [ ] 默认行为不变（开关默认 OFF 或严格等价）

### eval

- [ ] 结果 JSON 必含 `meta` 溯源块（`ts` / `git_sha` / `git_dirty` / `device` / `torch` / `python` / `benchmark`），由 `awareliquid_physics/observability.py::run_metadata` 生成
- [ ] 对多条轨迹/样本聚合的均值指标旁附 `_stderr`
- [ ] 对比实验只在同 device 内做（P5：GPU/CPU 数据不可比）；单 seed 结论措辞必须带 seed 号（P6）
- [ ] `python benchmarks/audit_results.py benchmarks --check` 退出码 0
- [ ] `docs/PRD.md` 结果表和/或 `docs/PRINCIPLES.md` 原则状态已更新——**原则状态变化写进 commit message**

### infra

- [ ] `python scripts/xxx.py --help`（或等价入口）正常退出
- [ ] 文档包含：用途、用法、示例、局限性（诚实边界）

### docs

- [ ] 交叉引用无断链；引用的函数名/参数/路径与代码一致
- [ ] 数字改动可溯源到结果 JSON + PRD 表

## commit message 约定

`<type>: <一句话>`，中文或英文均可（仓库现状两者混用）。原则状态变化是一等公民：

```bash
docs: P2 升为 SUPPORTED - 第六波 7 配置 32.6-85.6%
feat: LiquidNBodyModel 对势 N-body（P2-1）
eval: 第七波收敛极限 - 非可分 1.66e-5
```
