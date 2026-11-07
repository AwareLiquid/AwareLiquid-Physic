# 测试可追溯性 — v0.1 的 13 个测试 → 当前代码

> P0-4 验收要求「v0.1 全部 13 个测试保持通过」。round-8 gap review 指出
> 仅有测试总数（63→68）不够，需要**逐一映射**可追溯。本文件是权威映射表。
>
> v0.1 基线 = 初始提交 `a838d6c`（"feat: AwareLiquid-Physic v0.1.0"）中的
> 3 个测试文件、13 个测试函数。**全部 13 个在当前代码中同名存活、同文件
> 存活、持续通过**（命名从未改动，行为语义随 ADR 演进处已在表中注明）。

## 映射表

### `tests/test_hamiltonian.py`（5 个）

| # | v0.1 @ a838d6c | 当前 | 备注 |
|---|---|---|---|
| 1 | `test_symplectic_conserves_energy_better_than_forward_euler` | 同名存活 | 辛积分 vs 前向欧拉同场对照；PRINCIPLES P1 的测试锚 |
| 2 | `test_integrator_is_exactly_time_reversible` | 同名存活 | velocity-Verlet 精确时间可逆 |
| 3 | `test_gradients_reach_energy_params_and_shapes` | 同名存活 | 梯度达 T 与 V |
| 4 | `test_context_conditions_the_potential` | 同名存活 | context 真实重塑势能 |
| 5 | `test_mlp_field_baseline_runs_and_differs` | 同名存活 | 无结构对照可跑且有差异 |

### `tests/test_liquid_core.py`（4 个）

| # | v0.1 @ a838d6c | 当前 | 备注 |
|---|---|---|---|
| 6 | `test_pscan_matches_sequential_reference` | 同名存活 | M1 升级为通用 pscan（逐时间步 A，ADR 风险表项）后语义等价保持 |
| 7 | `test_liquid_core_shapes_and_finite` | 同名存活 | — |
| 8 | `test_liquid_core_is_causal` | 同名存活 | — |
| 9 | `test_liquid_core_gradient_flows` | 同名存活 | — |

### `tests/test_model.py`（4 个）

| # | v0.1 @ a838d6c | 当前 | 备注 |
|---|---|---|---|
| 10 | `test_forward_shapes` | 同名存活 | — |
| 11 | `test_prefix_changes_context_and_rollout` | 同名存活 | — |
| 12 | `test_gradient_flows_through_core_and_ham` | 同名存活 | — |
| 13 | `test_gru_baseline_interface_matches` | 同名存活 | — |

## 验证方式

```bash
# 逐个点名跑（13/13 通过为 P0-4 的验收口径）
pytest tests/test_hamiltonian.py::test_symplectic_conserves_energy_better_than_forward_euler \
       tests/test_hamiltonian.py::test_integrator_is_exactly_time_reversible \
       tests/test_hamiltonian.py::test_gradients_reach_energy_params_and_shapes \
       tests/test_hamiltonian.py::test_context_conditions_the_potential \
       tests/test_hamiltonian.py::test_mlp_field_baseline_runs_and_differs \
       tests/test_liquid_core.py::test_pscan_matches_sequential_reference \
       tests/test_liquid_core.py::test_liquid_core_shapes_and_finite \
       tests/test_liquid_core.py::test_liquid_core_is_causal \
       tests/test_liquid_core.py::test_liquid_core_gradient_flows \
       tests/test_model.py::test_forward_shapes \
       tests/test_model.py::test_prefix_changes_context_and_rollout \
       tests/test_model.py::test_gradient_flows_through_core_and_ham \
       tests/test_model.py::test_gru_baseline_interface_matches -q
```

## 新增测试的分层（P0-4 的「新增 ≥20」侧）

v0.1 之后新增的测试按来源分层：v0.2 主体（`test_operator_potential.py`、
`test_datasets_train.py`）、P2 组件（`test_pairwise_potential.py`、
`test_nonseparable.py`、`test_p2_34.py`、`test_operator_2d.py`）、验收补充
（`test_audit_results.py`、`test_benchmark_integration.py`）、**P0-2 验收
钉死**（`test_p02_acceptance.py`——trained@N→零样本@M 的场/粒子可变节点数
与 dim=1/2 兼容性，round-8 缺口 #3）。
