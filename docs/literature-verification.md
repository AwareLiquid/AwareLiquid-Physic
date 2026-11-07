# 文献核实 · v0.2 PRD 引用逐条验证（夜间学术尽调批次）

- **日期**: 2026-09-07 · **性质**: 只读核实（针对 PRD 引用的 6 篇文献）
- **结论**: **6/6 引用真实存在且内容与 PRD 用法相符**；1 处建议补全出版信息

| PRD 引用 | 核实结果 | 与 PRD 用法的一致性 |
|---|---|---|
| **LTC**（Hasani et al., AAAI 2021） | ✅ "Liquid Time-constant Networks", AAAI 35(9):7657-7666, arXiv:2006.04439（~726 引用）——LTC 是显式 ODE 耦合的时连续 RNN，"liquid"时间常数、通用逼近、稳定动力系统 | ✅ B1 修复路径（输入依赖时间常数）与论文主题一致 |
| **CfC**（Nat. Mach. Intell. 2022） | ✅ "Closed-form continuous-time neural networks", NMI 4(11):992-1003, arXiv:2106.13898——LTC ODE 的闭式近似，免除数值求解器 | ✅ "CfC 闭式解"表述准确；代码开源 raminmh/CfC |
| **FNO**（Li et al., ICLR 2021） | ✅ Fourier Neural Operator（ICLR 2021，Zongyi Li et al.）——傅里叶核积分、分辨率不变 | ✅ B2 修复路径表述准确（本次搜索未直接返回，属常识级确认，建议 PRD 补 arXiv:2010.08895 引用） |
| **Poseidon**（arXiv:2405.19101） | ✅ NeurIPS 2024，ETH Zürich CAMLab——multiscale operator transformer + **time-conditioned LayerNorm** + 多阶段高效预训练（scOT 语料），T/B/L 三档预训练模型 | ✅ 与 PRD P2-4"对标 Poseidon 的 time-conditioned LayerNorm"细节吻合 |
| **Aurora**（Nature 2025） | ✅ "A foundation model for the Earth system", Nature s41586-025-09005-y（Microsoft，100 万小时数据，气象/空气质量/海浪/台风多任务） | ✅ "Aurora Nature 2025"准确；arXiv:2405.13063 |
| **GenCast**（DeepMind, Nature） | ✅ "Probabilistic weather forecasting with machine learning", Nature s41586-024-08252-9（Price et al.）——扩散式 ensemble、15 天 0.25°、50+ 轨迹、胜过 ECMWF ENS，已上线 WeatherNext | ✅ P2-3"扩散式 ensemble，对标 GenCast 路线"表述准确 |

## 建议（1 条）

- PRD 的 FNO 引用补全为 `arXiv:2010.08895`（Li et al., "Fourier Neural Operator
  for Parametric Partial Differential Equations", ICLR 2021），其余引用
  建议在 PRD 参考文献节补全卷期/链接（上表已给全）。

## 方法

Web 检索逐条验证（出版 venue/卷期/内容要点），核实日期 2026-09-07。
全部主张为文献存在性与内容描述，非本仓性能主张（RESULTS 纪律不适用）。
