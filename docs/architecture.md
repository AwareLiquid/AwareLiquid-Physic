# AwareLiquid-Physic v0.2 — 技术架构文档

> 配套文档：`docs/PRD.md`（需求与验收标准）。本文档定义架构、模块接口、训练管线与关键决策（ADR）。

---

## 1. 架构总览

```
                        ┌─────────────────────────────────────────────┐
                        │   LiquidOperatorHamiltonianModel            │
                        │                                             │
  trajectory prefix     │   ┌──────────────┐      ┌───────────────┐   │
  (q,p)_{0..T_obs} ─────┼──▶│ LiquidCore v2 │─ctx─▶│  FiLM 调制    │───┼──┐
  (B, T_obs, 2*dim)     │   │  真液体时间常数 │      │ (ctx→scale/bias)│  │
                        │   │  pscan 并行扫描│      └───────┬───────┘   │  │
                        │   └──────────────┘              │           │  │
                        │                                  ▼           │  │
                        │   ┌───────────────────────────────────────┐ │  │
                        │   │  OperatorHamiltonianHead               │ │  │
                        │   │  H(q,p|ctx) = T(p) + V_operator(q|ctx) │ │  │
                        │   │    T: 动能 MLP（不变）                  │ │  │
                        │   │    V: lift → FNO 谱层 ×L → 逐点能量和   │ │  │
                        │   └───────────────────┬───────────────────┘ │  │
                        │                       │                     │  │
  future trajectory     │   ┌───────────────────▼───────────────────┐ │  │
  (qs, ps)_{0..k} ◀─────┼───│  velocity-Verlet 辛积分 rollout         │ │  │
  + 能量诊断            │   │  (kick-drift-kick, 守恒 by 构造)        │ │◀─┘
                        │   └───────────────────────────────────────┘ │
                        └─────────────────────────────────────────────┘
```

**一句话**：liquid 核心读前缀做系统辨识 → context 通过 FiLM 调制一个**傅里叶算子参数化的势能场** → 辛积分在"被算子学习的能量景观"里 rollout。神经算子解决**能量景观的空间结构**，辛积分解决**时间演化的守恒**——两条路线正交叠加。

## 2. 设计原则

1. **硬约束优先**：守恒进架构（辛积分），不进损失；物理进架构的路线在 v0.1 已验证，v0.2 不回头。
2. **真液体**：时间常数必须输入依赖（液体性质的定义），不再用静态 tau 冒充 LTC。
3. **函数空间参数化**：能量场 V 在函数空间参数化 → 分辨率不变性（P1-3）成为架构性质而非技巧。
4. **可复现的诚实基准**：CPU 可跑、物理指标（rollout MSE / 能量漂移 / 分辨率误差）、所有对照实验脚本化。
5. **增量兼容**：v0.1 的低维系统 benchmark 必须仍能复跑（回归基线），新能力以新模块/新 benchmark 增量加入。

## 3. 模块设计

### 3.1 LiquidCore v2 —— 真液体时间常数

**问题（B1）**：v0.1 的 `tau = softplus(log_tau) + tau_min` 是静态参数；每尺度是纯线性泄漏积分器。没有 LTC 的定义性质：时间常数随输入/状态**液态**变化。

**v2 设计**（保持 pscan 训练路线）：

```
tau_{s,t} = tau_min + softplus(log_tau_s + gate_tau(u_t))      # 输入依赖调制
decay_{s,t} = exp(-dt / tau_{s,t})                              # 逐时间步 (B,S,D,t)
h_{s,t} = decay_{s,t} * h_{s,t-1} + (1 - decay_{s,t}) * u_t     # 泄漏积分（A 随时间变）
h_t = Σ_s w_s(x_t) * h_{s,t}                                    # 输入门控混合（保持 v0.1 结构）
```

- `gate_tau`: 小 MLP，`d_model → n_scales`，输出加在 log_tau 上（tau 下界 tau_min 保持稳定性保证）
- **扫描兼容性**：A 随时间变化 → 从 `pscan_constant_A` 切换到通用 `pscan(A, X)`（仓库 `parallel_scan.py` 已有通用实现，且与 `pscan_sequential` 有对齐测试）
- **备选路线（M1 消融）**：CfC 闭式单元（Hasani NatMI 2022，有表达力与稳定性定理）。决策依据：若输入依赖 tau 的消融显示 context 质量提升不足（liquid vs static 差距 < 30%），M1.5 切换到 CfC
- **不变**：`encode()` 取最后隐状态作为 context；`out_proj` 保留

### 3.2 OperatorPotential —— FNO 能量场（新模块，M2）

**问题（B2）**：v0.1 的 V(q|ctx) 是逐点 MLP：无平移不变性、无分辨率不变性、`phase_dim` 固定。

**v2 设计**：把势能写成**谱算子**，输入是"节点集合"而非固定维度向量：

```
q ∈ (B, N, dim)          # N = 节点数（可变！粒子数/网格点数）
ctx ∈ (B, d_ctx)         # liquid 核心输出的系统辨识向量

V(q|ctx) = Σ_n v(x_n)    # 能量 = 逐点势能之和（扩展性 → N 可伸缩）

v(x_n) 的通道流程：
  lift:    (dim + d_ctx_film) → channels          # 局部逐点提升
  FNO×L:   FFT → 谱域线性 R·(F v) → IFFT + 局部线性 W + 激活
           （谱域线性捕获长程交互 = 全局感受野）
  FiLM:    ctx → (scale, bias) 调制每层通道        # context 条件化
  project: channels → 1                            # 逐点势能
```

**关键性质**：
- **分辨率不变**：谱层参数在频域（截断模态数固定），与 N 无关 → 训练 64 网格、推理 256 网格（P1-3）
- **平移不变**：卷积形式核 → 物理对称性进架构
- **可微性**：全链路（lift/FFT/线性/激活）光滑可微 → velocity-Verlet 所需的 `dV/dq` autograd 梯度成立（用 Tanh/GeLU，不用 ReLU，保持 C²）
- **低维兼容**：dim=1/2 弹簧/轨道系统中 N=1 → 谱层退化为逐点线性，行为与 v0.1 MLP 兼容（P0-2）
- **周期性处理**：真实物理域常非周期 → 标准解法：对 q 沿空间维做反射 padding 后再 FFT（或 grouped FNO），M2 定案
- **N-body 扩展**：节点即粒子，谱层的全局交互天然提供 O(1) 深度的对相互作用通道（P2-1 的底座）

### 3.3 HamiltonianHead v2

- **T(p)**：保持 v0.1 的 MLP（动能项简单、可分离假设不变）
- **V(q|ctx)**：替换为 `OperatorPotential`（3.2）
- **step/rollout**：velocity-Verlet kick-drift-kick **完全不变**（守恒核心不动）
- **非可分扩展点**（P2-2，预留接口）：`H(q,p) = T(p) + V(q) + C(q,p)`，C 为可选的 q-p 耦合项（速度依赖力/磁项），rollout 需换广义 leapfrog——**不在 v0.2 范围**

### 3.4 条件化机制（M2 消融定案）

| 方案 | 表达力 | 分辨率不变兼容 | 成本 |
|---|---|---|---|
| concat（v0.1 现状） | 弱 | ✅（沿通道维拼接） | 最低 |
| **FiLM 调制**（本设计默认） | 中-强（每层每通道 scale/bias） | ✅（调制作用在通道维） | 低 |
| hypernetwork（ctx → 谱层权重） | 最强 | ✅（权重在频域维） | 高（参数量/稳定性风险） |

默认 FiLM；M2 消融实验三选一定案并写入 ADR 修订。

### 3.5 模型装配

```python
class LiquidOperatorHamiltonianModel(nn.Module):
    def __init__(self, phase_dim, d_model=64, context_dim=16,
                 n_scales=4, fno_modes=12, fno_width=32, fno_depth=4,
                 dt=0.1, core_dt=1.0): ...
    def infer_context(q_obs, p_obs) -> ctx        # LiquidCore v2.encode
    def rollout(q0, p0, ctx, steps) -> (qs, ps)   # OperatorHamiltonianHead, 辛积分
    def forward(q_obs, p_obs, k) -> (qs, ps, ctx) # 与 v0.1 接口兼容
```

- 接口与 v0.1 `LiquidHamiltonianModel` 兼容 → benchmark 脚本可复用
- `phase_dim` 仅约束 q/p 的每节点自由度；节点数 N 由输入张量形状自由给定（分辨率不变）

## 4. 数据管线（physics_ops 为引擎）

```
physics_ops.py（零参数确定性引擎，不变）
   ├─ gen_spring_family     # 弹簧族：隐藏刚度 ω ∈ [ω_lo, ω_hi]（M1 沿用）
   ├─ gen_orbit             # 轨道族（M1 沿用）
   ├─ gen_field_2d          # 新增（M2）：2D 场系统（Burgers 类/热扩散/波动），
   │                        #   用 integrate_verlet 或谱方法生成真值
   ├─ gen_nbody             # 新增（P2-1）：pairwise_gravity + resolve_sphere_collisions
   └─ 能量/动量诊断           # kinetic_energy / momentum / 守恒量真值
```

- **预训练语料（M3）**：physics_ops 生成多系统族轨迹 → 统一张量格式 `(B, T, N, 2*dim)` 存储
- 数据即代码（生成函数脚本化）→ 可复现、无需外部数据集

## 5. 训练管线

### 5.1 半群 all2all 训练（M3，修复 B3）

时间演化半群性质：`Φ(t_j) = Φ(t_j - t_i) ∘ Φ(t_i)`，因此轨迹上**任意时间对**都是合法训练样本：

```
采样：(t_i, t_j) ~ Uniform(轨迹内所有时间对)
loss = MSE( rollout(q_{t_i}, p_{t_i}, ctx, steps=j-i), (q,p)_{t_i..t_j} )
```

- 每条轨迹从 1 个样本 → O(T²) 个样本（P1-1：样本效率 5×+）
- ctx 从相同前缀推断（t_i ≤ T_obs 时）；t_i > T_obs 时 ctx 从 [0, T_obs] 前缀推断后**跨段复用**（系统不变假设）
- 梯度经辛积分反传到能量场参数与 liquid 核心（`create_graph=True` 路径保持 v0.1 机制）

### 5.2 漂移惩罚（可选正则，M3）

```
L = L_mse + λ * mean( (H_t - H_0)² )    # 弱正则，λ 小
```

- 架构已保守恒（O(dt²) 有界），此项仅用于平滑能量景观、缓解"学到守恒但错误的能量"；λ 默认 0，消融决定是否启用

### 5.3 预训练-微调（M3，对标 Aurora/Poseidon 范式）

1. **预训练**：多系统族数据（弹簧族 + 轨道族 + 2D 场族）联合训练 liquid+ham
2. **微调**：few-shot（≤ 20 条轨迹）适配新系统（如新刚度分布/新场参数）
3. **评估**：跨系统泛化 benchmark，指标对标 Poseidon（20 样本 ≈ FNO 1024 样本）

## 6. ADR（架构决策记录）

| ID | 决策 | 理由 | 替代方案（为何否决） |
|---|---|---|---|
| ADR-1 | 守恒走架构不走损失（保持 v0.1） | v0.1 已实证 3-6× 漂移优势；PINN 软约束文献综述显示软惩罚需调参且不保证 | PINN 残差损失：训练难、无保证（v0.1 README 已记录此决定） |
| ADR-2 | LTC 升级为输入依赖 tau，而非换 Transformer 编码器 | 保持液体基底连续性；pscan 路线兼容；原版 LTC 表达力理论（AAAI 2021）有支撑 | Transformer 前缀编码：与仓库定位冲突，且失去 O(log T) 扫描优势 |
| ADR-3 | 能量场用 FNO 谱算子而非图神经网络 | FNO 提供分辨率不变 + 平移不变 + 全局感受野（ICLR 2021）；与 N 无关的参数量 | GNN（GraphCast 路线）：分辨率绑定图结构，超分需重构图；参数量随 N 增长 |
| ADR-4 | 条件化默认 FiLM 调制 | 通道维调制保持分辨率不变性；表达力强于 concat；成本低于 hypernetwork | concat：v0.1 实证表达力不足（5% 差距）；hypernetwork：M2 消融后备。**实测修订（M1 消融）**：短窗口训练（k_train=8）+ 小系统下 FiLM 比 concat 更易过拟合 context（train_loss 更低但 100 步 rollout 发散 9-18x）——FiLM 的每层调制使势能对 context 过度敏感。结论：FiLM 保留用于 M2（半群训练 + 场任务的正则效应下稳定）；M1 保留 concat；FiLM 需配长窗口/半群训练使用 |
| ADR-5 | 半群 all2all 训练 | Poseidon 已证数据效率收益；对辛积分 rollout 天然适配（任意步数合法） | 仅前缀→未来：每条轨迹 1 样本，数据效率瓶颈 |
| ADR-6 | 保持 velocity-Verlet（不换高阶/隐式积分器） | 2 阶辛格式守恒保证充分；v0.1 基准可复现 | RK4/隐式：非辛 → 长程漂移；高阶级 symplectic：复杂度收益比低 |
| ADR-7 | FiLM 层序：pre-activation（Tanh 前） | 实测：post-activation FiLM 无界 → 力场爆炸 → rollout 发散（mse 51.7）；Tanh 前调制保证每层激活有界 | post-activation：M1-FiLM 首版发散的直接原因 |
| ADR-8 | 非可分 H 用隐式中点（P2-2） | velocity-Verlet 只对 T(p)+V(q) 可分裂 H 辛；非可分 H(q,p)（磁/科里奥利）需隐式中点——它对任意光滑 H 辛，守恒仍是架构性质（随机权重漂移 1.6e-6） | velocity-Verlet：非辛 → 漂移；广义 leapfrog：需已知分裂结构 |
| ADR-9 | 时间条件化（P2-4）不保证守恒 | 显式时间依赖 H(q,p,t) 时 dH/dt=∂H/∂t≠0，能量不守恒是物理正确（受迫系统外力做功）；守恒保证在时间无关极限（V 忽略 t）恢复 | 强行守恒：会错误建模受迫系统 |

## 7. 风险与验证计划

| 风险 | 等级 | 缓解与验证 |
|---|---|---|
| 谱层 autograd 梯度成本（每 Verlet 步 2 次 dV/dq） | 中 | M2 第一步做梯度成本 micro-benchmark（vs v0.1 MLP 场）；超标则谱层用 torch.fft 的解析梯度路径 + 混合精度 |
| FNO 周期性假设 vs 非周期物理域 | 中 | 反射 padding / grouped FNO；M2 消融二选一定案 |
| 输入依赖 tau 的扫描稳定性（A 时间变 → pscan 数值行为） | 低 | 与 `pscan_sequential` 的对齐测试（容差 1e-5）保留为回归测试 |
| FiLM 调制破坏能量标量性/光滑性 | 低 | 能量值诊断（H 的 Lipschitz 经验估计）；守恒测试（同一初始状态不同 rollout 长度能量漂移有界） |
| 半群训练中 ctx 跨段复用引入偏差 | 中 | M3 消融：ctx 仅从前缀段推断 vs 全轨迹推断，对比 rollout MSE |

**验证矩阵**（每里程碑跑全量）：

1. **单元测试**：模块级（LiquidCore v2 tau 输入依赖断言、OperatorPotential 分辨率不变断言、守恒断言）
2. **守恒测试**：任意随机初始权重下 rollout 能量漂移 ≤ O(dt²) 上界（架构性质，不需训练）
3. **回归基准**：v0.1 全部 benchmark 复跑，指标不退化
4. **对照实验**：liquid_ham vs static_ham vs gru_seq（M1）；MLP-V vs FNO-V（M2）；prefix vs all2all（M3）；pretrain-finetune vs from-scratch（M3）

## 8. 实施路线（与 PRD 里程碑对齐）

| 阶段 | 交付 | 文件 |
|---|---|---|
| M1 | LiquidCore v2 + tau 输入依赖 + pscan 通用化 + v0.1 基准复跑 | `liquid_core.py`、`parallel_scan.py`（如有需要）、`tests/test_liquid_core.py` |
| M2 | OperatorPotential + FiLM + 2D 场 benchmark + 分辨率测试 | `operator_potential.py`（新）、`hamiltonian.py`、`benchmarks/field_eval.py`（新） |
| M3 | 半群训练 + 数据管线 + 预训练-微调 + few-shot benchmark | `model.py`、`benchmarks/liquid_physics_eval.py`（扩展） |

每阶段独立可验收（PRD §7 的验收标准），阶段间以 ADR 修订收口。
