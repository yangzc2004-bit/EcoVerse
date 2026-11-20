# EcoVerse 基础库开发计划书（v0 架构与实施指南 · rev3）

> **项目名称**：`EcoVerse`（Python 包名：`ecoverse`）  
> **课题定位**：面向跨生态系统底层动力学理解的可组合合成引擎与标准化 Benchmark（深度适配 PyTorch + GPU 训练）  
> **文档用途**：可直接照图施工的 `v0` 工程开发规范与验收指南

---

## 0. 本轮规格复核（rev3）

逐项检查了第 4–8 节的公式、零基索引、先验、采样和评分协议。独立代入得到光照和温度昼夜因子的峰时为 12 点和 15 点，30°C 的模型氧饱和值为 7.437402。内部反应的化学计量矩阵秩为 5，三个独立守恒向量与它相乘均为零。

| 位置 | 核查结果与采用的口径 |
|---|---|
| 4.2、8 | 云量和季节包络随时间变化，总光照峰可偏离正午。连续一阶导数适用于基线驱动，指定干预时刻允许阶跃或状态跳跃。 |
| 4.3、8 | Monod 和 Hill 在半饱和常数处取半值。Haldane 在该点为 $1/(2+K/K_I)$，峰位为 $\sqrt{K K_I}$。 |
| 4.4 | 通道索引从 0 到 22。氧、碳气体交换允许负值，关闭的通道对宏观通量的贡献严格为零。 |
| 4.5、4.6 | 对数均匀先验要求正下界。表中同步列出已在预设文件采用的校准值，关闭项使用常数零。区分暴雨频率与干预幅度倍数。 |
| 5.1、5.2 | 4 小时输出网格右开。索引 1080 对应第 180 天，最后一点为第 364 天 20 时。固定版本与配置下，落盘种子须能单系统重放。 |
| 6、7 | 各套餐分别生成锁定视图，变量可以重叠。反事实只输入干预前状态观测、外部驱动和已知干预；训练使用温和分支，考题 3 使用强干预。 |
| 7、8 | 探针按预设分层拟合并涵盖全部五个测试划分。验收分别记录真值库规模、数值质检、模型分数及实际硬件。 |

### 历史修订（rev1 → rev2）

| 类别 | 位置 | 原问题 | 修改 |
|---|---|---|---|
| 数据错误 | 4.2 光照 | `sin(2πt)` 让白天落在 0–12 点 | 相位平移为 6–18 点，正午最强 |
| 数据错误 | 4.2 光照 | 季节项可能使 $I>1$ | 季节包络除以 $(1+a_I)$ |
| 数据错误 | 4.2 驱动 | `clip` 和脉冲起点导数不连续 | 云量用 sigmoid；脉冲改平方型 |
| 数据错误 | 通道 16 | 饱和溶解氧近似在 30°C 误差约 5% | 改用标准三次多项式 |
| 数据错误 | 通道 17 | $C_{\text{sat}}$ 未定义 | 明确宏观无机碳库并给先验 |
| 数据错误 | 边界项 | 只定义了 $N_{\text{in}}$ | 补全 8 维输入浓度向量 |
| 数据错误 | 边界项 | $\sigma_i\in\{0,1\}$ 与河流 0.05–0.3 矛盾 | 统一为 $\sigma_i\in[0,1]$ |
| 数据错误 | 湖泊预设 | 浮游动物摄食沉水植物、沉水植物沉降不合理 | 默认关闭通道 7、19 |
| 数据错误 | 锁死池 | $100K_N$ 只能让门控约 0.99 | 编译期门控旁路，严格为 1 |
| 数据错误 | 6.2 | 标准化统计量来源未规定 | 只用训练集统计量 |
| 实现缺口 | 4.5 | 约 20 个参数无先验 | 新增 4.6 补充先验表 |
| 实现缺口 | 4.5 | 可选状态 "(P2)" 规则未定义 | 抽样时以概率 0.5 启用 |
| 实现缺口 | 4.2 | 食物网/恒化器不该有昼夜光照 | 新增 `diel` / `seasonal` 开关 |
| 实现缺口 | 5.1 | 干预强度训练/测试如何区分未说明 | 新增 5.3 干预强度表 |
| 实现缺口 | 7 | 2b 混搭只举例 | 列出完整清单 |
| 实现缺口 | 7 | 线性探针无对照 | 增加随机初始化模型对照 |
| 实现缺口 | 8 | 只有碳、氮两个守恒量 | 补充 $O+\alpha_O C$ |
| 工程优化 | 2 / 5.1 | RK45→BDF 手动回退 | 默认 `LSODA`，超时由进程池控制 |
| 工程优化 | 8 | "单系统 <0.2 s" 不现实 | 改为 <2 s，并给后备方案 |
| 工程优化 | 5.2 | 按单系统分块产生数万小文件 | 每块 32 个系统 |
| 工程优化 | 5.1 | 竞争排斥被误判为废品 | 记录有效状态掩码 |
| 工程优化 | 7 | 训练只用饱和型，先验太窄 | 训练加入 S 型，留出全部单峰型 |
| 工程优化 | 8 | 2 周工期偏乐观 | 调整为 4–6 周 |

---

## 1. 版本演进规划（确保 v0 向前兼容）

整个基础库始终坚持一套核心数学内核：存量–流量矩阵动力学 $\frac{d\mathbf{x}}{dt} = \mathbf{S}^\top \mathbf{r}(\mathbf{x}, \mathbf{u}, \boldsymbol{\theta}) + \text{边界吞吐}$。分四个阶段演进，**当前开发聚焦于 `v0`**：

| 版本 | 核心定位 | 系统结构与能力边界 | 关键产出 |
|---|---|---|---|
| **v0（当前开发）** | **稳打地基与全链路闭环** | 固定 $n_p=2, n_z=2$（8 维状态、23 条原子通道、5 大生态预设、0 维确定性 ODE）；完整 Numba 生产引擎 + Zarr 仓库 + PyTorch GPU 事件流管线 + 4 道 Benchmark 考题 | 跑通 5,000–10,000 个合成生态系统库；验证 Transformer / Latent ODE 能否提取潜空间机制 |
| **v1** | **复杂生态网扩展** | 引入 **Niche Model** 随机食物网拓扑生成器，允许任意物种数 $n_p, n_z$（10–50 维）；将 $\mathbf{S}$ 升级为稀疏矩阵索引表示 | 测试模型向任意维度、未见网络拓扑的零样本泛化 |
| **v2** | **随机性与空间耦合** | ODE 升级为 **SDE**（过程噪声 + 参数慢漂移，模拟噪声诱发突变）；0D 单箱沿有向图串联为 **1D 河网 / 空间斑块系统** | 缩小 Sim-to-Real 鸿沟，覆盖临界转变与空间水文耦合 |
| **v3** | **虚实迁移与闭环调控** | 接入真实监测数据集（USGS/StreamPULSE 河流、LTER 湖泊、FLUXNET 通量塔、课题组实测数据）+ 开放实时 Gym 干预调控接口 | 真实世界零样本迁移验证与基于世界模型的稳态调控 |

---

## 2. 技术栈与 PyTorch/GPU 适配选型

> [!IMPORTANT]
> **算力分工原则**：离线解 ODE 造数据在 **CPU（SciPy + Numba 多进程）** 完成，避免变结构自适应步长 ODE 在 GPU 上发生分支发散；数据存储、动态观测切片、模型训练和潜空间探针全部深度适配 **PyTorch + GPU**。

| 模块环节 | 推荐选型 | 核心理由与避坑指南 |
|---|---|---|
| **基础环境** | `Python 3.11+` + `uv` (或 `conda`) | 依赖管理干净、启动快 |
| **深度学习框架** | `PyTorch >= 2.1`（CUDA） | 原生支持 `scaled_dot_product_attention` (FlashAttention)，适合变长事件流 Transformer |
| **ODE 求解加速** | `NumPy` + `SciPy solve_ivp(method="LSODA")` + `Numba (@njit)` 右端项 | 求解用 `float64`；LSODA 自动在刚性/非刚性间切换；**Numba 禁用 `fastmath`**，保证多进程结果逐位一致；若吞吐不达标，后备方案为 `numbalsoda` 或自写 Numba 版 Dormand–Prince |
| **配置与 DNA 校验** | `PyYAML` + `Pydantic v2`（或 `dataclass`） | 强校验所有参数范围与数组维度，防止配置写错产出垃圾数据 |
| **真值仓库存储** | **`Zarr`**（轨迹张量）+ **`Parquet`**（`pyarrow` 元数据） | `Zarr` 支持多进程并发读，适配 `DataLoader(num_workers>0)`；`h5py` 也能用，但必须在每个 worker 内延迟打开文件。落盘统一转 `float32` |
| **测试与可视化** | `pytest` + `matplotlib` | 强制校验物质守恒、保正性、确定性种子与 GPU 吞吐 |

---

## 3. 代码仓库目录结构与模块接口规范

```text
ecoverse/
├── pyproject.toml                  # 包管理与依赖声明
├── configs/                        # 【配置层：改实验不动代码】
│   ├── presets.yaml                # 5 大生态预设：状态掩码、通道开关、σ、驱动开关、参数先验
│   ├── shapes.yaml                 # 训练可用形状 / 考题 2c 留出形状
│   ├── interventions.yaml          # 4 类干预及其训练范围 / 分布外范围
│   ├── sensors.yaml                # 标准异构监测站套餐定义
│   └── benchmark_v0.yaml           # 各划分系统数、种子区段、2b 混搭清单
│
├── ecoverse/                       # 【核心 Python 包】
│   ├── core/                       # 1. 数学核心层（无 I/O，纯数学定义）
│   │   ├── constants.py            # 状态索引、通道索引、形状枚举 ID、配比常数
│   │   ├── spec.py                 # SystemSpec DNA 对象 + compile_spec() 静态编译
│   │   ├── shapes.py               # 4 类基础形状算子 + 保正门控 g_O, g_C (@njit)
│   │   ├── forcing.py              # 光滑解析驱动 u(t) 生成器 + Intervention 干预算子
│   │   ├── dynamics.py             # 固定 8 维 23 通道内核 (@njit)：rhs() 返回 dx/dt，rates() 返回 r(t)
│   │   └── torch_dynamics.py       # (可选扩展) PyTorch 向量化可微版本，支持 GPU 反传
│   │
│   ├── sim/                        # 2. 离线批量生产引擎
│   │   ├── sampler.py              # 按预设与“相对比例抽样”规则生成 SystemSpec
│   │   ├── solver.py               # 预热 180d → 正式 365d → 干预分叉；在输出网格上回算 r(t)
│   │   ├── quality.py              # 质检关卡 + 有效状态掩码 + 动力学体制打标
│   │   └── pipeline.py             # SeedSequence 种子派生 + 进程池(含超时) + 分块写入
│   │
│   ├── obs/                        # 3. 异构观测与 PyTorch GPU 数据管线
│   │   ├── normalize.py            # 只用训练集计算各变量标准化统计量
│   │   ├── sensors.py              # 观测算子：变量掩码、不规则采样、加噪、块状缺测
│   │   ├── dataset.py              # EcoVerseDataset (torch.utils.data.Dataset)
│   │   ├── collate.py              # 变长事件流打包器：输出 (B, L_max, ...) + attn_mask
│   │   └── env.py                  # (v0 预留接口) 交互式调控沙盒 Gym Env 骨架
│   │
│   └── bench/                      # 4. Benchmark 考场与 GPU 评测器
│       ├── splits.py               # 读取 Parquet 锁定划分 Train / Val / Test 1, 2a, 2b, 2c, 3
│       ├── metrics.py              # 预测、未观测重构、通量反推、跨观测对齐、反事实指标
│       └── probe.py                # GPU 加速的潜空间线性探针 (Linear Probe)
│
├── scripts/                        # 【命令行入口】
│   ├── generate_v0.py              # 一键生成 v0 真值库与元信息
│   ├── inspect_data.py             # 可视化典型曲线、合格率与动力学多样性体检报告
│   └── run_eval.py                 # 一键在 GPU 上运行 Benchmark 四道考题
│
├── data/                           # 【本地数据资产，加入 .gitignore】
│   └── v0/
│       ├── metadata.parquet        # 所有系统的 DNA 配置、体制标签、划分键
│       ├── trajectories.zarr/      # 分块压缩的 float32 真值轨迹仓库
│       ├── norm_stats.json         # 训练集标准化统计量
│       └── splits.json             # 锁定的考题索引表
│
└── tests/                          # 【自动化单元测试】
    ├── test_shapes.py              # 形状值域、资源响应零点、Monod/Hill 半饱和值及单峰算子峰位
    ├── test_forcing.py             # 光照在 [0,1]、正午最大、驱动一阶导数连续
    ├── test_conservation.py        # 闭边界下全池启用时三个守恒量误差 < 1e-8
    ├── test_positivity.py          # 极端参数下所有状态非负
    ├── test_reproducibility.py     # 单核 vs 多核同种子逐比特一致性测试
    └── test_torch_loader.py        # PyTorch DataLoader 多进程 + pin_memory + CUDA 压测
```

### 模块间数据流架构

```mermaid
flowchart LR
    CFG["configs/*.yaml"] --> SAM["sim/sampler.py\n生成 SystemSpec"]
    SAM --> COMP["core/spec.py\n编译为静态数组"]
    COMP --> SOL["sim/solver.py\nNumba 求解器"]
    SOL --> QG["sim/quality.py\n质检与打标"]
    QG -->|"通过"| DB["data/v0/\nZarr + Parquet"]
    DB --> DS["obs/dataset.py\n实时/锁定观测切片"]
    DS --> COL["obs/collate.py\n变长事件流 + Mask"]
    COL -->|"DataLoader → CUDA"| GPU["PyTorch 模型训练"]
    GPU -->|"预测与 Latent z"| EV["bench/metrics.py & probe.py\n四道考题 + GPU 探针"]
```

---

## 4. `v0` 底层数学规格说明书

### 4.1 状态变量、单位与全局常数
- **固定 8 维状态向量**：$\mathbf{x}(t) = \big(N,\ P_1,\ P_2,\ Z_1,\ Z_2,\ D,\ O,\ C\big)^\top$
  - 生物与碎屑池 $(P_1, P_2, Z_1, Z_2, D)$：碳当量浓度 $\text{g C m}^{-3}$
  - 资源池 $N$：无机营养盐浓度 $\text{g N m}^{-3}$
  - 氧池 $O$：溶解氧浓度 $\text{g O}_2\text{ m}^{-3}$
  - 碳池 $C$：**宏观无机碳库** $\text{g C m}^{-3}$（不区分碳酸盐形态，这是有意的宏观抽象；数量级参照淡水溶解无机碳）
- **时间约定**：单位为**天（$\text{d}$）**；预热期 $180\ \text{d}$，正式记录期 $365\ \text{d}$；真值存储分辨率 $\Delta t = 1/6\ \text{d}$（每 4 小时 1 点，兼顾昼夜节律与磁盘体积，单条轨迹长度 $T = 2190$ 点）。
- **全局宏观常数与保正门控**（记 $x_+ = \max(x, 0)$）：
  - 模型采用的宏观质量配比：$\alpha_N = 0.15\ (\text{g N/g C})$，$\alpha_O = 2.67\ (\text{g O}_2/\text{g C})$
  - 低氧平滑关停门控：$g_O(O) = \frac{O_+}{0.1 + O_+}$
  - 低碳平滑关停门控：$g_C(C) = \frac{C_+}{0.1 + C_+}$

经典海洋 Redfield 比是摩尔比。按 106 mol C、16 mol N 换算约为 0.176 g N/g C，本模型采用的 0.15 和 2.67 是固定建模系数。[海洋颗粒有机物元素计量研究](https://www.nature.com/articles/s41467-025-61061-0)

### 4.2 连续光滑外部驱动 $\mathbf{u}(t)$ 与干预 `Intervention`
所有驱动写成关于连续时间 $t \in \mathbb{R}$ 的解析函数，在 `@njit` 中直接求值，**不做离散插值**（插值会在导数上制造尖角，拖慢自适应求解器）。每个预设带两个驱动开关：`diel`（昼夜变化）和 `seasonal`（季节变化）。

1. **归一化光照 $I(t) \in [0, 1]$**（昼夜因子在 6–18 点取正值，12 点达峰）：
   $$I(t) = \underbrace{\frac{1 - a_I \cos(2\pi t / 365)}{1 + a_I}}_{\text{季节}} \cdot \underbrace{\Big[\max\!\big(0,\ \sin(2\pi (t - 0.25))\big)\Big]^2}_{\text{昼夜}} \cdot c_{\text{cloud}}(t)$$
   - 云量包络用 sigmoid 压缩以保持光滑：$c_{\text{cloud}}(t) = 0.2 + 0.8\,\mathrm{sigmoid}\!\left(b_c + \sum_{k=1}^{8} w_k \sin(2\pi f_k t + \psi_k)\right)$，$f_k \in [0.02, 0.25]\ \text{d}^{-1}$
   - `diel=false` 时昼夜因子换成其日均值 $0.25$（白天半日内 $\sin^2$ 均值 0.5，再乘白天占比 0.5）；`seasonal=false` 时季节因子取 1
2. **温度 $T(t)\ (^\circ\text{C})$**（季节相位比光照滞后 $\ell_T$ 天，昼夜峰值约在 15 点）：
   $$T(t) = \bar{T} - A_{\text{season}}\cos\!\big(2\pi (t - \ell_T) / 365\big) + A_{\text{diel}}\sin\!\big(2\pi (t - 0.375)\big) + \eta_T(t)$$
   其中 $\eta_T(t) = \sum_{k=1}^{6} v_k \sin(2\pi g_k t + \chi_k)$ 为低频随机波动。
3. **冲刷率 $q(t)\ (\text{d}^{-1})$**（平方型脉冲，起点处函数值与一阶导数都为 0）：
   $$q(t) = q_{\text{base}} + \sum_{k=1}^{K_{\text{storm}}} A_k \left(\frac{t - t_k}{\tau_k}\right)^{2}\exp\!\left(2 - \frac{2(t - t_k)}{\tau_k}\right)\mathbb{I}(t > t_k)$$
   暴雨时刻 $t_k$ 用泊松过程在 $[-180, 365]$ 上预先抽好，峰值出现在 $t_k + \tau_k$，峰高为 $A_k$。
4. **外源资源浓度 $N_{\text{in}}(t)$**：常基线 $N_{\text{in},0}$（可叠加干预变化）。
5. **标准干预算子**（在正式轨迹第 $t_{\text{int}} = 180\ \text{d}$ 触发反事实分支；强度范围见 5.3）：
   - `nutrient_step`：外源 $N_{\text{in}}$ 阶跃变更为 $\lambda N_{\text{in},0}$（如削减 50% 或加倍）
   - `warming_shift`：温度在分叉后叠加常量 $\Delta T$，温和与强干预的范围见 5.3
   - `flow_regime`：冲刷率 $q(t)$ 整体缩放或消除暴雨脉冲（模拟建坝/干旱）
   - `bio_pulse`：在 $t_{\text{int}}$ 瞬间将某生物池状态 $P_1$ 或 $Z_1$ 乘以缩放系数（模拟生物操纵）

### 4.3 基础形状算子库 `shapes.py`（整数枚举编码）

| 形状族 | Enum ID | 名称 | 精确数学公式（$x_+ = \max(x, 0)$） | 备注 |
|---|---|---|---|---|
| **资源/捕食** $\phi_{\text{res}}$ | `0` | `MONOD`（饱和） | $\frac{x_+}{K + x_+}$ | 训练可用 |
| | `1` | `HILL`（S 型阈值） | $\frac{x_+^n}{K^n + x_+^n}\ (n \in [2, 3.5])$ | 训练可用 |
| | `2` | `HALDANE`（高值抑制，单峰） | $\frac{x_+}{K + x_+ + x_+^2 / K_I}$ | **考题 2c 留出** |
| **光照** $\phi_{\text{light}}$ | `0` | `SMITH`（饱和） | $\frac{I_+}{\sqrt{I_k^2 + I_+^2}}$ | 训练可用 |
| | `1` | `STEELE`（强光抑制，单峰） | $\frac{I_+}{I_{\text{opt}}}\exp\!\left(1 - \frac{I_+}{I_{\text{opt}}}\right)$ | **考题 2c 留出** |
| | `2` | `LINEAR`（线性） | $I_+$ | 训练可用（恒化器/食物网） |
| **温度** $\phi_{\text{temp}}$ | `0` | `Q10`（单调指数） | $\theta_T^{(T - 20)/10}$ | 训练可用；$T=20^\circ\text{C}$ 时为 1 |
| | `1` | `GAUSSIAN`（最适温度，单峰） | $\exp\!\left(-\frac{(T - T_{\text{opt}})^2}{2\sigma_T^2}\right)$ | **考题 2c 留出** |
| | `2` | `CONST`（不敏感） | $1.0$ | 训练可用（恒化器/食物网） |
| **密度损耗** $\phi_{\text{loss}}$ | `0` | `LINEAR`（一阶） | $m_a x_+$ | — |
| | `1` | `QUADRATIC`（二次自限） | $m_a x_+ + m_b x_+^2$ | 提供 Logistic 承载力 |
| **碎屑分解** $\psi_{\text{dec}}$ | `0` | `LINEAR`（一阶） | $D_+$ | 与下式在低浓度斜率统一 |
| | `1` | `SATURATING`（量纲对齐饱和） | $\frac{K_D D_+}{K_D + D_+}$ | 保持 $k_{\text{dec}}$ 单位严格为 $\text{d}^{-1}$ |

### 4.4 总方程与 23 条原子反应通道

$$
\frac{d\mathbf{x}}{dt} = \mathbf{m}_{\text{state}} \odot \left[\sum_{j=0}^{22} w_j \, \mathbf{s}_j^\top \, r_j(\mathbf{x}, \mathbf{u}, \boldsymbol{\theta}) + q(t)\big(\boldsymbol{\delta} \odot \mathbf{x}_{\text{in}}(t) - \boldsymbol{\sigma} \odot \mathbf{x}_+\big)\right]
$$

- **状态锁死与编译期门控旁路规则（A9 修复）**：当某预设不启用状态 $i$（$m_{\text{state},i}=0$）时，生物池初值锁定为 $0$；非生物池（$N, O, C$）不仅导数锁为 $0$，且在 `compile_spec()` 编译期直接将对应限制因子/门控设为**旁路模式**（即 $m_{\text{state},N}=0 \Rightarrow \phi_{\text{res}}(N)\equiv 1$，$m_{\text{state},O}=0 \Rightarrow g_O(O)\equiv 1$，$m_{\text{state},C}=0 \Rightarrow g_C(C)\equiv 1$），数学上严格等于 $1$（而非 $0.99$ 近似），方程精确退化为经典群落动力学方程。
- **8 维开放边界输入与连续冲刷系数（A6/A7 修复）**：
  - 边界输入向量固定为 $\mathbf{x}_{\text{in}}(t) = \big(N_{\text{in}}(t),\ 0,\ 0,\ 0,\ 0,\ D_{\text{in}},\ O_{\text{sat}}(T(t)),\ C_{\text{sat}}\big)^\top$，对应掩码 $\boldsymbol{\delta} \in \{0,1\}^8$；
  - 边界冲刷/流失系数泛化为连续区间 $\boldsymbol{\sigma} \in [0, 1]^8$（`1.0` 为完全随水体冲刷，`0.0` 为扎根固着，$(0,1)$ 为河流底栖/附着生物的部分滞留）。

| 编号 | 宏观大类 | 矩阵行 $\mathbf{s}_j$ 非零项 | 精确速率公式 $r_j(\mathbf{x}, \mathbf{u})$ | 可替换槽位 |
|---|---|---|---|---|
| `0` | 1. 自养生产 $P_1$ | $N\!:\!-\alpha_N,\ P_1\!:\!+1,\ O\!:\!+\alpha_O,\ C\!:\!-1$ | $v_{\max,1}\,\phi_{\text{light}}(I)\,\phi_{\text{res}}(N;K_{N1})\,g_C(C)\,\phi_{\text{temp}}(T)\,P_{1+}$ | 光 / 源 / 温 |
| `1` | 1. 自养生产 $P_2$ | $N\!:\!-\alpha_N,\ P_2\!:\!+1,\ O\!:\!+\alpha_O,\ C\!:\!-1$ | $v_{\max,2}\,\phi_{\text{light}}(I)\,\phi_{\text{res}}(N;K_{N2})\,g_C(C)\,\phi_{\text{temp}}(T)\,P_{2+}$ | 光 / 源 / 温 |
| `2` | 2. 生产者呼吸 $P_1$ | $N\!:\!+\alpha_N,\ P_1\!:\!-1,\ O\!:\!-\alpha_O,\ C\!:\!+1$ | $k_{\text{resp},P1}\,\phi_{\text{temp}}(T)\,g_O(O)\,P_{1+}$ | 温 |
| `3` | 2. 生产者呼吸 $P_2$ | $N\!:\!+\alpha_N,\ P_2\!:\!-1,\ O\!:\!-\alpha_O,\ C\!:\!+1$ | $k_{\text{resp},P2}\,\phi_{\text{temp}}(T)\,g_O(O)\,P_{2+}$ | 温 |
| `4` | 3. 生产者凋亡 $P_1$ | $P_1\!:\!-1,\ D\!:\!+1$ | $\phi_{\text{loss}}(P_1; m_{P1a}, m_{P1b})$ | 损耗 |
| `5` | 3. 生产者凋亡 $P_2$ | $P_2\!:\!-1,\ D\!:\!+1$ | $\phi_{\text{loss}}(P_2; m_{P2a}, m_{P2b})$ | 损耗 |
| `6` | 4. 初级捕食 $P_1\!\to\!Z_1$ | $P_1\!:\!-1,\ Z_1\!:\!+\gamma_1,\ D\!:\!+(1-\gamma_1)$ | $g_{\max,11}\,\phi_{\text{res}}(P_1; K_{P1})\,\phi_{\text{temp}}(T)\,Z_{1+}$ | 捕食 / 温 |
| `7` | 4. 初级捕食 $P_2\!\to\!Z_1$ | $P_2\!:\!-1,\ Z_1\!:\!+\gamma_1,\ D\!:\!+(1-\gamma_1)$ | $g_{\max,12}\,\phi_{\text{res}}(P_2; K_{P2})\,\phi_{\text{temp}}(T)\,Z_{1+}$ | 捕食 / 温 |
| `8` | 5. 次级捕食 $Z_1\!\to\!Z_2$ | $Z_1\!:\!-1,\ Z_2\!:\!+\gamma_2,\ D\!:\!+(1-\gamma_2)$ | $g_{\max,2}\,\phi_{\text{res}}(Z_1; K_{Z1})\,\phi_{\text{temp}}(T)\,Z_{2+}$ | 捕食 / 温 |
| `9` | 6. 同级干扰 $P_2\!\dashv\!P_1$ | $P_1\!:\!-1,\ D\!:\!+1$ | $c_{12}\,P_{1+}\,P_{2+}$ | — |
| `10` | 6. 同级干扰 $P_1\!\dashv\!P_2$ | $P_2\!:\!-1,\ D\!:\!+1$ | $c_{21}\,P_{2+}\,P_{1+}$ | — |
| `11` | 7a. 消费者呼吸 $Z_1$ | $N\!:\!+\alpha_N,\ Z_1\!:\!-1,\ O\!:\!-\alpha_O,\ C\!:\!+1$ | $k_{\text{resp},Z1}\,\phi_{\text{temp}}(T)\,g_O(O)\,Z_{1+}$ | 温 |
| `12` | 7a. 消费者呼吸 $Z_2$ | $N\!:\!+\alpha_N,\ Z_2\!:\!-1,\ O\!:\!-\alpha_O,\ C\!:\!+1$ | $k_{\text{resp},Z2}\,\phi_{\text{temp}}(T)\,g_O(O)\,Z_{2+}$ | 温 |
| `13` | 7b. 消费者死亡 $Z_1$ | $Z_1\!:\!-1,\ D\!:\!+1$ | $\phi_{\text{loss}}(Z_1; m_{Z1a}, m_{Z1b})$ | 损耗 |
| `14` | 7b. 消费者死亡 $Z_2$ | $Z_2\!:\!-1,\ D\!:\!+1$ | $\phi_{\text{loss}}(Z_2; m_{Z2a}, m_{Z2b})$ | 损耗 |
| `15` | 8. 碎屑分解 | $N\!:\!+\alpha_N,\ D\!:\!-1,\ O\!:\!-\alpha_O,\ C\!:\!+1$ | $k_{\text{dec}}\,\phi_{\text{temp}}(T)\,g_O(O)\,\psi_{\text{dec}}(D; K_D)$ | 温 / 分解 |
| `16` | 9. 氧气体交换 | $O\!:\!+1$ | $k_{\text{gas},O}\,\big(O_{\text{sat}}(T) - O\big)$，其中 $O_{\text{sat}}(T) = 14.652 - 0.41022T + 0.007991T^2 - 7.7774\times 10^{-5}T^3$ | — |
| `17` | 9. 碳气体交换 | $C\!:\!+1$ | $0.9\,k_{\text{gas},O}\,\big(C_{\text{sat}} - C\big)$（$C_{\text{sat}}$ 为环境平衡无机碳基线） | — |
| `18` | 10. 颗粒沉降 $P_1$ | $P_1\!:\!-1$ | $\frac{s_{P1}}{1 + \eta_q q(t)}\,P_{1+}$ | — |
| `19` | 10. 颗粒沉降 $P_2$ | $P_2\!:\!-1$ | $\frac{s_{P2}}{1 + \eta_q q(t)}\,P_{2+}$ | — |
| `20` | 10. 碎屑埋藏 $D$ | $D\!:\!-1$ | $\frac{s_D}{1 + \eta_q q(t)}\,D_+$ | — |
| `21` | 12a. 底界面耗氧产碳 | $O\!:\!-\alpha_O,\ C\!:\!+1$ | $b_{\text{SOD}}\,\phi_{\text{temp}}(T)\,g_O(O)$ | 温 |
| `22` | 12b. 低氧内源释放 | $N\!:\!+1$ | $b_N\,\big(1 - g_O(O)\big)$ | — |

- **宏观派生真值（自动由 `r_true` 汇总）**：$\text{GPP} = r_0 + r_1$，$\text{ER} = r_2 + r_3 + r_{11} + r_{12} + r_{15} + r_{21}$，$\text{NEP} = \text{GPP} - \text{ER}$。

### 4.5 五大生态预设开关与相对比例参数先验

- **可选状态 `(P2)` 抽样规则（B2 修复）**：表中带括号的 `(P2)` 表示抽样每个系统时以 **$0.5$ 的概率**启用 $P_2$；若不启用，则 $m_{\text{state},P_2}=0$ 且所有关联通道（`1, 3, 5, 7, 9, 10, 19`）自动关闭。
- **湖泊预设生态修正（A8 修复）**：浅水湖泊中 $P_2$ 代表大型沉水植物，默认关闭浮游动物 $Z_1$ 对沉水植物的摄食（通道 `7`）以及沉水植物的颗粒沉降（通道 `19`），保留其与浮游藻类 $P_1$ 的资源竞争与相互干扰（通道 `9, 10`）。

| 预设 | 活跃状态 ($\mathbf{m}_{\text{state}}=1$) | 开启通道 ID ($w_j=1$) | 边界冲刷系数 $\boldsymbol{\sigma}$ | 驱动开关 (`diel` / `seasonal`) |
|---|---|---|---|---|
| **river**（河流） | $N, P_1, Z_1, D, O, C$ | `0, 2, 4, 6, 11, 13, 15, 16, 17, 18, 20, 21` | $N,D,O,C=1.0$；$P_1,Z_1 \in [0.02, 0.12]$ | `true` / `true` |
| **lake**（浅水湖泊） | $N, P_1, P_2, Z_1, D, O, C$ | `0–6, 9–11, 13, 15–18, 20–22`（关闭 `7, 19`） | $N,P_1,Z_1,D,O,C=1.0$；$P_2=0.0$ | `true` / `true` |
| **terrestrial**（陆地） | $N, P_1, (P_2), Z_1, D, C$ | `0–7, 9–11, 13, 15, 17, 20, 21` | $P_1,P_2,Z_1,D=0.0$；$N=1.0, C=0.0$ | `true` / `true` |
| **foodweb**（经典食物网） | $P_1, (P_2), Z_1, Z_2$ | `0–1, 4–5(二次), 6–10, 13–14` | 全 `0.0` | `false` / `50% true` |
| **chemostat**（恒化器） | $N, P_1, (P_2), Z_1$ | `0–1, 4–7, 13` | $N,P_1,P_2,Z_1=1.0$（$q$ 为常数） | `false` / `false` |

**核心参数抽样顺序与先验范围**（`LU` = 对数均匀分布，`U` = 均匀分布；先抽锚点速率再按比例抽从属速率，确保高存活率）：

数值使用 configs/presets.yaml 的校准先验。变体继承基础预设并应用 variants_2b 中的覆盖项。

| 采样参数 | river | lake | terrestrial | foodweb | chemostat |
|---|---|---|---|---|---|
| v_max1 | LU[1.2, 3.5] | LU[0.5, 2.0] | LU[0.04, 0.25] | LU[0.3, 1.5] | LU[1.5, 4.5] |
| v_max2_ratio | U[0.4, 1.0] | U[0.1, 0.35] | U[0.3, 0.9] | U[0.5, 1.2] | U[0.6, 1.2] |
| k_resp_P_ratio | U[0.04, 0.15] | U[0.05, 0.2] | U[0.1, 0.3] | 0.0 | 0.0 |
| m_Pa_ratio | U[0.01, 0.08] | U[0.02, 0.12] | U[0.02, 0.1] | U[0.02, 0.1] | U[0.01, 0.08] |
| m_Pb | LU[0.005, 0.1] | LU[0.04, 0.35] | LU[0.01, 0.1] | LU[0.05, 0.5] | LU[0.002, 0.05] |
| g_max1_ratio | U[0.4, 1.2] | U[0.4, 1.4] | U[0.2, 0.8] | U[0.5, 1.5] | U[0.5, 1.5] |
| g_max2_ratio | U[0.3, 0.9] | U[0.3, 0.9] | U[0.3, 0.9] | U[0.3, 0.9] | U[0.3, 0.9] |
| gamma | U[0.3, 0.7] | U[0.3, 0.7] | U[0.2, 0.6] | U[0.3, 0.8] | U[0.3, 0.7] |
| z_loss_ratio | U[0.1, 0.4] | U[0.1, 0.4] | U[0.1, 0.4] | U[0.1, 0.45] | U[0.05, 0.3] |
| c_comp | LU[0.05, 0.4] | LU[0.015, 0.15] | LU[0.02, 0.3] | LU[0.05, 0.5] | 0.0 |
| k_dec | LU[0.05, 0.4] | LU[0.02, 0.2] | LU[0.005, 0.05] | 0.0 | 0.0 |
| K_N | LU[0.05, 0.4] | LU[0.03, 0.4] | LU[0.05, 0.5] | 0.2 | LU[0.05, 0.4] |
| K_P | LU[0.2, 2.0] | LU[0.2, 2.0] | LU[0.2, 2.0] | LU[0.3, 2.5] | LU[0.2, 2.0] |
| K_Z1 | LU[0.2, 2.0] | LU[0.2, 2.0] | LU[0.2, 2.0] | LU[0.3, 2.5] | LU[0.2, 2.0] |
| theta_T_prod | U[1.3, 1.8] | U[1.3, 1.8] | U[1.3, 1.9] | U[1.0, 1.5] | 1.0 |
| theta_T_resp | U[1.7, 2.6] | U[1.7, 2.6] | U[1.8, 2.8] | 1.0 | 1.0 |
| k_gas_O | LU[1.0, 15.0] | LU[0.1, 1.5] | LU[0.5, 5.0] | 0.0 | 0.0 |
| s_P | LU[0.005, 0.04] | LU[0.01, 0.08] | LU[0.001, 0.01] | 0.0 | 0.0 |
| s_D | LU[0.01, 0.1] | LU[0.05, 0.4] | LU[0.001, 0.01] | 0.0 | 0.0 |
| q_base | LU[0.15, 1.5] | LU[0.002, 0.04] | LU[0.005, 0.05] | 0.0 | LU[0.03, 0.25] |
| N_in_0 | LU[0.3, 4.0] | LU[0.2, 4.0] | LU[0.2, 3.0] | 0.0 | LU[0.8, 5.0] |
| b_SOD | LU[0.1, 2.0] | LU[0.2, 2.5] | LU[0.05, 0.5] | 0.0 | 0.0 |
| b_N | 0.0 | LU[0.05, 0.6] | 0.0 | 0.0 | 0.0 |

### 4.6 补充参数先验：形状参数、驱动参数与初始状态（B1 修复）

为确保代码实现无死角，所有未在 4.5 列出的辅助参数先验统一规定如下：

| 参数类别 | 符号 | 含义 | 先验范围与分布 |
|---|---|---|---|
| **形状参数** | $n_{\text{Hill}}$ | Hill（S 型）指数 | `U[2.0, 3.5]` |
| | $K_I$ | Haldane 高值抑制常数（考题 2c 留出） | `U[2.0, 6.0] · K`（相对对应半饱和常数 $K$） |
| | $I_k$ / $I_{\text{opt}}$ | Smith 半饱和光强 / Steele 最适光强 | $I_k \sim \text{U}[0.15, 0.45]$；$I_{\text{opt}} \sim \text{U}[0.45, 0.80]$ |
| | $T_{\text{opt}}$ / $\sigma_T$ | Gaussian 最适温度与宽度（考题 2c 留出） | $T_{\text{opt}} \sim \text{U}[\bar{T}-2, \bar{T}+5]\ (^\circ\text{C})$；$\sigma_T \sim \text{U}[5.0, 10.0]\ (^\circ\text{C})$ |
| | $K_D$ | 饱和型碎屑分解半饱和常数 | `LU[0.5, 5.0]` ($\text{g C m}^{-3}$) |
| **转化与沉降辅助** | $\beta_Z$ | 消费者总基础损耗中“呼吸”所占比例（$k_{\text{resp},Z} / (k_{\text{resp},Z}+m_{Za})$） | `U[0.4, 0.7]`（其余 $1-\beta_Z$ 为一阶死亡 $m_{Za}$；二次死亡 $m_{Zb} \sim \text{LU}[0.01, 0.2]$） |
| | $\eta_q$ | 流速对沉降的抑制系数 | `LU[0.5, 5.0]` ($\text{d}$) |
| | $C_{\text{sat}}$ / $D_{\text{in}}$ | 平衡无机碳基线 / 外源碎屑输入浓度 | $C_{\text{sat}} \sim \text{U}[10.0, 25.0]$；河流 $D_{\text{in}} \sim \text{LU}[0.1, 2.0]$，其余预设 $D_{\text{in}}=0$ |
| **外部驱动参数** | $a_I$ / $b_c$ / $w_k$ | 光照季节振幅 / 云量偏置 / 傅里叶幅值 | $a_I \sim \text{U}[0.2, 0.6]$；$b_c \sim \text{U}[0.5, 1.5]$；$w_k \sim \text{U}[-0.4, 0.4]$ |
| | $\bar{T}$ / $A_{\text{season}}$ / $A_{\text{diel}}$ / $\ell_T$ | 年均温 / 季节温差 / 昼夜温差 / 季节滞后 | $\bar{T} \sim \text{U}[10, 22]$；$A_{\text{season}} \sim \text{U}[4, 12]$；$A_{\text{diel}} \sim \text{U}[0.5, 2.5]$；$\ell_T \sim \text{U}[15, 35]\ \text{d}$ |
| | $\nu_{\text{storm}}$ / $A_k$ / $\tau_k$ | 暴雨泊松频率 / 脉冲峰高 / 退水时间尺度 | 河流 $\nu_{\text{storm}} \sim \text{U}[0.03, 0.12]\ \text{d}^{-1}$，$A_k \sim \text{U}[1.0, 4.0] \cdot q_{\text{base}}$，$\tau_k \sim \text{U}[1.0, 4.0]\ \text{d}$ |
| | $v_k,g_k$ 与相位 | 热扰动幅值、频率、云量和热扰动相位 | $v_k\sim U[-0.5,0.5]$，$g_k\sim U[0.01,0.12]$ 每天，相位 $U[0,2\pi)$；恒化器关闭热扰动。 |
| **初始状态** $\mathbf{x}(-180)$ | $N_0, P_{i0}, Z_{i0}, D_0, O_0, C_0$ | 预热期起点初值（仅对 $m_{\text{state},i}=1$ 的池抽样） | $N_0 \sim \text{LU}[0.2, 2.0]$，$P_{i0} \sim \text{LU}[0.1, 1.5]$，$Z_{i0} \sim \text{LU}[0.05, 0.8]$，$D_0 \sim \text{LU}[0.2, 2.0]$，$O_0 = O_{\text{sat}}(\bar{T})$，$C_0 = C_{\text{sat}}$ |

---

## 5. 模拟流水线、质检规则与存储 Schema

### 5.1 单系统求解与质检 (`sim/solver.py` + `sim/quality.py`)
1. **确定性种子与求解器（C1 修复）**：由 `master_seed` 经 `np.random.SeedSequence` 派生独立子种子；统一使用 `scipy.integrate.solve_ivp(method="LSODA", rtol=1e-6, atol=1e-8)`（自动在非刚性 Adams 与刚性 BDF 间无缝切换），单系统求解预算为 10 秒，由各工作进程在求解中检查截止时间；并行完成顺序不改变候选录取顺序。
2. **预热期（Burn-in）**：从随机初值 $\mathbf{x}(-180)$ 运行 $t \in [-180, 0]$，丢弃瞬态轨迹。
3. **质检四条硬阈值与竞争排斥合法化（C4 修复）**：
   - `status == 0` 且未触发进程超时（淘汰数值病态/发散）；
   - $\min(\mathbf{x}) \ge -10^{-4}$ 且 $\max(\mathbf{x}) \le 500$（保证物理有界非负；输出时将 $(-10^{-4}, 0)$ 截断为 $0$）；
   - **存活与有效状态掩码**：预热最后 30 天内，**至少有 1 个生产者池（$P_1$ 或 $P_2$）均值 $\ge 10^{-3}$**（若全部生产者灭绝则直接淘汰）。若双物种竞争（$P_1, P_2$）或高阶捕食（$Z_1, Z_2$）中某一方因竞争排斥在预热期自然灭绝（均值 $< 10^{-3}$），**不作为废品淘汰**，而是保留该系统并将预热期末仍存活的状态记录为 `effective_state_mask`（评测未观测变量重构时只对 `effective_state_mask == 1` 的状态计分）；
   - 至少 2 个有效状态的变异系数 $\text{CV} = \text{std}/\text{mean} \ge 0.02$（淘汰完全锁死的水平直线死水）。
4. **正式模拟与反事实分叉（B5 修复）**：
   - 从 $t=0$ 求解至 $t=365\ \text{d}$，在网格 $t_k = k/6\ (k=0,\dots,2189)$ 上记录 `x_true`，并调用 `@njit` 编译的 `rates(t_k, x_true[k], arrays)` 精确回算 `r_true` 与 `u_true`；
   - 在 $t_{\text{int}} = 180\ \text{d}$（即零基索引 $1080$）克隆状态 $\mathbf{x}(180)$，**保持底层随机云层、随机热浪和暴雨时刻序列与 Baseline 完全相同**，仅注入干预算子，求解 $[180, 365)$ 得到 `x_cf`、`r_cf`、`u_cf`。
5. **动力学体制自动打标**：根据正式轨迹与驱动频带的周期性、持续位移等诊断量打描述性标签；单年轨迹上的标签用于分布体检。枚举 `regime_tag` $\in \{\text{forced\_periodic},\ \text{endogenous\_oscillation},\ \text{regime\_shift}\}$。

### 5.2 Zarr + Parquet 存储 Schema (`data/v0/`)

- **`trajectories.zarr/`**（所有数组 `dtype=float32`；**C3 修复**：按 `chunk=(32, T, D)` 分块存储，避免单系统分块在 6,000 个系统下产生 3.6 万个碎文件，同时保证高吞吐读取）：
  - `x_true`: `(N_sys, 2190, 8)`
  - `r_true`: `(N_sys, 2190, 23)`
  - `u_true`: `(N_sys, 2190, 4)`
  - `x_cf`: `(N_sys, 1110, 8)`（对应 $t \in [180, 365)$ 共 1110 个点）
  - `r_cf`: `(N_sys, 1110, 23)`
  - `u_cf`: `(N_sys, 1110, 4)`
- **`metadata.parquet`**（每行对应一个系统 `sys_id`）：
  - `sys_id`, `seed`, `preset`, `split_tag`, `regime_tag`
  - `state_mask_json`, `effective_state_mask_json`, `channel_mask_json`, `shape_ids_json`, `params_json`, `intervention_json`, `shape_mode`, `interv_regime`, `spec_json`

完整系统可从 spec_json 还原。逐系统重采样使用 default_rng(seed)，复用行内的预设、形状模式及干预范围。

### 5.3 干预算子训练范围与考题 3（分布外）强度划分表（B4 修复）

每个系统在 $t_{\text{int}} = 180\ \text{d}$ 随机抽取以下 4 类干预之一生成反事实分支，**训练集采用温和干预，考题 3（`test_3_interv`）采用未见过的强干预**：

| 干预算子类型 | 作用对象与数学操作 | 训练集 / 验证集强度范围 | **考题 3 (`test_3_interv`) 分布外强度范围** |
|---|---|---|---|
| `nutrient_step` | $t \ge 180$ 时 $N_{\text{in}}(t) \leftarrow \lambda_N N_{\text{in}}(t)$ | $\lambda_N \in [0.6, 0.85] \cup [1.15, 1.6]$ | $\lambda_N \in [0.15, 0.4] \cup [2.2, 4.0]$（深度削减或剧烈富营养化冲击） |
| `warming_shift` | $t \ge 180$ 时 $T(t) \leftarrow T(t) + \Delta T$ | $\Delta T \in [-1.5, -0.5] \cup [+0.5, +2.0]\ ^\circ\text{C}$ | $\Delta T \in [+3.0, +6.0]\ ^\circ\text{C}$（极端气候变暖/持续热浪） |
| `flow_regime` | $t \ge 180$ 时 $q_{\text{base}} \leftarrow \lambda_q q_{\text{base}}$ 且暴雨幅值 $\times \lambda_{\text{storm}}$ | $\lambda_q \in [0.6, 1.5]$，$\lambda_{\text{storm}} \in [0.5, 1.5]$ | **极端干旱**（$\lambda_q \in [0.15, 0.35]$）或**水坝截流拉平**（$\lambda_{\text{storm}} = 0$） |
| `bio_pulse` | 在 $t = 180$ 瞬间令 $P_1(180^+) = \kappa P_1(180^-)$ 或 $Z_1(180^+) = \kappa Z_1(180^-)$ | $\kappa \in [0.5, 0.8] \cup [1.2, 1.8]$ | $\kappa \in [0.05, 0.25] \cup [2.5, 4.0]$（高强度生物操纵清除或投放） |

---

## 6. 异构观测算子与 PyTorch GPU 数据管线 (`obs/`)

### 6.1 虚拟监测站套餐 (`configs/sensors.yaml`)
- **套餐 A（高频水质站）**：观测 $\{O, C\}$ + 驱动 $\{I, T, q\}$，每 4 小时一次，低高斯噪声（$\sigma = 2\%$），随机 1–3 天传感器掉线块缺测；
- **套餐 B（传统生态采样）**：观测 $\{N, P_1, P_2, Z_1, D\}$，每 7–14 天不规则采样一次，中等对数正态噪声（$\sigma = 8\%$）；
- **套餐 C（单变量遥感/浮标）**：仅观测 $\{P_1\}$ 或 $\{O\}$，随机泊松间隔不规则采样 + 连续云层缺测；
- **训练时增强**：在 `EcoVerseDataset(mode="train")` 内，每次 `__getitem__` 现场随机抽变量子集、随机抽采样密度和噪声，1 条真值轨线衍生无限多观测视图。

套餐是观测生成协议。模型的 512 token 输入预算单独记录，满年观测视图可使用更高预算。套餐 A 在首选池均未启用时按配置回退到有效状态；套餐 C 只提供一个状态。

### 6.2 面向 Transformer/Latent ODE 的事件流 Token 协议与无泄露标准化
- **严格无泄露标准化（A10 修复，`obs/normalize.py`）**：
  - 仅在 **`train` 划分**中 `effective_state_mask == 1` 的有效轨迹上，按变量分别统计均值与标准差（对严格非负且跨数量级的生物/资源浓度建议采用 $\log(x + \epsilon)$ 后的均值与标准差，对温度 $T$、氧 $O$、碳 $C$ 采用原始值均值与标准差），保存至 `data/v0/norm_stats.json`；
  - 验证集与所有测试集（`test_1` ~ `test_3`）一律读取 `norm_stats.json` 进行变换，严禁使用测试集自身统计量。
- **全局 `var_id` 词表（共 12 个 ID）**：
  - `0..7` 对应状态 $(N, P_1, P_2, Z_1, Z_2, D, O, C)$，`is_driver = 0`
  - `8..11` 对应驱动 $(I, T, q, N_{\text{in}})$，`is_driver = 1`
- **`collate.py` 输出标准 PyTorch Batch 字典**（逐个张量调用 `.to(device, non_blocking=True)`）：
  - `obs_times`: `FloatTensor (B, L_max)` —— 归一化时间戳 $t / 365 \in [0, 1)$
  - `obs_var_ids`: `LongTensor (B, L_max)` —— 变量 ID（`0..11`）
  - `obs_values`: `FloatTensor (B, L_max)` —— 经 `norm_stats.json` 标准化后的观测值
  - `obs_is_driver`: `BoolTensor (B, L_max)` —— 区分状态与驱动
  - `obs_attn_mask`: `BoolTensor (B, L_max)` —— `True` 为有效 Token，`False` 为 Padding
  - `query_times`: `FloatTensor (B, Q)` —— 需要预测/重构的查询时刻
  - `target_x`: `FloatTensor (B, Q, 8)` & `target_r`: `FloatTensor (B, Q, 23)` & `target_state_mask`: `BoolTensor (B, 8)`（取 `effective_state_mask`）
- **反事实输入**：`intervention` 为 `FloatTensor (B, 11)`，记录四类动作的独热编码、归一化触发时刻、营养盐/温度/基流/暴雨的改变量和两种目标生物的脉冲改变量。基线视图全零。已知未来驱动覆盖六个日内采样相位。
- **GPU 加载配置**：`num_workers=4, persistent_workers=True, collate_fn=ecoverse_collate`。CUDA 压测开启 `pin_memory=True`，本机 MPS 压测关闭页锁定内存并独立记录设备。

---

## 7. Benchmark 四道考题与评测规范 (`bench/`)

建议 `v0` 首版规模生成 **6,000 个合格系统**，按以下严格无泄露规则锁定划分（`splits.py`）：

| 划分名称 | 数量 | 生成规则 | 对应考题与核心指标 |
|---|---|---|---|
| `train` / `val` | 4,000 / 400 | 5 大标准预设；允许使用**饱和型 + S 型阈值**（`MONOD`+`HILL`、`SMITH`+`LINEAR`、`Q10`+`CONST`），扩大训练动力学先验（C5 修复） | 预训练与早停 |
| `test_1_obs` | 300 | 同分布新系统，为每个系统绑定 3 套独立生成的锁定传感器套餐（套餐 A / B / C） | **考题 1（跨观测反推与对齐）**：未观测有效状态 NRMSE、过程通量 $\mathbf{r}(t)$ 的 $R^2$、同系统异观测潜空间余弦相似度 |
| `test_2a_param` | 300 | 5 大标准预设，全新随机参数种子 | **考题 2a（新参数泛化）**：零样本/少样本预测 NRMSE |
| `test_2b_comp` | 400 | **训练未见过的 4 类通道混搭**（每类各 100 个，见下表 B6 清单） | **考题 2b（组合泛化）**：未见结构组合上的预测与通量反推误差 |
| `test_2c_shape` | 300 | **训练未见过的“单峰/高值抑制”形状族**：替换为 `HALDANE(高值抑制)` + `STEELE(强光抑制)` + `GAUSSIAN(最适温度单峰)` | **考题 2c（方程错配稳健性）**：单调形状训练向非单调单峰形状泛化的衰减幅度 |
| `test_3_interv` | 300 | 新系统的反事实分支，$t \ge 180$ 施加 **5.3 表中分布外强度的 4 类干预** | **考题 3（因果干预推演）**：干预后轨迹 NRMSE + 相对 Baseline 稳态变化方向准确率 |
| **全测试集探针** | 复用上述 | 冻结预训练编码器，在 GPU 上拟合岭回归 $z_t \to \mathbf{r}(t)$，**并同时测试相同架构的“随机初始化未训练编码器”作为基线对照（B7 修复）** | **考题 4（潜空间机制理解度）**：23 条通量与 GPP/ER 的线性探针 $R^2_{\text{pretrain}}$ 及相对随机基线的净增益 $\Delta R^2 = R^2_{\text{pretrain}} - R^2_{\text{random}}$ |

各通道均报告指标适用性。五个预设和四类混搭均关闭通道 19，其全库真值恒为零。目标方差不足时 R² 无定义，记为 null 并从均值剔除。

状态 NRMSE 在训练集标准化后的空间计算。整体分数使用有效条目的 RMSE 除以同组真值标准差，分母下限为 0.1。考题 1 只纳入该系统对应套餐实际未观测的有效状态。考题 3 在干预后均匀取 32 个查询点，对末四分之一查询点的标准化状态差取均值，仅当真实差值绝对值超过 0.05 时计入方向准确率。

### 考题 2b（`test_2b_comp`）留出组合完整清单（B6 修复）

训练集中 5 大预设各自只包含其标准通道；`test_2b_comp` 专门构造以下 **4 类跨预设模块混搭**（各 100 个系统，共 400 个）：
1. **`river_competition`（河流 + 双生产者竞争）**：在 `river` 基础上开启 $P_2$ 及通道 `1, 3, 5, 9, 10`（测试急流冲刷与同级竞争的未见耦合）；
2. **`lake_cascade`（浅水湖泊 + 三级营养级联）**：在 `lake` 基础上开启顶级捕食者 $Z_2$ 及通道 `8, 12, 14`（测试顶级捕食引起的藻华/清水稳态级联）；
3. **`terrestrial_cascade`（陆地系统 + 三级营养级联）**：在 `terrestrial` 基础上开启 $Z_2$ 及通道 `8, 12, 14`（测试陆地慢周转碳库与多级食物链耦合）；
4. **`chemostat_detritus_recycle`（恒化器 + 碎屑内循环）**：在 `chemostat` 基础上开启碎屑池 $D$ 及分解通道 `15`（测试受控稀释系统加入内部物质再循环后的动态变化）。

---

## 8. `v0` 四阶段开发与验收计划（4–6 周稳健落地，C6 修复）

### Sprint 1（第 1 周）：数学核心闭环
- [ ] 初始化仓库结构与 `pyproject.toml`，配置 Python 3.11 + Numba + PyTorch CUDA 环境
- [ ] 实现 `core/constants.py`, `shapes.py`, `forcing.py`, `spec.py`, `dynamics.py`
- [ ] **验收标准**：
  1. 运行 `tests/test_shapes.py` 与 `test_forcing.py`：验证资源响应在 $x=0$ 归零、Monod 和 Hill 在 $x=K$ 取半值、Haldane 在其理论峰位达峰。固定云量和季节包络时光照在正午达峰，所有基线光照均在 $[0,1]$。检查日出、日落和暴雨起点处的一阶导数连续；
  2. 手动实例化 1 条河流与 1 个浅水湖泊，画出 365 天 8 个状态曲线与 GPP/ER 曲线，形态符合生态直觉；
  3. 启用全部八个状态的**三守恒律严格单测（B8 修复，`tests/test_conservation.py`）**：关闭外部与边界通道（通道 `16–22` 设 $0$ 且 $q(t)\equiv 0$）后，以下 **3 个独立守恒量**在 365 天内的最大相对漂移均 $< 10^{-8}$：
     - 总碳守恒：$M_C(t) = P_1 + P_2 + Z_1 + Z_2 + D + C \equiv \text{const}$
     - 总氮守恒：$M_N(t) = \frac{1}{\alpha_N}N + P_1 + P_2 + Z_1 + Z_2 + D \equiv \text{const}$
     - 氧–碳氧化还原守恒：$M_{OC}(t) = O + \alpha_O C \equiv \text{const}$
  4. 运行 `tests/test_positivity.py`：极端参数压力测试中，原始浓度允许的数值下界为 −0.0001，落盘浓度全部非负。

### Sprint 2（第 2–3 周）：5 大预设 + 求解器 + 质检关卡
- [ ] 编写 `configs/presets.yaml`, `shapes.yaml`, `interventions.yaml`，实现 `sim/sampler.py`, `solver.py`, `quality.py`
- [ ] **验收标准**：
  1. 5 大预设各随机抽样 100 个系统，统计质检合格率——要求每类预设合格率 $\ge 50\%$（若某类过低，微调 `presets.yaml` 中的比例先验）；
  2. 为每类预设及 4 类考题 2b 混搭各随机抽 5 个系统画轨迹图 + 反事实干预对比图，人工抽检确认 Baseline 与反事实分支在 $t < 180$ 完全重合、$t \ge 180$ 分化合理。

### Sprint 3（第 3–4 周）：Numba 并发生产与 Zarr 仓库落盘
- [ ] 实现 `sim/pipeline.py` 与 `scripts/generate_v0.py`, `scripts/inspect_data.py`
- [ ] **验收标准**：
  1. 运行 `tests/test_reproducibility.py`：同一 `master_seed` 下单进程与多进程生成结果逐比特完全一致（确认 Numba 未开启 `fastmath`）；
  2. **基准测速（C2 修复）**：`scipy.solve_ivp(LSODA)` + `@njit` 右端项下，单系统平均生成（含预热 + 正式 + 反事实）单核耗时 $< 2.0\ \text{s}$（8 核并行生成 6,000 个系统约 25–40 分钟；若需进一步提速至 $< 0.2\ \text{s}$，可切换后备方案 `numbalsoda`）；
  3. 一键生成完整 **6,000 个系统**，产出 `inspect_data` 体检报告（各预设占比、有效状态存活分布、动力学体制占比、参数覆盖度）。

### Sprint 4（第 5–6 周）：PyTorch GPU 管线 + Benchmark 考场封版
- [ ] 编写 `configs/sensors.yaml`, `benchmark_v0.yaml`
- [ ] 实现 `obs/normalize.py`, `sensors.py`, `dataset.py`, `collate.py` 与 `bench/splits.py`, `metrics.py`, `probe.py`
- [ ] **验收标准**：
  1. 确认 `norm_stats.json` 仅由 `train` 集生成，且 `splits.json` 各划分系统 ID 无交集；
  2. 运行 `tests/test_torch_loader.py`：`DataLoader(batch_size=64, num_workers=4, pin_memory=True)` 连续迭代 200 个 Batch 送入 CUDA，无锁死、无 NaN、无显存泄漏，确认张量形状与 `obs_attn_mask` 完全对齐；
  3. 用一个简单 Baseline（如小型 MLP / GRU）和随机初始化对照跑通 `scripts/run_eval.py`，确认四道考题打分表与 `probe.py`（输出 $R^2_{\text{pretrain}}$ 与 $\Delta R^2$）能端到端出分；
  4. **打 Git Tag `v0.1.0`** —— `EcoVerse v0` 正式完工，转入你的第一个 Transformer / Latent ODE 模型实验！

---

## 9. 给 `v1` 预留的三条小约定（写 `v0` 时顺手守住）

1. **变量与通道全部走元数据表驱动**：`obs/dataset.py` 和 `obs/collate.py` 不要把 `8` 和 `23` 写死在张量切片逻辑里，统一从 `metadata` 或数组 `.shape[-1]` 读取，这样 `v1` 扩到 30 个物种时 PyTorch 数据管线**一行都不用改**；
2. **`SystemSpec` 存的是稀疏边列表/字典描述**：在 JSON 层保持语义化描述，只在 `compile_spec()` 阶段才摊平成 `v0` 的定长数组，以后 `v1` 只需换一个 `compile_sparse_spec()`；
3. **事件流 `var_id` 采用连续整数分配**：保留 `0..N_state-1` 为状态池、负数或偏移区为外部驱动，天然适配任意维度的 Niche Model 生态网。
