# EcoVerse

[English](README.md) | [简体中文](README_zh.md)

**EcoVerse**（Python 包名：`ecoverse`）是一个面向跨生态系统底层动力学理解的**可组合合成数据引擎**与**标准化 PyTorch/GPU Benchmark**，用于训练和评测事件流生态世界模型、跨观测状态重构、过程机制可解释性以及反事实干预推演。

---

## 核心特性

- **统一存量–流量数学内核（`ecoverse/core/`）**：
  - **8 维状态存量**：无机营养盐 $N$、两类生产者 $P_1, P_2$、两级消费者 $Z_1, Z_2$、碎屑 $D$、溶解氧 $O$、宏观无机碳 $C$。
  - **23 条原子反应通道**：涵盖自养生产、呼吸、凋亡、捕食/次级捕食、同级竞争干扰、碎屑分解、双向氧/碳气液交换、颗粒沉降/埋藏、底泥耗氧与低氧内源释氮，并自动汇总宏观通量 `GPP`、`ER`、`NEP`。
  - **光滑连续解析驱动**：光照 $I(t)$、温度 $T(t)$、带平方型暴雨脉冲的冲刷率 $q(t)$ 与外源输入 $N_{\text{in}}(t)$ 均直接求值（无插值尖角）；基于 Numba（禁用 `fastmath`）实现跨进程逐比特一致，另提供可微 PyTorch GPU 版本（`torch_dynamics.py`）。
  - **严格物理守恒与门控旁路**：闭边界下总碳 $M_C$、总氮 $M_N$、氧–碳氧化还原守恒量 $M_{OC}$ 相对漂移 $< 10^{-8}$；不启用非生物池时在编译期将对应门控严格旁路为 `1`。
- **5 大生态预设与 4 类未见结构混搭（`configs/`）**：
  - 标准预设：`river`（河流）、`lake`（浅水湖泊）、`terrestrial`（陆地）、`foodweb`（经典食物网）、`chemostat`（恒化器）。
  - 考题 2b 组合混搭：`river_competition`、`lake_cascade`、`terrestrial_cascade`、`chemostat_detritus_recycle`。
- **离线并发生产与 Zarr 真值仓库（`ecoverse/sim/`）**：
  - 采用 `SciPy solve_ivp(method="LSODA")` 执行 180 天预热 → 365 天正式记录（$\Delta t = 1/6\ \text{d}$，共 2,190 点）→ 第 180 天平行反事实分叉（共 1,110 点，底层天气与暴雨序列与基线完全相同）。
  - 内置 4 条硬质检关卡，合法保留竞争排斥样本并记录 `effective_state_mask`，自动标注动力学体制（`forced_periodic` / `endogenous_oscillation` / `regime_shift`），支持 `(32, T, D)` 分块 Zarr + Parquet 存储与原子发布。
- **异构事件流观测管线与防泄露标准化（`ecoverse/obs/`）**：
  - 现场将真值轨迹切片为带噪、不规则采样、含块状掉线缺测的事件流 Token（`obs_times`, `obs_var_ids`, `obs_values`, `obs_is_driver`, `obs_attn_mask`），严格仅使用训练集统计 `norm_stats.json`。
  - 支持事实训练（`train`）、温和反事实训练（`counterfactual_train`）及 A（高频水质站）/ B（低频生态采样）/ C（单变量遥感浮标）三套锁定观测套餐。
- **四道标准化 Benchmark 考题与 GPU 线性探针（`ecoverse/bench/`）**：
  - 覆盖跨观测反推与潜空间对齐、新参数泛化、新结构组合泛化、单峰函数形状错配、分布外强干预反事实推演，以及对比同架构随机初始化模型的潜空间 Ridge 线性探针（$\Delta R^2$）。

---

## 仓库目录结构

```text
ecoverse/
├── pyproject.toml               # 包配置与依赖声明（Python 3.11–3.13）
├── configs/                     # 实验配置文件（预设、形状族、干预强度、传感器套餐、划分）
├── ecoverse/
│   ├── core/                    # 数学核心层：常数、形状算子、解析驱动、SystemSpec、Numba/PyTorch 内核
│   ├── sim/                     # 采样器、LSODA 求解器、质检与体制打标、多进程 Zarr 生产管线
│   ├── obs/                     # 训练集标准化、虚拟监测站、EcoVerseDataset、collate 与 Gym 骨架
│   └── bench/                   # SplitManager、NRMSE/R²/对齐/反事实指标、GPU Ridge 线性探针
├── scripts/
│   ├── generate_v0.py           # 离线批量生成 v0 真值库与预设合格率校准入口
│   ├── verify_v0.py             # 全库规格校验与 SHA-256 完整性核验
│   ├── inspect_data.py          # 数据体检报告与轨迹/干预对比图生成
│   └── run_eval.py              # 一键基线训练 + Benchmark 四道考题端到端评测
├── tests/                       # 自动化单元测试与集成测试
├── ecoverse_dev_plan.md         # v0 架构设计与数学规格说明书（rev3）
└── acceptance_v0.md             # 本机 6,000 系统全量验收记录
```

---

## 安装与环境自检

推荐使用 Python `3.12`（已在 `.python-version` 指定）与 [`uv`](https://github.com/astral-sh/uv)：

```bash
git clone https://github.com/yangzc2004-bit/EcoVerse.git
cd EcoVerse

# 安装运行依赖与测试依赖
uv sync --extra dev

# 运行全套单元测试与集成测试
uv run pytest -v
```

---

## 从零到测评完整使用流程

### 1.（可选）校准各预设合格率与耗时

```bash
uv run python scripts/generate_v0.py --calibrate --cal-n 100 --cal-base-only
```

### 2. 离线生成真值仓库

```bash
# 快速 10% 验证集（600 个系统，覆盖全部 7 个划分）
uv run python scripts/generate_v0.py --scale 0.1 --workers 8 --output data/v0

# 正式全量 v0 真值库（6,000 个系统）
uv run python scripts/generate_v0.py --scale 1.0 --workers 8 --output data/v0
```

生成完成后，`data/v0/` 下将包含 `trajectories.zarr/`、`metadata.parquet`、`splits.json`、`norm_stats.json` 与 `generation_summary.json`。

### 3. 全库校验与可视化体检

```bash
uv run python scripts/verify_v0.py --data-dir data/v0 --output outputs/verify.json
uv run python scripts/inspect_data.py --data-dir data/v0 --plot-dir outputs/inspect
```

### 4. 在 PyTorch 中接入训练

```python
import torch
from torch.utils.data import DataLoader
from ecoverse.bench.splits import SplitManager
from ecoverse.obs.collate import ecoverse_collate
from ecoverse.obs.dataset import EcoVerseDataset

splits = SplitManager("data/v0")

train_ds = EcoVerseDataset(
    data_dir="data/v0",
    sys_ids=splits.get_ids("train"),
    mode="train",              # 也可配合 "counterfactual_train" 一起训练温和干预响应
    sensor_package="random",
    num_queries=64,
    max_tokens=1024,
)

loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    collate_fn=ecoverse_collate,
)

for batch in loader:
    # 变长事件流输入:
    #   obs_times:     FloatTensor (B, L_max)  归一化时刻 t / 365 ∈ [0, 1]
    #   obs_var_ids:   LongTensor  (B, L_max)  0..7 为状态, 8..11 为驱动
    #   obs_values:    FloatTensor (B, L_max)  经训练集 norm_stats.json 标准化
    #   obs_is_driver: BoolTensor  (B, L_max)
    #   obs_attn_mask: BoolTensor  (B, L_max)
    # 查询与监督目标:
    #   query_times:       FloatTensor (B, Q)
    #   target_x:          FloatTensor (B, Q, 8)   标准化状态真值
    #   target_r:          FloatTensor (B, Q, 23)  物理量纲过程通量真值
    #   target_state_mask: BoolTensor  (B, 8)      effective_state_mask（算状态损失必用）
    #   channel_mask:      BoolTensor  (B, 23)     开启通道掩码（算通量损失必用）
    #   intervention:      FloatTensor (B, 11)     反事实模式下提供已知干预编码
    ...
```

### 5. 运行 Benchmark 四道考题端到端评测

```bash
uv run python scripts/run_eval.py \
  --data-dir data/v0 \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 4 \
  --device auto \
  --output outputs/eval_report.json
```

---

## Benchmark 数据划分与四道考题说明（`v0`）

| 划分 | 系统数 | 对应考题与核心指标 |
|---|---:|---|
| `train` / `val` | 4,000 / 400 | 5 大标准预设（饱和型 + S 型形状）与温和干预，用于预训练与早停 |
| `test_1_obs` | 300 | **考题 1（跨观测反推与对齐）**：分别在套餐 `A`、`B`、`C` 下评估未观测有效状态 NRMSE、通量 $R^2$ 与同系统潜空间余弦相似度 |
| `test_2a_param` | 300 | **考题 2a（新参数泛化）**：全新随机参数种子上的零样本预测 NRMSE 与通量 $R^2$ |
| `test_2b_comp` | 400 | **考题 2b（组合泛化）**：4 类未见跨预设混搭（`river_competition`、`lake_cascade`、`terrestrial_cascade`、`chemostat_detritus_recycle`） |
| `test_2c_shape` | 300 | **考题 2c（方程错配稳健性）**：替换为训练集未见过的单峰抑制形状（`HALDANE`、`STEELE`、`GAUSSIAN`） |
| `test_3_interv` | 300 | **考题 3（因果干预推演）**：$t \ge 180\ \text{d}$ 施加分布外强干预后的轨迹 `CF NRMSE` 与稳态变化方向准确率 |
| **线性探针** | 全部测试集 | **考题 4（潜空间机制理解度）**：GPU 岭回归 $z_t \to (\mathbf{r}(t), \text{GPP}, \text{ER}, \text{NEP})$，输出 $R^2_{\text{pretrain}}$、随机对照 $R^2_{\text{random}}$ 与净增益 $\Delta R^2$ |

更详细的数学定义与开发规范见 [ecoverse_dev_plan.md](ecoverse_dev_plan.md)，本机 6,000 系统全量核验记录见 [acceptance_v0.md](acceptance_v0.md)。
