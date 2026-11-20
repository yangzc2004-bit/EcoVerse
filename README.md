# EcoVerse

[English](README.md) | [简体中文](README_zh.md)

**EcoVerse** (`ecoverse`) is a composable stock–flow synthetic ecosystem engine and standardized PyTorch/GPU benchmark designed for learning universal ecosystem dynamics, cross-sensor state reconstruction, mechanism probing, and counterfactual intervention forecasting.

---

## Key Features

- **Unified Stock–Flow Core (`ecoverse/core/`)**:
  - 8 state pools: inorganic nutrients $N$, primary producers $P_1, P_2$, consumers $Z_1, Z_2$, detritus $D$, dissolved oxygen $O$, and inorganic carbon $C$.
  - 23 atomic reaction channels covering autotrophic production, respiration, mortality, grazing/predation, interference competition, detritus decomposition, bidirectional gas exchange ($O, C$), sinking/burial, sediment oxygen demand, and anoxic $N$ release (plus derived macro fluxes `GPP`, `ER`, `NEP`).
  - Smooth analytical continuous-time forcing $\mathbf{u}(t) = (I, T, q, N_{\text{in}})$ without piecewise interpolation corners, compiled with Numba (`fastmath=False`) for cross-process bit-exact reproducibility, alongside a differentiable PyTorch extension (`torch_dynamics.py`).
  - Strict physical invariants: total carbon $M_C$, total nitrogen $M_N$, and oxygen–carbon redox balance $M_{OC}$ hold to $< 10^{-8}$ relative drift under closed boundaries, with compile-time gate bypasses for inactive abiotic pools.
- **5 Ecosystem Presets & 4 Structural Combinations (`configs/`)**:
  - Standard presets: `river`, `lake`, `terrestrial`, `foodweb`, and `chemostat`.
  - Held-out `test_2b_comp` combinations: `river_competition`, `lake_cascade`, `terrestrial_cascade`, and `chemostat_detritus_recycle`.
- **Offline Multiprocessing Production & Zarr Warehouse (`ecoverse/sim/`)**:
  - `SciPy solve_ivp(method="LSODA")` with 180-day burn-in, 365-day formal recording ($\Delta t = 1/6\ \text{d}$, $T = 2190$), and $t = 180\ \text{d}$ counterfactual branching ($T_{\text{cf}} = 1110$) sharing identical stochastic weather/cloud/storm realizations.
  - 4 quality gates with competitive-exclusion handling (`effective_state_mask`), dynamical regime tagging (`forced_periodic`, `endogenous_oscillation`, `regime_shift`), chunked Zarr storage (`chunk=(32, T, D)`), Parquet metadata, and atomic publish.
- **Heterogeneous Observation Pipeline (`ecoverse/obs/`)**:
  - Irregular, noisy, multi-sensor event-stream tokenization (`obs_times`, `obs_var_ids`, `obs_values`, `obs_is_driver`, `obs_attn_mask`) with train-only normalization (`norm_stats.json`) and worker-safe lazy Zarr reading.
  - Supports factual training, counterfactual training (`counterfactual_train`), and standardized evaluation packages (`A`: high-frequency water quality, `B`: irregular ecological sampling, `C`: single-variable remote sensing/buoy).
- **Standardized 4-Test Benchmark & GPU Linear Probe (`ecoverse/bench/`)**:
  - Locked zero-leakage splits (`6,000` systems total by default) testing cross-observation alignment, parameter generalization, structural composition, held-out unimodal functional responses, out-of-distribution counterfactuals, and latent mechanism probing against an untrained random-initialization control.

---

## Repository Layout

```text
ecoverse/
├── pyproject.toml               # Package metadata and dependencies (Python 3.11–3.13)
├── configs/                     # Experiment configurations (presets, shapes, interventions, sensors, splits)
├── ecoverse/
│   ├── core/                    # Math core: constants, shapes, forcing, SystemSpec, Numba & PyTorch kernels
│   ├── sim/                     # Sampler, LSODA solver, quality gates, multiprocessing Zarr pipeline
│   ├── obs/                     # Train-only normalizer, virtual sensors, EcoVerseDataset, collate, Gym skeleton
│   └── bench/                   # SplitManager, masked NRMSE / flux R² / alignment / CF metrics, GPU Ridge probe
├── scripts/
│   ├── generate_v0.py           # Offline dataset generation & preset calibration CLI
│   ├── verify_v0.py             # Release verification & SHA-256 integrity checks
│   ├── inspect_data.py          # Dataset health report & diagnostic trajectory plots
│   └── run_eval.py              # End-to-end baseline training + 4-test GPU Benchmark evaluation
├── tests/                       # Unit & integration test suite
├── ecoverse_dev_plan.md         # Architecture & mathematical specification (rev3)
└── acceptance_v0.md             # Local 6,000-system verification & acceptance record
```

---

## Installation

`EcoVerse` targets Python `3.11`–`3.13` (tested on `3.12`) and uses [`uv`](https://github.com/astral-sh/uv) for environment management:

```bash
git clone https://github.com/yangzc2004-bit/EcoVerse.git
cd EcoVerse

# Install dependencies and dev extras
uv sync --extra dev

# Run the automated test suite
uv run pytest -v
```

---

## End-to-End Workflow

### 1. (Optional) Calibrate Preset Pass Rates

```bash
uv run python scripts/generate_v0.py --calibrate --cal-n 100 --cal-base-only
```

### 2. Generate the Offline Ground-Truth Warehouse

```bash
# Quick 10% debug dataset (600 systems across all 7 splits)
uv run python scripts/generate_v0.py --scale 0.1 --workers 8 --output data/v0

# Full v0 benchmark dataset (6,000 systems)
uv run python scripts/generate_v0.py --scale 1.0 --workers 8 --output data/v0
```

This creates `data/v0/` containing `trajectories.zarr/`, `metadata.parquet`, `splits.json`, `norm_stats.json`, and `generation_summary.json`.

### 3. Verify & Inspect the Generated Dataset

```bash
uv run python scripts/verify_v0.py --data-dir data/v0 --output outputs/verify.json
uv run python scripts/inspect_data.py --data-dir data/v0 --plot-dir outputs/inspect
```

### 4. Plug Into Your PyTorch Training Loop

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
    mode="train",              # or "counterfactual_train" for mild-intervention views
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
    # Event-stream inputs:
    #   obs_times:     FloatTensor (B, L_max)  normalized t / 365 in [0, 1]
    #   obs_var_ids:   LongTensor  (B, L_max)  0..7 states, 8..11 drivers
    #   obs_values:    FloatTensor (B, L_max)  standardized via train-only norm_stats.json
    #   obs_is_driver: BoolTensor  (B, L_max)
    #   obs_attn_mask: BoolTensor  (B, L_max)
    # Targets & masks:
    #   query_times:       FloatTensor (B, Q)
    #   target_x:          FloatTensor (B, Q, 8)   standardized states
    #   target_r:          FloatTensor (B, Q, 23)  physical reaction fluxes
    #   target_state_mask: BoolTensor  (B, 8)      effective_state_mask (use in state loss!)
    #   channel_mask:      BoolTensor  (B, 23)     active channels (use in flux loss!)
    #   intervention:      FloatTensor (B, 11)     present in counterfactual / counterfactual_train modes
    ...
```

### 5. Run the Standardized 4-Test Benchmark

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

## Benchmark Splits & Tasks (`v0`)

| Split | Systems | Task & Evaluation Protocol |
|---|---:|---|
| `train` / `val` | 4,000 / 400 | 5 standard presets (`MONOD`/`HILL`, `SMITH`/`LINEAR`, `Q10`/`CONST`) + mild interventions |
| `test_1_obs` | 300 | **Test 1 (Cross-Observation Alignment & Reconstruction)**: Unobserved-state NRMSE, flux $R^2$, and latent cosine similarity across sensor packages `A`, `B`, `C` |
| `test_2a_param` | 300 | **Test 2a (Unseen Parameters)**: Zero-shot state NRMSE and flux $R^2$ on unseen parameter draws |
| `test_2b_comp` | 400 | **Test 2b (Compositional Generalization)**: 4 unseen cross-preset structural combinations (`river_competition`, `lake_cascade`, `terrestrial_cascade`, `chemostat_detritus_recycle`) |
| `test_2c_shape` | 300 | **Test 2c (Equation Mismatch)**: Held-out unimodal functional responses (`HALDANE`, `STEELE`, `GAUSSIAN`) |
| `test_3_interv` | 300 | **Test 3 (Counterfactual Forecasting)**: Post-$t=180\ \text{d}$ out-of-distribution interventions (`CF NRMSE` & steady-state direction accuracy) |
| **Linear Probe** | All test splits | **Test 4 (Mechanism Understanding)**: GPU Ridge probe $z_t \to (\mathbf{r}(t), \text{GPP}, \text{ER}, \text{NEP})$ reporting $R^2_{\text{pretrain}}$, $R^2_{\text{random}}$, and $\Delta R^2$ |

See [ecoverse_dev_plan.md](ecoverse_dev_plan.md) for the full mathematical specification and [acceptance_v0.md](acceptance_v0.md) for local 6,000-system verification logs.
