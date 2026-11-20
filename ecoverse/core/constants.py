"""Core constants, indices, shape enums, and macro flux utilities for EcoVerse."""

from __future__ import annotations

from typing import Tuple, TypeVar

import numpy as np

# ---------------------------------------------------------------------------
# State vector layout (v0: 8 dimensions)
# ---------------------------------------------------------------------------
STATE_NAMES: Tuple[str, ...] = ("N", "P1", "P2", "Z1", "Z2", "D", "O", "C")
IDX_N: int = 0
IDX_P1: int = 1
IDX_P2: int = 2
IDX_Z1: int = 3
IDX_Z2: int = 4
IDX_D: int = 5
IDX_O: int = 6
IDX_C: int = 7
N_STATES: int = 8

BIOTIC_INDICES: Tuple[int, ...] = (IDX_P1, IDX_P2, IDX_Z1, IDX_Z2)
PRODUCER_INDICES: Tuple[int, ...] = (IDX_P1, IDX_P2)
ABIOTIC_INDICES: Tuple[int, ...] = (IDX_N, IDX_D, IDX_O, IDX_C)

# ---------------------------------------------------------------------------
# External driver layout (4 dimensions)
# ---------------------------------------------------------------------------
DRIVER_NAMES: Tuple[str, ...] = ("I", "T", "q", "N_in")
DRV_I: int = 0
DRV_T: int = 1
DRV_Q: int = 2
DRV_NIN: int = 3
N_DRIVERS: int = 4

# ---------------------------------------------------------------------------
# Reaction channel layout (v0: 23 atomic channels)
# ---------------------------------------------------------------------------
N_CHANNELS: int = 23
CHANNEL_NAMES: Tuple[str, ...] = (
    "prod_P1",          # 0: Autotrophic production P1
    "prod_P2",          # 1: Autotrophic production P2
    "resp_P1",          # 2: Producer respiration P1
    "resp_P2",          # 3: Producer respiration P2
    "loss_P1",          # 4: Producer mortality/exudation P1
    "loss_P2",          # 5: Producer mortality/exudation P2
    "graz_P1_Z1",       # 6: Primary grazing P1 -> Z1
    "graz_P2_Z1",       # 7: Primary grazing P2 -> Z1
    "pred_Z1_Z2",       # 8: Secondary predation Z1 -> Z2
    "comp_P2_on_P1",    # 9: Interference competition P2 -| P1
    "comp_P1_on_P2",    # 10: Interference competition P1 -| P2
    "resp_Z1",          # 11: Consumer respiration Z1
    "resp_Z2",          # 12: Consumer respiration Z2
    "loss_Z1",          # 13: Consumer mortality Z1
    "loss_Z2",          # 14: Consumer mortality Z2
    "decomp_D",         # 15: Detritus mineralization
    "gas_O",            # 16: Oxygen gas exchange
    "gas_C",            # 17: Carbon gas exchange
    "sink_P1",          # 18: Sinking P1
    "sink_P2",          # 19: Sinking P2
    "burial_D",         # 20: Detritus burial
    "sod_O_C",          # 21: Sediment O2 demand / DIC production
    "anoxic_N_rel",     # 22: Anoxic internal N release
)

GPP_CHANNELS: Tuple[int, ...] = (0, 1)
SIGNED_FLUX_CHANNELS: Tuple[int, ...] = (16, 17)
ER_CHANNELS: Tuple[int, ...] = (2, 3, 11, 12, 15, 21)
CLOSED_CHANNELS: Tuple[int, ...] = tuple(range(16))
OPEN_CHANNELS: Tuple[int, ...] = tuple(range(16, 23))
P2_CHANNELS: Tuple[int, ...] = (1, 3, 5, 7, 9, 10, 19)
Z2_CHANNELS: Tuple[int, ...] = (8, 12, 14)

# ---------------------------------------------------------------------------
# Physical stoichiometry & temporal grid constants
# ---------------------------------------------------------------------------
ALPHA_N: float = 0.15       # Model N:C mass ratio (g N / g C)
ALPHA_O: float = 2.67       # Model O2:C mass ratio (g O2 / g C)
GATE_EPS: float = 0.1       # Half-saturation for smooth positivity gating g_O, g_C

BURNIN_DAYS: float = 180.0
FORMAL_DAYS: float = 365.0
INTERVENTION_DAY: float = 180.0
DT_DAYS: float = 1.0 / 6.0  # 4-hour output resolution

N_STEPS_FORMAL: int = 2190  # k = 0..2189 -> t_k = k / 6.0
INTERVENTION_STEP: int = 1080  # t_1080 = 180.0 d
N_STEPS_CF: int = 1110      # k = 1080..2189

# ---------------------------------------------------------------------------
# Shape family integer enum IDs (matching spec Section 4.3)
# ---------------------------------------------------------------------------
RES_MONOD: int = 0
RES_HILL: int = 1
RES_HALDANE: int = 2

LIGHT_SMITH: int = 0
LIGHT_STEELE: int = 1
LIGHT_LINEAR: int = 2

TEMP_Q10: int = 0
TEMP_GAUSSIAN: int = 1
TEMP_CONST: int = 2

LOSS_LINEAR: int = 0
LOSS_QUADRATIC: int = 1

DEC_LINEAR: int = 0
DEC_SATURATING: int = 1

RES_SHAPE_MAP = {"MONOD": RES_MONOD, "HILL": RES_HILL, "HALDANE": RES_HALDANE}
LIGHT_SHAPE_MAP = {"SMITH": LIGHT_SMITH, "STEELE": LIGHT_STEELE, "LINEAR": LIGHT_LINEAR}
TEMP_SHAPE_MAP = {"Q10": TEMP_Q10, "GAUSSIAN": TEMP_GAUSSIAN, "CONST": TEMP_CONST}
LOSS_SHAPE_MAP = {"LINEAR": LOSS_LINEAR, "QUADRATIC": LOSS_QUADRATIC}
DEC_SHAPE_MAP = {"LINEAR": DEC_LINEAR, "SATURATING": DEC_SATURATING}

RES_SHAPE_INV = {v: k for k, v in RES_SHAPE_MAP.items()}
LIGHT_SHAPE_INV = {v: k for k, v in LIGHT_SHAPE_MAP.items()}
TEMP_SHAPE_INV = {v: k for k, v in TEMP_SHAPE_MAP.items()}
LOSS_SHAPE_INV = {v: k for k, v in LOSS_SHAPE_MAP.items()}
DEC_SHAPE_INV = {v: k for k, v in DEC_SHAPE_MAP.items()}

# ---------------------------------------------------------------------------
# Intervention integer enum IDs
# ---------------------------------------------------------------------------
INTERV_NONE: int = 0
INTERV_NUTRIENT_STEP: int = 1
INTERV_WARMING_SHIFT: int = 2
INTERV_FLOW_REGIME: int = 3
INTERV_BIO_PULSE: int = 4

INTERV_MAP = {
    "none": INTERV_NONE,
    "nutrient_step": INTERV_NUTRIENT_STEP,
    "warming_shift": INTERV_WARMING_SHIFT,
    "flow_regime": INTERV_FLOW_REGIME,
    "bio_pulse": INTERV_BIO_PULSE,
}
INTERV_INV = {v: k for k, v in INTERV_MAP.items()}

# One-hot action, normalized day and offsets from the no-action controls.
INTERVENTION_FEATURES = (
    "nutrient_step", "warming_shift", "flow_regime", "bio_pulse", "t_int",
    "lambda_N_minus_one", "delta_T", "lambda_q_minus_one", "lambda_storm_minus_one",
    "kappa_P1_minus_one", "kappa_Z1_minus_one",
)

ArrayLike = TypeVar("ArrayLike", bound=object)


def compute_macro_fluxes(r: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute GPP, ER, and NEP from 23-channel flux array/tensor.

    Works identically for NumPy arrays and PyTorch tensors of shape (..., >=22).
    """
    gpp = r[..., GPP_CHANNELS[0]]
    for idx in GPP_CHANNELS[1:]:
        gpp = gpp + r[..., idx]

    er = r[..., ER_CHANNELS[0]]
    for idx in ER_CHANNELS[1:]:
        er = er + r[..., idx]

    nep = gpp - er
    return gpp, er, nep


def make_formal_time_grid() -> np.ndarray:
    """Return formal 365-day output time grid (2190 points at dt=1/6 d)."""
    return np.arange(N_STEPS_FORMAL, dtype=np.float64) * DT_DAYS
