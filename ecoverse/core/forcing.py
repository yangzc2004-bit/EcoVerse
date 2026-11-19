"""Smooth analytical external forcing u(t) and intervention injection."""

from __future__ import annotations

import math
from typing import NamedTuple, Tuple

import numpy as np
from numba import njit

from ecoverse.core.constants import (
    INTERV_FLOW_REGIME,
    INTERV_NUTRIENT_STEP,
    INTERV_WARMING_SHIFT,
)

TWO_PI: float = 2.0 * math.pi


class CompiledForcing(NamedTuple):
    """Numba-friendly flat container for continuous forcing parameters."""

    diel: bool
    seasonal: bool
    a_I: float
    b_c: float
    cloud_w: np.ndarray     # float64 (8,)
    cloud_f: np.ndarray     # float64 (8,)
    cloud_psi: np.ndarray   # float64 (8,)
    T_bar: float
    A_season: float
    A_diel: float
    lag_T: float
    temp_v: np.ndarray      # float64 (6,)
    temp_g: np.ndarray      # float64 (6,)
    temp_chi: np.ndarray    # float64 (6,)
    q_base: float
    storm_t: np.ndarray     # float64 (K_storm,)
    storm_A: np.ndarray     # float64 (K_storm,)
    storm_tau: np.ndarray   # float64 (K_storm,)
    N_in_0: float
    D_in: float
    C_sat: float
    interv_type: int
    interv_t: float
    interv_lambda_N: float
    interv_delta_T: float
    interv_lambda_q: float
    interv_lambda_storm: float


@njit(fastmath=False, cache=True)
def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@njit(fastmath=False, cache=True)
def eval_light(t: float, f: CompiledForcing) -> float:
    """Normalized irradiance I(t) in [0, 1], daytime 6:00-18:00 peaking at noon."""
    if f.seasonal:
        seasonal_fac = (1.0 - f.a_I * math.cos(TWO_PI * t / 365.0)) / (1.0 + f.a_I)
    else:
        seasonal_fac = 1.0

    if f.diel:
        s = math.sin(TWO_PI * (t - 0.25))
        diel_fac = s * s if s > 0.0 else 0.0
    else:
        diel_fac = 0.25

    z = f.b_c
    for k in range(f.cloud_w.shape[0]):
        z += f.cloud_w[k] * math.sin(TWO_PI * f.cloud_f[k] * t + f.cloud_psi[k])
    c_cloud = 0.2 + 0.8 * _sigmoid(z)

    val = seasonal_fac * diel_fac * c_cloud
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


@njit(fastmath=False, cache=True)
def eval_temp(t: float, f: CompiledForcing) -> float:
    """Water/air temperature T(t) in deg C, seasonal lag & 15:00 diel peak."""
    T = f.T_bar
    if f.seasonal:
        T -= f.A_season * math.cos(TWO_PI * (t - f.lag_T) / 365.0)
    if f.diel:
        T += f.A_diel * math.sin(TWO_PI * (t - 0.375))
    for k in range(f.temp_v.shape[0]):
        T += f.temp_v[k] * math.sin(TWO_PI * f.temp_g[k] * t + f.temp_chi[k])

    if f.interv_type == INTERV_WARMING_SHIFT and t >= f.interv_t:
        T += f.interv_delta_T
    return T


@njit(fastmath=False, cache=True)
def eval_flow(t: float, f: CompiledForcing) -> float:
    """Dilution / flushing rate q(t) (d^-1) with C^1 quadratic storm pulses."""
    q_base = f.q_base
    storm_scale = 1.0
    if f.interv_type == INTERV_FLOW_REGIME and t >= f.interv_t:
        q_base *= f.interv_lambda_q
        storm_scale = f.interv_lambda_storm

    q_storm = 0.0
    for k in range(f.storm_t.shape[0]):
        tk = f.storm_t[k]
        if t > tk:
            u = (t - tk) / f.storm_tau[k]
            if u < 30.0:
                q_storm += storm_scale * f.storm_A[k] * (u * u) * math.exp(2.0 - 2.0 * u)
    return q_base + q_storm


@njit(fastmath=False, cache=True)
def eval_n_in(t: float, f: CompiledForcing) -> float:
    """External nutrient inflow concentration N_in(t) (g N m^-3)."""
    n_in = f.N_in_0
    if f.interv_type == INTERV_NUTRIENT_STEP and t >= f.interv_t:
        n_in *= f.interv_lambda_N
    return n_in


@njit(fastmath=False, cache=True)
def eval_forcing(t: float, f: CompiledForcing) -> Tuple[float, float, float, float]:
    """Evaluate all 4 analytical forcing variables (I, T, q, N_in) at time t."""
    return eval_light(t, f), eval_temp(t, f), eval_flow(t, f), eval_n_in(t, f)
