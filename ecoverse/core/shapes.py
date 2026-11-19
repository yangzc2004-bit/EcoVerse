"""Numba-compiled functional response shapes and positivity gates.

All functions strictly disable fastmath (fastmath=False) to ensure IEEE-754
reproducibility across processes.
"""

from __future__ import annotations

import math

from numba import njit

from ecoverse.core.constants import (
    DEC_LINEAR,
    DEC_SATURATING,
    GATE_EPS,
    LIGHT_LINEAR,
    LIGHT_SMITH,
    LIGHT_STEELE,
    LOSS_LINEAR,
    LOSS_QUADRATIC,
    RES_HALDANE,
    RES_HILL,
    RES_MONOD,
    TEMP_CONST,
    TEMP_GAUSSIAN,
    TEMP_Q10,
)


@njit(fastmath=False, cache=True)
def phi_res(x: float, shape_id: int, K: float, n_hill: float, K_I: float) -> float:
    """Resource uptake / predation functional response."""
    xp = x if x > 0.0 else 0.0
    if xp <= 0.0:
        return 0.0
    if shape_id == RES_MONOD:
        return xp / (K + xp)
    elif shape_id == RES_HILL:
        r = xp / K
        if r <= 1.0:
            rn = r ** n_hill
            return rn / (1.0 + rn)
        else:
            inv = (1.0 / r) ** n_hill
            return 1.0 / (1.0 + inv)
    elif shape_id == RES_HALDANE:
        return xp / (K + xp + (xp * xp) / K_I)
    return xp / (K + xp)


@njit(fastmath=False, cache=True)
def phi_light(I: float, shape_id: int, I_k: float, I_opt: float) -> float:
    """Irradiance limitation response."""
    Ip = I if I > 0.0 else 0.0
    if Ip <= 0.0:
        return 0.0
    if shape_id == LIGHT_SMITH:
        return Ip / math.sqrt(I_k * I_k + Ip * Ip)
    elif shape_id == LIGHT_STEELE:
        u = Ip / I_opt
        return u * math.exp(1.0 - u)
    elif shape_id == LIGHT_LINEAR:
        return Ip
    return Ip / math.sqrt(I_k * I_k + Ip * Ip)


@njit(fastmath=False, cache=True)
def phi_temp(T: float, shape_id: int, theta_T: float, T_opt: float, sigma_T: float) -> float:
    """Temperature scaling response."""
    if shape_id == TEMP_Q10:
        return theta_T ** ((T - 20.0) / 10.0)
    elif shape_id == TEMP_GAUSSIAN:
        d = (T - T_opt) / sigma_T
        return math.exp(-0.5 * d * d)
    elif shape_id == TEMP_CONST:
        return 1.0
    return 1.0


@njit(fastmath=False, cache=True)
def phi_loss(x: float, shape_id: int, m_a: float, m_b: float) -> float:
    """Density-dependent loss / mortality rate."""
    xp = x if x > 0.0 else 0.0
    if xp <= 0.0:
        return 0.0
    if shape_id == LOSS_LINEAR:
        return m_a * xp
    elif shape_id == LOSS_QUADRATIC:
        return m_a * xp + m_b * xp * xp
    return m_a * xp


@njit(fastmath=False, cache=True)
def psi_dec(D: float, shape_id: int, K_D: float) -> float:
    """Detritus mineralization substrate dependence (dimension-aligned at low D)."""
    Dp = D if D > 0.0 else 0.0
    if Dp <= 0.0:
        return 0.0
    if shape_id == DEC_LINEAR:
        return Dp
    elif shape_id == DEC_SATURATING:
        return (K_D * Dp) / (K_D + Dp)
    return Dp


@njit(fastmath=False, cache=True)
def gate_o(O: float, bypass: bool) -> float:
    """Smooth low-oxygen shutoff gate g_O(O), strictly 1.0 when bypassed."""
    if bypass:
        return 1.0
    Op = O if O > 0.0 else 0.0
    return Op / (GATE_EPS + Op)


@njit(fastmath=False, cache=True)
def gate_c(C: float, bypass: bool) -> float:
    """Smooth low-carbon shutoff gate g_C(C), strictly 1.0 when bypassed."""
    if bypass:
        return 1.0
    Cp = C if C > 0.0 else 0.0
    return Cp / (GATE_EPS + Cp)


@njit(fastmath=False, cache=True)
def o_sat_freshwater(T: float) -> float:
    """Freshwater dissolved oxygen saturation concentration (g O2 m^-3)."""
    val = 14.652 - 0.41022 * T + 0.007991 * T * T - 7.7774e-5 * T * T * T
    return val if val > 0.5 else 0.5
