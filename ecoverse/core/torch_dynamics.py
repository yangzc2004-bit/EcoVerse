"""Vectorized differentiable PyTorch stock-flow dynamics (optional GPU extension)."""

from __future__ import annotations

from typing import Tuple

import torch

from ecoverse.core.constants import (
    ALPHA_N,
    ALPHA_O,
    DEC_LINEAR,
    DEC_SATURATING,
    GATE_EPS,
    IDX_C,
    IDX_D,
    IDX_N,
    IDX_O,
    IDX_P1,
    IDX_P2,
    IDX_Z1,
    IDX_Z2,
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
from ecoverse.core.spec import CompiledForcing, CompiledParams


def phi_res_torch(
    x: torch.Tensor,
    shape_id: int,
    K: float | torch.Tensor,
    n_hill: float | torch.Tensor,
    K_I: float | torch.Tensor,
) -> torch.Tensor:
    xp = torch.clamp(x, min=0.0)
    if shape_id == RES_MONOD:
        return xp / (K + xp)
    elif shape_id == RES_HILL:
        r = xp / K
        rn = torch.pow(r, n_hill)
        return rn / (1.0 + rn)
    elif shape_id == RES_HALDANE:
        return xp / (K + xp + (xp * xp) / K_I)
    return xp / (K + xp)


def phi_light_torch(
    I: torch.Tensor,
    shape_id: int,
    I_k: float | torch.Tensor,
    I_opt: float | torch.Tensor,
) -> torch.Tensor:
    Ip = torch.clamp(I, min=0.0)
    if shape_id == LIGHT_SMITH:
        return Ip / torch.sqrt(I_k * I_k + Ip * Ip + 1e-16)
    elif shape_id == LIGHT_STEELE:
        u = Ip / I_opt
        return u * torch.exp(1.0 - u)
    elif shape_id == LIGHT_LINEAR:
        return Ip
    return Ip / torch.sqrt(I_k * I_k + Ip * Ip + 1e-16)


def phi_temp_torch(
    T: torch.Tensor,
    shape_id: int,
    theta_T: float | torch.Tensor,
    T_opt: float | torch.Tensor,
    sigma_T: float | torch.Tensor,
) -> torch.Tensor:
    if shape_id == TEMP_Q10:
        return torch.pow(torch.as_tensor(theta_T, dtype=T.dtype, device=T.device), (T - 20.0) / 10.0)
    elif shape_id == TEMP_GAUSSIAN:
        d = (T - T_opt) / sigma_T
        return torch.exp(-0.5 * d * d)
    elif shape_id == TEMP_CONST:
        return torch.ones_like(T)
    return torch.ones_like(T)


def phi_loss_torch(
    x: torch.Tensor,
    shape_id: int,
    m_a: float | torch.Tensor,
    m_b: float | torch.Tensor,
) -> torch.Tensor:
    xp = torch.clamp(x, min=0.0)
    if shape_id == LOSS_LINEAR:
        return m_a * xp
    elif shape_id == LOSS_QUADRATIC:
        return m_a * xp + m_b * xp * xp
    return m_a * xp


def psi_dec_torch(
    D: torch.Tensor,
    shape_id: int,
    K_D: float | torch.Tensor,
) -> torch.Tensor:
    Dp = torch.clamp(D, min=0.0)
    if shape_id == DEC_LINEAR:
        return Dp
    elif shape_id == DEC_SATURATING:
        return (K_D * Dp) / (K_D + Dp)
    return Dp


def o_sat_freshwater_torch(T: torch.Tensor) -> torch.Tensor:
    val = 14.652 - 0.41022 * T + 0.007991 * T * T - 7.7774e-5 * T * T * T
    return torch.clamp(val, min=0.5)


def rates_and_rhs_torch(
    x: torch.Tensor,
    u: torch.Tensor,
    p: CompiledParams,
    f: CompiledForcing,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute batched rates (..., 23) and dx/dt (..., 8) in PyTorch.

    Supports differentiable gradient backpropagation through x and u.
    """
    dtype = x.dtype
    device = x.device

    I = u[..., 0]
    T = u[..., 1]
    q = u[..., 2]
    N_in = u[..., 3]

    N = torch.clamp(x[..., IDX_N], min=0.0)
    P1 = torch.clamp(x[..., IDX_P1], min=0.0)
    P2 = torch.clamp(x[..., IDX_P2], min=0.0)
    Z1 = torch.clamp(x[..., IDX_Z1], min=0.0)
    Z2 = torch.clamp(x[..., IDX_Z2], min=0.0)
    D = torch.clamp(x[..., IDX_D], min=0.0)
    O = x[..., IDX_O]
    C = x[..., IDX_C]

    f_I = phi_light_torch(I, p.shape_light, p.I_k, p.I_opt)
    f_T_prod = phi_temp_torch(T, p.shape_temp_prod, p.theta_T_prod, p.T_opt, p.sigma_T)
    f_T_resp = phi_temp_torch(T, p.shape_temp_resp, p.theta_T_resp, p.T_opt, p.sigma_T)

    if p.bypass_O:
        gO = torch.ones_like(O)
    else:
        Op = torch.clamp(O, min=0.0)
        gO = Op / (GATE_EPS + Op)

    if p.bypass_C:
        gC = torch.ones_like(C)
    else:
        Cp = torch.clamp(C, min=0.0)
        gC = Cp / (GATE_EPS + Cp)

    if p.bypass_N:
        f_N1 = torch.ones_like(N)
        f_N2 = torch.ones_like(N)
    else:
        f_N1 = phi_res_torch(N, p.shape_res_N, p.K_N1, p.n_hill_N, p.K_I_N1)
        f_N2 = phi_res_torch(N, p.shape_res_N, p.K_N2, p.n_hill_N, p.K_I_N2)

    sink_denom = 1.0 + p.eta_q * q
    o_sat = o_sat_freshwater_torch(T)

    r_list = [
        p.v_max1 * f_I * f_N1 * gC * f_T_prod * P1,
        p.v_max2 * f_I * f_N2 * gC * f_T_prod * P2,
        p.k_resp_P1 * f_T_resp * gO * P1,
        p.k_resp_P2 * f_T_resp * gO * P2,
        phi_loss_torch(P1, p.shape_loss_P1, p.m_P1a, p.m_P1b),
        phi_loss_torch(P2, p.shape_loss_P2, p.m_P2a, p.m_P2b),
        p.g_max11 * phi_res_torch(P1, p.shape_res_P, p.K_P1, p.n_hill_P, p.K_I_P1) * f_T_prod * Z1,
        p.g_max12 * phi_res_torch(P2, p.shape_res_P, p.K_P2, p.n_hill_P, p.K_I_P2) * f_T_prod * Z1,
        p.g_max2 * phi_res_torch(Z1, p.shape_res_Z, p.K_Z1, p.n_hill_Z, p.K_I_Z1) * f_T_prod * Z2,
        p.c_12 * P1 * P2,
        p.c_21 * P2 * P1,
        p.k_resp_Z1 * f_T_resp * gO * Z1,
        p.k_resp_Z2 * f_T_resp * gO * Z2,
        phi_loss_torch(Z1, p.shape_loss_Z1, p.m_Z1a, p.m_Z1b),
        phi_loss_torch(Z2, p.shape_loss_Z2, p.m_Z2a, p.m_Z2b),
        p.k_dec * f_T_resp * gO * psi_dec_torch(D, p.shape_dec, p.K_D),
        p.k_gas_O * (o_sat - O),
        0.9 * p.k_gas_O * (f.C_sat - C),
        (p.s_P1 / sink_denom) * P1,
        (p.s_P2 / sink_denom) * P2,
        (p.s_D / sink_denom) * D,
        p.b_SOD * f_T_resp * gO,
        p.b_N * (1.0 - gO),
    ]
    cw = torch.as_tensor(p.channel_weights, dtype=dtype, device=device)
    r = torch.stack(r_list, dim=-1) * cw

    dx_N = (
        ALPHA_N * (-r[..., 0] - r[..., 1] + r[..., 2] + r[..., 3] + r[..., 11] + r[..., 12] + r[..., 15])
        + r[..., 22]
    )
    dx_P1 = r[..., 0] - r[..., 2] - r[..., 4] - r[..., 6] - r[..., 9] - r[..., 18]
    dx_P2 = r[..., 1] - r[..., 3] - r[..., 5] - r[..., 7] - r[..., 10] - r[..., 19]
    dx_Z1 = p.gamma_1 * (r[..., 6] + r[..., 7]) - r[..., 8] - r[..., 11] - r[..., 13]
    dx_Z2 = p.gamma_2 * r[..., 8] - r[..., 12] - r[..., 14]
    dx_D = (
        r[..., 4]
        + r[..., 5]
        + (1.0 - p.gamma_1) * (r[..., 6] + r[..., 7])
        + (1.0 - p.gamma_2) * r[..., 8]
        + r[..., 9]
        + r[..., 10]
        + r[..., 13]
        + r[..., 14]
        - r[..., 15]
        - r[..., 20]
    )
    dx_O = (
        ALPHA_O
        * (r[..., 0] + r[..., 1] - r[..., 2] - r[..., 3] - r[..., 11] - r[..., 12] - r[..., 15] - r[..., 21])
        + r[..., 16]
    )
    dx_C = -r[..., 0] - r[..., 1] + r[..., 2] + r[..., 3] + r[..., 11] + r[..., 12] + r[..., 15] + r[..., 17] + r[..., 21]

    dx = torch.stack([dx_N, dx_P1, dx_P2, dx_Z1, dx_Z2, dx_D, dx_O, dx_C], dim=-1)

    zeros = torch.zeros_like(N_in)
    D_in_t = torch.full_like(N_in, float(f.D_in))
    C_sat_t = torch.full_like(N_in, float(f.C_sat))
    x_in = torch.stack([N_in, zeros, zeros, zeros, zeros, D_in_t, o_sat, C_sat_t], dim=-1)

    delta_t = torch.as_tensor(p.delta, dtype=dtype, device=device)
    sigma_t = torch.as_tensor(p.sigma, dtype=dtype, device=device)
    sm_t = torch.as_tensor(p.state_mask, dtype=dtype, device=device)

    x_pos = torch.clamp(x, min=0.0)
    boundary = q.unsqueeze(-1) * (delta_t * x_in - sigma_t * x_pos)
    dx = (dx + boundary) * sm_t
    return r, dx
