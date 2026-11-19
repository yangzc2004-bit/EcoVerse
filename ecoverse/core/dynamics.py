"""Numba-compiled 8-state 23-channel stock-flow ODE kernel (fastmath=False)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit

from ecoverse.core.constants import (
    ALPHA_N,
    ALPHA_O,
    IDX_C,
    IDX_D,
    IDX_N,
    IDX_O,
    IDX_P1,
    IDX_P2,
    IDX_Z1,
    IDX_Z2,
    N_CHANNELS,
    N_DRIVERS,
    N_STATES,
)
from ecoverse.core.forcing import CompiledForcing, eval_forcing
from ecoverse.core.shapes import (
    gate_c,
    gate_o,
    o_sat_freshwater,
    phi_light,
    phi_loss,
    phi_res,
    phi_temp,
    psi_dec,
)
from ecoverse.core.spec import CompiledParams


@njit(fastmath=False, cache=True)
def compute_rates_and_forcing(
    t: float,
    x: np.ndarray,
    p: CompiledParams,
    f: CompiledForcing,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all 23 masked reaction rates r(t) and 4 drivers u(t)."""
    I, T, q, N_in = eval_forcing(t, f)

    u = np.empty(N_DRIVERS, dtype=np.float64)
    u[0] = I
    u[1] = T
    u[2] = q
    u[3] = N_in

    N = x[IDX_N] if x[IDX_N] > 0.0 else 0.0
    P1 = x[IDX_P1] if x[IDX_P1] > 0.0 else 0.0
    P2 = x[IDX_P2] if x[IDX_P2] > 0.0 else 0.0
    Z1 = x[IDX_Z1] if x[IDX_Z1] > 0.0 else 0.0
    Z2 = x[IDX_Z2] if x[IDX_Z2] > 0.0 else 0.0
    D = x[IDX_D] if x[IDX_D] > 0.0 else 0.0
    O = x[IDX_O]
    C = x[IDX_C]

    f_I = phi_light(I, p.shape_light, p.I_k, p.I_opt)
    f_T_prod = phi_temp(T, p.shape_temp_prod, p.theta_T_prod, p.T_opt, p.sigma_T)
    f_T_resp = phi_temp(T, p.shape_temp_resp, p.theta_T_resp, p.T_opt, p.sigma_T)

    gO = gate_o(O, p.bypass_O)
    gC = gate_c(C, p.bypass_C)

    if p.bypass_N:
        f_N1 = 1.0
        f_N2 = 1.0
    else:
        f_N1 = phi_res(N, p.shape_res_N, p.K_N1, p.n_hill_N, p.K_I_N1)
        f_N2 = phi_res(N, p.shape_res_N, p.K_N2, p.n_hill_N, p.K_I_N2)

    r = np.zeros(N_CHANNELS, dtype=np.float64)

    # 0, 1: Autotrophic production P1, P2
    r[0] = p.v_max1 * f_I * f_N1 * gC * f_T_prod * P1
    r[1] = p.v_max2 * f_I * f_N2 * gC * f_T_prod * P2

    # 2, 3: Producer respiration P1, P2
    r[2] = p.k_resp_P1 * f_T_resp * gO * P1
    r[3] = p.k_resp_P2 * f_T_resp * gO * P2

    # 4, 5: Producer mortality/exudation P1, P2
    r[4] = phi_loss(P1, p.shape_loss_P1, p.m_P1a, p.m_P1b)
    r[5] = phi_loss(P2, p.shape_loss_P2, p.m_P2a, p.m_P2b)

    # 6, 7: Primary grazing P1, P2 -> Z1
    r[6] = p.g_max11 * phi_res(P1, p.shape_res_P, p.K_P1, p.n_hill_P, p.K_I_P1) * f_T_prod * Z1
    r[7] = p.g_max12 * phi_res(P2, p.shape_res_P, p.K_P2, p.n_hill_P, p.K_I_P2) * f_T_prod * Z1

    # 8: Secondary predation Z1 -> Z2
    r[8] = p.g_max2 * phi_res(Z1, p.shape_res_Z, p.K_Z1, p.n_hill_Z, p.K_I_Z1) * f_T_prod * Z2

    # 9, 10: Interference competition
    r[9] = p.c_12 * P1 * P2
    r[10] = p.c_21 * P2 * P1

    # 11, 12: Consumer respiration Z1, Z2
    r[11] = p.k_resp_Z1 * f_T_resp * gO * Z1
    r[12] = p.k_resp_Z2 * f_T_resp * gO * Z2

    # 13, 14: Consumer mortality Z1, Z2
    r[13] = phi_loss(Z1, p.shape_loss_Z1, p.m_Z1a, p.m_Z1b)
    r[14] = phi_loss(Z2, p.shape_loss_Z2, p.m_Z2a, p.m_Z2b)

    # 15: Detritus decomposition
    r[15] = p.k_dec * f_T_resp * gO * psi_dec(D, p.shape_dec, p.K_D)

    # 16, 17: Gas exchange
    r[16] = p.k_gas_O * (o_sat_freshwater(T) - O)
    r[17] = 0.9 * p.k_gas_O * (f.C_sat - C)

    # 18, 19, 20: Sinking & burial (attenuated by flow)
    sink_denom = 1.0 + p.eta_q * q
    r[18] = (p.s_P1 / sink_denom) * P1
    r[19] = (p.s_P2 / sink_denom) * P2
    r[20] = (p.s_D / sink_denom) * D

    # 21, 22: Benthic SOD & anoxic N release
    r[21] = p.b_SOD * f_T_resp * gO
    r[22] = p.b_N * (1.0 - gO)

    for j in range(N_CHANNELS):
        r[j] *= p.channel_weights[j]

    return r, u


@njit(fastmath=False, cache=True)
def rhs_jit(t: float, x: np.ndarray, p: CompiledParams, f: CompiledForcing) -> np.ndarray:
    """Compute dx/dt = m_state * (S^T r + q * (delta * x_in - sigma * x_+))."""
    r, u = compute_rates_and_forcing(t, x, p, f)
    T = u[1]
    q = u[2]
    N_in = u[3]

    dx = np.zeros(N_STATES, dtype=np.float64)

    # Stoichiometric accumulation matching Table 4.4
    dx[IDX_N] = (
        ALPHA_N * (-r[0] - r[1] + r[2] + r[3] + r[11] + r[12] + r[15])
        + r[22]
    )
    dx[IDX_P1] = r[0] - r[2] - r[4] - r[6] - r[9] - r[18]
    dx[IDX_P2] = r[1] - r[3] - r[5] - r[7] - r[10] - r[19]
    dx[IDX_Z1] = p.gamma_1 * (r[6] + r[7]) - r[8] - r[11] - r[13]
    dx[IDX_Z2] = p.gamma_2 * r[8] - r[12] - r[14]
    dx[IDX_D] = (
        r[4]
        + r[5]
        + (1.0 - p.gamma_1) * (r[6] + r[7])
        + (1.0 - p.gamma_2) * r[8]
        + r[9]
        + r[10]
        + r[13]
        + r[14]
        - r[15]
        - r[20]
    )
    dx[IDX_O] = (
        ALPHA_O * (r[0] + r[1] - r[2] - r[3] - r[11] - r[12] - r[15] - r[21])
        + r[16]
    )
    dx[IDX_C] = (
        -r[0]
        - r[1]
        + r[2]
        + r[3]
        + r[11]
        + r[12]
        + r[15]
        + r[17]
        + r[21]
    )

    # 8-D boundary throughput
    if q > 0.0:
        x_in_0 = N_in
        x_in_5 = f.D_in
        x_in_6 = o_sat_freshwater(T)
        x_in_7 = f.C_sat

        dx[IDX_N] += q * (p.delta[IDX_N] * x_in_0 - p.sigma[IDX_N] * (x[IDX_N] if x[IDX_N] > 0.0 else 0.0))
        dx[IDX_P1] += q * (-p.sigma[IDX_P1] * (x[IDX_P1] if x[IDX_P1] > 0.0 else 0.0))
        dx[IDX_P2] += q * (-p.sigma[IDX_P2] * (x[IDX_P2] if x[IDX_P2] > 0.0 else 0.0))
        dx[IDX_Z1] += q * (-p.sigma[IDX_Z1] * (x[IDX_Z1] if x[IDX_Z1] > 0.0 else 0.0))
        dx[IDX_Z2] += q * (-p.sigma[IDX_Z2] * (x[IDX_Z2] if x[IDX_Z2] > 0.0 else 0.0))
        dx[IDX_D] += q * (p.delta[IDX_D] * x_in_5 - p.sigma[IDX_D] * (x[IDX_D] if x[IDX_D] > 0.0 else 0.0))
        dx[IDX_O] += q * (p.delta[IDX_O] * x_in_6 - p.sigma[IDX_O] * (x[IDX_O] if x[IDX_O] > 0.0 else 0.0))
        dx[IDX_C] += q * (p.delta[IDX_C] * x_in_7 - p.sigma[IDX_C] * (x[IDX_C] if x[IDX_C] > 0.0 else 0.0))

    for i in range(N_STATES):
        dx[i] *= p.state_mask[i]

    return dx


@njit(fastmath=False, cache=True)
def eval_trajectory_rates_jit(
    t_grid: np.ndarray,
    x_traj: np.ndarray,
    p: CompiledParams,
    f: CompiledForcing,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate r_true (T, 23) and u_true (T, 4) along an output trajectory."""
    n_t = t_grid.shape[0]
    r_traj = np.empty((n_t, N_CHANNELS), dtype=np.float64)
    u_traj = np.empty((n_t, N_DRIVERS), dtype=np.float64)

    for k in range(n_t):
        r_k, u_k = compute_rates_and_forcing(t_grid[k], x_traj[k], p, f)
        for j in range(N_CHANNELS):
            r_traj[k, j] = r_k[j]
        for d in range(N_DRIVERS):
            u_traj[k, d] = u_k[d]

    return r_traj, u_traj


def warmup_jit() -> None:
    """Trigger Numba compilation once on minimal dummy arrays."""
    dummy_p = CompiledParams(
        state_mask=np.ones(N_STATES, dtype=np.float64),
        channel_weights=np.ones(N_CHANNELS, dtype=np.float64),
        sigma=np.ones(N_STATES, dtype=np.float64),
        delta=np.ones(N_STATES, dtype=np.float64),
        bypass_N=False,
        bypass_O=False,
        bypass_C=False,
        shape_res_N=0,
        shape_res_P=0,
        shape_res_Z=0,
        shape_light=0,
        shape_temp_prod=0,
        shape_temp_resp=0,
        shape_loss_P1=0,
        shape_loss_P2=0,
        shape_loss_Z1=0,
        shape_loss_Z2=0,
        shape_dec=0,
        v_max1=1.0,
        v_max2=0.5,
        k_resp_P1=0.1,
        k_resp_P2=0.05,
        m_P1a=0.05,
        m_P1b=0.01,
        m_P2a=0.05,
        m_P2b=0.01,
        g_max11=0.6,
        g_max12=0.3,
        g_max2=0.2,
        gamma_1=0.5,
        gamma_2=0.5,
        c_12=0.1,
        c_21=0.1,
        k_resp_Z1=0.1,
        k_resp_Z2=0.05,
        m_Z1a=0.05,
        m_Z1b=0.02,
        m_Z2a=0.03,
        m_Z2b=0.02,
        k_dec=0.1,
        k_gas_O=1.0,
        s_P1=0.05,
        s_P2=0.05,
        s_D=0.05,
        eta_q=1.0,
        b_SOD=0.2,
        b_N=0.0,
        K_N1=0.2,
        K_N2=0.2,
        K_P1=0.5,
        K_P2=0.5,
        K_Z1=0.5,
        K_D=1.0,
        n_hill_N=2.5,
        n_hill_P=2.5,
        n_hill_Z=2.5,
        K_I_N1=1.0,
        K_I_N2=1.0,
        K_I_P1=2.0,
        K_I_P2=2.0,
        K_I_Z1=2.0,
        I_k=0.25,
        I_opt=0.6,
        theta_T_prod=1.5,
        theta_T_resp=2.0,
        T_opt=18.0,
        sigma_T=7.0,
    )
    dummy_f = CompiledForcing(
        diel=True,
        seasonal=True,
        a_I=0.4,
        b_c=1.0,
        cloud_w=np.zeros(8, dtype=np.float64),
        cloud_f=np.linspace(0.02, 0.25, 8, dtype=np.float64),
        cloud_psi=np.zeros(8, dtype=np.float64),
        T_bar=16.0,
        A_season=6.0,
        A_diel=1.5,
        lag_T=25.0,
        temp_v=np.zeros(6, dtype=np.float64),
        temp_g=np.linspace(0.01, 0.1, 6, dtype=np.float64),
        temp_chi=np.zeros(6, dtype=np.float64),
        q_base=0.2,
        storm_t=np.array([10.0], dtype=np.float64),
        storm_A=np.array([0.5], dtype=np.float64),
        storm_tau=np.array([2.0], dtype=np.float64),
        N_in_0=1.0,
        D_in=0.2,
        C_sat=15.0,
        interv_type=0,
        interv_t=180.0,
        interv_lambda_N=1.0,
        interv_delta_T=0.0,
        interv_lambda_q=1.0,
        interv_lambda_storm=1.0,
    )
    t_grid = np.array([0.0, 1.0], dtype=np.float64)
    x_traj = np.ones((2, N_STATES), dtype=np.float64)
    rhs_jit(0.0, x_traj[0], dummy_p, dummy_f)
    eval_trajectory_rates_jit(t_grid, x_traj, dummy_p, dummy_f)
