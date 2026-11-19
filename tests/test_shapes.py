"""Unit tests for functional response shapes and PyTorch parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecoverse.core.constants import (
    DEC_LINEAR,
    DEC_SATURATING,
    LIGHT_LINEAR,
    LIGHT_SMITH,
    LIGHT_STEELE,
    LOSS_LINEAR,
    LOSS_QUADRATIC,
    RES_HALDANE,
    RES_HILL,
    RES_MONOD,
    STATE_NAMES,
    TEMP_CONST,
    TEMP_GAUSSIAN,
    TEMP_Q10,
)
from ecoverse.core.dynamics import compute_rates_and_forcing, rhs_jit
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
from ecoverse.core.spec import ForcingSpec, InterventionSpec, SystemSpec, compile_spec
from ecoverse.core.torch_dynamics import rates_and_rhs_torch


def test_phi_res_properties() -> None:
    K = 0.4
    n_hill = 2.8
    K_I = 2.0
    xs = np.linspace(-1.0, 20.0, 500)

    for shape_id in (RES_MONOD, RES_HILL, RES_HALDANE):
        assert phi_res(0.0, shape_id, K, n_hill, K_I) == 0.0
        assert phi_res(-0.5, shape_id, K, n_hill, K_I) == 0.0
        vals = [phi_res(float(x), shape_id, K, n_hill, K_I) for x in xs]
        assert min(vals) >= 0.0
        assert max(vals) <= 1.0

    # Exact half-saturation at x = K for Monod & Hill
    assert phi_res(K, RES_MONOD, K, n_hill, K_I) == pytest.approx(0.5, abs=1e-12)
    assert phi_res(K, RES_HILL, K, n_hill, K_I) == pytest.approx(0.5, abs=1e-12)

    # Very large x should not overflow in Hill
    assert phi_res(1e12, RES_HILL, K, 3.5, K_I) == pytest.approx(1.0, abs=1e-12)

    # Haldane is unimodal peaking at sqrt(K * K_I)
    x_peak = float(np.sqrt(K * K_I))
    v_peak = phi_res(x_peak, RES_HALDANE, K, n_hill, K_I)
    assert v_peak > phi_res(0.5 * x_peak, RES_HALDANE, K, n_hill, K_I)
    assert v_peak > phi_res(2.0 * x_peak, RES_HALDANE, K, n_hill, K_I)


def test_phi_light_properties() -> None:
    I_k = 0.25
    I_opt = 0.6
    for shape_id in (LIGHT_SMITH, LIGHT_STEELE, LIGHT_LINEAR):
        assert phi_light(0.0, shape_id, I_k, I_opt) == 0.0
        assert phi_light(-0.2, shape_id, I_k, I_opt) == 0.0

    for I in np.linspace(0.0, 1.0, 100):
        assert 0.0 <= phi_light(float(I), LIGHT_SMITH, I_k, I_opt) <= 1.0
        assert 0.0 <= phi_light(float(I), LIGHT_STEELE, I_k, I_opt) <= 1.0
        assert phi_light(float(I), LIGHT_LINEAR, I_k, I_opt) == pytest.approx(float(I))

    assert phi_light(I_opt, LIGHT_STEELE, I_k, I_opt) == pytest.approx(1.0, abs=1e-12)
    assert phi_light(0.5 * I_opt, LIGHT_STEELE, I_k, I_opt) < 1.0
    assert phi_light(1.5 * I_opt, LIGHT_STEELE, I_k, I_opt) < 1.0


def test_phi_temp_properties() -> None:
    theta_T = 1.6
    T_opt = 22.0
    sigma_T = 6.0

    assert phi_temp(20.0, TEMP_Q10, theta_T, T_opt, sigma_T) == pytest.approx(1.0, abs=1e-12)
    assert phi_temp(30.0, TEMP_Q10, theta_T, T_opt, sigma_T) == pytest.approx(theta_T, abs=1e-12)
    assert phi_temp(T_opt, TEMP_GAUSSIAN, theta_T, T_opt, sigma_T) == pytest.approx(1.0, abs=1e-12)
    assert phi_temp(T_opt + 10.0, TEMP_GAUSSIAN, theta_T, T_opt, sigma_T) < 1.0
    assert phi_temp(5.0, TEMP_CONST, theta_T, T_opt, sigma_T) == 1.0
    assert phi_temp(35.0, TEMP_CONST, theta_T, T_opt, sigma_T) == 1.0


def test_loss_dec_gates_and_osat() -> None:
    assert phi_loss(0.0, LOSS_LINEAR, 0.1, 0.2) == 0.0
    assert phi_loss(-1.0, LOSS_QUADRATIC, 0.1, 0.2) == 0.0
    assert phi_loss(2.0, LOSS_QUADRATIC, 0.1, 0.2) == pytest.approx(0.1 * 2.0 + 0.2 * 4.0)

    K_D = 2.0
    assert psi_dec(0.0, DEC_LINEAR, K_D) == 0.0
    assert psi_dec(0.0, DEC_SATURATING, K_D) == 0.0
    # Low-D slope alignment
    d_small = 1e-6
    assert psi_dec(d_small, DEC_SATURATING, K_D) / d_small == pytest.approx(1.0, abs=1e-5)
    assert psi_dec(1e6, DEC_SATURATING, K_D) == pytest.approx(K_D, rel=1e-5)

    # Gates
    assert gate_o(0.0, True) == 1.0
    assert gate_c(0.0, True) == 1.0
    assert gate_o(0.0, False) == 0.0
    assert gate_o(0.1, False) == pytest.approx(0.5)

    # Freshwater O2 saturation check using exact dev_plan cubic polynomial
    assert o_sat_freshwater(0.0) == pytest.approx(14.652, abs=1e-6)
    assert o_sat_freshwater(30.0) == pytest.approx(7.437402, abs=1e-6)


def test_torch_dynamics_parity_and_autograd() -> None:
    spec = SystemSpec(
        preset="river",
        active_states={s: True for s in STATE_NAMES},
        active_channels=list(range(23)),
        shapes={
            "res_N": "HILL",
            "res_P": "HALDANE",
            "res_Z": "MONOD",
            "light": "STEELE",
            "temp_prod": "GAUSSIAN",
            "temp_resp": "Q10",
            "loss_P1": "QUADRATIC",
            "loss_P2": "LINEAR",
            "loss_Z1": "QUADRATIC",
            "loss_Z2": "LINEAR",
            "dec": "SATURATING",
        },
        params={
            "v_max1": 1.4, "v_max2": 0.9, "k_resp_P1": 0.15, "k_resp_P2": 0.1,
            "m_P1a": 0.05, "m_P1b": 0.02, "m_P2a": 0.04, "m_P2b": 0.0,
            "g_max11": 0.8, "g_max12": 0.5, "g_max2": 0.4,
            "gamma_1": 0.55, "gamma_2": 0.5, "c_12": 0.1, "c_21": 0.12,
            "k_resp_Z1": 0.1, "k_resp_Z2": 0.08,
            "m_Z1a": 0.06, "m_Z1b": 0.03, "m_Z2a": 0.05, "m_Z2b": 0.0,
            "k_dec": 0.12, "k_gas_O": 2.5, "s_P1": 0.05, "s_P2": 0.03, "s_D": 0.08,
            "eta_q": 1.5, "b_SOD": 0.4, "b_N": 0.1,
            "K_N1": 0.2, "K_N2": 0.25, "K_P1": 0.5, "K_P2": 0.6, "K_Z1": 0.7, "K_D": 1.5,
            "n_hill_N": 2.4, "n_hill_P": 2.2, "n_hill_Z": 2.5,
            "K_I_N1": 1.0, "K_I_N2": 1.2, "K_I_P1": 2.0, "K_I_P2": 2.5, "K_I_Z1": 3.0,
            "I_k": 0.25, "I_opt": 0.6, "theta_T_prod": 1.5, "theta_T_resp": 2.1,
            "T_opt": 18.0, "sigma_T": 7.0,
        },
        sigma={s: 0.5 for s in STATE_NAMES},
        delta={s: (1.0 if s in ("N", "D", "O", "C") else 0.0) for s in STATE_NAMES},
        forcing=ForcingSpec(
            diel=True, seasonal=True, a_I=0.4, b_c=1.0,
            cloud_w=[0.1] * 8, cloud_f=list(np.linspace(0.02, 0.2, 8)), cloud_psi=[0.2] * 8,
            T_bar=16.0, A_season=6.0, A_diel=1.2, lag_T=25.0,
            temp_v=[0.1] * 6, temp_g=list(np.linspace(0.01, 0.1, 6)), temp_chi=[0.3] * 6,
            q_base=0.4, storm_t=[5.0], storm_A=[0.8], storm_tau=[2.0],
            N_in_0=1.2, D_in=0.3, C_sat=15.0,
        ),
        initial_state={s: 1.0 for s in STATE_NAMES},
        intervention=InterventionSpec(type="none"),
    )
    compiled = compile_spec(spec)
    x_np = np.array([0.8, 1.3, 0.7, 0.4, 0.2, 0.9, 8.5, 14.0], dtype=np.float64)
    r_nb, u_nb = compute_rates_and_forcing(12.5, x_np, compiled.params, compiled.forcing)
    dx_nb = rhs_jit(12.5, x_np, compiled.params, compiled.forcing)

    x_th = torch.tensor(x_np[None, :], dtype=torch.float64, requires_grad=True)
    u_th = torch.tensor(u_nb[None, :], dtype=torch.float64, requires_grad=True)
    r_th, dx_th = rates_and_rhs_torch(x_th, u_th, compiled.params, compiled.forcing)

    np.testing.assert_allclose(r_th.squeeze(0).detach().numpy(), r_nb, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dx_th.squeeze(0).detach().numpy(), dx_nb, atol=1e-10, rtol=1e-10)

    loss = dx_th.sum() + r_th.sum()
    loss.backward()
    assert torch.isfinite(x_th.grad).all()
    assert torch.isfinite(u_th.grad).all()
