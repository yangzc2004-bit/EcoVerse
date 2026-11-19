"""Unit tests for strict mass/redox conservation and compile-time gate bypass."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ecoverse.core.constants import (
    ALPHA_N,
    ALPHA_O,
    CLOSED_CHANNELS,
    IDX_C,
    IDX_D,
    IDX_N,
    IDX_O,
    IDX_P1,
    IDX_P2,
    IDX_Z1,
    IDX_Z2,
    STATE_NAMES,
)
from ecoverse.core.dynamics import compute_rates_and_forcing, rhs_jit
from ecoverse.core.spec import ForcingSpec, InterventionSpec, SystemSpec, compile_spec


def _closed_system_spec(shapes: dict[str, str]) -> SystemSpec:
    return SystemSpec(
        preset="closed_test",
        active_states={s: True for s in STATE_NAMES},
        active_channels=list(CLOSED_CHANNELS),  # 0..15 only; 16..22 disabled
        shapes=shapes,
        params={
            "v_max1": 1.2, "v_max2": 0.9, "k_resp_P1": 0.12, "k_resp_P2": 0.10,
            "m_P1a": 0.06, "m_P1b": 0.02, "m_P2a": 0.05, "m_P2b": 0.02,
            "g_max11": 0.7, "g_max12": 0.4, "g_max2": 0.35,
            "gamma_1": 0.5, "gamma_2": 0.45, "c_12": 0.08, "c_21": 0.06,
            "k_resp_Z1": 0.10, "k_resp_Z2": 0.08,
            "m_Z1a": 0.05, "m_Z1b": 0.02, "m_Z2a": 0.04, "m_Z2b": 0.02,
            "k_dec": 0.15, "k_gas_O": 0.0, "s_P1": 0.0, "s_P2": 0.0, "s_D": 0.0,
            "eta_q": 1.0, "b_SOD": 0.0, "b_N": 0.0,
            "K_N1": 0.2, "K_N2": 0.25, "K_P1": 0.4, "K_P2": 0.5, "K_Z1": 0.6, "K_D": 1.5,
            "n_hill_N": 2.5, "n_hill_P": 2.2, "n_hill_Z": 2.5,
            "K_I_N1": 1.0, "K_I_N2": 1.2, "K_I_P1": 2.0, "K_I_P2": 2.5, "K_I_Z1": 3.0,
            "I_k": 0.25, "I_opt": 0.6, "theta_T_prod": 1.5, "theta_T_resp": 2.0,
            "T_opt": 18.0, "sigma_T": 7.0,
        },
        sigma={s: 0.0 for s in STATE_NAMES},
        delta={s: 0.0 for s in STATE_NAMES},
        forcing=ForcingSpec(
            diel=True, seasonal=True, a_I=0.4, b_c=1.0,
            cloud_w=[0.1] * 8, cloud_f=list(np.linspace(0.02, 0.2, 8)), cloud_psi=[0.1] * 8,
            T_bar=16.0, A_season=6.0, A_diel=1.5, lag_T=25.0,
            temp_v=[0.2] * 6, temp_g=list(np.linspace(0.01, 0.1, 6)), temp_chi=[0.2] * 6,
            q_base=0.0, storm_t=[], storm_A=[], storm_tau=[],
            N_in_0=0.0, D_in=0.0, C_sat=15.0,
        ),
        initial_state={
            "N": 1.0, "P1": 0.8, "P2": 0.6, "Z1": 0.4, "Z2": 0.2, "D": 1.0, "O": 9.0, "C": 14.0,
        },
        intervention=InterventionSpec(type="none"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        {
            "res_N": "MONOD", "res_P": "MONOD", "res_Z": "MONOD",
            "light": "SMITH", "temp_prod": "Q10", "temp_resp": "Q10",
            "loss_P1": "LINEAR", "loss_P2": "LINEAR", "loss_Z1": "LINEAR", "loss_Z2": "LINEAR",
            "dec": "LINEAR",
        },
        {
            "res_N": "HILL", "res_P": "HALDANE", "res_Z": "HILL",
            "light": "STEELE", "temp_prod": "GAUSSIAN", "temp_resp": "Q10",
            "loss_P1": "QUADRATIC", "loss_P2": "QUADRATIC", "loss_Z1": "QUADRATIC", "loss_Z2": "QUADRATIC",
            "dec": "SATURATING",
        },
    ],
)
def test_three_invariants_closed_boundary(shapes: dict[str, str]) -> None:
    spec = _closed_system_spec(shapes)
    compiled = compile_spec(spec)
    t_eval = np.linspace(0.0, 365.0, 2191)

    sol = solve_ivp(
        lambda t, y: rhs_jit(t, y, compiled.params, compiled.forcing),
        (0.0, 365.0),
        compiled.x0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    assert sol.status == 0
    y = sol.y.T  # (T, 8)

    bio_det = y[:, IDX_P1] + y[:, IDX_P2] + y[:, IDX_Z1] + y[:, IDX_Z2] + y[:, IDX_D]
    M_C = bio_det + y[:, IDX_C]
    M_N = y[:, IDX_N] / ALPHA_N + bio_det
    M_OC = y[:, IDX_O] + ALPHA_O * y[:, IDX_C]

    drift_C = np.max(np.abs(M_C - M_C[0])) / M_C[0]
    drift_N = np.max(np.abs(M_N - M_N[0])) / M_N[0]
    drift_OC = np.max(np.abs(M_OC - M_OC[0])) / M_OC[0]

    assert drift_C < 1e-8
    assert drift_N < 1e-8
    assert drift_OC < 1e-8


def test_compile_time_gate_bypass_strictly_one() -> None:
    spec = _closed_system_spec(
        {
            "res_N": "MONOD", "res_P": "MONOD", "res_Z": "MONOD",
            "light": "LINEAR", "temp_prod": "CONST", "temp_resp": "CONST",
            "loss_P1": "LINEAR", "loss_P2": "LINEAR", "loss_Z1": "LINEAR", "loss_Z2": "LINEAR",
            "dec": "LINEAR",
        }
    )
    # Disable N, O, C like foodweb
    spec = spec.model_copy(
        update={
            "active_states": {
                "N": False, "P1": True, "P2": False, "Z1": True, "Z2": True,
                "D": False, "O": False, "C": False,
            },
            "initial_state": {
                "N": 0.0, "P1": 1.0, "P2": 0.0, "Z1": 0.5, "Z2": 0.2,
                "D": 0.0, "O": 0.0, "C": 0.0,
            },
        }
    )
    compiled = compile_spec(spec)
    assert compiled.params.bypass_N is True
    assert compiled.params.bypass_O is True
    assert compiled.params.bypass_C is True

    r, u = compute_rates_and_forcing(0.5, compiled.x0, compiled.params, compiled.forcing)
    # Under LIGHT_LINEAR & TEMP_CONST, r[0] must equal v_max1 * I * 1.0 * 1.0 * 1.0 * P1
    expected_r0 = compiled.params.v_max1 * u[0] * compiled.x0[IDX_P1]
    assert r[0] == pytest.approx(expected_r0, abs=1e-14)
