"""Unit tests for strict non-negativity and boundedness under extreme stress."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ecoverse.core.constants import STATE_NAMES
from ecoverse.core.dynamics import rhs_jit
from ecoverse.core.spec import (
    ForcingSpec,
    InterventionSpec,
    SystemSpec,
    apply_bio_pulse,
    compile_spec,
)


def test_positivity_under_extreme_stress() -> None:
    spec = SystemSpec(
        preset="stress_test",
        active_states={s: True for s in STATE_NAMES},
        active_channels=list(range(23)),
        shapes={
            "res_N": "HILL",
            "res_P": "HILL",
            "res_Z": "MONOD",
            "light": "SMITH",
            "temp_prod": "Q10",
            "temp_resp": "Q10",
            "loss_P1": "QUADRATIC",
            "loss_P2": "QUADRATIC",
            "loss_Z1": "QUADRATIC",
            "loss_Z2": "QUADRATIC",
            "dec": "SATURATING",
        },
        params={
            "v_max1": 4.0, "v_max2": 3.0, "k_resp_P1": 0.5, "k_resp_P2": 0.4,
            "m_P1a": 0.2, "m_P1b": 0.3, "m_P2a": 0.2, "m_P2b": 0.3,
            "g_max11": 2.5, "g_max12": 2.0, "g_max2": 1.5,
            "gamma_1": 0.4, "gamma_2": 0.4, "c_12": 0.4, "c_21": 0.4,
            "k_resp_Z1": 0.3, "k_resp_Z2": 0.3,
            "m_Z1a": 0.2, "m_Z1b": 0.2, "m_Z2a": 0.2, "m_Z2b": 0.2,
            "k_dec": 0.6, "k_gas_O": 0.1, "s_P1": 0.3, "s_P2": 0.3, "s_D": 0.2,
            "eta_q": 0.5, "b_SOD": 3.0, "b_N": 1.0,
            "K_N1": 0.02, "K_N2": 0.02, "K_P1": 0.05, "K_P2": 0.05, "K_Z1": 0.05, "K_D": 0.3,
            "n_hill_N": 3.5, "n_hill_P": 3.5, "n_hill_Z": 2.0,
            "K_I_N1": 1.0, "K_I_N2": 1.0, "K_I_P1": 2.0, "K_I_P2": 2.0, "K_I_Z1": 2.0,
            "I_k": 0.15, "I_opt": 0.5, "theta_T_prod": 1.9, "theta_T_resp": 2.6,
            "T_opt": 20.0, "sigma_T": 6.0,
        },
        sigma={s: 1.0 for s in STATE_NAMES},
        delta={"N": 1.0, "P1": 0.0, "P2": 0.0, "Z1": 0.0, "Z2": 0.0, "D": 1.0, "O": 1.0, "C": 1.0},
        forcing=ForcingSpec(
            diel=True, seasonal=True, a_I=0.5, b_c=0.5,
            cloud_w=[0.3] * 8, cloud_f=list(np.linspace(0.02, 0.25, 8)), cloud_psi=[0.5] * 8,
            T_bar=24.0, A_season=10.0, A_diel=2.5, lag_T=25.0,
            temp_v=[0.4] * 6, temp_g=list(np.linspace(0.01, 0.12, 6)), temp_chi=[0.5] * 6,
            q_base=2.5, storm_t=[20.0, 100.0, 220.0], storm_A=[8.0, 10.0, 6.0], storm_tau=[1.5, 2.0, 2.0],
            N_in_0=0.05, D_in=0.05, C_sat=12.0,
        ),
        initial_state={
            "N": 1e-4, "P1": 5.0, "P2": 4.0, "Z1": 3.0, "Z2": 2.0, "D": 1e-4, "O": 0.05, "C": 0.05,
        },
        intervention=InterventionSpec(type="bio_pulse", target_state="P1", kappa=4.0, regime="ood"),
    )

    compiled = compile_spec(spec, apply_intervention=False)
    t_eval = np.arange(0, 1081) / 6.0
    sol1 = solve_ivp(
        lambda t, y: rhs_jit(t, y, compiled.params, compiled.forcing),
        (0.0, t_eval[-1]),
        compiled.x0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    assert sol1.status == 0
    assert np.isfinite(sol1.y).all()
    assert np.min(sol1.y) >= -1e-4

    # Apply extreme bio_pulse at t = 180
    x_cf0 = apply_bio_pulse(np.maximum(sol1.y[:, -1], 0.0), spec)
    compiled_cf = compile_spec(spec, apply_intervention=True)
    t_eval_cf = np.arange(1080, 2190) / 6.0
    sol2 = solve_ivp(
        lambda t, y: rhs_jit(t, y, compiled_cf.params, compiled_cf.forcing),
        (180.0, t_eval_cf[-1]),
        x_cf0,
        method="LSODA",
        t_eval=t_eval_cf,
        rtol=1e-6,
        atol=1e-8,
    )
    assert sol2.status == 0
    assert np.isfinite(sol2.y).all()
    assert np.min(sol2.y) >= -1e-4
    assert np.max(sol2.y) <= 500.0
