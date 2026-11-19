"""Unit tests for smooth analytical external forcing u(t)."""

from __future__ import annotations

import numpy as np
import pytest

from ecoverse.core.forcing import CompiledForcing, eval_flow, eval_forcing, eval_light, eval_temp


def _make_forcing(**overrides: object) -> CompiledForcing:
    defaults = dict(
        diel=True,
        seasonal=True,
        a_I=0.4,
        b_c=1.0,
        cloud_w=np.array([0.2, -0.15, 0.1, 0.05, -0.2, 0.12, -0.08, 0.18], dtype=np.float64),
        cloud_f=np.linspace(0.02, 0.24, 8, dtype=np.float64),
        cloud_psi=np.linspace(0.1, 2.5, 8, dtype=np.float64),
        T_bar=16.0,
        A_season=8.0,
        A_diel=1.5,
        lag_T=25.0,
        temp_v=np.array([0.3, -0.2, 0.15, -0.1, 0.25, -0.12], dtype=np.float64),
        temp_g=np.linspace(0.01, 0.12, 6, dtype=np.float64),
        temp_chi=np.linspace(0.0, 2.0, 6, dtype=np.float64),
        q_base=0.5,
        storm_t=np.array([10.0, 45.0], dtype=np.float64),
        storm_A=np.array([1.5, 2.0], dtype=np.float64),
        storm_tau=np.array([2.0, 3.0], dtype=np.float64),
        N_in_0=1.2,
        D_in=0.3,
        C_sat=15.0,
        interv_type=0,
        interv_t=180.0,
        interv_lambda_N=1.0,
        interv_delta_T=0.0,
        interv_lambda_q=1.0,
        interv_lambda_storm=1.0,
    )
    defaults.update(overrides)
    return CompiledForcing(**defaults)


def test_light_bounds_and_noon_peak() -> None:
    f = _make_forcing()
    ts = np.linspace(-180.0, 365.0, 10000)
    vals = np.array([eval_light(float(t), f) for t in ts])
    assert np.all(vals >= 0.0)
    assert np.all(vals <= 1.0)

    # Clear sky, no seasonal modulation: check daytime 6:00-18:00, night == 0, noon peak
    f_clear = _make_forcing(seasonal=False, cloud_w=np.zeros(8, dtype=np.float64))
    assert eval_light(0.0, f_clear) == 0.0    # midnight
    assert eval_light(0.24, f_clear) == 0.0   # 05:45
    assert eval_light(0.76, f_clear) == 0.0   # 18:15
    day_grid = np.linspace(0.0, 1.0, 961)
    day_vals = np.array([eval_light(float(t), f_clear) for t in day_grid])
    t_peak = day_grid[int(np.argmax(day_vals))]
    assert t_peak == pytest.approx(0.5, abs=2e-3)

    # diel=False replaces diel factor with 0.25
    f_nodiel = _make_forcing(diel=False, seasonal=False, cloud_w=np.zeros(8, dtype=np.float64))
    assert eval_light(0.5, f_nodiel) == pytest.approx(0.25 * eval_light(0.5, f_clear), abs=1e-12)
    assert eval_light(0.0, f_nodiel) == pytest.approx(eval_light(0.5, f_nodiel), abs=1e-12)


def test_temperature_diel_peak_and_storm_peak() -> None:
    f_temp = _make_forcing(seasonal=False, temp_v=np.zeros(6, dtype=np.float64))
    day_grid = np.linspace(0.0, 1.0, 961)
    t_vals = np.array([eval_temp(float(t), f_temp) for t in day_grid])
    t_peak = day_grid[int(np.argmax(t_vals))]
    assert t_peak == pytest.approx(0.625, abs=2e-3)  # 15:00

    # Storm pulse check
    f_storm = _make_forcing(
        q_base=0.4,
        storm_t=np.array([10.0], dtype=np.float64),
        storm_A=np.array([2.5], dtype=np.float64),
        storm_tau=np.array([2.0], dtype=np.float64),
    )
    assert eval_flow(9.9, f_storm) == 0.4
    assert eval_flow(10.0, f_storm) == 0.4
    assert eval_flow(12.0, f_storm) == pytest.approx(0.4 + 2.5, abs=1e-12)  # peak at t_k + tau_k


def test_forcing_c1_continuity() -> None:
    f = _make_forcing()
    eps = 1e-6

    # Light C^1 at sunrise (0.25) and sunset (0.75)
    for t0 in (0.25, 0.75, 100.25, 100.75):
        d_left = (eval_light(t0, f) - eval_light(t0 - eps, f)) / eps
        d_right = (eval_light(t0 + eps, f) - eval_light(t0, f)) / eps
        assert d_left == pytest.approx(d_right, abs=1e-4)

    # Flow C^1 at storm onsets
    for tk in f.storm_t:
        d_left = (eval_flow(float(tk), f) - eval_flow(float(tk) - eps, f)) / eps
        d_right = (eval_flow(float(tk) + eps, f) - eval_flow(float(tk), f)) / eps
        assert d_left == pytest.approx(d_right, abs=1e-4)

    # Full eval_forcing tuple consistency
    I, T, q, N_in = eval_forcing(12.0, f)
    assert np.isfinite([I, T, q, N_in]).all()
