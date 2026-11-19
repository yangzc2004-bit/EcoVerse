"""Single-system ODE solver (burn-in -> formal 365d -> counterfactual fork)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ecoverse.core.constants import (
    BURNIN_DAYS,
    INTERVENTION_DAY,
    INTERVENTION_STEP,
    N_STEPS_FORMAL,
    make_formal_time_grid,
)
from ecoverse.core.dynamics import eval_trajectory_rates_jit, rhs_jit
from ecoverse.core.spec import SystemSpec, apply_bio_pulse, compile_spec


class SolveTimeoutError(RuntimeError):
    """Raised when a single ODE integration exceeds its wall-clock budget."""


@dataclass
class SimulationResult:
    """Outputs and diagnostics from solving burn-in, baseline, and counterfactual."""

    ok: bool
    fail_reason: str = ""
    x_burnin_last30: Optional[np.ndarray] = None  # (181, 8) float64
    x_true: Optional[np.ndarray] = None           # (2190, 8) float32
    r_true: Optional[np.ndarray] = None           # (2190, 23) float32
    u_true: Optional[np.ndarray] = None           # (2190, 4) float32
    x_cf: Optional[np.ndarray] = None             # (1110, 8) float32
    r_cf: Optional[np.ndarray] = None             # (1110, 23) float32
    u_cf: Optional[np.ndarray] = None             # (1110, 4) float32
    raw_min: float = 0.0
    raw_max: float = 0.0
    solve_seconds: float = 0.0


def _make_rhs_with_deadline(params: object, forcing: object, deadline: float):
    counter = 0

    def _rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal counter
        counter += 1
        if (counter & 255) == 0 and time.monotonic() > deadline:
            raise SolveTimeoutError("ODE solve exceeded timeout budget.")
        return rhs_jit(t, y, params, forcing)

    return _rhs


def solve_system(spec: SystemSpec, timeout_s: float = 10.0) -> SimulationResult:
    """Integrate burn-in [-180, 0], formal [0, 365], and counterfactual [180, 365]."""
    t_start = time.monotonic()
    deadline = t_start + timeout_s

    compiled_base = compile_spec(spec, apply_intervention=False)
    t_burn_eval = np.linspace(-30.0, 0.0, 181, dtype=np.float64)
    t_formal = make_formal_time_grid()
    assert t_formal.shape[0] == N_STEPS_FORMAL
    t_cf = t_formal[INTERVENTION_STEP:]

    rhs_base = _make_rhs_with_deadline(compiled_base.params, compiled_base.forcing, deadline)

    try:
        # 1. Burn-in [-180, 0]
        sol_burn = solve_ivp(
            rhs_base,
            (-BURNIN_DAYS, 0.0),
            compiled_base.x0,
            method="LSODA",
            t_eval=t_burn_eval,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol_burn.status != 0 or not np.isfinite(sol_burn.y).all():
            return SimulationResult(
                ok=False,
                fail_reason="burnin_solver_failed",
                solve_seconds=time.monotonic() - t_start,
            )

        burn_min = float(np.min(sol_burn.y))
        burn_max = float(np.max(sol_burn.y))
        if burn_min < -1e-4 or burn_max > 500.0:
            return SimulationResult(
                ok=False,
                fail_reason="burnin_bounds_violated",
                raw_min=burn_min,
                raw_max=burn_max,
                solve_seconds=time.monotonic() - t_start,
            )

        x_burn_last30 = np.maximum(sol_burn.y.T, 0.0)
        x_formal_0 = x_burn_last30[-1]

        # 2. Formal baseline [0, 365)
        sol_base = solve_ivp(
            rhs_base,
            (0.0, float(t_formal[-1])),
            x_formal_0,
            method="LSODA",
            t_eval=t_formal,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol_base.status != 0 or not np.isfinite(sol_base.y).all():
            return SimulationResult(
                ok=False,
                fail_reason="formal_solver_failed",
                solve_seconds=time.monotonic() - t_start,
            )

        x_base_raw = sol_base.y.T  # (2190, 8)
        base_min = float(np.min(x_base_raw))
        base_max = float(np.max(x_base_raw))
        if base_min < -1e-4 or base_max > 500.0:
            return SimulationResult(
                ok=False,
                fail_reason="formal_bounds_violated",
                raw_min=min(burn_min, base_min),
                raw_max=max(burn_max, base_max),
                solve_seconds=time.monotonic() - t_start,
            )

        # 3. Counterfactual branch [180, 365)
        compiled_cf = compile_spec(spec, apply_intervention=True)
        rhs_cf = _make_rhs_with_deadline(compiled_cf.params, compiled_cf.forcing, deadline)
        x_cf_0 = apply_bio_pulse(np.maximum(x_base_raw[INTERVENTION_STEP], 0.0), spec)

        sol_cf = solve_ivp(
            rhs_cf,
            (INTERVENTION_DAY, float(t_cf[-1])),
            x_cf_0,
            method="LSODA",
            t_eval=t_cf,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol_cf.status != 0 or not np.isfinite(sol_cf.y).all():
            return SimulationResult(
                ok=False,
                fail_reason="cf_solver_failed",
                solve_seconds=time.monotonic() - t_start,
            )

        x_cf_raw = sol_cf.y.T  # (1110, 8)
        cf_min = float(np.min(x_cf_raw))
        cf_max = float(np.max(x_cf_raw))
        raw_min = min(burn_min, base_min, cf_min)
        raw_max = max(burn_max, base_max, cf_max)
        if raw_min < -1e-4 or raw_max > 500.0:
            return SimulationResult(
                ok=False,
                fail_reason="cf_bounds_violated",
                raw_min=raw_min,
                raw_max=raw_max,
                solve_seconds=time.monotonic() - t_start,
            )

        # Clamp tiny negative solver tolerances (-1e-4, 0) -> 0.0
        x_base_clean = np.maximum(x_base_raw, 0.0)
        x_cf_clean = np.maximum(x_cf_raw, 0.0)

        r_base, u_base = eval_trajectory_rates_jit(
            t_formal, x_base_clean, compiled_base.params, compiled_base.forcing
        )
        r_cf, u_cf = eval_trajectory_rates_jit(
            t_cf, x_cf_clean, compiled_cf.params, compiled_cf.forcing
        )

        return SimulationResult(
            ok=True,
            fail_reason="",
            x_burnin_last30=x_burn_last30,
            x_true=x_base_clean.astype(np.float32),
            r_true=r_base.astype(np.float32),
            u_true=u_base.astype(np.float32),
            x_cf=x_cf_clean.astype(np.float32),
            r_cf=r_cf.astype(np.float32),
            u_cf=u_cf.astype(np.float32),
            raw_min=raw_min,
            raw_max=raw_max,
            solve_seconds=time.monotonic() - t_start,
        )
    except SolveTimeoutError:
        return SimulationResult(
            ok=False,
            fail_reason="timeout",
            solve_seconds=time.monotonic() - t_start,
        )
