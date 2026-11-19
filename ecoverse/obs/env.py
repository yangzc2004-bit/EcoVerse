"""Reserved v0 Gym-style interactive control sandbox skeleton (for v3 closed-loop control)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from ecoverse.core.constants import IDX_P1, IDX_Z1, N_STATES
from ecoverse.core.dynamics import compute_rates_and_forcing, rhs_jit
from ecoverse.core.spec import SystemSpec, compile_spec


class EcoVerseEnv:
    """Minimal step-wise interactive ecosystem control skeleton.

    Action dict / array supports dynamic nutrient scaling (lambda_N),
    flow scaling (lambda_q), and bio-manipulation scaling (kappa_P1, kappa_Z1).
    """

    def __init__(
        self,
        spec: SystemSpec,
        step_days: float = 1.0,
        max_days: float = 365.0,
    ) -> None:
        self.base_spec = spec
        self.step_days = float(step_days)
        self.max_days = float(max_days)
        self.t: float = 0.0
        self.state: np.ndarray = np.zeros(N_STATES, dtype=np.float64)
        self._compiled = compile_spec(spec, apply_intervention=False)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._compiled = compile_spec(self.base_spec, apply_intervention=False)
        self.t = 0.0
        self.state = self._compiled.x0.copy()
        return self.state.copy(), {"t": self.t, "seed": seed}

    def step(
        self, action: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = action or {}
        forcing = self._compiled.forcing
        params = self._compiled.params

        # Apply transient step control via forcing overrides if provided
        if "lambda_N" in act or "lambda_q" in act or "delta_T" in act:
            forcing = forcing._replace(
                N_in_0=self.base_spec.forcing.N_in_0 * float(act.get("lambda_N", 1.0)),
                q_base=self.base_spec.forcing.q_base * float(act.get("lambda_q", 1.0)),
                T_bar=self.base_spec.forcing.T_bar + float(act.get("delta_T", 0.0)),
            )
        if "kappa_P1" in act:
            self.state[IDX_P1] = max(0.0, self.state[IDX_P1] * float(act["kappa_P1"]))
        if "kappa_Z1" in act:
            self.state[IDX_Z1] = max(0.0, self.state[IDX_Z1] * float(act["kappa_Z1"]))

        t_next = min(self.t + self.step_days, self.max_days)
        sol = solve_ivp(
            lambda t, y: rhs_jit(t, y, params, forcing),
            (self.t, t_next),
            self.state,
            method="LSODA",
            rtol=1e-6,
            atol=1e-8,
        )
        self.t = t_next
        self.state = np.maximum(sol.y[:, -1], 0.0)
        rates, drivers = compute_rates_and_forcing(self.t, self.state, params, forcing)

        # Simple ecological balance reward placeholder (NEP / dissolved oxygen health)
        reward = float(rates[0] + rates[1] - (rates[2] + rates[3] + rates[11] + rates[12] + rates[15]))
        terminated = self.t >= self.max_days
        truncated = sol.status != 0
        return self.state.copy(), reward, terminated, truncated, {
            "t": self.t,
            "rates": rates,
            "drivers": drivers,
        }
