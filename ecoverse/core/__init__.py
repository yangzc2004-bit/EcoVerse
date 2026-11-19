"""Core mathematical layer for EcoVerse."""

from ecoverse.core.constants import (
    ALPHA_N,
    ALPHA_O,
    CHANNEL_NAMES,
    DRIVER_NAMES,
    N_CHANNELS,
    N_DRIVERS,
    N_STATES,
    STATE_NAMES,
    compute_macro_fluxes,
    make_formal_time_grid,
)
from ecoverse.core.dynamics import (
    compute_rates_and_forcing,
    eval_trajectory_rates_jit,
    rhs_jit,
    warmup_jit,
)
from ecoverse.core.forcing import CompiledForcing, eval_forcing
from ecoverse.core.spec import (
    CompiledParams,
    CompiledSystem,
    ForcingSpec,
    InterventionSpec,
    SystemSpec,
    apply_bio_pulse,
    compile_spec,
)

__all__ = [
    "ALPHA_N",
    "ALPHA_O",
    "CHANNEL_NAMES",
    "DRIVER_NAMES",
    "N_CHANNELS",
    "N_DRIVERS",
    "N_STATES",
    "STATE_NAMES",
    "compute_macro_fluxes",
    "make_formal_time_grid",
    "compute_rates_and_forcing",
    "eval_trajectory_rates_jit",
    "rhs_jit",
    "warmup_jit",
    "CompiledForcing",
    "eval_forcing",
    "CompiledParams",
    "CompiledSystem",
    "ForcingSpec",
    "InterventionSpec",
    "SystemSpec",
    "apply_bio_pulse",
    "compile_spec",
]
