"""System DNA specification (SystemSpec) and static compilation for Numba."""

from __future__ import annotations

from typing import Dict, List, Literal, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ecoverse.core.constants import (
    ALPHA_N,
    ALPHA_O,
    DEC_SHAPE_MAP,
    IDX_C,
    IDX_D,
    IDX_N,
    IDX_O,
    IDX_P1,
    IDX_P2,
    IDX_Z1,
    IDX_Z2,
    INTERV_MAP,
    INTERV_NONE,
    LIGHT_SHAPE_MAP,
    LOSS_SHAPE_MAP,
    N_CHANNELS,
    N_STATES,
    RES_SHAPE_MAP,
    STATE_NAMES,
    TEMP_SHAPE_MAP,
)
from ecoverse.core.forcing import CompiledForcing


class InterventionSpec(BaseModel):
    """Counterfactual intervention descriptor applied at t >= t_int."""

    type: Literal["none", "nutrient_step", "warming_shift", "flow_regime", "bio_pulse"] = "none"
    t_int: float = 180.0
    regime: Literal["train", "ood"] = "train"
    lambda_N: float = 1.0
    delta_T: float = 0.0
    lambda_q: float = 1.0
    lambda_storm: float = 1.0
    target_state: Literal["P1", "Z1"] = "P1"
    kappa: float = 1.0


class ForcingSpec(BaseModel):
    """Analytical external forcing specification u(t)."""

    diel: bool = True
    seasonal: bool = True
    a_I: float = Field(default=0.4, ge=0.0, lt=1.0)
    b_c: float = 1.0
    cloud_w: List[float] = Field(min_length=8, max_length=8)
    cloud_f: List[float] = Field(min_length=8, max_length=8)
    cloud_psi: List[float] = Field(min_length=8, max_length=8)
    T_bar: float = 16.0
    A_season: float = Field(default=8.0, ge=0.0)
    A_diel: float = Field(default=1.5, ge=0.0)
    lag_T: float = 25.0
    temp_v: List[float] = Field(min_length=6, max_length=6)
    temp_g: List[float] = Field(min_length=6, max_length=6)
    temp_chi: List[float] = Field(min_length=6, max_length=6)
    q_base: float = Field(default=0.1, ge=0.0)
    storm_t: List[float] = Field(default_factory=list)
    storm_A: List[float] = Field(default_factory=list)
    storm_tau: List[float] = Field(default_factory=list)
    N_in_0: float = Field(default=1.0, ge=0.0)
    D_in: float = Field(default=0.0, ge=0.0)
    C_sat: float = Field(default=15.0, ge=0.0)

    @model_validator(mode="after")
    def _check_storm_lengths(self) -> "ForcingSpec":
        if not (len(self.storm_t) == len(self.storm_A) == len(self.storm_tau)):
            raise ValueError("storm_t, storm_A, and storm_tau must have equal length.")
        if any(tau <= 0.0 for tau in self.storm_tau):
            raise ValueError("All storm_tau values must be strictly positive.")
        return self


class SystemSpec(BaseModel):
    """Semantic JSON-serializable DNA of an EcoVerse ecosystem instance."""

    sys_id: int = 0
    seed: int = 0
    preset: str
    split_tag: str = "train"
    state_names: List[str] = Field(default_factory=lambda: list(STATE_NAMES))
    active_states: Dict[str, bool]
    active_channels: List[int]
    shapes: Dict[str, str]
    params: Dict[str, float]
    sigma: Dict[str, float]
    delta: Dict[str, float]
    forcing: ForcingSpec
    initial_state: Dict[str, float]
    intervention: InterventionSpec = Field(default_factory=InterventionSpec)

    @model_validator(mode="after")
    def _validate_structure(self) -> "SystemSpec":
        if self.state_names != list(STATE_NAMES):
            raise ValueError(f"v0 expects state_names={list(STATE_NAMES)}, got {self.state_names}")
        for name in STATE_NAMES:
            if name not in self.active_states:
                raise ValueError(f"Missing active_states entry for '{name}'")
            if name not in self.sigma:
                raise ValueError(f"Missing sigma entry for '{name}'")
            if not (0.0 <= self.sigma[name] <= 1.0):
                raise ValueError(f"sigma['{name}']={self.sigma[name]} outside [0, 1]")
            if name not in self.delta:
                raise ValueError(f"Missing delta entry for '{name}'")
            if self.delta[name] not in (0.0, 1.0):
                raise ValueError(f"delta['{name}']={self.delta[name]} must be 0.0 or 1.0")
            if name not in self.initial_state:
                raise ValueError(f"Missing initial_state entry for '{name}'")
            if self.initial_state[name] < 0.0:
                raise ValueError(f"initial_state['{name}']={self.initial_state[name]} must be >= 0")

        for ch in self.active_channels:
            if not (0 <= ch < N_CHANNELS):
                raise ValueError(f"Invalid channel index {ch}; must be in [0, {N_CHANNELS - 1}]")

        required_shape_keys = (
            "res_N",
            "res_P",
            "res_Z",
            "light",
            "temp_prod",
            "temp_resp",
            "loss_P1",
            "loss_P2",
            "loss_Z1",
            "loss_Z2",
            "dec",
        )
        for sk in required_shape_keys:
            if sk not in self.shapes:
                raise ValueError(f"Missing shape key '{sk}'")

        if self.shapes["res_N"] not in RES_SHAPE_MAP:
            raise ValueError(f"Invalid res_N shape '{self.shapes['res_N']}'")
        if self.shapes["res_P"] not in RES_SHAPE_MAP:
            raise ValueError(f"Invalid res_P shape '{self.shapes['res_P']}'")
        if self.shapes["res_Z"] not in RES_SHAPE_MAP:
            raise ValueError(f"Invalid res_Z shape '{self.shapes['res_Z']}'")
        if self.shapes["light"] not in LIGHT_SHAPE_MAP:
            raise ValueError(f"Invalid light shape '{self.shapes['light']}'")
        if self.shapes["temp_prod"] not in TEMP_SHAPE_MAP:
            raise ValueError(f"Invalid temp_prod shape '{self.shapes['temp_prod']}'")
        if self.shapes["temp_resp"] not in TEMP_SHAPE_MAP:
            raise ValueError(f"Invalid temp_resp shape '{self.shapes['temp_resp']}'")
        for lk in ("loss_P1", "loss_P2", "loss_Z1", "loss_Z2"):
            if self.shapes[lk] not in LOSS_SHAPE_MAP:
                raise ValueError(f"Invalid {lk} shape '{self.shapes[lk]}'")
        if self.shapes["dec"] not in DEC_SHAPE_MAP:
            raise ValueError(f"Invalid dec shape '{self.shapes['dec']}'")

        positive_k_params = (
            "K_N1",
            "K_N2",
            "K_P1",
            "K_P2",
            "K_Z1",
            "K_D",
            "I_k",
            "I_opt",
            "sigma_T",
        )
        for pk in positive_k_params:
            if pk in self.params and self.params[pk] <= 0.0:
                raise ValueError(f"Parameter '{pk}' must be > 0, got {self.params[pk]}")

        for pk, val in self.params.items():
            if not np.isfinite(val):
                raise ValueError(f"Non-finite parameter '{pk}': {val}")
            if pk != "T_opt" and val < 0.0:
                raise ValueError(f"Parameter '{pk}' must be >= 0, got {val}")

        return self


class CompiledParams(NamedTuple):
    """Flat float64/int64/bool array-backed parameter struct for Numba."""

    state_mask: np.ndarray          # float64 (8,)
    channel_weights: np.ndarray     # float64 (23,)
    sigma: np.ndarray               # float64 (8,)
    delta: np.ndarray               # float64 (8,)
    bypass_N: bool
    bypass_O: bool
    bypass_C: bool

    # Shape enum IDs
    shape_res_N: int
    shape_res_P: int
    shape_res_Z: int
    shape_light: int
    shape_temp_prod: int
    shape_temp_resp: int
    shape_loss_P1: int
    shape_loss_P2: int
    shape_loss_Z1: int
    shape_loss_Z2: int
    shape_dec: int

    # Kinetic parameters
    v_max1: float
    v_max2: float
    k_resp_P1: float
    k_resp_P2: float
    m_P1a: float
    m_P1b: float
    m_P2a: float
    m_P2b: float
    g_max11: float
    g_max12: float
    g_max2: float
    gamma_1: float
    gamma_2: float
    c_12: float
    c_21: float
    k_resp_Z1: float
    k_resp_Z2: float
    m_Z1a: float
    m_Z1b: float
    m_Z2a: float
    m_Z2b: float
    k_dec: float
    k_gas_O: float
    s_P1: float
    s_P2: float
    s_D: float
    eta_q: float
    b_SOD: float
    b_N: float

    # Half-saturation & shape parameters
    K_N1: float
    K_N2: float
    K_P1: float
    K_P2: float
    K_Z1: float
    K_D: float
    n_hill_N: float
    n_hill_P: float
    n_hill_Z: float
    K_I_N1: float
    K_I_N2: float
    K_I_P1: float
    K_I_P2: float
    K_I_Z1: float
    I_k: float
    I_opt: float
    theta_T_prod: float
    theta_T_resp: float
    T_opt: float
    sigma_T: float


class CompiledSystem(NamedTuple):
    """Complete compiled representation passed into the Numba solver."""

    params: CompiledParams
    forcing: CompiledForcing
    x0: np.ndarray


def build_stoich_matrix(gamma_1: float, gamma_2: float) -> np.ndarray:
    """Return dense (23, 8) stoichiometric matrix S matching Section 4.4."""
    S = np.zeros((N_CHANNELS, N_STATES), dtype=np.float64)
    # 0, 1: Autotrophic production P1, P2
    S[0, IDX_N] = -ALPHA_N; S[0, IDX_P1] = 1.0; S[0, IDX_O] = ALPHA_O; S[0, IDX_C] = -1.0
    S[1, IDX_N] = -ALPHA_N; S[1, IDX_P2] = 1.0; S[1, IDX_O] = ALPHA_O; S[1, IDX_C] = -1.0
    # 2, 3: Producer respiration P1, P2
    S[2, IDX_N] = ALPHA_N; S[2, IDX_P1] = -1.0; S[2, IDX_O] = -ALPHA_O; S[2, IDX_C] = 1.0
    S[3, IDX_N] = ALPHA_N; S[3, IDX_P2] = -1.0; S[3, IDX_O] = -ALPHA_O; S[3, IDX_C] = 1.0
    # 4, 5: Producer mortality/exudation
    S[4, IDX_P1] = -1.0; S[4, IDX_D] = 1.0
    S[5, IDX_P2] = -1.0; S[5, IDX_D] = 1.0
    # 6, 7: Primary grazing P1, P2 -> Z1
    S[6, IDX_P1] = -1.0; S[6, IDX_Z1] = gamma_1; S[6, IDX_D] = 1.0 - gamma_1
    S[7, IDX_P2] = -1.0; S[7, IDX_Z1] = gamma_1; S[7, IDX_D] = 1.0 - gamma_1
    # 8: Secondary predation Z1 -> Z2
    S[8, IDX_Z1] = -1.0; S[8, IDX_Z2] = gamma_2; S[8, IDX_D] = 1.0 - gamma_2
    # 9, 10: Interference competition
    S[9, IDX_P1] = -1.0; S[9, IDX_D] = 1.0
    S[10, IDX_P2] = -1.0; S[10, IDX_D] = 1.0
    # 11, 12: Consumer respiration Z1, Z2
    S[11, IDX_N] = ALPHA_N; S[11, IDX_Z1] = -1.0; S[11, IDX_O] = -ALPHA_O; S[11, IDX_C] = 1.0
    S[12, IDX_N] = ALPHA_N; S[12, IDX_Z2] = -1.0; S[12, IDX_O] = -ALPHA_O; S[12, IDX_C] = 1.0
    # 13, 14: Consumer mortality Z1, Z2
    S[13, IDX_Z1] = -1.0; S[13, IDX_D] = 1.0
    S[14, IDX_Z2] = -1.0; S[14, IDX_D] = 1.0
    # 15: Detritus decomposition
    S[15, IDX_N] = ALPHA_N; S[15, IDX_D] = -1.0; S[15, IDX_O] = -ALPHA_O; S[15, IDX_C] = 1.0
    # 16, 17: Gas exchange O, C
    S[16, IDX_O] = 1.0
    S[17, IDX_C] = 1.0
    # 18, 19, 20: Sinking & burial
    S[18, IDX_P1] = -1.0
    S[19, IDX_P2] = -1.0
    S[20, IDX_D] = -1.0
    # 21, 22: Sediment O2 demand & anoxic N release
    S[21, IDX_O] = -ALPHA_O; S[21, IDX_C] = 1.0
    S[22, IDX_N] = 1.0
    return S


def compile_spec(spec: SystemSpec, apply_intervention: bool = False) -> CompiledSystem:
    """Compile a semantic SystemSpec into flat float64 arrays for Numba."""
    state_mask = np.array(
        [1.0 if spec.active_states[name] else 0.0 for name in STATE_NAMES],
        dtype=np.float64,
    )
    channel_weights = np.zeros(N_CHANNELS, dtype=np.float64)
    for ch in spec.active_channels:
        channel_weights[ch] = 1.0

    sigma = np.array([spec.sigma[name] for name in STATE_NAMES], dtype=np.float64)
    delta = np.array([spec.delta[name] for name in STATE_NAMES], dtype=np.float64)

    bypass_N = bool(state_mask[IDX_N] == 0.0)
    bypass_O = bool(state_mask[IDX_O] == 0.0)
    bypass_C = bool(state_mask[IDX_C] == 0.0)

    p = spec.params
    n_hill_default = float(p.get("n_hill", 2.5))
    k_i_ratio_default = float(p.get("K_I_ratio", 4.0))

    K_N1 = float(p.get("K_N1", 0.2))
    K_N2 = float(p.get("K_N2", K_N1))
    K_P1 = float(p.get("K_P1", 0.5))
    K_P2 = float(p.get("K_P2", K_P1))
    K_Z1 = float(p.get("K_Z1", 0.5))

    compiled_params = CompiledParams(
        state_mask=state_mask,
        channel_weights=channel_weights,
        sigma=sigma,
        delta=delta,
        bypass_N=bypass_N,
        bypass_O=bypass_O,
        bypass_C=bypass_C,
        shape_res_N=RES_SHAPE_MAP[spec.shapes["res_N"]],
        shape_res_P=RES_SHAPE_MAP[spec.shapes["res_P"]],
        shape_res_Z=RES_SHAPE_MAP[spec.shapes["res_Z"]],
        shape_light=LIGHT_SHAPE_MAP[spec.shapes["light"]],
        shape_temp_prod=TEMP_SHAPE_MAP[spec.shapes["temp_prod"]],
        shape_temp_resp=TEMP_SHAPE_MAP[spec.shapes["temp_resp"]],
        shape_loss_P1=LOSS_SHAPE_MAP[spec.shapes["loss_P1"]],
        shape_loss_P2=LOSS_SHAPE_MAP[spec.shapes["loss_P2"]],
        shape_loss_Z1=LOSS_SHAPE_MAP[spec.shapes["loss_Z1"]],
        shape_loss_Z2=LOSS_SHAPE_MAP[spec.shapes["loss_Z2"]],
        shape_dec=DEC_SHAPE_MAP[spec.shapes["dec"]],
        v_max1=float(p.get("v_max1", 0.0)),
        v_max2=float(p.get("v_max2", 0.0)),
        k_resp_P1=float(p.get("k_resp_P1", 0.0)),
        k_resp_P2=float(p.get("k_resp_P2", 0.0)),
        m_P1a=float(p.get("m_P1a", 0.0)),
        m_P1b=float(p.get("m_P1b", 0.0)),
        m_P2a=float(p.get("m_P2a", 0.0)),
        m_P2b=float(p.get("m_P2b", 0.0)),
        g_max11=float(p.get("g_max11", 0.0)),
        g_max12=float(p.get("g_max12", 0.0)),
        g_max2=float(p.get("g_max2", 0.0)),
        gamma_1=float(p.get("gamma_1", 0.5)),
        gamma_2=float(p.get("gamma_2", 0.5)),
        c_12=float(p.get("c_12", 0.0)),
        c_21=float(p.get("c_21", 0.0)),
        k_resp_Z1=float(p.get("k_resp_Z1", 0.0)),
        k_resp_Z2=float(p.get("k_resp_Z2", 0.0)),
        m_Z1a=float(p.get("m_Z1a", 0.0)),
        m_Z1b=float(p.get("m_Z1b", 0.0)),
        m_Z2a=float(p.get("m_Z2a", 0.0)),
        m_Z2b=float(p.get("m_Z2b", 0.0)),
        k_dec=float(p.get("k_dec", 0.0)),
        k_gas_O=float(p.get("k_gas_O", 0.0)),
        s_P1=float(p.get("s_P1", 0.0)),
        s_P2=float(p.get("s_P2", 0.0)),
        s_D=float(p.get("s_D", 0.0)),
        eta_q=float(p.get("eta_q", 1.0)),
        b_SOD=float(p.get("b_SOD", 0.0)),
        b_N=float(p.get("b_N", 0.0)),
        K_N1=K_N1,
        K_N2=K_N2,
        K_P1=K_P1,
        K_P2=K_P2,
        K_Z1=K_Z1,
        K_D=float(p.get("K_D", 1.0)),
        n_hill_N=float(p.get("n_hill_N", n_hill_default)),
        n_hill_P=float(p.get("n_hill_P", n_hill_default)),
        n_hill_Z=float(p.get("n_hill_Z", n_hill_default)),
        K_I_N1=float(p.get("K_I_N1", k_i_ratio_default * K_N1)),
        K_I_N2=float(p.get("K_I_N2", k_i_ratio_default * K_N2)),
        K_I_P1=float(p.get("K_I_P1", k_i_ratio_default * K_P1)),
        K_I_P2=float(p.get("K_I_P2", k_i_ratio_default * K_P2)),
        K_I_Z1=float(p.get("K_I_Z1", k_i_ratio_default * K_Z1)),
        I_k=float(p.get("I_k", 0.25)),
        I_opt=float(p.get("I_opt", 0.6)),
        theta_T_prod=float(p.get("theta_T_prod", 1.5)),
        theta_T_resp=float(p.get("theta_T_resp", 2.0)),
        T_opt=float(p.get("T_opt", spec.forcing.T_bar)),
        sigma_T=float(p.get("sigma_T", 7.5)),
    )

    f = spec.forcing
    if apply_intervention:
        interv_type = INTERV_MAP[spec.intervention.type]
        interv_t = float(spec.intervention.t_int)
        interv_lambda_N = float(spec.intervention.lambda_N)
        interv_delta_T = float(spec.intervention.delta_T)
        interv_lambda_q = float(spec.intervention.lambda_q)
        interv_lambda_storm = float(spec.intervention.lambda_storm)
    else:
        interv_type = INTERV_NONE
        interv_t = float(spec.intervention.t_int)
        interv_lambda_N = 1.0
        interv_delta_T = 0.0
        interv_lambda_q = 1.0
        interv_lambda_storm = 1.0

    compiled_forcing = CompiledForcing(
        diel=bool(f.diel),
        seasonal=bool(f.seasonal),
        a_I=float(f.a_I),
        b_c=float(f.b_c),
        cloud_w=np.asarray(f.cloud_w, dtype=np.float64),
        cloud_f=np.asarray(f.cloud_f, dtype=np.float64),
        cloud_psi=np.asarray(f.cloud_psi, dtype=np.float64),
        T_bar=float(f.T_bar),
        A_season=float(f.A_season),
        A_diel=float(f.A_diel),
        lag_T=float(f.lag_T),
        temp_v=np.asarray(f.temp_v, dtype=np.float64),
        temp_g=np.asarray(f.temp_g, dtype=np.float64),
        temp_chi=np.asarray(f.temp_chi, dtype=np.float64),
        q_base=float(f.q_base),
        storm_t=np.asarray(f.storm_t, dtype=np.float64),
        storm_A=np.asarray(f.storm_A, dtype=np.float64),
        storm_tau=np.asarray(f.storm_tau, dtype=np.float64),
        N_in_0=float(f.N_in_0),
        D_in=float(f.D_in),
        C_sat=float(f.C_sat),
        interv_type=interv_type,
        interv_t=interv_t,
        interv_lambda_N=interv_lambda_N,
        interv_delta_T=interv_delta_T,
        interv_lambda_q=interv_lambda_q,
        interv_lambda_storm=interv_lambda_storm,
    )

    x0 = np.array(
        [
            float(spec.initial_state[name]) if spec.active_states[name] else 0.0
            for name in STATE_NAMES
        ],
        dtype=np.float64,
    )

    return CompiledSystem(params=compiled_params, forcing=compiled_forcing, x0=x0)


def apply_bio_pulse(x_at_tint: np.ndarray, spec: SystemSpec) -> np.ndarray:
    """Return cloned state vector with instantaneous bio_pulse applied if configured."""
    x_cf0 = np.array(x_at_tint, dtype=np.float64, copy=True)
    if spec.intervention.type == "bio_pulse":
        target = spec.intervention.target_state
        idx = IDX_P1 if target == "P1" else IDX_Z1
        if spec.active_states.get(target, False):
            x_cf0[idx] = max(0.0, x_cf0[idx] * float(spec.intervention.kappa))
    return x_cf0
