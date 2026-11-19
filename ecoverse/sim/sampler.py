"""Config-driven parameter sampling for EcoVerse v0 presets and benchmark variants."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ecoverse.core.constants import P2_CHANNELS, STATE_NAMES
from ecoverse.core.shapes import o_sat_freshwater
from ecoverse.core.spec import ForcingSpec, InterventionSpec, SystemSpec

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sample_dist(spec: Any, rng: np.random.Generator) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    dist = spec["dist"]
    if dist == "CONST":
        return float(spec["value"])
    lo, hi = float(spec["range"][0]), float(spec["range"][1])
    if dist == "U":
        return float(rng.uniform(lo, hi))
    if dist == "LU":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    raise ValueError(f"Unknown distribution type '{dist}'")


def _resolve_preset_config(
    preset_or_variant: str, presets_cfg: Dict[str, Any]
) -> tuple[str, Dict[str, Any]]:
    presets = presets_cfg["presets"]
    variants_2b = presets_cfg.get("variants_2b", {})
    if preset_or_variant in presets:
        return preset_or_variant, copy.deepcopy(presets[preset_or_variant])
    if preset_or_variant in variants_2b:
        var = variants_2b[preset_or_variant]
        base_name = var["base"]
        cfg = copy.deepcopy(presets[base_name])
        for k, v in var.get("active_states_override", {}).items():
            cfg["active_states"][k] = v
        if "optional_p2_prob" in var:
            cfg["optional_p2_prob"] = var["optional_p2_prob"]
        channels = set(cfg["active_channels"]) | set(var.get("add_channels", []))
        cfg["active_channels"] = sorted(channels)
        for k, v in var.get("sigma_override", {}).items():
            cfg["sigma"][k] = v
        for k, v in var.get("delta_override", {}).items():
            cfg["delta"][k] = v
        for k, v in var.get("priors_override", {}).items():
            cfg["priors"][k] = v
        return base_name, cfg
    raise KeyError(f"Unknown preset or variant '{preset_or_variant}'")


def _sample_intervention(
    base_preset: str,
    interv_regime: str,
    shapes: Dict[str, str],
    rng: np.random.Generator,
    interv_cfg: Dict[str, Any],
) -> InterventionSpec:
    valid = list(interv_cfg["preset_valid_interventions"][base_preset])
    # Avoid vacuous warming_shift when both temp shapes are CONST
    if (
        "warming_shift" in valid
        and shapes["temp_prod"] == "CONST"
        and shapes["temp_resp"] == "CONST"
    ):
        valid = [t for t in valid if t != "warming_shift"]
    if not valid:
        valid = ["bio_pulse"]

    itype = str(rng.choice(valid))
    t_int = float(interv_cfg.get("t_int", 180.0))
    rcfg = interv_cfg["regimes"][interv_regime][itype]

    if itype == "nutrient_step":
        rng_pair = rcfg["lambda_N_ranges"][int(rng.integers(0, len(rcfg["lambda_N_ranges"])))]
        lam = float(rng.uniform(rng_pair[0], rng_pair[1]))
        return InterventionSpec(
            type="nutrient_step", t_int=t_int, regime=interv_regime, lambda_N=lam
        )

    if itype == "warming_shift":
        rng_pair = rcfg["delta_T_ranges"][int(rng.integers(0, len(rcfg["delta_T_ranges"])))]
        dT = float(rng.uniform(rng_pair[0], rng_pair[1]))
        return InterventionSpec(
            type="warming_shift", t_int=t_int, regime=interv_regime, delta_T=dT
        )

    if itype == "flow_regime":
        lq_lo, lq_hi = rcfg["lambda_q_range"]
        ls_lo, ls_hi = rcfg["lambda_storm_range"]
        lam_q = float(rng.uniform(lq_lo, lq_hi))
        if rcfg.get("allow_zero_storm", False) and rng.random() < 0.5:
            lam_storm = 0.0
        else:
            lam_storm = float(rng.uniform(ls_lo, ls_hi))
        return InterventionSpec(
            type="flow_regime",
            t_int=t_int,
            regime=interv_regime,
            lambda_q=lam_q,
            lambda_storm=lam_storm,
        )

    if itype == "bio_pulse":
        rng_pair = rcfg["kappa_ranges"][int(rng.integers(0, len(rcfg["kappa_ranges"])))]
        kappa = float(rng.uniform(rng_pair[0], rng_pair[1]))
        target = str(rng.choice(rcfg["targets"]))
        return InterventionSpec(
            type="bio_pulse",
            t_int=t_int,
            regime=interv_regime,
            target_state=target,  # type: ignore[arg-type]
            kappa=kappa,
        )

    raise ValueError(f"Unsupported intervention type '{itype}'")


def sample_system_spec(
    preset_or_variant: str,
    rng: np.random.Generator,
    seed: int = 0,
    sys_id: int = 0,
    split_tag: str = "train",
    shape_mode: str = "train",
    interv_regime: str = "train",
    config_dir: Optional[Path] = None,
) -> SystemSpec:
    """Sample a complete SystemSpec according to preset priors and Section 4.5-4.6 rules."""
    cdir = config_dir or CONFIG_DIR
    presets_cfg = load_yaml(cdir / "presets.yaml")
    shapes_cfg = load_yaml(cdir / "shapes.yaml")
    interv_cfg = load_yaml(cdir / "interventions.yaml")

    base_preset, pcfg = _resolve_preset_config(preset_or_variant, presets_cfg)
    aux = presets_cfg["auxiliary_priors"]
    priors = pcfg["priors"]

    # 1. Active states & optional P2 rule (Section 4.5)
    active_states: Dict[str, bool] = {s: bool(pcfg["active_states"][s]) for s in STATE_NAMES}
    p2_prob = float(pcfg.get("optional_p2_prob", 0.0))
    if active_states["P2"]:
        enable_p2 = bool(rng.random() < p2_prob)
        active_states["P2"] = enable_p2

    active_channels = list(pcfg["active_channels"])
    if not active_states["P2"]:
        p2_set = set(P2_CHANNELS)
        active_channels = [ch for ch in active_channels if ch not in p2_set]
    ch_set = set(active_channels)

    # 2. Functional response shapes
    shape_pool = shapes_cfg["train_allowed" if shape_mode == "train" else "heldout_2c"]
    overrides = pcfg.get("shape_overrides_train", {}) if shape_mode == "train" else {}

    def _pick_shape(family: str, override_key: Optional[str] = None) -> str:
        key = override_key or family
        choices = overrides.get(key, shape_pool[family])
        return str(rng.choice(choices))

    shapes: Dict[str, str] = {
        "res_N": _pick_shape("res"),
        "res_P": _pick_shape("res"),
        "res_Z": _pick_shape("res"),
        "light": _pick_shape("light"),
        "temp_prod": _pick_shape("temp", "temp_prod"),
        "temp_resp": _pick_shape("temp", "temp_resp"),
        "loss_P1": "QUADRATIC" if pcfg.get("force_quadratic_producer_loss", False) else _pick_shape("loss"),
        "loss_P2": "QUADRATIC" if pcfg.get("force_quadratic_producer_loss", False) else _pick_shape("loss"),
        "loss_Z1": _pick_shape("loss"),
        "loss_Z2": _pick_shape("loss"),
        "dec": _pick_shape("dec"),
    }

    # 3. Sample forcing parameters (Section 4.6)
    diel = bool(pcfg["diel"])
    seasonal = bool(rng.random() < float(pcfg.get("seasonal_prob", 1.0)))

    T_bar = _sample_dist(aux["T_bar"], rng)
    A_season = _sample_dist(aux["A_season"], rng) if seasonal else 0.0
    A_diel = _sample_dist(aux["A_diel"], rng) if diel else 0.0
    lag_T = _sample_dist(aux["lag_T"], rng)

    a_I = _sample_dist(aux["a_I"], rng) if seasonal else 0.0
    b_c = _sample_dist(aux["b_c"], rng)
    cloud_w = [float(rng.uniform(*aux["cloud_w"]["range"])) for _ in range(8)]
    cloud_f = [float(rng.uniform(*aux["cloud_f"]["range"])) for _ in range(8)]
    cloud_psi = [float(rng.uniform(0.0, 2.0 * np.pi)) for _ in range(8)]

    if base_preset == "chemostat" and not seasonal and not diel:
        temp_v = [0.0] * 6
    else:
        temp_v = [float(rng.uniform(*aux["temp_v"]["range"])) for _ in range(6)]
    temp_g = [float(rng.uniform(*aux["temp_g"]["range"])) for _ in range(6)]
    temp_chi = [float(rng.uniform(0.0, 2.0 * np.pi)) for _ in range(6)]

    q_base = _sample_dist(priors["q_base"], rng)
    N_in_0 = _sample_dist(priors["N_in_0"], rng)
    D_in = _sample_dist(pcfg["D_in"], rng)
    C_sat = _sample_dist(aux["C_sat"], rng)

    storm_rate_range = pcfg.get("storm_rate", [0.0, 0.0])
    storm_rate = float(rng.uniform(storm_rate_range[0], storm_rate_range[1])) if storm_rate_range[1] > 0.0 else 0.0
    storm_t: List[float] = []
    storm_A: List[float] = []
    storm_tau: List[float] = []
    if storm_rate > 0.0 and q_base > 0.0:
        n_storms = int(rng.poisson(storm_rate * (365.0 + 180.0)))
        if n_storms > 0:
            t_arr = np.sort(rng.uniform(-180.0, 365.0, size=n_storms))
            amp_lo, amp_hi = pcfg["storm_amp_mult"]
            tau_lo, tau_hi = pcfg["storm_tau"]
            storm_t = [float(t) for t in t_arr]
            storm_A = [float(rng.uniform(amp_lo, amp_hi) * q_base) for _ in range(n_storms)]
            storm_tau = [float(rng.uniform(tau_lo, tau_hi)) for _ in range(n_storms)]

    forcing = ForcingSpec(
        diel=diel,
        seasonal=seasonal,
        a_I=a_I,
        b_c=b_c,
        cloud_w=cloud_w,
        cloud_f=cloud_f,
        cloud_psi=cloud_psi,
        T_bar=T_bar,
        A_season=A_season,
        A_diel=A_diel,
        lag_T=lag_T,
        temp_v=temp_v,
        temp_g=temp_g,
        temp_chi=temp_chi,
        q_base=q_base,
        storm_t=storm_t,
        storm_A=storm_A,
        storm_tau=storm_tau,
        N_in_0=N_in_0,
        D_in=D_in,
        C_sat=C_sat,
    )

    # 4. Anchor & relative kinetic parameters (Section 4.5 + 4.6)
    v_max1 = _sample_dist(priors["v_max1"], rng)
    v_max2 = v_max1 * _sample_dist(priors["v_max2_ratio"], rng) if active_states["P2"] else 0.0

    k_resp_P1 = v_max1 * _sample_dist(priors["k_resp_P_ratio"], rng) if (2 in ch_set) else 0.0
    k_resp_P2 = v_max2 * _sample_dist(priors["k_resp_P_ratio"], rng) if (3 in ch_set and active_states["P2"]) else 0.0

    m_P1a = v_max1 * _sample_dist(priors["m_Pa_ratio"], rng)
    m_P1b = _sample_dist(priors["m_Pb"], rng) if shapes["loss_P1"] == "QUADRATIC" else 0.0

    if active_states["P2"]:
        m_P2a = v_max2 * _sample_dist(priors["m_Pa_ratio"], rng)
        m_P2b = _sample_dist(priors["m_Pb"], rng) if shapes["loss_P2"] == "QUADRATIC" else 0.0
    else:
        m_P2a = 0.0
        m_P2b = 0.0

    g_max11 = v_max1 * _sample_dist(priors["g_max1_ratio"], rng)
    g_max12 = (v_max2 * _sample_dist(priors["g_max1_ratio"], rng)) if (7 in ch_set and active_states["P2"]) else 0.0
    g_max2 = (g_max11 * _sample_dist(priors["g_max2_ratio"], rng)) if (8 in ch_set and active_states["Z2"]) else 0.0

    gamma_1 = _sample_dist(priors["gamma"], rng)
    gamma_2 = _sample_dist(priors["gamma"], rng)

    z1_loss_tot = _sample_dist(priors["z_loss_ratio"], rng) * gamma_1 * g_max11
    if 11 in ch_set:
        beta_Z1 = _sample_dist(aux["beta_Z"], rng)
        k_resp_Z1 = beta_Z1 * z1_loss_tot
        m_Z1a = (1.0 - beta_Z1) * z1_loss_tot
    else:
        k_resp_Z1 = 0.0
        m_Z1a = z1_loss_tot
    m_Z1b = _sample_dist(aux["m_Zb"], rng) if shapes["loss_Z1"] == "QUADRATIC" else 0.0

    if active_states["Z2"]:
        z2_loss_tot = _sample_dist(priors["z_loss_ratio"], rng) * gamma_2 * g_max2
        if 12 in ch_set:
            beta_Z2 = _sample_dist(aux["beta_Z"], rng)
            k_resp_Z2 = beta_Z2 * z2_loss_tot
            m_Z2a = (1.0 - beta_Z2) * z2_loss_tot
        else:
            k_resp_Z2 = 0.0
            m_Z2a = z2_loss_tot
        m_Z2b = _sample_dist(aux["m_Zb"], rng) if shapes["loss_Z2"] == "QUADRATIC" else 0.0
    else:
        k_resp_Z2 = 0.0
        m_Z2a = 0.0
        m_Z2b = 0.0

    c_12 = _sample_dist(priors["c_comp"], rng) if (9 in ch_set and active_states["P2"]) else 0.0
    c_21 = _sample_dist(priors["c_comp"], rng) if (10 in ch_set and active_states["P2"]) else 0.0

    k_dec = _sample_dist(priors["k_dec"], rng) if (15 in ch_set) else 0.0
    k_gas_O = _sample_dist(priors["k_gas_O"], rng) if ((16 in ch_set) or (17 in ch_set)) else 0.0

    s_P1 = _sample_dist(priors["s_P"], rng) if (18 in ch_set) else 0.0
    s_P2 = _sample_dist(priors["s_P"], rng) if (19 in ch_set and active_states["P2"]) else 0.0
    s_D = _sample_dist(priors["s_D"], rng) if (20 in ch_set) else 0.0
    eta_q = _sample_dist(aux["eta_q"], rng)

    b_SOD = _sample_dist(priors["b_SOD"], rng) if (21 in ch_set) else 0.0
    b_N = _sample_dist(priors["b_N"], rng) if (22 in ch_set) else 0.0

    K_N1 = _sample_dist(priors["K_N"], rng)
    K_N2 = _sample_dist(priors["K_N"], rng)
    K_P1 = _sample_dist(priors["K_P"], rng)
    K_P2 = _sample_dist(priors["K_P"], rng)
    K_Z1 = _sample_dist(priors["K_Z1"], rng)
    K_D = _sample_dist(aux["K_D"], rng)

    n_hill_N = _sample_dist(aux["n_hill"], rng)
    n_hill_P = _sample_dist(aux["n_hill"], rng)
    n_hill_Z = _sample_dist(aux["n_hill"], rng)

    K_I_N1 = K_N1 * _sample_dist(aux["K_I_ratio"], rng)
    K_I_N2 = K_N2 * _sample_dist(aux["K_I_ratio"], rng)
    K_I_P1 = K_P1 * _sample_dist(aux["K_I_ratio"], rng)
    K_I_P2 = K_P2 * _sample_dist(aux["K_I_ratio"], rng)
    K_I_Z1 = K_Z1 * _sample_dist(aux["K_I_ratio"], rng)

    I_k = _sample_dist(aux["I_k"], rng)
    I_opt = _sample_dist(aux["I_opt"], rng)
    theta_T_prod = _sample_dist(priors["theta_T_prod"], rng)
    theta_T_resp = _sample_dist(priors["theta_T_resp"], rng)
    T_opt = T_bar + _sample_dist(aux["T_opt_offset"], rng)
    sigma_T = _sample_dist(aux["sigma_T"], rng)

    params: Dict[str, float] = {
        "v_max1": v_max1, "v_max2": v_max2,
        "k_resp_P1": k_resp_P1, "k_resp_P2": k_resp_P2,
        "m_P1a": m_P1a, "m_P1b": m_P1b, "m_P2a": m_P2a, "m_P2b": m_P2b,
        "g_max11": g_max11, "g_max12": g_max12, "g_max2": g_max2,
        "gamma_1": gamma_1, "gamma_2": gamma_2,
        "c_12": c_12, "c_21": c_21,
        "k_resp_Z1": k_resp_Z1, "k_resp_Z2": k_resp_Z2,
        "m_Z1a": m_Z1a, "m_Z1b": m_Z1b, "m_Z2a": m_Z2a, "m_Z2b": m_Z2b,
        "k_dec": k_dec, "k_gas_O": k_gas_O,
        "s_P1": s_P1, "s_P2": s_P2, "s_D": s_D, "eta_q": eta_q,
        "b_SOD": b_SOD, "b_N": b_N,
        "K_N1": K_N1, "K_N2": K_N2, "K_P1": K_P1, "K_P2": K_P2, "K_Z1": K_Z1, "K_D": K_D,
        "n_hill_N": n_hill_N, "n_hill_P": n_hill_P, "n_hill_Z": n_hill_Z,
        "K_I_N1": K_I_N1, "K_I_N2": K_I_N2, "K_I_P1": K_I_P1, "K_I_P2": K_I_P2, "K_I_Z1": K_I_Z1,
        "I_k": I_k, "I_opt": I_opt,
        "theta_T_prod": theta_T_prod, "theta_T_resp": theta_T_resp,
        "T_opt": T_opt, "sigma_T": sigma_T,
    }

    # 5. Boundary sigma & delta
    sigma: Dict[str, float] = {}
    delta: Dict[str, float] = {}
    for s in STATE_NAMES:
        sv = pcfg["sigma"][s]
        if isinstance(sv, list):
            sigma[s] = float(rng.uniform(sv[0], sv[1])) if active_states[s] else 0.0
        else:
            sigma[s] = float(sv) if active_states[s] else 0.0
        delta[s] = float(pcfg["delta"][s]) if active_states[s] else 0.0

    # 6. Initial state x(-180)
    x0_priors = aux["initial_state"]
    initial_state: Dict[str, float] = {}
    for s in STATE_NAMES:
        if not active_states[s]:
            initial_state[s] = 0.0
        elif s == "O":
            initial_state[s] = float(o_sat_freshwater(T_bar))
        elif s == "C":
            initial_state[s] = float(C_sat)
        else:
            initial_state[s] = _sample_dist(x0_priors[s], rng)

    # 7. Intervention
    intervention = _sample_intervention(base_preset, interv_regime, shapes, rng, interv_cfg)

    return SystemSpec(
        sys_id=sys_id,
        seed=seed,
        preset=preset_or_variant,
        split_tag=split_tag,
        active_states=active_states,
        active_channels=active_channels,
        shapes=shapes,
        params=params,
        sigma=sigma,
        delta=delta,
        forcing=forcing,
        initial_state=initial_state,
        intervention=intervention,
    )
