"""Heterogeneous virtual monitoring station observation operators."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ecoverse.config import CONFIG_DIR
from ecoverse.obs.normalize import Normalizer


@lru_cache(maxsize=4)
def load_sensors_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else (CONFIG_DIR / "sensors.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _apply_noise(
    raw: np.ndarray,
    noise_type: str,
    sigma: float,
    is_log_var: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma <= 0.0:
        return raw.copy()
    if noise_type == "lognormal":
        mult = np.exp(rng.normal(0.0, sigma, size=raw.shape))
        noisy = raw * mult
        return np.maximum(noisy, 0.0)
    # Gaussian noise: proportional + tiny floor
    scale = sigma * np.maximum(np.abs(raw), 1e-2)
    noisy = raw + rng.normal(0.0, scale)
    if is_log_var:
        noisy = np.maximum(noisy, 0.0)
    return noisy


def _apply_dropout_blocks(
    t_days: np.ndarray,
    n_blocks: int,
    dur_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    keep = np.ones(t_days.shape[0], dtype=bool)
    if n_blocks <= 0 or t_days.shape[0] < 4:
        return keep
    t_min, t_max = float(t_days[0]), float(t_days[-1])
    span = max(t_max - t_min, 1e-6)
    d_lo, d_hi = float(dur_range[0]), float(dur_range[1])
    for _ in range(n_blocks):
        dur = float(rng.uniform(d_lo, d_hi))
        start = float(rng.uniform(t_min, max(t_min, t_max - dur)))
        keep &= ~((t_days >= start) & (t_days <= start + dur))
    if not np.any(keep):
        keep[0] = True
    return keep


def _irregular_indices(
    t_days: np.ndarray,
    interval_range: Tuple[float, float],
    rng: np.random.Generator,
    poisson: bool = False,
) -> np.ndarray:
    n_t = t_days.shape[0]
    if n_t == 0:
        return np.zeros(0, dtype=np.int64)
    idx_list: List[int] = []
    t_cur = float(t_days[0]) + float(rng.uniform(0.0, 0.5 * interval_range[1]))
    t_end = float(t_days[-1])
    mean_dt = float(rng.uniform(interval_range[0], interval_range[1]))

    while t_cur <= t_end:
        k = int(np.searchsorted(t_days, t_cur))
        if 0 <= k < n_t and (not idx_list or k != idx_list[-1]):
            idx_list.append(k)
        step = float(rng.exponential(mean_dt)) if poisson else float(rng.uniform(interval_range[0], interval_range[1]))
        t_cur += max(step, 1.0 / 6.0)

    if not idx_list:
        idx_list.append(0)
    return np.asarray(idx_list, dtype=np.int64)


def sample_observation(
    x_obs_window: np.ndarray,
    u_obs_window: np.ndarray,
    t_obs_days: np.ndarray,
    effective_state_mask: np.ndarray,
    state_names: Sequence[str],
    driver_names: Sequence[str],
    normalizer: Normalizer,
    rng: np.random.Generator,
    package: str = "random",
    norm_time_denom: float = 365.0,
    max_tokens: int = 2048,
    future_u: Optional[np.ndarray] = None,
    future_t_days: Optional[np.ndarray] = None,
    sensors_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Generate irregular masked multi-sensor event stream tokens.

    All dimensions are inferred from effective_state_mask, state_names, driver_names
    (no hardcoded 8 or 23 for v1 forward compatibility).
    """
    cfg = sensors_cfg or load_sensors_config()
    n_states = len(state_names)
    n_drivers = len(driver_names)
    state_to_id = {name: i for i, name in enumerate(state_names)}
    drv_to_id = {name: n_states + d for d, name in enumerate(driver_names)}

    eff_indices = [i for i in range(n_states) if bool(effective_state_mask[i])]
    if not eff_indices:
        eff_indices = [0]

    # Determine observed states, drivers, cadence, noise, and dropout
    if package in ("A", "B", "C"):
        pcfg = cfg["packages"][package]
        pref = [state_to_id[s] for s in pcfg["preferred_states"] if s in state_to_id and effective_state_mask[state_to_id[s]]]
        if not pref:
            pref = [state_to_id[s] for s in pcfg.get("fallback_states", []) if s in state_to_id and effective_state_mask[state_to_id[s]]]
        if not pref:
            pref = [eff_indices[0]]

        if pcfg.get("pick_single_state", False) and len(pref) > 1:
            obs_state_ids = [int(rng.choice(pref))]
        else:
            obs_state_ids = pref

        obs_drv_ids = [drv_to_id[d] for d in pcfg.get("drivers", []) if d in drv_to_id]
        noise_type = str(pcfg["noise_type"])
        noise_sigma = float(pcfg["noise_sigma"])
        b_lo, b_hi = pcfg.get("dropout_blocks", [0, 0])
        n_blocks = int(rng.integers(b_lo, b_hi + 1)) if b_hi > 0 else 0
        dur_range = tuple(pcfg.get("dropout_days", [0.0, 0.0]))

        if "cadence_steps" in pcfg:
            step = max(1, int(pcfg["cadence_steps"]))
            base_idx = np.arange(0, t_obs_days.shape[0], step, dtype=np.int64)
        elif "mean_interval_days" in pcfg:
            base_idx = _irregular_indices(t_obs_days, tuple(pcfg["mean_interval_days"]), rng, poisson=True)
        else:
            base_idx = _irregular_indices(t_obs_days, tuple(pcfg["interval_days"]), rng, poisson=False)

    elif package == "random":
        acfg = cfg["train_augmentation"]
        k_max = min(int(acfg["max_observed_states"]), len(eff_indices))
        k_min = min(int(acfg["min_observed_states"]), k_max)
        k_pick = int(rng.integers(k_min, k_max + 1))
        obs_state_ids = sorted(int(v) for v in rng.choice(eff_indices, size=k_pick, replace=False))

        if rng.random() < float(acfg["driver_include_prob"]):
            k_drv = int(rng.integers(1, n_drivers + 1))
            obs_drv_ids = sorted(
                n_states + int(d) for d in rng.choice(n_drivers, size=k_drv, replace=False)
            )
        else:
            obs_drv_ids = []

        cadence_steps = int(rng.integers(acfg["min_steps_interval"], acfg["max_steps_interval"] + 1))
        if cadence_steps == 1 or rng.random() < 0.4:
            base_idx = np.arange(0, t_obs_days.shape[0], cadence_steps, dtype=np.int64)
        else:
            dt_approx = max(1.0 / 6.0, float(t_obs_days[1] - t_obs_days[0])) if t_obs_days.shape[0] > 1 else 1.0 / 6.0
            mean_d = cadence_steps * dt_approx
            base_idx = _irregular_indices(t_obs_days, (0.6 * mean_d, 1.4 * mean_d), rng, poisson=True)

        noise_type = "gaussian" if rng.random() < 0.5 else "lognormal"
        noise_sigma = float(rng.uniform(*acfg["noise_sigma_range"]))
        if rng.random() < float(acfg["dropout_prob"]):
            b_lo, b_hi = acfg["dropout_blocks"]
            n_blocks = int(rng.integers(b_lo, b_hi + 1))
            dur_range = tuple(acfg["dropout_days"])
        else:
            n_blocks = 0
            dur_range = (0.0, 0.0)
    else:
        raise ValueError(f"Unknown sensor package '{package}'")

    keep_mask = _apply_dropout_blocks(t_obs_days[base_idx], n_blocks, dur_range, rng)
    sample_idx = base_idx[keep_mask]

    times_list: List[np.ndarray] = []
    var_ids_list: List[np.ndarray] = []
    values_list: List[np.ndarray] = []
    is_drv_list: List[np.ndarray] = []

    obs_state_mask = np.zeros(n_states, dtype=bool)
    for sid in obs_state_ids:
        obs_state_mask[sid] = True
        raw = x_obs_window[sample_idx, sid].astype(np.float32)
        is_log = normalizer.by_id[sid].transform == "log"
        noisy = _apply_noise(raw, noise_type, noise_sigma, is_log, rng)
        normed = normalizer.transform_var(sid, noisy)
        times_list.append((t_obs_days[sample_idx] / norm_time_denom).astype(np.float32))
        var_ids_list.append(np.full(sample_idx.shape[0], sid, dtype=np.int64))
        values_list.append(normed.astype(np.float32))
        is_drv_list.append(np.zeros(sample_idx.shape[0], dtype=bool))

    for did_full in obs_drv_ids:
        d_local = did_full - n_states
        raw = u_obs_window[sample_idx, d_local].astype(np.float32)
        is_log = normalizer.by_id[did_full].transform == "log"
        noisy = _apply_noise(raw, "gaussian", min(noise_sigma, 0.03), is_log, rng)
        normed = normalizer.transform_var(did_full, noisy)
        times_list.append((t_obs_days[sample_idx] / norm_time_denom).astype(np.float32))
        var_ids_list.append(np.full(sample_idx.shape[0], did_full, dtype=np.int64))
        values_list.append(normed.astype(np.float32))
        is_drv_list.append(np.ones(sample_idx.shape[0], dtype=bool))

    # Reserve history tokens and spread future measurements across dates and time of day.
    future_tokens = 0
    if future_u is not None and future_t_days is not None and future_u.shape[0] > 0:
        n_times = min(180, len(future_t_days), max_tokens // (2 * max(1, n_drivers)))
        phases = np.floor(np.remainder(future_t_days, 1.0) * 6 + 1e-3).astype(int) % 6
        indices = []
        if n_times:
            for k, window in enumerate(np.array_split(np.arange(len(phases)), n_times)):
                preferred = window[phases[window] == k % 6]
                indices.append(rng.choice(preferred if preferred.size else window))
        fut_idx = np.asarray(indices, dtype=np.int64)
        future_tokens = len(fut_idx) * n_drivers
        for d_local in range(n_drivers):
            did_full = n_states + d_local
            raw = future_u[fut_idx, d_local].astype(np.float32)
            normed = normalizer.transform_var(did_full, raw)
            times_list.append((future_t_days[fut_idx] / norm_time_denom).astype(np.float32))
            var_ids_list.append(np.full(fut_idx.shape[0], did_full, dtype=np.int64))
            values_list.append(normed.astype(np.float32))
            is_drv_list.append(np.ones(fut_idx.shape[0], dtype=bool))

    obs_times = np.concatenate(times_list)
    obs_var_ids = np.concatenate(var_ids_list)
    obs_values = np.concatenate(values_list)
    obs_is_driver = np.concatenate(is_drv_list)

    history_size = len(obs_times) - future_tokens
    history_budget = max_tokens - future_tokens
    if history_size > history_budget:
        kept_history = np.sort(rng.choice(history_size, size=history_budget, replace=False))
        keep = np.concatenate((kept_history, np.arange(history_size, len(obs_times))))
        obs_times = obs_times[keep]
        obs_var_ids = obs_var_ids[keep]
        obs_values = obs_values[keep]
        obs_is_driver = obs_is_driver[keep]
    obs_state_mask[:] = False
    obs_state_mask[obs_var_ids[~obs_is_driver]] = True

    # Sort chronologically (stable sort preserves var_id ordering at equal timestamps)
    order = np.argsort(obs_times, kind="mergesort")
    obs_times = obs_times[order]
    obs_var_ids = obs_var_ids[order]
    obs_values = obs_values[order]
    obs_is_driver = obs_is_driver[order]

    return {
        "obs_times": obs_times,
        "obs_var_ids": obs_var_ids,
        "obs_values": obs_values,
        "obs_is_driver": obs_is_driver,
        "obs_state_mask": obs_state_mask,
    }
