"""Multiprocessing batch generation pipeline writing Zarr + Parquet + splits/norm."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr

from ecoverse.bench.splits import build_splits_json
from ecoverse.core.constants import (
    CHANNEL_NAMES,
    DRIVER_NAMES,
    DT_DAYS,
    INTERVENTION_DAY,
    N_CHANNELS,
    N_DRIVERS,
    N_STATES,
    N_STEPS_CF,
    N_STEPS_FORMAL,
    STATE_NAMES,
)
from ecoverse.core.dynamics import warmup_jit
from ecoverse.obs.normalize import Normalizer
from ecoverse.sim.quality import QualityReport, evaluate_quality
from ecoverse.sim.sampler import CONFIG_DIR, load_yaml, sample_system_spec
from ecoverse.sim.solver import SimulationResult, solve_system


def _worker_init() -> None:
    """Compile Numba kernels once per worker process."""
    warmup_jit()


def _run_candidate_worker(
    task: Tuple[int, int, int, str, str, str, str, float, str],
) -> Tuple[int, bool, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Deterministic worker evaluating one candidate attempt_idx.

    Returns lightweight tuple to avoid shipping large arrays on failed attempts.
    """
    (
        master_seed,
        group_id,
        attempt_idx,
        preset_name,
        split_tag,
        shape_mode,
        interv_regime,
        timeout_s,
        config_dir_str,
    ) = task
    ss = np.random.SeedSequence([master_seed, group_id, attempt_idx])
    child_seed = int(ss.generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(ss)

    spec = sample_system_spec(
        preset_or_variant=preset_name,
        rng=rng,
        seed=child_seed,
        sys_id=0,
        split_tag=split_tag,
        shape_mode=shape_mode,
        interv_regime=interv_regime,
        config_dir=Path(config_dir_str),
    )
    res: SimulationResult = solve_system(spec, timeout_s=timeout_s)
    qual: QualityReport = evaluate_quality(spec, res)

    if not qual.passed:
        return (attempt_idx, False, qual.fail_reason, None, None)

    arrays_payload = {
        "x_true": res.x_true,
        "r_true": res.r_true,
        "u_true": res.u_true,
        "x_cf": res.x_cf,
        "r_cf": res.r_cf,
        "u_cf": res.u_cf,
        "solve_seconds": res.solve_seconds,
    }
    meta_payload = {
        "seed": child_seed,
        "preset": spec.preset,
        "split_tag": spec.split_tag,
        "regime_tag": qual.regime_tag,
        "state_mask": [bool(spec.active_states[s]) for s in STATE_NAMES],
        "effective_state_mask": [bool(v) for v in qual.effective_state_mask],
        "channel_mask": [i in spec.active_channels for i in range(N_CHANNELS)],
        "shape_ids": spec.shapes,
        "params": spec.params,
        "intervention": spec.intervention.model_dump(),
        "sigma": spec.sigma,
        "delta": spec.delta,
    }
    return (attempt_idx, True, "", arrays_payload, meta_payload)


def _create_zarr_arrays(root: zarr.Group, n_total: int, chunk_size: int) -> Dict[str, Any]:
    """Create Zarr arrays compatibly across Zarr v2 and v3."""
    shapes = {
        "x_true": (n_total, N_STEPS_FORMAL, N_STATES),
        "r_true": (n_total, N_STEPS_FORMAL, N_CHANNELS),
        "u_true": (n_total, N_STEPS_FORMAL, N_DRIVERS),
        "x_cf": (n_total, N_STEPS_CF, N_STATES),
        "r_cf": (n_total, N_STEPS_CF, N_CHANNELS),
        "u_cf": (n_total, N_STEPS_CF, N_DRIVERS),
    }
    is_v3 = int(zarr.__version__.split(".")[0]) >= 3
    arrays: Dict[str, Any] = {}
    for name, shape in shapes.items():
        chunks = (min(chunk_size, n_total), shape[1], shape[2])
        if is_v3:
            arrays[name] = root.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                overwrite=True,
            )
        else:
            arrays[name] = root.create_dataset(
                name=name,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                overwrite=True,
            )
    return arrays


def generate_dataset(
    config_path: Optional[Path] = None,
    output_dir: Path | str = Path("data/v0"),
    num_workers: int = 1,
    scale: float = 1.0,
    master_seed_override: Optional[int] = None,
    progress: bool = True,
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run deterministic dataset generation across all configured splits and groups."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg_file = Path(config_path) if config_path else (cdir / "benchmark_v0.yaml")
    bench_cfg = load_yaml(cfg_file)

    master_seed = int(master_seed_override if master_seed_override is not None else bench_cfg["master_seed"])
    chunk_size = int(bench_cfg.get("chunk_size", 32))
    timeout_s = float(bench_cfg.get("timeout_seconds", 10.0))

    # Build deterministic generation schedule
    plan_items: List[Tuple[str, str, int, int, str, str]] = []
    n_total = 0
    for split_tag, scfg in bench_cfg["splits"].items():
        shape_mode = str(scfg["shape_mode"])
        interv_regime = str(scfg["interv_regime"])
        for grp in scfg["groups"]:
            raw_cnt = int(grp["count"])
            target_cnt = max(1, int(round(raw_cnt * scale))) if scale != 1.0 else raw_cnt
            plan_items.append(
                (split_tag, str(grp["name"]), target_cnt, int(grp["group_id"]), shape_mode, interv_regime)
            )
            n_total += target_cnt

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    zarr_path = out_path / "trajectories.zarr"

    # Warmup JIT in main process if single-worker
    warmup_jit()

    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["version"] = str(bench_cfg.get("version", "v0.1.0"))
    root.attrs["state_names"] = list(STATE_NAMES)
    root.attrs["driver_names"] = list(DRIVER_NAMES)
    root.attrs["channel_names"] = list(CHANNEL_NAMES)
    root.attrs["n_states"] = N_STATES
    root.attrs["n_drivers"] = N_DRIVERS
    root.attrs["n_channels"] = N_CHANNELS
    root.attrs["dt_days"] = DT_DAYS
    root.attrs["n_steps_formal"] = N_STEPS_FORMAL
    root.attrs["n_steps_cf"] = N_STEPS_CF
    root.attrs["intervention_day"] = INTERVENTION_DAY
    zarr_arrays = _create_zarr_arrays(root, n_total, chunk_size)

    # Chunk-aligned buffers to minimize Zarr write calls
    buf_arrays = {
        "x_true": np.empty((chunk_size, N_STEPS_FORMAL, N_STATES), dtype=np.float32),
        "r_true": np.empty((chunk_size, N_STEPS_FORMAL, N_CHANNELS), dtype=np.float32),
        "u_true": np.empty((chunk_size, N_STEPS_FORMAL, N_DRIVERS), dtype=np.float32),
        "x_cf": np.empty((chunk_size, N_STEPS_CF, N_STATES), dtype=np.float32),
        "r_cf": np.empty((chunk_size, N_STEPS_CF, N_CHANNELS), dtype=np.float32),
        "u_cf": np.empty((chunk_size, N_STEPS_CF, N_DRIVERS), dtype=np.float32),
    }
    buf_start = 0
    buf_fill = 0

    def _flush_buffer() -> None:
        nonlocal buf_start, buf_fill
        if buf_fill == 0:
            return
        end = buf_start + buf_fill
        for key, arr in zarr_arrays.items():
            arr[buf_start:end] = buf_arrays[key][:buf_fill]
        buf_start = end
        buf_fill = 0

    metadata_rows: List[Dict[str, Any]] = []
    group_stats: List[Dict[str, Any]] = []
    next_sys_id = 0
    t0_all = time.monotonic()

    executor: Optional[ProcessPoolExecutor] = None
    if num_workers > 1:
        executor = ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init)

    try:
        for split_tag, grp_name, target_cnt, group_id, shape_mode, interv_regime in plan_items:
            accepted = 0
            attempt_idx = 0
            attempts_total = 0
            fail_counts: Dict[str, int] = {}
            solve_times: List[float] = []
            batch_size = max(16, num_workers * 4) if num_workers > 1 else 16

            while accepted < target_cnt:
                # Deterministic batching: run fixed batch, sort by attempt_idx, take in order
                remaining = target_cnt - accepted
                cur_batch = (
                    min(max(num_workers, remaining * 2), 64)
                    if num_workers > 1
                    else min(max(1, remaining * 2), 32)
                )
                tasks = [
                    (
                        master_seed,
                        group_id,
                        attempt_idx + k,
                        grp_name,
                        split_tag,
                        shape_mode,
                        interv_regime,
                        timeout_s,
                        str(cdir),
                    )
                    for k in range(cur_batch)
                ]
                attempt_idx += cur_batch

                if executor is not None:
                    results = list(executor.map(_run_candidate_worker, tasks))
                else:
                    results = [_run_candidate_worker(tk) for tk in tasks]

                results.sort(key=lambda r: r[0])

                for _, passed, fail_reason, arrs, meta in results:
                    attempts_total += 1
                    if not passed:
                        fail_counts[fail_reason] = fail_counts.get(fail_reason, 0) + 1
                        continue
                    assert arrs is not None and meta is not None

                    sys_id = next_sys_id
                    next_sys_id += 1
                    accepted += 1
                    solve_times.append(float(arrs["solve_seconds"]))

                    for key in buf_arrays:
                        buf_arrays[key][buf_fill] = arrs[key]
                    buf_fill += 1
                    if buf_fill == chunk_size:
                        _flush_buffer()

                    metadata_rows.append(
                        {
                            "sys_id": sys_id,
                            "seed": int(meta["seed"]),
                            "preset": str(meta["preset"]),
                            "split_tag": str(meta["split_tag"]),
                            "regime_tag": str(meta["regime_tag"]),
                            "state_mask_json": json.dumps(meta["state_mask"]),
                            "effective_state_mask_json": json.dumps(meta["effective_state_mask"]),
                            "channel_mask_json": json.dumps(meta["channel_mask"]),
                            "shape_ids_json": json.dumps(meta["shape_ids"]),
                            "params_json": json.dumps(meta["params"]),
                            "intervention_json": json.dumps(meta["intervention"]),
                            "sigma_json": json.dumps(meta["sigma"]),
                            "delta_json": json.dumps(meta["delta"]),
                        }
                    )
                    if accepted >= target_cnt:
                        break

                if attempts_total > max(500, target_cnt * 100) and accepted == 0:
                    raise RuntimeError(
                        f"Too many failures for {split_tag}/{grp_name}: {fail_counts}"
                    )

            _flush_buffer()
            pass_rate = accepted / max(attempts_total, 1)
            avg_solve = float(np.mean(solve_times)) if solve_times else 0.0
            group_stats.append(
                {
                    "split": split_tag,
                    "group": grp_name,
                    "accepted": accepted,
                    "attempts": attempts_total,
                    "pass_rate": pass_rate,
                    "avg_solve_s": avg_solve,
                    "fails": fail_counts,
                }
            )
            if progress:
                print(
                    f"[{split_tag}:{grp_name}] {accepted}/{attempts_total} passed "
                    f"({pass_rate*100:.1f}%), avg solve={avg_solve:.3f}s",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # Save metadata.parquet
    df = pd.DataFrame(metadata_rows)
    meta_path = out_path / "metadata.parquet"
    df.to_parquet(meta_path, index=False)

    # Save splits.json
    splits_path = out_path / "splits.json"
    splits_dict = build_splits_json(df, splits_path)

    # Compute and save train-only norm_stats.json (strictly no leakage)
    train_ids = splits_dict["train"]
    normalizer = Normalizer.fit_from_zarr(zarr_path=zarr_path, metadata_path=meta_path, train_ids=train_ids)
    norm_path = out_path / "norm_stats.json"
    normalizer.save(norm_path)

    elapsed_total = time.monotonic() - t0_all
    summary = {
        "n_total": n_total,
        "master_seed": master_seed,
        "num_workers": num_workers,
        "elapsed_seconds": elapsed_total,
        "groups": group_stats,
        "output_dir": str(out_path),
    }
    if progress:
        print(
            f"Completed {n_total} systems in {elapsed_total:.2f}s "
            f"({elapsed_total / max(n_total, 1):.3f}s/sys wall).",
            flush=True,
        )
    return summary
