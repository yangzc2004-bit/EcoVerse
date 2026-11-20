"""Verify a full release dataset against its benchmark configuration and stored truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ecoverse.bench.splits import SplitManager
from ecoverse.config import CONFIG_DIR
from ecoverse.core.constants import (
    INTERVENTION_STEP, N_CHANNELS, N_DRIVERS, N_STATES, N_STEPS_CF, N_STEPS_FORMAL,
    SIGNED_FLUX_CHANNELS, STATE_NAMES,
)
from ecoverse.core.spec import SystemSpec
from ecoverse.obs.dataset import EcoVerseDataset
from ecoverse.obs.normalize import Normalizer
from ecoverse.sim.sampler import load_yaml, sample_system_spec
from ecoverse.sim.solver import solve_system


def verify_release(data_dir: Path, config_path: Path) -> dict:
    config = load_yaml(config_path)
    shape_pools = load_yaml(CONFIG_DIR / "shapes.yaml")
    interventions = load_yaml(CONFIG_DIR / "interventions.yaml")
    variants = load_yaml(CONFIG_DIR / "presets.yaml")["variants_2b"]
    df = pd.read_parquet(data_dir / "metadata.parquet").sort_values("sys_id")
    splits = SplitManager(data_dir)
    counts = {name: group["total"] for name, group in config["splits"].items()}
    n = sum(counts.values())
    assert splits.summary() == counts and len(df) == n
    np.testing.assert_array_equal(df.sys_id, np.arange(n))
    assert df.seed.is_unique
    for name, group in config["splits"].items():
        rows = df[df.split_tag == name]
        assert set(rows.sys_id) == set(splits.get_ids(name))
        assert rows.preset.value_counts().to_dict() == {g["name"]: g["count"] for g in group["groups"]}
        assert rows.shape_mode.eq(group["shape_mode"]).all()
        assert rows.interv_regime.eq(group["interv_regime"]).all()

    store = zarr.open_group(str(data_dir / "trajectories.zarr"), mode="r")
    stats = Normalizer.load(data_dir / "norm_stats.json").stats
    sums, squares, sizes = np.zeros((3, len(stats)), dtype=np.float64)
    shapes = {}
    for suffix, steps in (("true", N_STEPS_FORMAL), ("cf", N_STEPS_CF)):
        for key, dim in (("x", N_STATES), ("r", N_CHANNELS), ("u", N_DRIVERS)):
            array = store[f"{key}_{suffix}"]
            assert array.shape == (n, steps, dim) and array.dtype == np.float32
            assert array.chunks == (min(config["chunk_size"], n), steps, dim)
            shapes[f"{key}_{suffix}"] = array.shape

    ranges = {key: [float("inf"), -float("inf")] for key in shapes}
    hashes = {key: hashlib.sha256() for key in shapes}
    negatives = np.zeros(2, dtype=np.int64)
    intervention_counts = defaultdict(Counter)
    max_fork_error = 0.0
    max_driver_error = 0.0
    for start in range(0, n, config["chunk_size"]):
        rows = df.iloc[start:start + config["chunk_size"]]
        block = {key: np.asarray(store[key][start:start + len(rows)]) for key in shapes}
        active = np.array([json.loads(s) for s in rows.state_mask_json])
        effective = np.array([json.loads(s) for s in rows.effective_state_mask_json])
        channel = np.array([json.loads(s) for s in rows.channel_mask_json])
        assert not (effective & ~active).any() and np.all(effective[:, [1, 2]].any(axis=1))
        for key, arr in block.items():
            assert np.isfinite(arr).all(), (start, key)
            hashes[key].update(np.ascontiguousarray(arr, dtype="<f4").data)
            lo, hi = ranges[key]
            ranges[key] = [min(lo, float(arr.min())), max(hi, float(arr.max()))]
            if key.startswith("x"):
                assert arr.min() >= 0 and arr.max() <= 500
                assert not np.any(arr * ~active[:, None, :])
            if key.startswith("r"):
                nonnegative = [j for j in range(N_CHANNELS) if j not in SIGNED_FLUX_CHANNELS]
                assert arr[..., nonnegative].min() >= 0
                assert not np.any(arr * ~channel[:, None, :])
        negatives += (block["r_true"][..., list(SIGNED_FLUX_CHANNELS)] < 0).sum(axis=(0, 1))
        train = rows.split_tag.eq("train").to_numpy()
        for stat in stats:
            idx = stat.var_id
            if stat.is_driver:
                raw = block["u_true"][train, :, idx - N_STATES]
            else:
                raw = block["x_true"][train & effective[:, idx], :, idx]
            vals = raw.astype(np.float64)
            if stat.transform == "log":
                vals = np.log(np.maximum(vals, 0) + stat.eps)
            sums[idx] += vals.sum()
            squares[idx] += np.square(vals).sum()
            sizes[idx] += vals.size

        mean = block["x_true"].mean(axis=1, dtype=np.float64)
        std = block["x_true"].std(axis=1, dtype=np.float64)
        assert np.all(((std >= .02 * mean) & (mean > 1e-9) & effective).sum(axis=1) >= 2)
        for i, row in enumerate(rows.itertuples()):
            action = json.loads(row.intervention_json)
            spec = json.loads(row.spec_json)
            assert action == spec["intervention"] and action["regime"] == row.interv_regime
            assert row.seed == spec["seed"] and row.sys_id == spec["sys_id"]
            assert row.preset == spec["preset"] and row.split_tag == spec["split_tag"]
            assert spec["shapes"] == json.loads(row.shape_ids_json)
            pool = shape_pools["train_allowed" if row.shape_mode == "train" else "heldout_2c"]
            for key, shape in spec["shapes"].items():
                assert shape in pool[key.split("_")[0]], (row.sys_id, key, shape)
            np.testing.assert_array_equal(active[i], [spec["active_states"][s] for s in STATE_NAMES])
            np.testing.assert_array_equal(np.flatnonzero(channel[i]), sorted(spec["active_channels"]))
            base = variants.get(row.preset, {}).get("base", row.preset)
            kind = action["type"]
            assert kind in interventions["preset_valid_interventions"][base]
            assert action["t_int"] == interventions["t_int"]
            limits = interventions["regimes"][row.interv_regime][kind]
            for key in ("lambda_N", "delta_T", "kappa"):
                if key + "_ranges" in limits:
                    assert any(lo <= action[key] <= hi for lo, hi in limits[key + "_ranges"])
            if kind == "flow_regime":
                for key in ("lambda_q", "lambda_storm"):
                    lo, hi = limits[key + "_range"]
                    assert lo <= action[key] <= hi
                assert action["lambda_storm"] > 0 or limits["allow_zero_storm"]
            if kind == "bio_pulse":
                assert action["target_state"] in limits["targets"] and spec["active_states"][action["target_state"]]
            intervention_counts[row.split_tag][kind] += 1
            expected_x = block["x_true"][i, INTERVENTION_STEP].astype(np.float64)
            expected_u = block["u_true"][i, INTERVENTION_STEP:].astype(np.float64)
            if action["type"] == "bio_pulse":
                expected_x[STATE_NAMES.index(action["target_state"])] *= action["kappa"]
            elif action["type"] == "nutrient_step":
                expected_u[:, 3] *= action["lambda_N"]
            elif action["type"] == "warming_shift":
                expected_u[:, 1] += action["delta_T"]
            elif action["type"] == "flow_regime":
                q_base = spec["forcing"]["q_base"]
                expected_u[:, 2] = q_base * action["lambda_q"] + (expected_u[:, 2] - q_base) * action["lambda_storm"]
            np.testing.assert_allclose(block["x_cf"][i, 0], expected_x, rtol=2e-6, atol=1e-6)
            np.testing.assert_allclose(block["u_cf"][i], expected_u, rtol=2e-6, atol=2e-6)
            max_fork_error = max(max_fork_error, float(np.max(np.abs(block["x_cf"][i, 0] - expected_x))))
            max_driver_error = max(max_driver_error, float(np.max(np.abs(block["u_cf"][i] - expected_u))))

    means = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)
    variance = np.divide(squares, sizes, out=np.ones_like(sums), where=sizes > 0) - means ** 2
    deviations = np.maximum(np.sqrt(np.maximum(0, variance)), 1e-6)
    np.testing.assert_allclose(means, [s.mean for s in stats], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(deviations, [s.std for s in stats], rtol=1e-8, atol=1e-9)

    # Replay one independent example from every configured split/preset group.
    replayed = []
    for _, group in df.groupby(["split_tag", "preset"]):
        row = group.sample(1, random_state=41).iloc[0]
        seed = int(row.seed)
        spec = sample_system_spec(row.preset, np.random.default_rng(seed), seed, int(row.sys_id),
                                  row.split_tag, row.shape_mode, row.interv_regime)
        assert spec == SystemSpec.model_validate_json(row.spec_json)
        result = solve_system(spec)
        assert result.ok, (row.sys_id, result.fail_reason)
        for key in shapes:
            np.testing.assert_array_equal(getattr(result, key), store[key][int(row.sys_id)])
        replayed.append(int(row.sys_id))
    ood_views = EcoVerseDataset(data_dir, splits.get_ids("test_3_interv"), mode="counterfactual",
                                sensor_package="A", num_queries=32, max_tokens=512, seed=42)
    for view in ood_views:
        steps = np.rint(view["obs_times"].numpy() * N_STEPS_FORMAL).astype(int)
        future = steps >= INTERVENTION_STEP
        assert view["obs_is_driver"].numpy()[future].all()
        for driver in range(N_DRIVERS):
            observed = future & (view["obs_var_ids"].numpy() == N_STATES + driver)
            assert set(steps[observed] % 6) == set(range(6))
    assert set(intervention_counts["test_3_interv"]) == set(interventions["regimes"]["ood"])
    return dict(systems=n, splits=counts, shapes=shapes, ranges=ranges,
                trajectory_sha256={key: digest.hexdigest() for key, digest in hashes.items()},
                train_only_normalization_checked=True, unique_seeds=True, checked_trajectory_pairs=n,
                shape_pools_and_intervention_ranges_checked=True, interventions_by_split=intervention_counts,
                test3_views_with_only_known_future_drivers_at_all_six_phases=len(ood_views),
                max_fork_state_abs_error=max_fork_error, max_driver_abs_error=max_driver_error,
                negative_factual_gas_exchange_values=negatives.tolist(), bit_identical_replays=replayed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/v0"))
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "benchmark_v0.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/acceptance/verification.json"))
    args = parser.parse_args()
    report = verify_release(args.data_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Verified {report['systems']} systems and replayed {len(report['bit_identical_replays'])} systems bit-for-bit.")
