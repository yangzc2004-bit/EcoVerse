"""CLI entry point to generate EcoVerse v0 dataset or run pass-rate calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ecoverse.core.dynamics import warmup_jit
from ecoverse.sim.pipeline import generate_dataset
from ecoverse.sim.quality import evaluate_quality
from ecoverse.sim.sampler import sample_system_spec
from ecoverse.sim.solver import solve_system


def run_calibration(n_per_group: int = 100, seed: int = 20260101, base_only: bool = False) -> dict:
    """Test pass rates and single-core solve timing across presets and variants."""
    warmup_jit()
    groups = [
        ("river", "train", "train"),
        ("lake", "train", "train"),
        ("terrestrial", "train", "train"),
        ("foodweb", "train", "train"),
        ("chemostat", "train", "train"),
        ("river_competition", "train", "train"),
        ("lake_cascade", "train", "train"),
        ("terrestrial_cascade", "train", "train"),
        ("chemostat_detritus_recycle", "train", "train"),
        ("river", "heldout_2c", "train"),
        ("lake", "heldout_2c", "train"),
        ("terrestrial", "heldout_2c", "train"),
        ("foodweb", "heldout_2c", "train"),
        ("chemostat", "heldout_2c", "train"),
    ]
    if base_only:
        groups = groups[:5]
    rows = []
    print(f"Running calibration ({n_per_group} samples/group, single-core)...")
    for idx, (preset, shape_mode, interv_regime) in enumerate(groups):
        ss = np.random.SeedSequence([seed, 9000 + idx])
        rng = np.random.default_rng(ss)
        passed = 0
        fails: dict[str, int] = {}
        regimes: dict[str, int] = {}
        solve_times: list[float] = []
        t0 = time.monotonic()
        for k in range(n_per_group):
            child_seed = int(rng.integers(0, 2**31 - 1))
            spec = sample_system_spec(
                preset_or_variant=preset,
                rng=np.random.default_rng(child_seed),
                seed=child_seed,
                sys_id=k,
                split_tag="cal",
                shape_mode=shape_mode,
                interv_regime=interv_regime,
            )
            res = solve_system(spec, timeout_s=10.0)
            qual = evaluate_quality(spec, res)
            solve_times.append(res.solve_seconds)
            if qual.passed:
                passed += 1
                regimes[qual.regime_tag] = regimes.get(qual.regime_tag, 0) + 1
            else:
                fails[qual.fail_reason] = fails.get(qual.fail_reason, 0) + 1
        elapsed = time.monotonic() - t0
        rate = 100.0 * passed / n_per_group
        avg_s = float(np.mean(solve_times))
        time_per_pass = elapsed / max(passed, 1)
        rows.append(dict(preset=preset, shape_mode=shape_mode, attempts=n_per_group, passed=passed,
                         mean_solve_seconds=avg_s, wall_seconds=elapsed, seconds_per_pass=time_per_pass,
                         failures=fails, regimes=regimes))
        print(
            f"  {preset:28s} [{shape_mode:10s}]: pass={passed:3d}/{n_per_group} ({rate:5.1f}%) | "
            f"avg_solve={avg_s:.3f}s | wall/pass={time_per_pass:.3f}s | fails={fails} | regimes={regimes}"
        )
    return {"seed": seed, "workers": 1, "groups": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EcoVerse v0 dataset")
    parser.add_argument("--config", type=Path, help="Default: the packaged benchmark_v0.yaml")
    parser.add_argument("--output", type=Path, default=Path("data/v0"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (1.0 = 6000 systems)")
    parser.add_argument("--seed", type=int, default=None, help="Override master seed")
    parser.add_argument("--calibrate", action="store_true", help="Run pass-rate calibration only")
    parser.add_argument("--cal-n", type=int, default=100, help="Samples per group in calibration mode")
    parser.add_argument("--cal-base-only", action="store_true", help="Calibrate the five standard presets")
    parser.add_argument("--cal-report", type=Path, help="Write calibration counts and timings as JSON")
    args = parser.parse_args()

    if args.calibrate:
        report = run_calibration(args.cal_n, args.seed if args.seed is not None else 20260101, args.cal_base_only)
        if args.cal_report:
            args.cal_report.parent.mkdir(parents=True, exist_ok=True)
            args.cal_report.write_text(json.dumps(report, indent=2))
        return

    generate_dataset(
        config_path=args.config,
        output_dir=args.output,
        num_workers=args.workers,
        scale=args.scale,
        master_seed_override=args.seed,
        progress=True,
    )


if __name__ == "__main__":
    main()
