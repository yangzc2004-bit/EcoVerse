"""Bit-for-bit single-worker vs multi-worker reproducibility test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zarr

from ecoverse.sim.pipeline import generate_dataset


def _write_compact_config(path: Path) -> Path:
    cfg = {
        "version": "v0.1.0-test",
        "master_seed": 777777,
        "chunk_size": 8,
        "timeout_seconds": 10.0,
        "splits": {
            "train": {
                "total": 10,
                "shape_mode": "train",
                "interv_regime": "train",
                "groups": [
                    {"name": "river", "count": 2, "group_id": 101},
                    {"name": "lake", "count": 2, "group_id": 102},
                    {"name": "terrestrial", "count": 2, "group_id": 103},
                    {"name": "foodweb", "count": 2, "group_id": 104},
                    {"name": "chemostat", "count": 2, "group_id": 105},
                ],
            },
            "val": {
                "total": 5,
                "shape_mode": "train",
                "interv_regime": "train",
                "groups": [
                    {"name": "river", "count": 1, "group_id": 201},
                    {"name": "lake", "count": 1, "group_id": 202},
                    {"name": "terrestrial", "count": 1, "group_id": 203},
                    {"name": "foodweb", "count": 1, "group_id": 204},
                    {"name": "chemostat", "count": 1, "group_id": 205},
                ],
            },
        },
    }
    cfg_path = path / "compact_bench.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg_path


def test_single_vs_multi_worker_bit_identical(tmp_path: Path) -> None:
    cfg_path = _write_compact_config(tmp_path)
    dir_1 = tmp_path / "run_1w"
    dir_4 = tmp_path / "run_4w"

    generate_dataset(
        config_path=cfg_path,
        output_dir=dir_1,
        num_workers=1,
        master_seed_override=777777,
        progress=False,
    )
    generate_dataset(
        config_path=cfg_path,
        output_dir=dir_4,
        num_workers=4,
        master_seed_override=777777,
        progress=False,
    )

    z1 = zarr.open_group(str(dir_1 / "trajectories.zarr"), mode="r")
    z4 = zarr.open_group(str(dir_4 / "trajectories.zarr"), mode="r")

    for key in ("x_true", "r_true", "u_true", "x_cf", "r_cf", "u_cf"):
        a1 = np.asarray(z1[key])
        a4 = np.asarray(z4[key])
        assert a1.shape == a4.shape
        assert np.array_equal(a1, a4), f"Mismatch in Zarr array '{key}'"

    df1 = pd.read_parquet(dir_1 / "metadata.parquet")
    df4 = pd.read_parquet(dir_4 / "metadata.parquet")
    pd.testing.assert_frame_equal(df1, df4)
