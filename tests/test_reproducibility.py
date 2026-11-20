"""Bit-for-bit single-worker vs multi-worker reproducibility test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import zarr

from ecoverse.bench.splits import SplitManager
from ecoverse.obs.dataset import EcoVerseDataset
from ecoverse.obs.normalize import Normalizer
from ecoverse.sim.pipeline import generate_dataset
from ecoverse.sim.sampler import sample_system_spec


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
    for row in df1.itertuples():
        seed = int(row.seed)
        spec = sample_system_spec(row.preset, np.random.default_rng(seed), seed, row.sys_id,
                                  row.split_tag, row.shape_mode, row.interv_regime)
        assert spec.model_dump() == json.loads(row.spec_json)

    split = SplitManager(dir_1)
    probe_ids = split.stratified_ids("train", 5, 42)
    assert df1.set_index("sys_id").loc[probe_ids].preset.nunique() == 5
    for mode in ("counterfactual", "counterfactual_train"):
        sample = EcoVerseDataset(dir_1, probe_ids, mode=mode, sensor_package="A", max_tokens=512)[0]
        future = sample["obs_times"] >= 180 / 365
        assert sample["obs_is_driver"][future].all()
        light = future & (sample["obs_var_ids"] == 8)
        phases = torch.round(sample["obs_times"][light] * 365 * 6).long() % 6
        assert phases.unique().numel() == 6
        assert sample["intervention"][:4].sum() == 1
        assert ("baseline_x" in sample) == (mode == "counterfactual")

    # A failed replacement must retain the previous store, metadata and split indices.
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["splits"]["train"]["groups"][0]["name"] = "unknown-preset"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(KeyError, match="unknown-preset"):
        generate_dataset(cfg_path, dir_1, num_workers=1, progress=False)
    retained = zarr.open_group(str(dir_1 / "trajectories.zarr"), mode="r")
    np.testing.assert_array_equal(np.asarray(retained["x_true"]), np.asarray(z4["x_true"]))
    pd.testing.assert_frame_equal(pd.read_parquet(dir_1 / "metadata.parquet"), df4)
    assert (dir_1 / "splits.json").read_bytes() == (dir_4 / "splits.json").read_bytes()


def test_failed_final_write_preserves_all_release_files(tmp_path, monkeypatch):
    cfg_path = _write_compact_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    train = cfg["splits"]["train"]
    train.update(total=1, groups=[{"name": "chemostat", "count": 1, "group_id": 105}])
    cfg["splits"] = {"train": train}
    cfg_path.write_text(yaml.safe_dump(cfg))
    release = tmp_path / "release"
    generate_dataset(cfg_path, release, progress=False)

    def snapshot():
        return {p.relative_to(release): p.read_bytes() for p in release.rglob("*") if p.is_file()}

    def fail_save(self, path):
        assert path.with_name("metadata.parquet").exists() and path.with_name("splits.json").exists()
        raise OSError("final write failed")

    before = snapshot()
    monkeypatch.setattr(Normalizer, "save", fail_save)
    with pytest.raises(OSError, match="final write failed"):
        generate_dataset(cfg_path, release, master_seed_override=91, progress=False)
    assert snapshot() == before
