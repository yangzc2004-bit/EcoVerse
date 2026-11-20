"""PyTorch DataLoader multi-worker stress test, mask alignment, and zero-leakage checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

from ecoverse.bench.splits import SplitManager
from ecoverse.obs.collate import ecoverse_collate
from ecoverse.obs.dataset import EcoVerseDataset
from ecoverse.obs.normalize import Normalizer
from ecoverse.sim.pipeline import generate_dataset


def _write_compact_config(path: Path) -> Path:
    cfg = {
        "version": "v0.1.0-test",
        "master_seed": 202609,
        "chunk_size": 16,
        "timeout_seconds": 10.0,
        "splits": {
            "train": {
                "total": 20,
                "shape_mode": "train",
                "interv_regime": "train",
                "groups": [
                    {"name": "river", "count": 4, "group_id": 101},
                    {"name": "lake", "count": 4, "group_id": 102},
                    {"name": "terrestrial", "count": 4, "group_id": 103},
                    {"name": "foodweb", "count": 4, "group_id": 104},
                    {"name": "chemostat", "count": 4, "group_id": 105},
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
            "test_1_obs": {
                "total": 5,
                "shape_mode": "train",
                "interv_regime": "train",
                "groups": [
                    {"name": "river", "count": 1, "group_id": 301},
                    {"name": "lake", "count": 1, "group_id": 302},
                    {"name": "terrestrial", "count": 1, "group_id": 303},
                    {"name": "foodweb", "count": 1, "group_id": 304},
                    {"name": "chemostat", "count": 1, "group_id": 305},
                ],
            },
        },
    }
    cfg_path = path / "compact_loader_bench.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg_path


def test_torch_loader_and_zero_leakage(tmp_path: Path) -> None:
    if full_data := os.environ.get("ECOVERSE_ACCEPTANCE_DATA_DIR"):
        data_dir = Path(full_data)
    else:
        cfg_path = _write_compact_config(tmp_path)
        data_dir = tmp_path / "v0_small"
        generate_dataset(config_path=cfg_path, output_dir=data_dir, num_workers=4,
                         master_seed_override=202609, progress=False)

    # 1. Disjoint split verification
    split_mgr = SplitManager(data_dir)
    split_mgr.verify_disjoint()
    train_ids = split_mgr.get_ids("train")
    val_ids = split_mgr.get_ids("val")
    assert len(set(train_ids).intersection(val_ids)) == 0

    # 2. Zero-leakage check on Normalizer
    with pytest.raises(ValueError, match="Zero-leakage"):
        Normalizer.fit_from_zarr(
            data_dir / "trajectories.zarr",
            data_dir / "metadata.parquet",
            train_ids=train_ids + val_ids,
        )

    with open(data_dir / "norm_stats.json", "r", encoding="utf-8") as fh:
        norm_json = json.load(fh)
    assert len(norm_json["vars"]) == 12

    # 3. 200-batch PyTorch DataLoader stress test with num_workers=4 & persistent_workers=True
    ds = EcoVerseDataset(
        data_dir=data_dir,
        sys_ids=train_ids,
        mode="train",
        sensor_package="random",
        num_queries=32,
        max_tokens=512,
        seed=99,
    )
    sampler = RandomSampler(ds, replacement=True, num_samples=200 * 64,
                            generator=torch.Generator().manual_seed(99))
    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=64,
        sampler=sampler,
        num_workers=4,
        pin_memory=pin,
        persistent_workers=True,
        collate_fn=ecoverse_collate,
        generator=torch.Generator().manual_seed(99),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    live_bytes = []
    batch_bytes = []
    n_batches = 0
    for raw_batch in loader:
        batch = {
            k: v.to(device, non_blocking=(device.type == "cuda")) if isinstance(v, torch.Tensor) else v
            for k, v in raw_batch.items()
        }
        bsz, l_max = batch["obs_times"].shape
        q_len = batch["query_times"].shape[1]
        assert bsz == 64
        assert l_max >= 1
        assert batch["obs_var_ids"].shape == (bsz, l_max)
        assert batch["obs_values"].shape == (bsz, l_max)
        assert batch["obs_is_driver"].shape == (bsz, l_max)
        assert batch["obs_attn_mask"].shape == (bsz, l_max)
        assert batch["obs_attn_mask"].dtype == torch.bool
        assert batch["obs_var_ids"].dtype == torch.long
        assert batch["target_x"].shape == (bsz, q_len, ds.n_states)
        assert batch["target_r"].shape == (bsz, q_len, ds.n_channels)
        assert batch["target_state_mask"].shape == (bsz, ds.n_states)
        assert batch["obs_state_mask"].shape == (bsz, ds.n_states)
        assert batch["channel_mask"].shape == (bsz, ds.n_channels)

        # Verify mask alignment: padded slots must be zero and at least 1 valid token per item
        attn = batch["obs_attn_mask"]
        assert torch.all(attn.sum(dim=1) >= 1)
        if not torch.all(attn):
            pad_mask = ~attn
            assert torch.all(batch["obs_values"][pad_mask] == 0.0)
            assert torch.all(batch["obs_times"][pad_mask] == 0.0)
            del pad_mask

        assert torch.isfinite(batch["obs_values"]).all()
        assert torch.isfinite(batch["target_x"]).all()
        assert torch.isfinite(batch["target_r"]).all()
        n_batches += 1
        sample_memory = n_batches % 10 == 0 and device.type in ("mps", "cuda")
        if sample_memory:
            backend = torch.mps if device.type == "mps" else torch.cuda
            allocated = torch.mps.current_allocated_memory if device.type == "mps" else torch.cuda.memory_allocated
            backend.synchronize()
            batch_bytes.append(allocated())
        del batch, attn
        if sample_memory:
            backend.synchronize()
            live_bytes.append(allocated())

    assert n_batches == 200
    if live_bytes:
        assert all(before > after for before, after in zip(batch_bytes, live_bytes))
        assert live_bytes[-1] <= live_bytes[0] + 1024**2
    if out := os.environ.get("ECOVERSE_LOADER_REPORT"):
        Path(out).write_text(json.dumps({
            "device": str(device), "systems_in_train": len(ds), "batch_size": 64, "batches": n_batches,
            "num_workers": 4, "persistent_workers": True, "pin_memory": pin,
            "batch_device_bytes_every_10_batches": batch_bytes,
            "live_device_bytes_every_10_batches": live_bytes,
        }, indent=2))
    del loader
