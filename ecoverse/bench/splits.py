"""Dataset split management and strict zero-overlap verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

SPLIT_NAMES = (
    "train",
    "val",
    "test_1_obs",
    "test_2a_param",
    "test_2b_comp",
    "test_2c_shape",
    "test_3_interv",
)


def build_splits_json(metadata_df: pd.DataFrame, output_path: Path | str) -> Dict[str, List[int]]:
    """Build and verify disjoint split index map from metadata DataFrame."""
    splits: Dict[str, List[int]] = {}
    seen: set[int] = set()

    for tag, grp in metadata_df.groupby("split_tag", sort=False):
        ids = sorted(int(i) for i in grp["sys_id"].tolist())
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Split '{tag}' overlaps with previous splits on sys_ids: {sorted(overlap)[:5]}")
        seen.update(ids)
        splits[str(tag)] = ids

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    return splits


class SplitManager:
    """Loads locked split indices from splits.json and guarantees disjointness."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.splits: Dict[str, List[int]] = {k: [int(i) for i in v] for k, v in raw.items()}
        else:
            df = pd.read_parquet(self.data_dir / "metadata.parquet")
            self.splits = build_splits_json(df, splits_path)
        self.verify_disjoint()

    def verify_disjoint(self) -> None:
        seen: set[int] = set()
        for name, ids in self.splits.items():
            id_set = set(ids)
            if len(id_set) != len(ids):
                raise ValueError(f"Duplicate sys_id within split '{name}'")
            overlap = seen.intersection(id_set)
            if overlap:
                raise ValueError(f"Overlap detected involving split '{name}': {sorted(overlap)[:5]}")
            seen.update(id_set)

    def get_ids(self, split_name: str) -> List[int]:
        if split_name not in self.splits:
            raise KeyError(f"Unknown split '{split_name}'. Available: {list(self.splits.keys())}")
        return list(self.splits[split_name])

    def summary(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.splits.items()}
