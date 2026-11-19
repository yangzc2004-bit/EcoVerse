"""Train-only zero-leakage variable normalization for EcoVerse."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import zarr

LINEAR_VAR_NAMES = {"O", "C", "I", "T"}
DEFAULT_LOG_EPS = 1e-4


@dataclass
class VariableNormStat:
    name: str
    var_id: int
    is_driver: bool
    transform: str  # "log" or "linear"
    eps: float
    mean: float
    std: float


class Normalizer:
    """Per-variable normalizer computed strictly from the train split."""

    def __init__(self, stats: Sequence[VariableNormStat], n_states: int, n_drivers: int) -> None:
        self.stats: List[VariableNormStat] = sorted(stats, key=lambda s: s.var_id)
        self.n_states = int(n_states)
        self.n_drivers = int(n_drivers)
        self.by_id: Dict[int, VariableNormStat] = {s.var_id: s for s in self.stats}

        self._state_is_log = np.array(
            [self.by_id[i].transform == "log" for i in range(self.n_states)], dtype=bool
        )
        self._state_eps = np.array(
            [self.by_id[i].eps for i in range(self.n_states)], dtype=np.float32
        )
        self._state_mean = np.array(
            [self.by_id[i].mean for i in range(self.n_states)], dtype=np.float32
        )
        self._state_std = np.array(
            [self.by_id[i].std for i in range(self.n_states)], dtype=np.float32
        )

        self._drv_is_log = np.array(
            [self.by_id[self.n_states + d].transform == "log" for d in range(self.n_drivers)],
            dtype=bool,
        )
        self._drv_eps = np.array(
            [self.by_id[self.n_states + d].eps for d in range(self.n_drivers)], dtype=np.float32
        )
        self._drv_mean = np.array(
            [self.by_id[self.n_states + d].mean for d in range(self.n_drivers)], dtype=np.float32
        )
        self._drv_std = np.array(
            [self.by_id[self.n_states + d].std for d in range(self.n_drivers)], dtype=np.float32
        )

    @classmethod
    def fit_from_zarr(
        cls,
        zarr_path: Path | str,
        metadata_path: Path | str,
        train_ids: Iterable[int],
        batch_size: int = 256,
    ) -> "Normalizer":
        """Compute normalization stats using ONLY train_ids and effective_state_mask==1."""
        train_arr = np.asarray(sorted(set(int(i) for i in train_ids)), dtype=np.int64)
        if train_arr.size == 0:
            raise ValueError("train_ids must be non-empty when fitting Normalizer.")

        root = zarr.open_group(str(zarr_path), mode="r")
        x_ds = root["x_true"]
        u_ds = root["u_true"]
        n_states = int(root.attrs.get("n_states", x_ds.shape[-1]))
        n_drivers = int(root.attrs.get("n_drivers", u_ds.shape[-1]))
        state_names = list(root.attrs.get("state_names", [f"S{i}" for i in range(n_states)]))
        driver_names = list(root.attrs.get("driver_names", [f"D{i}" for i in range(n_drivers)]))

        df = pd.read_parquet(metadata_path)
        if not set(train_arr.tolist()).issubset(set(df["sys_id"].astype(int).tolist())):
            raise ValueError("Some train_ids are missing from metadata.parquet.")
        split_check = df.loc[df["sys_id"].isin(train_arr), "split_tag"].unique().tolist()
        if any(tag != "train" for tag in split_check):
            raise ValueError(f"Zero-leakage violation: non-train split tags in train_ids: {split_check}")

        df_idx = df.set_index("sys_id")
        eff_masks = np.zeros((len(train_arr), n_states), dtype=bool)
        for row_i, sid in enumerate(train_arr):
            eff_masks[row_i] = np.asarray(
                json.loads(df_idx.loc[int(sid), "effective_state_mask_json"]), dtype=bool
            )

        state_sum = np.zeros(n_states, dtype=np.float64)
        state_sq = np.zeros(n_states, dtype=np.float64)
        state_cnt = np.zeros(n_states, dtype=np.int64)

        drv_sum = np.zeros(n_drivers, dtype=np.float64)
        drv_sq = np.zeros(n_drivers, dtype=np.float64)
        drv_cnt = np.zeros(n_drivers, dtype=np.int64)

        state_is_log = [name not in LINEAR_VAR_NAMES for name in state_names]
        drv_is_log = [name not in LINEAR_VAR_NAMES for name in driver_names]

        for start in range(0, len(train_arr), batch_size):
            batch_ids = train_arr[start : start + batch_size]
            batch_masks = eff_masks[start : start + batch_size]
            # Load chunk from Zarr
            xb = np.asarray(x_ds[batch_ids], dtype=np.float64)  # (B, T, n_states)
            ub = np.asarray(u_ds[batch_ids], dtype=np.float64)  # (B, T, n_drivers)

            for i in range(n_states):
                valid = batch_masks[:, i]
                if not np.any(valid):
                    continue
                vals = xb[valid, :, i].reshape(-1)
                if state_is_log[i]:
                    vals = np.log(np.maximum(vals, 0.0) + DEFAULT_LOG_EPS)
                state_sum[i] += float(np.sum(vals))
                state_sq[i] += float(np.sum(vals * vals))
                state_cnt[i] += int(vals.size)

            for d in range(n_drivers):
                vals = ub[:, :, d].reshape(-1)
                if drv_is_log[d]:
                    vals = np.log(np.maximum(vals, 0.0) + DEFAULT_LOG_EPS)
                drv_sum[d] += float(np.sum(vals))
                drv_sq[d] += float(np.sum(vals * vals))
                drv_cnt[d] += int(vals.size)

        stats: List[VariableNormStat] = []
        for i in range(n_states):
            if state_cnt[i] > 0:
                mean = state_sum[i] / state_cnt[i]
                var = max(0.0, state_sq[i] / state_cnt[i] - mean * mean)
                std = max(float(np.sqrt(var)), 1e-6)
            else:
                mean, std = 0.0, 1.0
            stats.append(
                VariableNormStat(
                    name=state_names[i],
                    var_id=i,
                    is_driver=False,
                    transform="log" if state_is_log[i] else "linear",
                    eps=DEFAULT_LOG_EPS if state_is_log[i] else 0.0,
                    mean=float(mean),
                    std=float(std),
                )
            )

        for d in range(n_drivers):
            if drv_cnt[d] > 0:
                mean = drv_sum[d] / drv_cnt[d]
                var = max(0.0, drv_sq[d] / drv_cnt[d] - mean * mean)
                std = max(float(np.sqrt(var)), 1e-6)
            else:
                mean, std = 0.0, 1.0
            stats.append(
                VariableNormStat(
                    name=driver_names[d],
                    var_id=n_states + d,
                    is_driver=True,
                    transform="log" if drv_is_log[d] else "linear",
                    eps=DEFAULT_LOG_EPS if drv_is_log[d] else 0.0,
                    mean=float(mean),
                    std=float(std),
                )
            )

        return cls(stats=stats, n_states=n_states, n_drivers=n_drivers)

    def save(self, path: Path | str) -> None:
        payload = {
            "n_states": self.n_states,
            "n_drivers": self.n_drivers,
            "vars": [asdict(s) for s in self.stats],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "Normalizer":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        stats = [VariableNormStat(**item) for item in payload["vars"]]
        return cls(
            stats=stats,
            n_states=int(payload["n_states"]),
            n_drivers=int(payload["n_drivers"]),
        )

    def transform_var(self, var_id: int, values: np.ndarray) -> np.ndarray:
        stat = self.by_id[int(var_id)]
        arr = np.asarray(values, dtype=np.float32)
        if stat.transform == "log":
            arr = np.log(np.maximum(arr, 0.0) + np.float32(stat.eps))
        return (arr - np.float32(stat.mean)) / np.float32(stat.std)

    def transform_state_matrix(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).copy()
        if np.any(self._state_is_log):
            idx = self._state_is_log
            arr[..., idx] = np.log(np.maximum(arr[..., idx], 0.0) + self._state_eps[idx])
        return (arr - self._state_mean) / self._state_std

    def inverse_transform_state_matrix(self, x_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_norm, dtype=np.float32) * self._state_std + self._state_mean
        if np.any(self._state_is_log):
            idx = self._state_is_log
            arr[..., idx] = np.maximum(np.exp(arr[..., idx]) - self._state_eps[idx], 0.0)
        return arr

    def inverse_transform_state_torch(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self._state_mean, dtype=x_norm.dtype, device=x_norm.device)
        std = torch.as_tensor(self._state_std, dtype=x_norm.dtype, device=x_norm.device)
        eps = torch.as_tensor(self._state_eps, dtype=x_norm.dtype, device=x_norm.device)
        is_log = torch.as_tensor(self._state_is_log, dtype=torch.bool, device=x_norm.device)

        x_raw = x_norm * std + mean
        if torch.any(is_log):
            exp_vals = torch.clamp(torch.exp(torch.clamp(x_raw, max=20.0)) - eps, min=0.0)
            x_raw = torch.where(is_log, exp_vals, x_raw)
        return x_raw

    def transform_driver_matrix(self, u: np.ndarray) -> np.ndarray:
        arr = np.asarray(u, dtype=np.float32).copy()
        if np.any(self._drv_is_log):
            idx = self._drv_is_log
            arr[..., idx] = np.log(np.maximum(arr[..., idx], 0.0) + self._drv_eps[idx])
        return (arr - self._drv_mean) / self._drv_std
