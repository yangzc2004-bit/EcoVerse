"""PyTorch Dataset reading Zarr + Parquet lazily for safe multi-worker loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset, get_worker_info

from ecoverse.obs.normalize import Normalizer
from ecoverse.obs.sensors import load_sensors_config, sample_observation


class EcoVerseDataset(Dataset):
    """Event-stream PyTorch Dataset over EcoVerse Zarr trajectories.

    All state/channel/driver counts are dynamically read from Zarr metadata
    attributes and array shapes (forward-compatible with v1 arbitrary topologies).
    """

    def __init__(
        self,
        data_dir: Path | str,
        sys_ids: Sequence[int],
        mode: str = "train",
        sensor_package: str = "random",
        num_queries: int = 64,
        max_tokens: int = 1024,
        normalizer: Optional[Normalizer] = None,
        seed: int = 12345,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.zarr_path = self.data_dir / "trajectories.zarr"
        self.meta_path = self.data_dir / "metadata.parquet"
        self.sys_ids = [int(i) for i in sys_ids]
        self.mode = mode
        self.sensor_package = sensor_package
        self.num_queries = int(num_queries)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)

        self.normalizer = normalizer or Normalizer.load(self.data_dir / "norm_stats.json")
        self.sensors_cfg = load_sensors_config()

        # Read metadata and Zarr schema attrs once, then release Zarr handle for worker safety
        root = zarr.open_group(str(self.zarr_path), mode="r")
        x_ds = root["x_true"]
        r_ds = root["r_true"]
        u_ds = root["u_true"]
        x_cf_ds = root["x_cf"]

        self.n_steps_formal = int(root.attrs.get("n_steps_formal", x_ds.shape[1]))
        self.n_steps_cf = int(root.attrs.get("n_steps_cf", x_cf_ds.shape[1]))
        self.n_states = int(root.attrs.get("n_states", x_ds.shape[-1]))
        self.n_channels = int(root.attrs.get("n_channels", r_ds.shape[-1]))
        self.n_drivers = int(root.attrs.get("n_drivers", u_ds.shape[-1]))
        self.dt_days = float(root.attrs.get("dt_days", 1.0 / 6.0))
        self.intervention_day = float(root.attrs.get("intervention_day", 180.0))
        self.intervention_step = self.n_steps_formal - self.n_steps_cf
        self.formal_days = float(self.n_steps_formal) * self.dt_days

        self.state_names: List[str] = list(
            root.attrs.get("state_names", [f"S{i}" for i in range(self.n_states)])
        )
        self.driver_names: List[str] = list(
            root.attrs.get("driver_names", [f"D{i}" for i in range(self.n_drivers)])
        )
        self.t_formal_days = np.arange(self.n_steps_formal, dtype=np.float32) * np.float32(self.dt_days)
        self.t_cf_days = self.t_formal_days[self.intervention_step :]

        df = pd.read_parquet(self.meta_path).set_index("sys_id")
        self._eff_masks: Dict[int, np.ndarray] = {}
        self._ch_masks: Dict[int, np.ndarray] = {}
        for sid in self.sys_ids:
            row = df.loc[sid]
            self._eff_masks[sid] = np.asarray(
                json.loads(row["effective_state_mask_json"]), dtype=bool
            )
            self._ch_masks[sid] = np.asarray(json.loads(row["channel_mask_json"]), dtype=bool)

        # Lazy per-worker Zarr handles
        self._zarr_root: Optional[zarr.Group] = None
        self._worker_rng: Optional[np.random.Generator] = None

    def _get_zarr(self) -> zarr.Group:
        if self._zarr_root is None:
            self._zarr_root = zarr.open_group(str(self.zarr_path), mode="r")
        return self._zarr_root

    def _get_rng(self, sys_id: int) -> np.random.Generator:
        if self.mode == "train":
            if self._worker_rng is None:
                wi = get_worker_info()
                wid = wi.id if wi is not None else 0
                self._worker_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, wid, int(torch.initial_seed() & 0xFFFFFFFF)])
                )
            return self._worker_rng
        pkg_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(self.sensor_package))
        return np.random.default_rng(np.random.SeedSequence([self.seed, sys_id, pkg_hash]))

    def __len__(self) -> int:
        return len(self.sys_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sid = self.sys_ids[index]
        root = self._get_zarr()
        rng = self._get_rng(sid)

        eff_mask = self._eff_masks[sid]
        ch_mask = self._ch_masks[sid]

        if self.mode == "counterfactual":
            x_pre = np.asarray(root["x_true"][sid, : self.intervention_step], dtype=np.float32)
            u_pre = np.asarray(root["u_true"][sid, : self.intervention_step], dtype=np.float32)
            t_pre = self.t_formal_days[: self.intervention_step]

            x_cf = np.asarray(root["x_cf"][sid], dtype=np.float32)
            r_cf = np.asarray(root["r_cf"][sid], dtype=np.float32)
            u_cf = np.asarray(root["u_cf"][sid], dtype=np.float32)
            x_base_post = np.asarray(root["x_true"][sid, self.intervention_step :], dtype=np.float32)

            obs = sample_observation(
                x_obs_window=x_pre,
                u_obs_window=u_pre,
                t_obs_days=t_pre,
                effective_state_mask=eff_mask,
                state_names=self.state_names,
                driver_names=self.driver_names,
                normalizer=self.normalizer,
                rng=rng,
                package=self.sensor_package,
                norm_time_denom=self.formal_days,
                max_tokens=self.max_tokens,
                future_u=u_cf,
                future_t_days=self.t_cf_days,
                sensors_cfg=self.sensors_cfg,
            )

            q_count = min(self.num_queries, self.n_steps_cf)
            q_idx = np.linspace(0, self.n_steps_cf - 1, q_count, dtype=np.int64)
            query_times = (self.t_cf_days[q_idx] / self.formal_days).astype(np.float32)
            target_x = self.normalizer.transform_state_matrix(x_cf[q_idx])
            target_r = r_cf[q_idx].astype(np.float32)
            baseline_x = self.normalizer.transform_state_matrix(x_base_post[q_idx])

            return {
                "obs_times": torch.from_numpy(obs["obs_times"]),
                "obs_var_ids": torch.from_numpy(obs["obs_var_ids"]),
                "obs_values": torch.from_numpy(obs["obs_values"]),
                "obs_is_driver": torch.from_numpy(obs["obs_is_driver"]),
                "obs_state_mask": torch.from_numpy(obs["obs_state_mask"]),
                "query_times": torch.from_numpy(query_times),
                "target_x": torch.from_numpy(target_x),
                "target_r": torch.from_numpy(target_r),
                "target_state_mask": torch.from_numpy(eff_mask.copy()),
                "channel_mask": torch.from_numpy(ch_mask.copy()),
                "baseline_x": torch.from_numpy(baseline_x),
                "sys_id": sid,
            }

        # Standard baseline mode ("train" or "eval")
        x_full = np.asarray(root["x_true"][sid], dtype=np.float32)
        r_full = np.asarray(root["r_true"][sid], dtype=np.float32)
        u_full = np.asarray(root["u_true"][sid], dtype=np.float32)

        obs = sample_observation(
            x_obs_window=x_full,
            u_obs_window=u_full,
            t_obs_days=self.t_formal_days,
            effective_state_mask=eff_mask,
            state_names=self.state_names,
            driver_names=self.driver_names,
            normalizer=self.normalizer,
            rng=rng,
            package=self.sensor_package,
            norm_time_denom=self.formal_days,
            max_tokens=self.max_tokens,
            sensors_cfg=self.sensors_cfg,
        )

        q_count = min(self.num_queries, self.n_steps_formal)
        if self.mode == "train":
            q_idx = np.sort(rng.choice(self.n_steps_formal, size=q_count, replace=False))
        else:
            q_idx = np.linspace(0, self.n_steps_formal - 1, q_count, dtype=np.int64)

        query_times = (self.t_formal_days[q_idx] / self.formal_days).astype(np.float32)
        target_x = self.normalizer.transform_state_matrix(x_full[q_idx])
        target_r = r_full[q_idx].astype(np.float32)

        return {
            "obs_times": torch.from_numpy(obs["obs_times"]),
            "obs_var_ids": torch.from_numpy(obs["obs_var_ids"]),
            "obs_values": torch.from_numpy(obs["obs_values"]),
            "obs_is_driver": torch.from_numpy(obs["obs_is_driver"]),
            "obs_state_mask": torch.from_numpy(obs["obs_state_mask"]),
            "query_times": torch.from_numpy(query_times),
            "target_x": torch.from_numpy(target_x),
            "target_r": torch.from_numpy(target_r),
            "target_state_mask": torch.from_numpy(eff_mask.copy()),
            "channel_mask": torch.from_numpy(ch_mask.copy()),
            "sys_id": sid,
        }
