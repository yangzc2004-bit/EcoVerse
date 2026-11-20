"""Variable-length event-stream collate function for PyTorch DataLoader."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def ecoverse_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length observation event streams into padded GPU-ready tensors.

    Dimension-agnostic across state and channel counts (compatible with v1+).
    """
    bsz = len(batch)
    l_max = max(1, max(int(item["obs_times"].shape[0]) for item in batch))

    obs_times = torch.zeros((bsz, l_max), dtype=torch.float32)
    obs_var_ids = torch.zeros((bsz, l_max), dtype=torch.long)
    obs_values = torch.zeros((bsz, l_max), dtype=torch.float32)
    obs_is_driver = torch.zeros((bsz, l_max), dtype=torch.bool)
    obs_attn_mask = torch.zeros((bsz, l_max), dtype=torch.bool)

    for b, item in enumerate(batch):
        length = int(item["obs_times"].shape[0])
        if length > 0:
            obs_times[b, :length] = item["obs_times"]
            obs_var_ids[b, :length] = item["obs_var_ids"]
            obs_values[b, :length] = item["obs_values"]
            obs_is_driver[b, :length] = item["obs_is_driver"]
            obs_attn_mask[b, :length] = True

    out: Dict[str, torch.Tensor] = {
        "obs_times": obs_times,
        "obs_var_ids": obs_var_ids,
        "obs_values": obs_values,
        "obs_is_driver": obs_is_driver,
        "obs_attn_mask": obs_attn_mask,
        "query_times": torch.stack([item["query_times"] for item in batch], dim=0),
        "target_x": torch.stack([item["target_x"] for item in batch], dim=0),
        "target_r": torch.stack([item["target_r"] for item in batch], dim=0),
        "target_state_mask": torch.stack([item["target_state_mask"] for item in batch], dim=0),
        "obs_state_mask": torch.stack([item["obs_state_mask"] for item in batch], dim=0),
        "channel_mask": torch.stack([item["channel_mask"] for item in batch], dim=0),
        "sys_id": torch.tensor([int(item["sys_id"]) for item in batch], dtype=torch.long),
    }

    if "intervention" in batch[0]:
        out["intervention"] = torch.stack([item["intervention"] for item in batch])
    if all("baseline_x" in item for item in batch):
        out["baseline_x"] = torch.stack([item["baseline_x"] for item in batch], dim=0)

    return out
