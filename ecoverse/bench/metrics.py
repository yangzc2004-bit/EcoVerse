"""Vectorized PyTorch benchmark evaluation metrics for EcoVerse v0."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ecoverse.core.constants import STATE_NAMES, compute_macro_fluxes


def compute_masked_nrmse(
    pred_x: torch.Tensor,
    target_x: torch.Tensor,
    mask: torch.Tensor,
    state_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compute NRMSE over valid masked entries (B, Q, N_state).

    mask can be (B, N_state) or (B, Q, N_state) boolean tensor.
    """
    if mask.ndim == 2:
        mask_3d = mask.unsqueeze(1).expand_as(target_x)
    else:
        mask_3d = mask
    mask_3d = mask_3d.bool()

    n_states = target_x.shape[-1]
    names = list(state_names) if state_names is not None else [
        STATE_NAMES[i] if i < len(STATE_NAMES) else f"S{i}" for i in range(n_states)
    ]

    sq_err = (pred_x - target_x) ** 2
    per_state: Dict[str, float] = {}
    valid_nrmses = []

    for i in range(n_states):
        m_i = mask_3d[..., i]
        cnt = int(m_i.sum().item())
        if cnt < 2:
            continue
        t_i = target_x[..., i][m_i]
        e_i = sq_err[..., i][m_i]
        rmse_i = torch.sqrt(torch.mean(e_i))
        denom_i = torch.clamp(torch.std(t_i, unbiased=False), min=0.1)
        val = float((rmse_i / denom_i).item())
        per_state[names[i]] = val
        valid_nrmses.append(val)

    total_cnt = int(mask_3d.sum().item())
    if total_cnt >= 2:
        t_all = target_x[mask_3d]
        e_all = sq_err[mask_3d]
        overall = float((torch.sqrt(torch.mean(e_all)) / torch.clamp(torch.std(t_all, unbiased=False), min=0.1)).item())
    else:
        overall = float("nan")

    mean_state_nrmse = float(sum(valid_nrmses) / len(valid_nrmses)) if valid_nrmses else float("nan")
    return {
        "overall_nrmse": overall,
        "mean_state_nrmse": mean_state_nrmse,
        "per_state_nrmse": per_state,
    }


def _safe_r2(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    if target.numel() < 10:
        return None
    t_mean = torch.mean(target)
    ss_tot = torch.sum((target - t_mean) ** 2)
    if float(ss_tot.item()) / target.numel() < 1e-5:
        return None
    ss_res = torch.sum((target - pred) ** 2)
    return float((1.0 - ss_res / ss_tot).item())


def compute_flux_r2(
    pred_r: torch.Tensor,
    true_r: torch.Tensor,
    channel_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Compute per-channel R^2 and macro GPP/ER/NEP R^2."""
    n_chan = true_r.shape[-1]
    if channel_mask is not None:
        if channel_mask.ndim == 2:
            cmask = channel_mask.unsqueeze(1).expand_as(true_r)
        else:
            cmask = channel_mask
        cmask = cmask.bool()
    else:
        cmask = torch.ones_like(true_r, dtype=torch.bool)

    per_channel: Dict[int, float] = {}
    vals = []
    for j in range(n_chan):
        mj = cmask[..., j]
        r2_j = _safe_r2(pred_r[..., j][mj], true_r[..., j][mj])
        if r2_j is not None:
            per_channel[j] = r2_j
            vals.append(r2_j)

    pred_gpp, pred_er, pred_nep = compute_macro_fluxes(pred_r)
    true_gpp, true_er, true_nep = compute_macro_fluxes(true_r)

    gpp_r2 = _safe_r2(pred_gpp.reshape(-1), true_gpp.reshape(-1))
    er_r2 = _safe_r2(pred_er.reshape(-1), true_er.reshape(-1))
    nep_r2 = _safe_r2(pred_nep.reshape(-1), true_nep.reshape(-1))

    return {
        "mean_channel_r2": float(sum(vals) / len(vals)) if vals else float("nan"),
        "n_scored_channels": len(vals),
        "per_channel_r2": per_channel,
        "gpp_r2": gpp_r2 if gpp_r2 is not None else float("nan"),
        "er_r2": er_r2 if er_r2 is not None else float("nan"),
        "nep_r2": nep_r2 if nep_r2 is not None else float("nan"),
    }


def compute_cross_obs_alignment(
    z_by_pkg: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute pairwise cosine similarity of latent vectors across sensor packages."""
    pkgs = sorted(z_by_pkg.keys())
    pair_scores: Dict[str, float] = {}
    all_vals = []
    for i in range(len(pkgs)):
        for j in range(i + 1, len(pkgs)):
            p1, p2 = pkgs[i], pkgs[j]
            z1 = F.normalize(z_by_pkg[p1].float(), dim=-1)
            z2 = F.normalize(z_by_pkg[p2].float(), dim=-1)
            cos = float(torch.mean(torch.sum(z1 * z2, dim=-1)).item())
            pair_scores[f"{p1}_{p2}_cosine"] = cos
            all_vals.append(cos)
    pair_scores["mean_cross_obs_cosine"] = (
        float(sum(all_vals) / len(all_vals)) if all_vals else float("nan")
    )
    return pair_scores


def compute_counterfactual_metrics(
    pred_x_cf: torch.Tensor,
    true_x_cf: torch.Tensor,
    base_x: torch.Tensor,
    state_mask: torch.Tensor,
    delta_threshold: float = 0.05,
) -> Dict[str, float]:
    """Evaluate post-intervention trajectory NRMSE and steady-state direction accuracy."""
    nrmse_res = compute_masked_nrmse(pred_x_cf, true_x_cf, state_mask)

    q_len = true_x_cf.shape[1]
    tail = max(1, q_len // 4)
    true_delta = torch.mean(true_x_cf[:, -tail:, :] - base_x[:, -tail:, :], dim=1)  # (B, N_state)
    pred_delta = torch.mean(pred_x_cf[:, -tail:, :] - base_x[:, -tail:, :], dim=1)  # (B, N_state)

    valid = state_mask.bool() & (torch.abs(true_delta) > delta_threshold)
    n_valid = int(valid.sum().item())
    if n_valid > 0:
        correct = (torch.sign(pred_delta[valid]) == torch.sign(true_delta[valid])).float()
        dir_acc = float(torch.mean(correct).item())
    else:
        dir_acc = float("nan")

    return {
        "cf_overall_nrmse": float(nrmse_res["overall_nrmse"]),  # type: ignore[arg-type]
        "cf_mean_state_nrmse": float(nrmse_res["mean_state_nrmse"]),  # type: ignore[arg-type]
        "cf_direction_accuracy": dir_acc,
        "cf_n_scored_directions": float(n_valid),
    }
