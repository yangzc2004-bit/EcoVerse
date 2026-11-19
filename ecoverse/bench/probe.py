"""GPU-accelerated Ridge Regression linear probe for Test 4 mechanism understanding."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from ecoverse.core.constants import compute_macro_fluxes


def _build_target_matrix(R: torch.Tensor) -> torch.Tensor:
    """Append GPP, ER, NEP to atomic flux targets: (N, n_chan + 3)."""
    gpp, er, nep = compute_macro_fluxes(R)
    return torch.cat([R, gpp.unsqueeze(-1), er.unsqueeze(-1), nep.unsqueeze(-1)], dim=-1)


def _solve_ridge(
    Z_c: torch.Tensor,
    Y_c: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Solve centered ridge regression W = (Z^T Z + alpha I)^-1 Z^T Y."""
    d = Z_c.shape[1]
    gram = Z_c.T @ Z_c
    gram = gram + alpha * torch.eye(d, dtype=Z_c.dtype, device=Z_c.device)
    rhs = Z_c.T @ Y_c
    try:
        return torch.linalg.solve(gram, rhs)
    except NotImplementedError:
        return torch.linalg.solve(gram.cpu(), rhs.cpu()).to(Z_c.device)


def _compute_target_r2(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, Dict[int, float], float, float, float]:
    n_tot = target.shape[1]
    n_chan = n_tot - 3

    t_mean = torch.mean(target, dim=0, keepdim=True)
    ss_tot = torch.sum((target - t_mean) ** 2, dim=0)
    ss_res = torch.sum((target - pred) ** 2, dim=0)

    per_channel: Dict[int, float] = {}
    vals = []
    for j in range(n_chan):
        if float(ss_tot[j].item()) / max(target.shape[0], 1) > 1e-5:
            r2_j = float((1.0 - ss_res[j] / ss_tot[j]).item())
            per_channel[j] = r2_j
            vals.append(r2_j)

    mean_chan = float(sum(vals) / len(vals)) if vals else float("nan")

    def _idx_r2(idx: int) -> float:
        if float(ss_tot[idx].item()) / max(target.shape[0], 1) > 1e-5:
            return float((1.0 - ss_res[idx] / ss_tot[idx]).item())
        return float("nan")

    return mean_chan, per_channel, _idx_r2(n_chan), _idx_r2(n_chan + 1), _idx_r2(n_chan + 2)


@torch.no_grad()
def fit_and_eval_linear_probe(
    Z_train: torch.Tensor,
    R_train: torch.Tensor,
    Z_eval: torch.Tensor,
    R_eval: torch.Tensor,
    device: Optional[torch.device | str] = None,
    alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> Dict[str, object]:
    """Fit Ridge probe on (Z_train -> R_train + GPP/ER/NEP) on GPU and evaluate on Z_eval."""
    dev = torch.device(device) if device is not None else Z_train.device

    Zt = Z_train.to(dev, dtype=torch.float32)
    Rt = _build_target_matrix(R_train.to(dev, dtype=torch.float32))
    Ze = Z_eval.to(dev, dtype=torch.float32)
    Re = _build_target_matrix(R_eval.to(dev, dtype=torch.float32))

    # Deterministic 80/20 split inside Z_train to select alpha with zero test leakage
    n_tr = Zt.shape[0]
    n_fit = max(2, int(0.8 * n_tr))
    if n_tr - n_fit >= 2:
        Z_fit, Z_val = Zt[:n_fit], Zt[n_fit:]
        R_fit, R_val = Rt[:n_fit], Rt[n_fit:]
        z_m = torch.mean(Z_fit, dim=0, keepdim=True)
        r_m = torch.mean(R_fit, dim=0, keepdim=True)
        best_alpha = float(alphas[0])
        best_mse = float("inf")
        for alpha in alphas:
            W = _solve_ridge(Z_fit - z_m, R_fit - r_m, float(alpha))
            pred_val = (Z_val - z_m) @ W + r_m
            mse = float(torch.mean((pred_val - R_val) ** 2).item())
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
    else:
        best_alpha = float(alphas[len(alphas) // 2])

    # Refit on full Z_train with best_alpha
    z_mean = torch.mean(Zt, dim=0, keepdim=True)
    r_mean = torch.mean(Rt, dim=0, keepdim=True)
    W_full = _solve_ridge(Zt - z_mean, Rt - r_mean, best_alpha)
    pred_eval = (Ze - z_mean) @ W_full + r_mean

    mean_chan, per_chan, gpp_r2, er_r2, nep_r2 = _compute_target_r2(pred_eval, Re)
    return {
        "best_alpha": best_alpha,
        "mean_channel_r2": mean_chan,
        "per_channel_r2": per_chan,
        "gpp_r2": gpp_r2,
        "er_r2": er_r2,
        "nep_r2": nep_r2,
    }


@torch.no_grad()
def compare_pretrained_vs_random_probe(
    Z_train_pretrain: torch.Tensor,
    Z_eval_pretrain: torch.Tensor,
    Z_train_random: torch.Tensor,
    Z_eval_random: torch.Tensor,
    R_train: torch.Tensor,
    R_eval: torch.Tensor,
    device: Optional[torch.device | str] = None,
    alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> Dict[str, object]:
    """Run Test 4 probe comparing pretrained encoder vs randomly initialized baseline."""
    pre = fit_and_eval_linear_probe(
        Z_train_pretrain, R_train, Z_eval_pretrain, R_eval, device=device, alphas=alphas
    )
    rnd = fit_and_eval_linear_probe(
        Z_train_random, R_train, Z_eval_random, R_eval, device=device, alphas=alphas
    )

    delta_chan = float(pre["mean_channel_r2"]) - float(rnd["mean_channel_r2"])  # type: ignore[arg-type]
    delta_gpp = float(pre["gpp_r2"]) - float(rnd["gpp_r2"])  # type: ignore[arg-type]
    delta_er = float(pre["er_r2"]) - float(rnd["er_r2"])  # type: ignore[arg-type]
    delta_nep = float(pre["nep_r2"]) - float(rnd["nep_r2"])  # type: ignore[arg-type]

    return {
        "r2_pretrain": pre["mean_channel_r2"],
        "r2_random": rnd["mean_channel_r2"],
        "delta_r2": delta_chan,
        "gpp_r2_pretrain": pre["gpp_r2"],
        "gpp_r2_random": rnd["gpp_r2"],
        "gpp_delta_r2": delta_gpp,
        "er_r2_pretrain": pre["er_r2"],
        "er_r2_random": rnd["er_r2"],
        "er_delta_r2": delta_er,
        "nep_r2_pretrain": pre["nep_r2"],
        "nep_r2_random": rnd["nep_r2"],
        "nep_delta_r2": delta_nep,
        "pretrain_detail": pre,
        "random_detail": rnd,
    }
