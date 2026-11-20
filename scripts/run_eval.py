"""End-to-end GPU Benchmark evaluation harness (Tests 1-4) with baseline & random control."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from ecoverse.bench.metrics import (
    compute_counterfactual_metrics,
    compute_cross_obs_alignment,
    compute_flux_r2,
    compute_masked_nrmse,
)
from ecoverse.bench.probe import compare_pretrained_vs_random_probe
from ecoverse.bench.splits import SplitManager
from ecoverse.core.constants import INTERVENTION_FEATURES, SIGNED_FLUX_CHANNELS
from ecoverse.obs.collate import ecoverse_collate
from ecoverse.obs.dataset import EcoVerseDataset
from ecoverse.obs.normalize import Normalizer


def _select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EventStreamBaseline(nn.Module):
    """Lightweight event-stream Transformer encoder + query decoder baseline."""

    def __init__(
        self,
        n_vars: int,
        n_states: int,
        n_channels: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.var_emb = nn.Embedding(n_vars, d_model)
        self.cont_proj = nn.Linear(3, d_model)
        self.in_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.query_net = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.state_head = nn.Linear(d_model, n_states)
        self.rate_head = nn.Linear(d_model, n_channels)
        self.action_proj = nn.Linear(len(INTERVENTION_FEATURES), d_model, bias=False)
        signed = torch.tensor([j in SIGNED_FLUX_CHANNELS for j in range(n_channels)])
        self.register_buffer("signed_channels", signed)
        nn.init.constant_(self.rate_head.bias, -3.0)
        with torch.no_grad():
            self.rate_head.bias[signed] = 0

    def encode(
        self,
        obs_times: torch.Tensor,
        obs_var_ids: torch.Tensor,
        obs_values: torch.Tensor,
        obs_is_driver: torch.Tensor,
        obs_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        cont = torch.stack(
            [obs_values, obs_times, obs_is_driver.float()], dim=-1
        )  # (B, L, 3)
        h = self.in_norm(self.var_emb(obs_var_ids) + self.cont_proj(cont))
        pad_mask = ~obs_attn_mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        w = obs_attn_mask.unsqueeze(-1).float()
        pooled = (h * w).sum(dim=1) / torch.clamp(w.sum(dim=1), min=1.0)
        return pooled  # (B, d_model)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_sys = self.encode(
            batch["obs_times"],
            batch["obs_var_ids"],
            batch["obs_values"],
            batch["obs_is_driver"],
            batch["obs_attn_mask"],
        )  # (B, D)
        if "intervention" in batch:
            z_sys = z_sys + self.action_proj(batch["intervention"])
        qt = batch["query_times"].unsqueeze(-1)  # (B, Q, 1)
        z_exp = z_sys.unsqueeze(1).expand(-1, qt.shape[1], -1)
        z_t = self.query_net(torch.cat([z_exp, qt], dim=-1))  # (B, Q, D)
        pred_x = self.state_head(z_t)
        raw_r = self.rate_head(z_t)
        pred_r = torch.where(self.signed_channels, raw_r, F.softplus(raw_r))
        return pred_x, pred_r, z_sys, z_t


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    non_block = device.type == "cuda"
    return {
        k: v.to(device, non_blocking=non_block) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _train_baseline(
    model: EventStreamBaseline,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_x, pred_r, _, _ = model(batch)
            mask_x = batch["target_state_mask"].unsqueeze(1).expand_as(pred_x)
            loss_x = F.mse_loss(pred_x[mask_x], batch["target_x"][mask_x])
            mask_r = batch["channel_mask"].unsqueeze(1).expand_as(pred_r)
            loss_r = F.mse_loss(
                torch.asinh(pred_r[mask_r]), torch.asinh(batch["target_r"][mask_r])
            )
            loss = loss_x + 0.5 * loss_r
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += float(loss.item())
            n_batches += 1

        model.eval()
        val_error = 0.0
        val_count = 0
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _to_device(raw_batch, device)
                pred_x, _, _, _ = model(batch)
                mask_x = batch["target_state_mask"].unsqueeze(1).expand_as(pred_x)
                val_error += float(F.mse_loss(pred_x[mask_x], batch["target_x"][mask_x], reduction="sum"))
                val_count += int(mask_x.sum())
        mean_val = val_error / max(val_count, 1)
        history.append(dict(epoch=ep, train_loss=tot_loss / max(n_batches, 1), val_state_mse=mean_val))
        if mean_val < best_val:
            best_val, best_epoch = mean_val, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"  Epoch {ep}/{epochs} | train_loss={tot_loss / max(n_batches, 1):.4f} | val_x_mse={mean_val:.4f}")
        if ep - best_epoch >= 3:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "selected_epoch": best_epoch, "patience": 3}


@torch.no_grad()
def _collect_predictions(
    model: EventStreamBaseline,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    keys = [
        "pred_x", "target_x", "pred_r", "target_r",
        "target_state_mask", "obs_state_mask", "channel_mask", "z_sys", "z_t",
    ]
    buckets: Dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    base_list = []

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        pred_x, pred_r, z_sys, z_t = model(batch)
        buckets["pred_x"].append(pred_x.cpu())
        buckets["target_x"].append(batch["target_x"].cpu())
        buckets["pred_r"].append(pred_r.cpu())
        buckets["target_r"].append(batch["target_r"].cpu())
        buckets["target_state_mask"].append(batch["target_state_mask"].cpu())
        buckets["obs_state_mask"].append(batch["obs_state_mask"].cpu())
        buckets["channel_mask"].append(batch["channel_mask"].cpu())
        buckets["z_sys"].append(z_sys.cpu())
        buckets["z_t"].append(z_t.cpu())
        if "baseline_x" in batch:
            base_list.append(batch["baseline_x"].cpu())

    out = {k: torch.cat(v, dim=0) for k, v in buckets.items()}
    if len(base_list) == len(buckets["pred_x"]):
        out["baseline_x"] = torch.cat(base_list, dim=0)
    return out


def run_benchmark_eval(
    data_dir: Path,
    output_path: Path,
    epochs: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    device_str: str = "auto",
    seed: int = 42,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    device = _select_device(device_str)
    pin = device.type == "cuda"

    print(f"Running EcoVerse Benchmark Evaluation on device={device} (data={data_dir})")
    split_mgr = SplitManager(data_dir)
    normalizer = Normalizer.load(data_dir / "norm_stats.json")

    def dataset(ids, mode, pkg, num_q=32):
        return EcoVerseDataset(
            data_dir=data_dir,
            sys_ids=ids,
            mode=mode,
            sensor_package=pkg,
            num_queries=num_q,
            max_tokens=512,
            normalizer=normalizer,
            seed=seed,
        )

    def loader(ds, shuffle=False, persistent=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent and num_workers > 0,
            collate_fn=ecoverse_collate,
        )

    train_ids = split_mgr.get_ids("train")
    val_ids = split_mgr.get_ids("val")
    factual_train = dataset(train_ids, "train", "random")
    train_loader = loader(ConcatDataset([factual_train, dataset(train_ids, "counterfactual_train", "random")]),
                          shuffle=True, persistent=True)
    val_loader = loader(ConcatDataset([dataset(val_ids, "eval", "random"),
                                       dataset(val_ids, "counterfactual", "random")]), persistent=True)

    n_states = normalizer.n_states
    n_vars = normalizer.n_states + normalizer.n_drivers
    n_channels = factual_train.n_channels

    model = EventStreamBaseline(n_vars=n_vars, n_states=n_states, n_channels=n_channels).to(device)
    random_model = EventStreamBaseline(n_vars=n_vars, n_states=n_states, n_channels=n_channels).to(device)

    print("Training baseline EventStreamBaseline...")
    training = _train_baseline(model, train_loader, val_loader, device, epochs=epochs)

    del train_loader, val_loader
    report: Dict[str, object] = {
        "device": str(device), "epochs": epochs, "seed": seed, "batch_size": batch_size,
        "num_workers": num_workers, "max_tokens": 512, "splits": split_mgr.summary(),
        "training_views_per_epoch": {"factual": len(train_ids), "counterfactual": len(train_ids)},
        "training": training,
    }

    # Test 1: Cross-observation reconstruction & alignment (Packages A, B, C)
    print("\n[Test 1] Evaluating test_1_obs across Packages A / B / C...")
    pkg_results = {}
    z_by_pkg: Dict[str, torch.Tensor] = {}
    for pkg in ("A", "B", "C"):
        ldr = loader(dataset(split_mgr.get_ids("test_1_obs"), "eval", pkg))
        out = _collect_predictions(model, ldr, device)
        unobs_mask = out["target_state_mask"] & (~out["obs_state_mask"])
        nrmse_unobs = compute_masked_nrmse(out["pred_x"], out["target_x"], unobs_mask)
        flux_r2 = compute_flux_r2(out["pred_r"], out["target_r"], out["channel_mask"])
        z_by_pkg[pkg] = out["z_sys"]
        pkg_results[pkg] = {
            "unobs_overall_nrmse": nrmse_unobs["overall_nrmse"],
            "n_unobserved_state_system_pairs": int(unobs_mask.sum()),
            "mean_flux_r2": flux_r2["mean_channel_r2"],
            "gpp_r2": flux_r2["gpp_r2"],
            "er_r2": flux_r2["er_r2"],
            "nep_r2": flux_r2["nep_r2"],
            "per_channel_r2": {j: flux_r2["per_channel_r2"].get(j) for j in range(n_channels)},
        }
    align = compute_cross_obs_alignment(z_by_pkg)
    report["test_1_obs"] = {"packages": pkg_results, "alignment": align}

    # Tests 2a, 2b, 2c: Zero-shot generalization
    for sname in ("test_2a_param", "test_2b_comp", "test_2c_shape"):
        print(f"[Test {sname}] Evaluating...")
        ldr = loader(dataset(split_mgr.get_ids(sname), "eval", "B"))
        out = _collect_predictions(model, ldr, device)
        nrmse = compute_masked_nrmse(out["pred_x"], out["target_x"], out["target_state_mask"])
        flux = compute_flux_r2(out["pred_r"], out["target_r"], out["channel_mask"])
        report[sname] = {
            "overall_nrmse": nrmse["overall_nrmse"],
            "mean_state_nrmse": nrmse["mean_state_nrmse"],
            "mean_flux_r2": flux["mean_channel_r2"],
            "gpp_r2": flux["gpp_r2"],
            "er_r2": flux["er_r2"],
            "nep_r2": flux["nep_r2"],
            "per_channel_r2": {j: flux["per_channel_r2"].get(j) for j in range(n_channels)},
        }
    report["test_2c_shape"]["delta_nrmse_vs_2a"] = (
        report["test_2c_shape"]["overall_nrmse"] - report["test_2a_param"]["overall_nrmse"]
    )

    # Test 3: Counterfactual intervention forecasting
    print("[Test 3] Evaluating test_3_interv counterfactual forecasting...")
    cf_ldr = loader(dataset(split_mgr.get_ids("test_3_interv"), "counterfactual", "A"))
    cf_out = _collect_predictions(model, cf_ldr, device)
    cf_metrics = compute_counterfactual_metrics(
        cf_out["pred_x"], cf_out["target_x"], cf_out["baseline_x"], cf_out["target_state_mask"]
    )
    report["test_3_interv"] = cf_metrics

    # Test 4: GPU Linear Probe (Pretrained vs Random Untrained Control)
    print("[Test 4] Running GPU Linear Probe (pretrained vs random initialization)...")
    train_probe_ids = split_mgr.stratified_ids("train", limit=200, seed=seed)
    probe_fit_ldr = loader(dataset(train_probe_ids, "eval", "A", 16))
    test_splits = [name for name in split_mgr.splits if name.startswith("test_")]
    probe_sets = [
        dataset(split_mgr.get_ids(name), "counterfactual" if name == "test_3_interv" else "eval", "A", 16)
        for name in test_splits
    ]
    probe_eval_ldr = loader(ConcatDataset(probe_sets))

    fit_pre = _collect_predictions(model, probe_fit_ldr, device)
    eval_pre = _collect_predictions(model, probe_eval_ldr, device)
    fit_rnd = _collect_predictions(random_model, probe_fit_ldr, device)
    eval_rnd = _collect_predictions(random_model, probe_eval_ldr, device)

    Z_tr_pre = fit_pre["z_t"].reshape(-1, fit_pre["z_t"].shape[-1])
    Z_ev_pre = eval_pre["z_t"].reshape(-1, eval_pre["z_t"].shape[-1])
    Z_tr_rnd = fit_rnd["z_t"].reshape(-1, fit_rnd["z_t"].shape[-1])
    Z_ev_rnd = eval_rnd["z_t"].reshape(-1, eval_rnd["z_t"].shape[-1])
    R_tr = fit_pre["target_r"].reshape(-1, n_channels)
    R_ev = eval_pre["target_r"].reshape(-1, n_channels)

    probe_res = compare_pretrained_vs_random_probe(
        Z_tr_pre, Z_ev_pre, Z_tr_rnd, Z_ev_rnd, R_tr, R_ev, device=device
    )
    report["test_4_probe"] = {
        "train_sys_ids": train_probe_ids,
        "eval_systems_by_split": {name: len(ds) for name, ds in zip(test_splits, probe_sets)},
        "per_channel_r2_pretrain": {j: probe_res["pretrain_detail"]["per_channel_r2"].get(j)
                                    for j in range(n_channels)},
        "per_channel_r2_random": {j: probe_res["random_detail"]["per_channel_r2"].get(j)
                                  for j in range(n_channels)},
        "r2_pretrain": probe_res["r2_pretrain"],
        "r2_random": probe_res["r2_random"],
        "delta_r2": probe_res["delta_r2"],
        "gpp_r2_pretrain": probe_res["gpp_r2_pretrain"],
        "gpp_r2_random": probe_res["gpp_r2_random"],
        "gpp_delta_r2": probe_res["gpp_delta_r2"],
        "er_r2_pretrain": probe_res["er_r2_pretrain"],
        "er_r2_random": probe_res["er_r2_random"],
        "er_delta_r2": probe_res["er_delta_r2"],
        "nep_r2_pretrain": probe_res["nep_r2_pretrain"],
        "nep_r2_random": probe_res["nep_r2_random"],
        "nep_delta_r2": probe_res["nep_delta_r2"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 72)
    print("EcoVerse v0 Benchmark Scorecard Summary")
    print("=" * 72)
    print(f"Test 1 (Cross-Obs Alignment Cosine): {align['mean_cross_obs_cosine']:.4f}")
    for pkg, vals in pkg_results.items():
        print(f"  Package {pkg}: Unobs NRMSE={vals['unobs_overall_nrmse']:.4f}, Flux R2={vals['mean_flux_r2']:.4f}")
    for sname in ("test_2a_param", "test_2b_comp", "test_2c_shape"):
        s = report[sname]
        print(f"{sname:16s}: NRMSE={s['overall_nrmse']:.4f}, Flux R2={s['mean_flux_r2']:.4f}, GPP R2={s['gpp_r2']:.4f}")
    print(
        f"test_3_interv   : CF NRMSE={cf_metrics['cf_overall_nrmse']:.4f}, "
        f"Direction Acc={cf_metrics['cf_direction_accuracy']:.4f}"
    )
    p4 = report["test_4_probe"]
    print(
        f"Test 4 Probe    : R2_pretrain={p4['r2_pretrain']:.4f}, "
        f"R2_random={p4['r2_random']:.4f}, Delta_R2={p4['delta_r2']:+.4f} "
        f"(GPP Delta={p4['gpp_delta_r2']:+.4f}, ER Delta={p4['er_delta_r2']:+.4f})"
    )
    print(f"\nSaved full evaluation report to {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EcoVerse v0 GPU Benchmark suite")
    parser.add_argument("--data-dir", type=Path, default=Path("data/v0"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_report.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark_eval(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
