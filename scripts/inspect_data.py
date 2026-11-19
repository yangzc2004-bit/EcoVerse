"""Generate health report and diagnostic figures for an EcoVerse v0 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from ecoverse.core.constants import (
    INTERVENTION_STEP,
    STATE_NAMES,
    compute_macro_fluxes,
    make_formal_time_grid,
)
from ecoverse.obs.normalize import Normalizer


def inspect_dataset(data_dir: Path, plot_dir: Path) -> None:
    meta_path = data_dir / "metadata.parquet"
    zarr_path = data_dir / "trajectories.zarr"
    norm_path = data_dir / "norm_stats.json"
    splits_path = data_dir / "splits.json"

    df = pd.read_parquet(meta_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    normalizer = Normalizer.load(norm_path)
    with open(splits_path, "r", encoding="utf-8") as fh:
        splits = json.load(fh)

    print("=" * 72)
    print(f"EcoVerse Dataset Inspection: {data_dir}")
    print(f"Total systems: {len(df)} | Zarr x_true shape: {root['x_true'].shape}")
    print("=" * 72)

    print("\n1. Split counts:")
    for k, ids in splits.items():
        print(f"  {k:16s}: {len(ids)}")

    print("\n2. Preset counts & regime distribution:")
    for preset, grp in df.groupby("preset"):
        reg_counts = grp["regime_tag"].value_counts().to_dict()
        eff_mat = np.array([json.loads(s) for s in grp["effective_state_mask_json"]], dtype=float)
        surv_rates = {STATE_NAMES[i]: round(float(eff_mat[:, i].mean()), 2) for i in range(len(STATE_NAMES))}
        print(f"  {preset:28s} (n={len(grp):4d}): regimes={reg_counts}")
        print(f"    effective state survival: {surv_rates}")

    print("\n3. Normalization summary (train-only):")
    for st in normalizer.stats:
        print(
            f"  var_id={st.var_id:2d} {st.name:5s} ({st.transform:6s}, drv={str(st.is_driver):5s}): "
            f"mean={st.mean:8.4f}, std={st.std:8.4f}"
        )

    # Check counterfactual consistency at t = 180 for non-bio_pulse interventions
    non_pulse = df[
        df["intervention_json"].apply(lambda s: json.loads(s)["type"] != "bio_pulse")
    ]
    if len(non_pulse) > 0:
        sample_ids = non_pulse["sys_id"].head(20).tolist()
        max_branch_diff = 0.0
        for sid in sample_ids:
            x_180 = np.asarray(root["x_true"][sid, INTERVENTION_STEP])
            xcf_0 = np.asarray(root["x_cf"][sid, 0])
            max_branch_diff = max(max_branch_diff, float(np.max(np.abs(x_180 - xcf_0))))
        print(f"\n4. Counterfactual fork t=180 continuity check (non-bio_pulse): max |x_cf[0] - x_true[1080]| = {max_branch_diff:.2e}")

    # Generate diagnostic plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    t_grid = make_formal_time_grid()
    t_cf = t_grid[INTERVENTION_STEP:]

    # Plot A: Typical River & Lake (8 states + GPP/ER/NEP)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row_idx, pname in enumerate(["river", "lake"]):
        sub = df[df["preset"] == pname]
        if len(sub) == 0:
            continue
        sid = int(sub.iloc[0]["sys_id"])
        x = np.asarray(root["x_true"][sid])
        r = np.asarray(root["r_true"][sid])
        gpp, er, nep = compute_macro_fluxes(r)
        mask = json.loads(sub.iloc[0]["effective_state_mask_json"])

        ax_s = axes[row_idx, 0]
        for i, sname in enumerate(STATE_NAMES):
            if mask[i]:
                ax_s.plot(t_grid, x[:, i], label=sname, linewidth=1.1)
        ax_s.set_title(f"{pname} (sys_id={sid}) - Active States")
        ax_s.set_xlabel("Time (days)")
        ax_s.set_ylabel("Concentration")
        ax_s.legend(ncol=4, fontsize=8)

        ax_f = axes[row_idx, 1]
        ax_f.plot(t_grid, gpp, label="GPP", linewidth=1.1)
        ax_f.plot(t_grid, er, label="ER", linewidth=1.1)
        ax_f.plot(t_grid, nep, label="NEP", linewidth=1.0, alpha=0.8)
        ax_f.axhline(0.0, color="k", linestyle="--", linewidth=0.7)
        ax_f.set_title(f"{pname} (sys_id={sid}) - Macro Fluxes (GPP / ER / NEP)")
        ax_f.set_xlabel("Time (days)")
        ax_f.set_ylabel("g C m^-3 d^-1")
        ax_f.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_dir / "typical_river_lake.png", dpi=150)
    plt.close(fig)

    # Plot B: Baseline vs Counterfactual across 5 presets + 2b variants
    all_presets = [
        "river", "lake", "terrestrial", "foodweb", "chemostat",
        "river_competition", "lake_cascade", "terrestrial_cascade", "chemostat_detritus_recycle",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for idx, pname in enumerate(all_presets):
        ax = axes[idx // 3, idx % 3]
        sub = df[df["preset"] == pname]
        if len(sub) == 0:
            ax.set_title(f"{pname} (missing)")
            continue
        sid = int(sub.iloc[0]["sys_id"])
        interv = json.loads(sub.iloc[0]["intervention_json"])
        eff_mask = json.loads(sub.iloc[0]["effective_state_mask_json"])
        if eff_mask[1]:
            plot_idx = 1
        elif eff_mask[2]:
            plot_idx = 2
        else:
            plot_idx = next((i for i, v in enumerate(eff_mask) if v), 0)
        plot_name = STATE_NAMES[plot_idx]

        x = np.asarray(root["x_true"][sid])
        xcf = np.asarray(root["x_cf"][sid])
        ax.plot(t_grid, x[:, plot_idx], label=f"{plot_name} base", color="tab:blue", linewidth=1.2)
        ax.plot(t_cf, xcf[:, plot_idx], label=f"{plot_name} cf", color="tab:red", linestyle="--", linewidth=1.2)
        ax.axvline(180.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{pname}\n({interv['type']})", fontsize=9)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(f"{plot_name} (g m^-3)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_dir / "counterfactual_presets.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved inspection plots to {plot_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated EcoVerse v0 dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/v0"))
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/inspect"))
    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.plot_dir)


if __name__ == "__main__":
    main()
