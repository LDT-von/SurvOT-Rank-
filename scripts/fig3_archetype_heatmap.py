#!/usr/bin/env python3
"""Fig 3: Archetype Composition Heatmap + Clinical Tracks.

Generates a six-panel figure for the paper:
  Panel 1-6: Each archetype k (k=0..K-1) showing:
    - Top row: Archetype hazard trajectory h_{k,t} (logit over stages)
    - Middle row: Patient α composition scatter (UMAP/PCA coloured by α_{:,k})
    - Bottom row: Clinical track overlay (stage, grade as annotation bars)

Run:
    python scripts/fig3_archetype_heatmap.py --cancer blca --fold 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_act_surv_v5_mechanism import (
    build_dataloader,
    detect_dims_from_state,
    load_checkpoint_pretrained_state,
)
from scripts.clinical_enrichment_act_surv_v5 import (
    _find_clinical_csv,
    _load_clinical_labels,
    STAGE_ORDINAL_MAP,
    GRADE_MAP,
)


def _find_ckpt_path(cancer: str, fold: int) -> Path:
    candidates = [
        REPO_ROOT / f"results/act_surv_v5_1/{cancer}",
        REPO_ROOT / f"results/act_surv_v5/{cancer}",
    ]
    for parent in candidates:
        if not parent.exists():
            continue
        for d in parent.iterdir():
            if d.is_dir() and d.name.endswith(f"_fold{fold}"):
                ckpts = list(d.glob("model_best_s*.pth"))
                if ckpts:
                    return ckpts[0]
    raise FileNotFoundError(f"No checkpoint for {cancer} fold {fold}")


def build_model_and_forward(
    ckpt_path: Path,
    cancer: str,
    fold: int,
    device: str,
    max_batches: int = 32,
):
    """Load checkpoint and run forward on validation set, collecting alpha + hazards."""
    from types import SimpleNamespace

    state = load_checkpoint_pretrained_state(ckpt_path)
    dims = detect_dims_from_state(state)

    from survot_rank.research.methods.archetypal_transport_composition_v5.model import (
        ArchetypalTransportCompositionV5,
    )
    model_args = SimpleNamespace(
        omic_sizes=dims.get("omic_sizes", [16] * 10),
        encoding_dim=dims.get("encoding_dim", 1536),
        wsi_projection_dim=dims.get("wsi_projection_dim", 256),
        act5_num_archetypes=dims.get("act5_num_archetypes", 6),
        act5_epsilon=0.10,
        act5_hazard_scale=1.0,
        act5_warmup_epochs=0,
        act5_lambda_balance=0.05,
        act5_lambda_rank=0.00,
        act5_rank_margin=0.02,
        act5_rank_temperature=0.50,
        act5_rank_max_pairs=4096,
        rna_format=dims.get("rna_format", "Pathways"),
        omics_input_dim=dims.get("omic_input_dim", 0),
        n_classes=dims.get("n_classes", 4),
    )
    model = ArchetypalTransportCompositionV5(model_args).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    try:
        val_loader = build_dataloader(cancer, fold, batch_size=4, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build dataloader: {e}")

    alpha_rows = []
    case_ids = []

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            try:
                model(**kwargs)
            except Exception:
                continue

            comp = model.last_explanations["composition"].cpu().numpy()  # [B, K]
            cids = batch.get("case_id", [f"p_{bi}_{i}" for i in range(comp.shape[0])])
            for i, cid in enumerate(cids):
                row = {"case_id": str(cid)}
                for k in range(comp.shape[1]):
                    row[f"alpha_{k}"] = float(comp[i, k])
                alpha_rows.append(row)
                case_ids.append(str(cid))

    hazards = model.last_explanations["archetype_hazard_logits"].cpu().numpy()  # [K, C]
    hazard_vals = model.last_explanations["archetype_hazards"].cpu().numpy()     # [K, C]

    return {
        "model": model,
        "alpha_df": alpha_rows,
        "case_ids": case_ids,
        "hazard_logits": hazards,
        "hazard_values": hazard_vals,
        "K": model.num_archetypes,
        "num_classes": model.num_classes,
    }


def make_fig3(
    cancer: str,
    fold: int,
    ckpt_path: Path,
    device: str,
    max_batches: int = 32,
    output_dir: Path | None = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print(f"\n[Fig 3] {cancer} fold {fold} — loading checkpoint: {ckpt_path.name}")
    data = build_model_and_forward(ckpt_path, cancer, fold, device, max_batches)

    alpha_df_rows = data["alpha_df"]
    if not alpha_df_rows:
        raise RuntimeError("No alpha rows collected")
    import pandas as pd
    alpha_df = pd.DataFrame(alpha_df_rows)

    # Merge clinical labels
    try:
        clinical_df = _load_clinical_labels(cancer)
        merged = alpha_df.merge(clinical_df, on="case_id", how="left")
    except Exception as e:
        print(f"  WARNING: Could not load clinical labels: {e}")
        merged = alpha_df.copy()
        merged["stage_ordinal"] = np.nan
        merged["grade_ordinal"] = np.nan

    K = data["K"]
    C = data["num_classes"]
    hazard_logits = data["hazard_logits"]   # [K, C]
    hazard_vals = data["hazard_values"]       # [K, C]

    # UMAP for visualization (use PCA if UMAP fails)
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, min_dist=0.3, n_neighbors=15)
        alpha_matrix = np.stack([alpha_df[f"alpha_{k}"].values for k in range(K)], axis=1)
        embed_2d = reducer.fit_transform(alpha_matrix)
    except ImportError:
        pca = PCA(n_components=2, random_state=42)
        alpha_matrix = np.stack([alpha_df[f"alpha_{k}"].values for k in range(K)], axis=1)
        embed_2d = pca.fit_transform(alpha_matrix)

    # Stage colour map
    stage_vals = merged["stage_ordinal"].fillna(-1).values
    stage_colours = np.array([
        "#4CAF50" if s < 1.5 else "#FFC107" if s < 3 else "#F44336"
        for s in stage_vals
    ])
    stage_labels = merged["stage_ordinal"].fillna(-1).values

    # Grade colour bar
    grade_vals = merged["grade_ordinal"].fillna(-1).values

    if output_dir is None:
        output_dir = REPO_ROOT / "results" / "act_surv_v5" / "fig3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Six-panel figure: 3 rows × 2 cols ─────────────────────────
    # Rows: (1) Hazard trajectories, (2) UMAP composition, (3) Clinical tracks
    # Cols: 2 archetypes at a time (side by side for comparison)
    fig, axes = plt.subplots(3, K, figsize=(4 * K, 9), squeeze=False)
    stage_ticks = ["Stage I", "II", "III", "IV"]
    time_axis = np.arange(C)

    for k in range(K):
        ax_haz = axes[0, k]
        ax_umap = axes[1, k]
        ax_track = axes[2, k]

        # ── Row 1: Archetype hazard trajectory ──────────────────────
        colour = plt.cm.tab10(k % 10)
        ax_haz.plot(time_axis, hazard_logits[k], "o-", color=colour, linewidth=2, markersize=6)
        ax_haz.fill_between(time_axis, hazard_logits[k], alpha=0.15, color=colour)
        ax_haz.set_title(f"Archetype {k}\nHazard Trajectory", fontsize=11)
        ax_haz.set_xlabel("Stage"); ax_haz.set_ylabel("Hazard Logit")
        ax_haz.set_xticks(time_axis)
        ax_haz.set_xticklabels(stage_ticks[:C])
        ax_haz.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax_haz.grid(True, alpha=0.3)

        # ── Row 2: UMAP coloured by α_{:,k} ────────────────────────
        alpha_k = alpha_df[f"alpha_{k}"].values
        sc = ax_umap.scatter(
            embed_2d[:, 0], embed_2d[:, 1],
            c=alpha_k, cmap="YlOrRd", s=15, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax_umap, fraction=0.046, pad=0.04, label=f"α_{k}")
        ax_umap.set_title(f"Patients coloured by α_{{{k},:}}\n(UMAP)", fontsize=10)
        ax_umap.set_xlabel("UMAP-1"); ax_umap.set_ylabel("UMAP-2")

        # ── Row 3: Clinical track (stage + grade bar per patient) ──
        # Show top-30 patients sorted by α_{:,k} for clarity
        sort_idx = np.argsort(alpha_k)[::-1][:min(30, len(alpha_k))]
        track_data = []
        for idx in sort_idx:
            s = stage_vals[idx]
            g = grade_vals[idx]
            if s < 0 and g < 0:
                colour = "#CCCCCC"
            elif g >= 0.5:
                colour = "#E91E63" if s >= 3 else "#FF9800"  # High grade, high/low stage
            else:
                colour = "#2196F3" if s >= 3 else "#4CAF50"  # Low grade, high/low stage
            track_data.append({"stage": s, "grade": g, "colour": colour})

        colours = [t["colour"] for t in track_data]
        ax_track.barh(range(len(colours)), [1.0]*len(colours), color=colours, height=0.8)
        ax_track.set_yticks([]); ax_track.set_xticks([])
        ax_track.set_title(f"Clinical Track (top-α_{k})\n■ HighGrade+HighStage ■ LowGrade+HighStage\n"
                            f"■ HighGrade+LowStage ■ LowGrade+LowStage", fontsize=8)

    # Legend for clinical track
    legend_patches = [
        mpatches.Patch(color="#E91E63", label="HighGrade + HighStage"),
        mpatches.Patch(color="#FF9800", label="HighGrade + LowStage"),
        mpatches.Patch(color="#2196F3", label="LowGrade + HighStage"),
        mpatches.Patch(color="#4CAF50", label="LowGrade + LowStage"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"Fig 3 — Archetype Pathological Semantics: {cancer.upper()} fold {fold}\n"
        f"Top: Hazard trajectories | Middle: Patient composition (UMAP) | Bottom: Clinical tracks",
        fontsize=12, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fig = output_dir / f"fig3_archetype_heatmap_{cancer}_fold{fold}_{stamp}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Compute archetype statistics
    stats = {}
    for k in range(K):
        alpha_k = alpha_df[f"alpha_{k}"].values
        top5_mean = float(np.mean(np.sort(alpha_k)[::-1][:5]))
        stats[f"archetype_{k}"] = {
            "mean_alpha": float(alpha_k.mean()),
            "std_alpha": float(alpha_k.std()),
            "max_alpha": float(alpha_k.max()),
            "top5_mean": top5_mean,
            "hazard_logit_range": [
                float(hazard_logits[k].min()),
                float(hazard_logits[k].max()),
            ],
        }

    return {
        "figure_path": str(out_fig),
        "cancer": cancer,
        "fold": fold,
        "K": K,
        "num_patients": int(len(alpha_df)),
        "archetype_stats": stats,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Fig 3 — Archetype Composition Heatmap")
    p.add_argument("--cancer", default="blca")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",
                   default=str(REPO_ROOT / "results" / "act_surv_v5" / "fig3"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("WARNING: falling back to cpu")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = _find_ckpt_path(args.cancer, args.fold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = make_fig3(args.cancer, args.fold, ckpt, device, args.max_batches, output_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"fig3_metadata_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFigure saved: {result['figure_path']}")
    print(f"Metadata:     {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
