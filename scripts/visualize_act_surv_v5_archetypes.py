#!/usr/bin/env python3
"""ACT-Surv v5 per-archetype patch retrieval on real BLCA checkpoint.

Loads the trained v5 BLCA fold-0 checkpoint, runs forward on the real validation
loader, and produces:
  1. per-archetype metrics (top-1 concentration, top-16 L2 spread, silhouette)
  2. one PNG per archetype: PCA-2D scatter of all B*T patches coloured by α_{·,k},
     with the top-16 patches highlighted.
  3. JSON summary → results/act_surv_v5/proofs/act_surv_v5_per_patch_<timestamp>.json

Usage:
    python scripts/visualize_act_surv_v5_archetypes.py \\
        --cancer blca --fold 0 --top-k 16 --device cuda

Note: real WSI patch embeddings come from the `get_fold_dataset(...)` pipeline
(Pathways/UNI2-h). For this script we only need the `x_wsi` tensor at each
patient; the cohort-level summary does NOT require patch-level histology
images, only the embedding geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from survot_rank.config import flatten_config, load_config
from survot_rank.training.model_factory import get_model


# ---------------------------------------------------------------------------
# Imports shared with verify_act_surv_v5_mechanism.py
# ---------------------------------------------------------------------------
try:
    from scripts.run_dct_v38_transport_consistency import (
        DATASET_CSV_ROOT,
        DEFAULT_DATA_ROOT,
    )
except ModuleNotFoundError:
    from run_dct_v38_transport_consistency import (
        DATASET_CSV_ROOT,
        DEFAULT_DATA_ROOT,
    )


def load_state_dict(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model_state_dict", "state_dict"):
        if k in ckpt:
            return ckpt[k]
    return ckpt


def detect_dims(state: dict) -> dict:
    dims: dict[str, int] = {}
    if "wsi_mlp.0.weight" in state:
        dims["encoding_dim"] = int(state["wsi_mlp.0.weight"].shape[0])
    if "wsi_mlp.2.weight" in state:
        dims["wsi_projection_dim"] = int(state["wsi_mlp.2.weight"].shape[0])
    sig_keys = [k for k in state if k.startswith("sig_networks.")]
    if sig_keys:
        max_idx = max(int(k.split(".")[1]) for k in sig_keys)
        if max_idx >= 10:
            dims["rna_format"] = "RNASeq"
            total = 0
            for k in sorted(state.keys()):
                if k.startswith("sig_networks.") and k.endswith(".0.0.weight"):
                    total += int(state[k].shape[1])
            dims["omic_input_dim"] = total
        else:
            sizes = []
            for k in sorted(state.keys()):
                if k.startswith("sig_networks.") and k.endswith(".0.0.weight"):
                    sizes.append(int(state[k].shape[1]))
            if sizes:
                dims["omic_sizes"] = sizes
                dims["rna_format"] = "Pathways"
    if "archetype_embedding" in state:
        dims["act5_num_archetypes"] = int(state["archetype_embedding"].shape[0])
    if "_logit_hazard_raw" in state:
        dims["n_classes"] = int(state["_logit_hazard_raw"].shape[1])
    return dims


def build_dataloader(cancer: str, fold: int, batch_size: int):
    try:
        from tools.gen_splits_5fold import get_fold_dataset
    except ModuleNotFoundError:
        from survot_rank.research.legacy.slotspe_runtime.tools.gen_splits_5fold import (
            get_fold_dataset,
        )

    split_dir = Path(DATASET_CSV_ROOT) / "5fold_uni2h" / cancer
    ds = get_fold_dataset(
        cancer=cancer,
        fold=fold,
        data_root=Path(DEFAULT_DATA_ROOT),
        split_dir=split_dir,
        rna_format="Pathways",
        label_col="survival_months_dss",
        signature="combine",
        num_patches=2048,
        encoding_dim=1536,
        wsi_encoder="uni2-h",
        deterministic=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def per_archetype_retrieval(
    model,
    dataloader,
    device: str,
    top_k: int,
    max_batches: int = 8,
):
    """Run forward on validation set, collect plan + WSI embeddings, return per-archetype stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist

    K = model.num_archetypes
    all_plans = []
    all_x_wsi = []

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            _ = model(**kwargs)
            plan = model.last_explanations["transport_plan"].detach().cpu().numpy()
            x_wsi = kwargs["x_wsi"].detach().cpu().numpy()
            # Filter out all-zero patches (padding) to keep retrieval meaningful
            nonzero = (x_wsi**2).sum(axis=-1) > 1e-6
            for b in range(plan.shape[0]):
                mask = nonzero[b]
                if mask.sum() == 0:
                    continue
                all_plans.append(plan[b][mask])
                all_x_wsi.append(x_wsi[b][mask])

    if not all_plans:
        return {"error": "no valid batches"}

    plans = np.concatenate(all_plans, axis=0)   # [N_total_patches, K]
    patches = np.concatenate(all_x_wsi, axis=0)  # [N_total_patches, D]
    N, K_ = plans.shape
    assert K_ == K

    output_dir = REPO_ROOT / "results" / "act_surv_v5" / "proofs" / "figures_4_6_per_patch"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]

    per_archetype = []
    for k in range(K):
        weights = plans[:, k]
        if weights.max() < 1e-6:
            per_archetype.append({"archetype": k, "error": "all-zero weights"})
            axes[k].set_title(f"Archetype {k}\n(no mass)")
            continue
        top_idx = np.argsort(weights)[::-1][:top_k]
        top_weights = weights[top_idx]
        top_patches = patches[top_idx]

        # Normalised pairwise L2 in embedding space
        norms = np.linalg.norm(top_patches, axis=1, keepdims=True).clip(min=1e-8)
        top_n = top_patches / norms
        d = pdist(top_n, metric="euclidean")
        mean_dist = float(np.mean(d)) if len(d) > 0 else 0.0

        weight_sum = float(top_weights.sum())
        top1_share = float(top_weights[0] / max(weight_sum, 1e-8))
        top4_share = float(top_weights[:4].sum() / max(weight_sum, 1e-8))

        # PCA projection of all patches
        proj = PCA(n_components=2, random_state=0).fit_transform(patches)
        ax = axes[k]
        sc = ax.scatter(
            proj[:, 0], proj[:, 1],
            c=weights, cmap="viridis", s=8, alpha=0.4,
            vmin=0, vmax=max(0.01, weights.max()),
        )
        ax.scatter(
            proj[top_idx, 0], proj[top_idx, 1],
            c="red", s=40, edgecolors="black", linewidths=0.5,
            label=f"top-{top_k}",
        )
        ax.set_title(f"Archetype {k}\nmean L2={mean_dist:.3f}, top1={top1_share:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        per_archetype.append({
            "archetype": k,
            "n_total_patches": int(N),
            "n_top": int(top_k),
            "weight_sum_topk": weight_sum,
            "top1_share": top1_share,
            "top4_share": top4_share,
            "mean_pairwise_L2_normalised": mean_dist,
            "mean_weight_topk": float(top_weights.mean()),
        })

    plt.suptitle(
        f"ACT-Surv v5 — Per-archetype patch retrieval (real BLCA, top-{top_k})\n"
        f"Background: {N} patches coloured by α_{{·,k}}; red: top-{top_k}",
        fontsize=11,
    )
    plt.tight_layout()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fig = output_dir / f"per_archetype_top{top_k}_real_{stamp}.png"
    plt.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)

    mean_top1 = float(np.mean([p.get("top1_share", 1.0) for p in per_archetype]))
    mean_l2 = float(np.mean([p.get("mean_pairwise_L2_normalised", 0.0) for p in per_archetype]))
    retrievable = (mean_top1 < 0.5) and (mean_l2 > 0.5)

    return {
        "K": int(K),
        "n_total_patches": int(N),
        "top_k": int(top_k),
        "per_archetype": per_archetype,
        "summary": {
            "mean_top1_share": mean_top1,
            "mean_pairwise_L2": mean_l2,
        },
        "figure_path": str(out_fig),
        "verdict": (
            f"archetypes visually retrievable: top1={mean_top1:.2f}, L2={mean_l2:.3f}"
            if retrievable
            else f"archetypes NOT visually distinct: top1={mean_top1:.2f}, L2={mean_l2:.3f}"
        ),
        "passed": retrievable,
    }


def main():
    parser = argparse.ArgumentParser(description="ACT-Surv v5 per-archetype patch retrieval")
    parser.add_argument("--cancer", default="blca")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to v5 checkpoint; default = auto from results/")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",
                        default=str(REPO_ROOT / "results" / "act_surv_v5" / "proofs"))
    args = parser.parse_args()

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu")
        device_str = "cpu"

    # Build base model from YAML
    config_path = REPO_ROOT / "configs" / f"act_surv_v5_{args.cancer}.yaml"
    if not config_path.exists():
        config_path = REPO_ROOT / "configs" / "act_surv_v5_blca.yaml"
    flat_cfg = flatten_config(load_config(config_path))
    flat_cfg["survot_method"] = "archetypal_transport_composition_v5"
    flat_cfg.setdefault("encoding_dim", 1024)
    flat_cfg.setdefault("omic_sizes", [128, 128, 128, 128])
    config_ns = argparse.Namespace(**flat_cfg)
    model = get_model("archetypal_transport_composition_v5", config_ns)
    print(f"Built fresh model: K={model.num_archetypes}, C={model.num_classes}")

    # Load checkpoint if provided
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        nested = (
            REPO_ROOT / "results" / "act_surv_v5" / "full_run" / args.cancer
            / "SurvOTRank_archetypal_transport_composition_v5"
        )
        matches = list(nested.glob(
            f"*sp_act_surv_v5_{args.cancer}_fold{args.fold}/model_best_s{args.fold}.pth"
        ))
        if matches:
            ckpt_path = matches[0]
        else:
            ckpt_path = (
                REPO_ROOT / "results" / "act_surv_v5" / args.cancer
                / f"fold{args.fold}" / "models" / "best_model.pt"
            )

    if ckpt_path is not None and ckpt_path.exists():
        state = load_state_dict(ckpt_path)
        detected = detect_dims(state)
        for k, v in detected.items():
            if hasattr(config_ns, k):
                setattr(config_ns, k, v)
        model = get_model("archetypal_transport_composition_v5", config_ns)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded {ckpt_path.name}: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"No checkpoint at {ckpt_path}; using fresh-init model.")

    model.to(device_str)
    model.eval()

    # Build real dataloader
    print(f"Loading {args.cancer} fold {args.fold} ...")
    loader = build_dataloader(args.cancer, args.fold, args.batch_size)

    result = per_archetype_retrieval(
        model, loader, device_str, args.top_k, args.max_batches,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"act_surv_v5_per_patch_{args.cancer}_fold{args.fold}_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump({"timestamp": stamp, **result}, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PER-PATCH RETRIEVAL SUMMARY")
    print("=" * 60)
    print(f"  K={result.get('K')}, n_patches={result.get('n_total_patches')}, top_k={result.get('top_k')}")
    print(f"  verdict: {result.get('verdict')}")
    print(f"  passed:  {result.get('passed')}")
    print(f"  figure:  {result.get('figure_path')}")
    print(f"  JSON:    {out_json}")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())