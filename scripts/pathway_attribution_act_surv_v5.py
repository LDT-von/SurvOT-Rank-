#!/usr/bin/env python3
"""Pathway Program Attribution per Archetype (Section 5.3 / Fig 3).

For each archetype k:
  1. Identify top pathway tokens with highest P_{i,k} (transport mass to archetype k)
  2. For each high-mass pathway, trace back to the gene-level input
  3. Compute per-gene contribution to archetype k via pathway × token mass
  4. Rank genes by contribution; identify top gene programs
  5. Annotate with MSigDB pathway names (from combine_signatures.csv)

Run:
    python scripts/pathway_attribution_act_surv_v5.py --cancer blca --fold 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_act_surv_v5_mechanism import (
    build_dataloader,
    detect_dims_from_state,
    load_checkpoint_pretrained_state,
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


def _load_signatures(signature: str = "combine") -> pd.DataFrame | None:
    """Load pathway gene set definitions from signatures CSV.

    Expected columns: gene names (rows), pathway names (cols).
    Returns DataFrame [gene × pathway] with 0/1 entries.
    """
    roots = [
        REPO_ROOT / "survot_rank/research/legacy/slotspe_runtime/dataset_csv",
        REPO_ROOT / "dataset_csv",
    ]
    for root in roots:
        sig_path = root / "signatures" / f"{signature}_signatures.csv"
        if sig_path.exists():
            df = pd.read_csv(sig_path, index_col=0)
            print(f"[signatures] Loaded {df.shape[0]} genes × {df.shape[1]} pathways from {sig_path.name}")
            return df
    print("[signatures] WARNING: signatures CSV not found — gene-level attribution skipped")
    return None


def pathway_attribution(
    ckpt_path: Path,
    cancer: str,
    fold: int,
    device: str,
    max_batches: int = 32,
    mass_threshold: float = 0.05,
) -> dict:
    """Run pathway attribution for one fold."""
    print(f"\n{'='*60}")
    print(f"Pathway Attribution: {cancer} fold {fold}")
    print(f"{'='*60}")

    from types import SimpleNamespace
    state = load_checkpoint_pretrained_state(ckpt_path)
    dims = detect_dims_from_state(state)
    print(f"[ckpt] Dims: {dims}")

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
        return {"experiment": "pathway_attribution", "skipped": True, "reason": str(e)}

    # Collect (transport_plan, pathway_gene_expression) per batch
    # The model stores: plan [B, tokens, K], with omic tokens starting at index num_wsi
    all_omic_mass = []   # [N_samples, num_pathways, K]
    all_omic_input = []  # [N_samples, num_pathways, num_genes_per_pathway]

    # We need the raw omic input per pathway per patient
    # Get pathway sizes from model
    omic_sizes = model.omic_sizes
    num_pathways = len(omic_sizes)
    K = model.num_archetypes

    print(f"[model] K={K}, num_pathways={num_pathways}, omic_sizes={omic_sizes[:5]}...")

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            try:
                model(**kwargs)
            except Exception as e:
                print(f"  WARN batch {bi}: {e}")
                continue

            plan = model.last_explanations["transport_plan"].cpu()  # [B, tokens, K]
            num_wsi = model.last_explanations.get("num_wsi_tokens", 0)
            B = plan.size(0)

            # Omics mass: plan[:, num_wsi:, :] → [B, num_pathways, K]
            omic_plan = plan[:, num_wsi:, :].numpy()  # [B, P, K]
            all_omic_mass.append(omic_plan)

            # Collect raw omic inputs
            for i in range(B):
                pathway_exprs = []
                for p_idx in range(num_pathways):
                    key = f"x_omic{p_idx+1}"
                    if key in kwargs:
                        val = kwargs[key][i].cpu().float()
                        # clip extreme values for numerical stability
                        val = torch.nan_to_num(val, nan=0.0, posinf=10.0, neginf=-10.0)
                        pathway_exprs.append(val.numpy())
                    else:
                        pathway_exprs.append(np.zeros(omic_sizes[p_idx]))
                all_omic_input.append(np.stack(pathway_exprs, axis=0))  # [P, G_p]

    if not all_omic_mass:
        return {"experiment": "pathway_attribution", "skipped": True,
                "reason": "No batches processed"}

    omic_mass = np.concatenate(all_omic_mass, axis=0)   # [N, P, K]
    omic_input = np.concatenate(all_omic_input, axis=0)  # [N, P, G_p]
    N, P, K_ = omic_mass.shape
    assert K_ == K
    print(f"[data] {N} patients × {P} pathways × K={K} archetypes")

    # ── Per-archetype pathway attribution ─────────────────────────
    # Mean mass per pathway per archetype across patients
    mean_mass = omic_mass.mean(axis=0)  # [P, K]
    pathway_mass_ranked = {}

    # Load signature definitions
    sig_df = _load_signatures("combine")

    results_per_archetype = {}
    for k in range(K):
        pathway_mass_k = mean_mass[:, k]  # [P]
        top_pathway_indices = np.argsort(pathway_mass_k)[::-1]  # descending

        # Aggregate gene-level contribution for top pathways
        # Contribution = pathway_expression(gene) × mass_to_archetype_k
        gene_contributions = {}

        for p_idx in top_pathway_indices[:10]:  # top 10 pathways
            mass_k_p = pathway_mass_k[p_idx]
            if mass_k_p < mass_threshold:
                continue

            # Average gene expression in this pathway across patients
            avg_expr_p = omic_input[:, p_idx, :].mean(axis=0)  # [G_p]

            # Gene contribution: expr(g) × mass(p→k)
            contrib_p = avg_expr_p * mass_k_p
            gene_indices = np.argsort(contrib_p)[::-1][:20]  # top 20 genes

            g_per_p = omic_sizes[p_idx]
            genes_in_pathway = np.arange(g_per_p)

            for rank_idx, g_local in enumerate(gene_indices[:10]):
                if contrib_p[g_local] < 1e-6:
                    break
                gene_contributions[f"pathway_{p_idx}_gene_{g_local}"] = {
                    "gene_idx": int(g_local),
                    "pathway_idx": int(p_idx),
                    "contribution": float(contrib_p[g_local]),
                    "avg_expression": float(avg_expr_p[g_local]),
                    "pathway_mass_to_archetype": float(mass_k_p),
                    "rank_in_pathway": rank_idx + 1,
                }

        pathway_mass_ranked[f"archetype_{k}"] = {
            "mean_mass": float(pathway_mass_k.sum()),  # total mass to this archetype
            "top_pathways": [
                {
                    "pathway_idx": int(p_idx),
                    "mass": float(pathway_mass_k[p_idx]),
                    "mass_fraction": float(pathway_mass_k[p_idx] / max(pathway_mass_k.sum(), 1e-8)),
                    "pathway_name": sig_df.columns[p_idx] if sig_df is not None else f"Pathway_{p_idx}",
                }
                for p_idx in top_pathway_indices[:10]
                if pathway_mass_k[p_idx] >= mass_threshold
            ],
            "top_genes": dict(
                sorted(
                    gene_contributions.items(),
                    key=lambda x: x[1]["contribution"],
                    reverse=True
                )[:20]
            ),
        }

    # ── Summary: which archetypes are omics-dominated vs WSI-dominated ──
    wsi_contrib_sum = []
    omic_contrib_sum = []

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
            wsi_c = model.last_explanations.get("wsi_token_contribution", torch.zeros(1, K))
            omic_c = model.last_explanations.get("omic_token_contribution", torch.zeros(1, K))
            wsi_contrib_sum.append(wsi_c.cpu())
            omic_contrib_sum.append(omic_c.cpu())

    wsi_all = torch.cat(wsi_contrib_sum, dim=0)   # [N, K]
    omic_all = torch.cat(omic_contrib_sum, dim=0)  # [N, K]
    total_all = wsi_all + omic_all
    total_all = total_all.clamp(min=1e-8)
    omic_fraction = (omic_all / total_all).mean(dim=0).numpy()  # [K]
    wsi_fraction = 1 - omic_fraction

    print(f"\n[Modality Split per Archetype]")
    for k in range(K):
        print(f"  A{k}: WSI={wsi_fraction[k]:.3f}  Omics={omic_fraction[k]:.3f}")

    return {
        "experiment": "pathway_attribution",
        "cancer": cancer,
        "fold": fold,
        "K": K,
        "num_pathways": P,
        "n_patients": N,
        "mass_threshold": mass_threshold,
        "omic_sizes_sample": omic_sizes[:5],
        "omic_fraction_per_archetype": {
            f"A{k}": {"omic": float(omic_fraction[k]), "wsi": float(wsi_fraction[k])}
            for k in range(K)
        },
        "archetype_pathway_top": {
            k: pathway_mass_ranked[f"archetype_{k}"]
            for k in range(K)
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Pathway attribution per archetype")
    p.add_argument("--cancer", default="blca")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--mass-threshold", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",
                   default=str(REPO_ROOT / "results" / "act_surv_v5" / "pathway_attr"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = _find_ckpt_path(args.cancer, args.fold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pathway_attribution(ckpt, args.cancer, args.fold, device,
                                 args.max_batches, args.mass_threshold)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"pathway_attribution_{args.cancer}_fold{args.fold}_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary
    print(f"\n[Summary]")
    for k in range(result.get("K", 0)):
        key = f"A{k}"
        ofrac = result["omic_fraction_per_archetype"][key]["omic"]
        wfrac = result["omic_fraction_per_archetype"][key]["wsi"]
        top_p = result["archetype_pathway_top"][k]["top_pathways"][:3]
        top_names = [p["pathway_name"] for p in top_p] if top_p else []
        print(f"  {key}: Omics={ofrac:.2f}, WSI={wfrac:.2f}")
        print(f"    Top pathways: {top_names}")

    print(f"\nSaved: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
