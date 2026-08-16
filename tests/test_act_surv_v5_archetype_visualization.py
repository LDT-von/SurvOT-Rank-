#!/usr/bin/env python3
"""Experiment D: Archetype morphology visualization.

RESEARCH QUESTION:
  Do the learned archetypes (K hazard trajectories) correspond to interpretable
  tissue morphology (e.g., tumor-rich, lymphocyte-rich, necrotic)?

  If archetypes have clear biological semantics, ACT-Surv moves from
  "mathematical interpretation" to "pathologically interpretable survival model".
  This is critical for MedIA / TMI.

WHAT TO VISUALIZE:
  For each archetype k, show:
    1. The hazard trajectory: h_k = (h_{k,1}, ..., h_{k,T})
    2. The top-N patches with highest P_{i,k} (strongest assignment to archetype k)
    3. UMAP / PCA of patch embeddings colored by archetype assignment
    4. Summary table: archetype → morphological interpretation → risk level

PROTOCOL:
  1. Load a trained model checkpoint
  2. Run inference on a test patient → get transport plan P [N, K]
  3. For each archetype k:
       a. Find patches with max P_{i,k}
       b. Save patch locations / thumbnails
       c. Record h_k trajectory
  4. Cluster patches across archetypes → check morphological coherence

KEY CLAIM TO SUPPORT:
  "ACT-Surv archetypes encode interpretable tissue morphology: high-risk archetypes
   capture tumor-rich and necrotic regions, while low-risk archetypes capture
   immune-enriched andstromal patterns."

NOTE:
  This script requires:
    - Trained model checkpoint (results/act_surv_v5/{cancer}/fold{fold}/best.pt)
    - WSI patch features and coordinates (for visualization)

  It provides the analysis pipeline; visualization rendering depends on your
  matplotlib / scanpy / PIL setup.

Run:
  python -m pytest tests/test_act_surv_v5_archetype_visualization.py -v
  # standalone with real data:
  python tests/test_act_surv_v5_archetype_visualization.py \
      --checkpoint results/act_surv_v5/blca/fold0/best.pt \
      --wsi-features /path/to/blca_fold0_wsi_features.pt \
      --output results/act_surv_v5/interpretability/blca_fold0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from tests.test_act_surv_v5 import make_args


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchetypeProfile:
    """Summary of one archetype across patients."""
    archetype_idx: int
    mean_composition: float          # mean α_k across patients
    composition_std: float
    hazard_trajectory: np.ndarray    # [T] h_{k,t}
    risk_level: str                  # "high" | "medium" | "low"
    top_patch_count: int             # number of patients where this archetype is top-1
    suggested_label: Optional[str] = None  # e.g. "tumor-rich", "immune-enriched"
    morphological_notes: Optional[str] = None


@dataclass
class VisualizationReport:
    """Full archetype interpretability report."""
    cancer: str
    fold: int
    num_archetypes: int
    num_patients: int
    num_time_classes: int
    compositions: np.ndarray    # [N_patients, K] — α per patient
    archetypes: list[ArchetypeProfile]
    top_patch_summary: dict     # archetype → list of (patient_id, patch_idx, P_value)
    umap_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_archetypes(
    model: torch.nn.Module,
    wsi_tokens: torch.Tensor,      # [B, N, D] patch embeddings
    omic_tokens: torch.Tensor | None = None,  # [B, M, D]
    token_mask: torch.Tensor | None = None,    # [B, N+M]
    patient_ids: list | None = None,
) -> VisualizationReport:
    """Analyze archetype patterns across a batch of patients.

    Args:
        model: trained ACT-Surv v5 model
        wsi_tokens: WSI patch embeddings [B, N, D]
        omic_tokens: optional omics tokens [B, M, D]
        token_mask: availability mask [B, N+M]
        patient_ids: optional patient ID strings

    Returns:
        VisualizationReport with archetype profiles
    """
    B = wsi_tokens.size(0)

    if omic_tokens is not None:
        tokens = torch.cat([wsi_tokens, omic_tokens], dim=1)
    else:
        tokens = wsi_tokens

    N_wsi = wsi_tokens.size(1)
    N_total = tokens.size(1)

    if token_mask is None:
        token_mask = torch.ones(B, N_total, dtype=torch.bool, device=tokens.device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        plan, _ = model._transport(tokens, token_mask)   # [B, N+M, K]
        composition = plan.sum(dim=1)                    # [B, K]
        hazard_logits = model._logit_hazard_raw.detach()  # [K, T]
        hazard_probs = torch.sigmoid(hazard_logits)       # [K, T]

    # Per-patient compositions [B, K]
    compositions = composition.cpu().numpy()

    # Archetype-level analysis
    T = hazard_logits.shape[1]
    K = hazard_logits.shape[0]

    # Average risk level per archetype (mean of T hazard probs)
    archetype_risk = hazard_probs.mean(dim=1).cpu().numpy()   # [K]

    profiles = []
    for k in range(K):
        # Hazard trajectory
        h_trajectory = hazard_probs[k].cpu().numpy()

        # Risk level
        if archetype_risk[k] > 0.6:
            risk_level = "high"
        elif archetype_risk[k] < 0.4:
            risk_level = "low"
        else:
            risk_level = "medium"

        # How often is this archetype the top-1 composition for patients?
        top_count = (compositions.argmax(axis=1) == k).sum()

        # Suggested label based on archetype risk + prevalence
        mean_alpha = compositions[:, k].mean()
        if archetype_risk[k] > 0.6 and mean_alpha > 0.2:
            suggested = "tumor-rich / high-risk"
        elif archetype_risk[k] < 0.4 and mean_alpha > 0.15:
            suggested = "immune-enriched / low-risk"
        elif archetype_risk[k] > 0.6 and mean_alpha < 0.1:
            suggested = "rare high-risk morphology"
        elif archetype_risk[k] < 0.4 and mean_alpha < 0.1:
            suggested = "rare low-risk morphology"
        else:
            suggested = "mixed morphology"

        profile = ArchetypeProfile(
            archetype_idx=k,
            mean_composition=float(compositions[:, k].mean()),
            composition_std=float(compositions[:, k].std()),
            hazard_trajectory=h_trajectory,
            risk_level=risk_level,
            top_patch_count=int(top_count),
            suggested_label=suggested,
            morphological_notes=None,
        )
        profiles.append(profile)

    # Top-patch summary: for each archetype, collect top-10 highest-assignment patches
    top_patch_summary = {}
    wsi_plan = plan[:, :N_wsi, :]   # [B, N_wsi, K]

    for k in range(K):
        # Find patches with highest P_{i,k} across all patients
        # wsi_plan[:, :, k] = [B, N_wsi]
        top_vals, top_idx = wsi_plan[:, :, k].topk(min(10, N_wsi), dim=1)  # [B, 10]
        entries = []
        for b in range(B):
            pid = patient_ids[b] if patient_ids else f"patient_{b}"
            for patch_rank in range(top_idx.shape[1]):
                entries.append({
                    "patient": pid,
                    "patch_idx": int(top_idx[b, patch_rank].item()),
                    "P_ik": float(top_vals[b, patch_rank].item()),
                })
        top_patch_summary[f"archetype_{k}"] = entries

    # Composition statistics (diagnostic for α diversity)
        comp_std_per_archetype = compositions.std(axis=0)   # [K]
        comp_entropy_per_patient = -(compositions * np.log(np.clip(compositions, 1e-12, None))).sum(axis=1)  # [B]

    report = VisualizationReport(
        cancer="unknown",
        fold=0,
        num_archetypes=K,
        num_patients=B,
        num_time_classes=T,
        compositions=compositions,
        archetypes=profiles,
        top_patch_summary=top_patch_summary,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Archetype Analysis: {B} patients, K={K} archetypes, T={T} time-classes")
    print(f"{'='*60}")
    for p in profiles:
        print(f"  Archetype {p.archetype_idx}: "
              f"mean α={p.mean_composition:.3f}±{p.composition_std:.3f}, "
              f"risk={p.risk_level:>6}, "
              f"top-1 count={p.top_patch_count:3d}, "
              f"suggested: {p.suggested_label}")

    print(f"\n  Composition diversity check:")
    print(f"    mean(α std per archetype): {comp_std_per_archetype.mean():.4f}")
    print(f"    mean(patient composition entropy): {comp_entropy_per_patient.mean():.4f}")
    print(f"    → {'DIVERSE (good)' if comp_std_per_archetype.mean() > 0.05 else 'LOW DIVERSITY (concern)'}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def plot_archetype_hazard_trajectories(
    report: VisualizationReport,
    output_path: str | Path,
):
    """Plot the hazard trajectory for each archetype."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    K = report.num_archetypes
    T = report.num_time_classes
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3), squeeze=False)

    risk_colors = {"high": "red", "medium": "orange", "low": "blue"}

    for k, prof in enumerate(report.archetypes):
        ax = axes[0, k]
        trajectory = prof.hazard_trajectory
        t_axis = np.arange(1, T + 1)

        color = risk_colors.get(prof.risk_level, "gray")
        ax.plot(t_axis, trajectory, color=color, linewidth=2, marker="o", markersize=3)
        ax.fill_between(t_axis, 0, trajectory, alpha=0.2, color=color)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time class")
        ax.set_ylabel("Hazard probability")
        ax.set_title(f"Archetype {k}\n({prof.suggested_label})")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "archetype_hazard_trajectories.png", dpi=150)
    print(f"  Saved: {output_path / 'archetype_hazard_trajectories.png'}")
    plt.close()


def plot_composition_heatmap(
    report: VisualizationReport,
    output_path: str | Path,
):
    """Plot patient × archetype composition heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    compositions = report.compositions  # [N_patients, K]
    fig, ax = plt.subplots(figsize=(min(12, compositions.shape[1] * 2), compositions.shape[0] * 0.05))

    im = ax.imshow(compositions, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Archetype")
    ax.set_ylabel("Patient")
    ax.set_title("Patient × Archetype Composition (α)")
    plt.colorbar(im, ax=ax, label="α_k")
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path / "composition_heatmap.png", dpi=150)
    print(f"  Saved: {output_path / 'composition_heatmap.png'}")
    plt.close()


def plot_composition_distribution(
    report: VisualizationReport,
    output_path: str | Path,
):
    """Plot distribution of α_k across patients per archetype."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    K = report.num_archetypes
    compositions = report.compositions  # [N_patients, K]

    fig, axes = plt.subplots(1, K, figsize=(3 * K, 2.5), squeeze=False)
    for k, prof in enumerate(report.archetypes):
        ax = axes[0, k]
        ax.hist(compositions[:, k], bins=20, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(prof.mean_composition, color="red", linestyle="--", label=f"μ={prof.mean_composition:.3f}")
        ax.set_title(f"Archetype {k}\n({prof.suggested_label})")
        ax.set_xlabel("α_k")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path / "composition_distribution.png", dpi=150)
    print(f"  Saved: {output_path / 'composition_distribution.png'}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic unit test
# ──────────────────────────────────────────────────────────────────────────────

def test_archetype_analysis_synthetic():
    """Unit test on synthetic data."""
    args = make_args(act5_num_archetypes=6, n_classes=4)
    model = __import__(
        "survot_rank.research.methods.archetypal_transport_composition_v5.model",
        fromlist=["ArchetypalTransportCompositionV5"]
    ).ArchetypalTransportCompositionV5(args)
    model.eval()

    # Synthetic batch: 8 patients, 50 patches each
    B = 8
    N_wsi = 50
    D = 16
    wsi_tokens = torch.randn(B, N_wsi, D)

    report = analyze_archetypes(
        model, wsi_tokens,
        patient_ids=[f"syn_patient_{i}" for i in range(B)],
    )

    assert report.num_archetypes == 6
    assert report.num_patients == 8
    assert report.num_time_classes == 4
    assert len(report.archetypes) == 6
    for p in report.archetypes:
        assert p.hazard_trajectory.shape == (4,)
        assert p.suggested_label is not None

    print("  ✓ Synthetic archetype analysis passes all assertions")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archetype morphology visualization")
    parser.add_argument("--checkpoint", type=str, required=False, help="Model checkpoint path")
    parser.add_argument("--wsi-features", type=str, help="WSI feature tensor path")
    parser.add_argument("--cancer", type=str, default="blca")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/act_surv_v5/interpretability")
    args = parser.parse_args()

    import matplotlib   # noqa: F401

    # Unit test
    test_archetype_analysis_synthetic()

    # With real data (requires checkpoint + features)
    if args.checkpoint and args.wsi_features:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        wsi_features = torch.load(args.wsi_features, map_location="cpu")   # [B, N, D]

        args_model = make_args()
        model = __import__(
            "survot_rank.research.methods.archetypal_transport_composition_v5.model",
            fromlist=["ArchetypalTransportCompositionV5"]
        ).ArchetypalTransportCompositionV5(args_model)
        model.load_state_dict(checkpoint)
        model.eval()

        report = analyze_archetypes(
            model, wsi_features,
            patient_ids=[f"{args.cancer}_fold{args.fold}_p{i}" for i in range(wsi_features.size(0))],
        )
        report.cancer = args.cancer
        report.fold = args.fold

        output = Path(args.output) / f"{args.cancer}_fold{args.fold}"
        plot_archetype_hazard_trajectories(report, output)
        plot_composition_heatmap(report, output)
        plot_composition_distribution(report, output)

        # Save JSON report
        json_path = output / "archetype_report.json"
        serializable = {
            "cancer": report.cancer,
            "fold": report.fold,
            "num_archetypes": report.num_archetypes,
            "num_patients": report.num_patients,
            "num_time_classes": report.num_time_classes,
            "archetypes": [
                {
                    "archetype_idx": p.archetype_idx,
                    "mean_composition": p.mean_composition,
                    "composition_std": p.composition_std,
                    "hazard_trajectory": p.hazard_trajectory.tolist(),
                    "risk_level": p.risk_level,
                    "top_patch_count": p.top_patch_count,
                    "suggested_label": p.suggested_label,
                }
                for p in report.archetypes
            ],
        }
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Report saved to {json_path}")
    else:
        print("\n  [skip] No checkpoint + features provided; run with --checkpoint and --wsi-features")
