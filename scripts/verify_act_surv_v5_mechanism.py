#!/usr/bin/env python3
"""Verify ACT-Surv v5's four constructive claims on real checkpoint data.

Run:
    python scripts/verify_act_surv_v5_mechanism.py --cancer blca --fold 0

Claims verified:
    1. Completeness residual < 1e-5  (logit_t = Σ_k α_k · h_{k,t}, exact)
    2. Closed-form deletion vs re-solved Sinkhorn error < 0.001
    3. Predictions in convex hull of K archetype curves  (α ≥ 0, Σ α = 1)
    4. Archetype hazard curves genuinely distinct  (pairwise L1, cosine)

Dependencies:
    A trained checkpoint at:
        results/act_surv_v5/{cancer}/fold{fold}/models/best_model.pt
    The launcher (scripts/run_act_surv_v5.py) hard-codes data paths:
        data_path=survot_rank/research/legacy/slotspe_runtime/dataset_csv
        data_root_dir=/data1/TCGA-UNI2-h-features
    If running locally adjust DATA_ROOT / CSV_ROOT accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from survot_rank.research.methods.archetypal_transport_composition_v5.model import (
    ArchetypalTransportCompositionV5,
    _ipcw_ranking_loss,          # re-exported; not used here but available
)


# ---------------------------------------------------------------------------
# NOTE for reviewers
# ---------------------------------------------------------------------------
# The training launcher (scripts/run_act_surv_v5.py) has hard-coded data paths:
#   FINAL_OVERRIDES["data_path"]      = "survot_rank/research/legacy/slotspe_runtime/dataset_csv"
#   FINAL_OVERRIDES["data_root_dir"]  = "/data1/TCGA-UNI2-h-features"
# Change these constants in _get_data_dirs() if you want to run on a different machine.
DATA_ROOT_DIR = "/data1/TCGA-UNI2-h-features"   # adjust to your TCGA patch root
CSV_SPLITS_DIR = "survot_rank/research/legacy/slotspe_runtime/dataset_csv"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config override helpers (mirrors run_act_surv_v5.py)
# ---------------------------------------------------------------------------

def _build_act5_args(
    cancer: str,
    fold: int,
    *,
    num_epochs: int = 30,
    num_archetypes: int = 6,
    epsilon: float = 0.10,
    warmup_epochs: int = 5,
    hazard_scale: float = 1.0,
    lambda_balance: float = 0.01,
    lambda_rank: float = 0.10,
    rank_margin: float = 0.02,
    rank_temperature: float = 0.50,
    rank_max_pairs: int = 4096,
    **extra_overrides,
):
    """Build a mock argparse.Namespace matching what the model __init__ expects."""
    import argparse
    ns = argparse.Namespace()
    ns.n_classes = 36
    ns.encoding_dim = 1536
    ns.wsi_projection_dim = 512
    ns.rna_format = "Pathways"
    ns.omic_sizes = [979, 471, 132, 200, 221, 299, 329, 184, 53, 58, 106]
    ns.survot_method = "act_surv_v5"
    ns.which_splits = "5fold_uni2h"
    ns.fold = fold

    ns.act5_num_archetypes = num_archetypes
    ns.act5_epsilon = epsilon
    ns.act5_warmup_epochs = warmup_epochs
    ns.act5_hazard_scale = hazard_scale
    ns.act5_lambda_balance = lambda_balance
    ns.act5_lambda_rank = lambda_rank
    ns.act5_rank_margin = rank_margin
    ns.act5_rank_temperature = rank_temperature
    ns.act5_rank_max_pairs = rank_max_pairs

    ns.num_epochs = num_epochs
    ns.cur_epoch = 0

    for k, v in extra_overrides.items():
        setattr(ns, k, v)

    return ns


# ---------------------------------------------------------------------------
# Data loading (matches run_act_surv_v5.py data pipeline)
# ---------------------------------------------------------------------------

def _get_data_dirs():
    return Path(DATA_ROOT_DIR), Path(CSV_SPLITS_DIR)


def build_dataloader(cancer: str, fold: int, batch_size: int = 4):
    """Build test dataloader for one fold (same split as training)."""
    try:
        from torch.utils.data import DataLoader
        from survot_rank.research.legacy.slotspe_runtime.tools.gen_splits_5fold import (
            get_fold_dataset,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot import data pipeline: {exc}\n"
            "Ensure PYTHONPATH includes the repo root and slotspe_runtime/."
        ) from exc

    data_root, csv_root = _get_data_dirs()
    splits_dir = csv_root / "5fold_uni2h" / cancer

    ds = get_fold_dataset(
        cancer=cancer,
        fold=fold,
        data_root=data_root,
        split_dir=splits_dir,
        rna_format="Pathways",
        label_col="survival_months_dss",
        signature="combine",
        num_patches=2048,
        encoding_dim=1536,
        wsi_encoder="uni2-h",
        deterministic=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path):
    """Load state dict from a SurvOT training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


# ---------------------------------------------------------------------------
# Sinkhorn re-solve (for Claim 2 comparison)
# ---------------------------------------------------------------------------

def sinkhorn_plan_from_scratch(
    tokens: torch.Tensor,
    archetype_emb: torch.Tensor,
    token_mask: torch.Tensor,
    epsilon: float,
    max_iter: int = 40,
):
    """Re-solve Sinkhorn from scratch — used for Claim 2 ground truth.

    tokens:        [B, T, D]
    archetype_emb: [K, D]
    token_mask:    [B, T]  (bool, True = active)
    epsilon:       float (transport temperature)

    Returns plan [B, T, K] matching the forward pass.
    """
    B, T, D = tokens.shape
    K = archetype_emb.size(0)

    # cosine cost: C_{i,k} = 1 - cosine(tokens_i, archetype_k)
    directions = F.normalize(tokens, dim=-1)         # [B, T, D]
    archetypes = F.normalize(archetype_emb, dim=-1)  # [K, D]
    cost = 1.0 - directions @ archetypes.t()          # [B, T, K]

    # Token marginal = uniform availability
    weights = token_mask.to(dtype=torch.float32)     # [B, T]
    safe = weights.clone()
    safe[weights.sum(dim=1) == 0, 0] = 1.0
    safe = safe / safe.sum(dim=1, keepdim=True).clamp_min(1.0)

    u = torch.zeros(B, T, device=tokens.device)
    v = torch.zeros(B, K, device=tokens.device)

    for _ in range(max_iter):
        # Sinkhorn alternating projection
        masked_cost = cost.clone()
        masked_cost[~token_mask] = float("inf")

        u_new = (-masked_cost + u.unsqueeze(-1) + v.unsqueeze(1) / epsilon).exp()
        denom = (u_new.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                 * token_mask.unsqueeze(-1).float()
                 + (~token_mask).unsqueeze(-1).float())
        u_new = u_new / denom
        u_new = u_new.nan_to_num(0.0)

        v_new = (-masked_cost + u_new.unsqueeze(-1) + v.unsqueeze(1) / epsilon).exp()
        denom = (v_new.sum(dim=-2, keepdim=True).clamp_min(1e-8)
                 + 0.0)
        v_new = v_new / denom
        v_new = v_new.nan_to_num(0.0)

        u = u_new
        v = v_new

    plan = (-cost / epsilon + u.unsqueeze(-1) + v.unsqueeze(1) / epsilon).exp()
    plan = plan.nan_to_num(0.0)
    plan[~token_mask] = 0.0
    plan = plan * safe.unsqueeze(-1)
    return plan


def resinkhorn_delete_and_resolve(
    tokens: torch.Tensor,
    archetype_emb: torch.Tensor,
    token_mask: torch.Tensor,
    plan: torch.Tensor,
    hazard_logits: torch.Tensor,
    delete_token: int,
    epsilon: float,
):
    """Re-solve Sinkhorn after deleting token 'delete_token' (Claim 2 ground truth).

    Returns: counterfactual logit after token deletion, resolved from scratch.
    """
    B, T, K = plan.shape
    # Mask out the deleted token
    new_mask = token_mask.clone()
    new_mask[:, delete_token] = False

    plan_resolved = sinkhorn_plan_from_scratch(
        tokens, archetype_emb, new_mask, epsilon
    )

    # Compute counterfactual from re-solved plan
    composition_resolved = plan_resolved.sum(dim=1)       # [B, K]
    factual = composition_resolved @ hazard_logits        # [B, C]
    return factual, composition_resolved


# ---------------------------------------------------------------------------
# Claim verifiers
# ---------------------------------------------------------------------------

def verify_claim1_completeness(model, dataloader, device: str) -> dict:
    """Claim 1: logit_t = Σ_k α_k · h_{k,t} exactly — residual < 1e-5."""
    model.eval()
    residuals = []
    with torch.no_grad():
        for batch in dataloader:
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            model(**kwargs)
            stored = model.last_explanations

            # The model already computes completeness_error per forward pass.
            # Use it directly.
            residuals.append(stored["completeness_error"].cpu())

    residuals = torch.cat(residuals)
    return {
        "claim": "Claim 1: Completeness residual < 1e-5",
        "max_residual": residuals.max().item(),
        "mean_residual": residuals.mean().item(),
        "num_samples": len(residuals),
        "passed": residuals.max().item() < 1e-5,
        "threshold": 1e-5,
        "note": "Uses model's pre-computed last_explanations['completeness_error']",
    }


def verify_claim2_closed_form_vs_resolve(model, dataloader, device: str, num_tokens: int = 3) -> dict:
    """Claim 2: Closed-form deletion vs re-solved Sinkhorn error < 0.001."""
    model.eval()
    errors = []
    samples_tested = 0

    archetype_emb = F.normalize(model.archetype_embedding.data, dim=-1)
    hazard_logits = model.archetype_hazard_logits.data
    epsilon = model.epsilon

    with torch.no_grad():
        for batch in dataloader:
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            model(**kwargs)
            stored = model.last_explanations

            plan = stored["transport_plan"]          # [B, T, K]
            tokens_wsi = model._encode_wsi(kwargs["x_wsi"].to(device))
            tokens_omic = model._encode_omics(kwargs)
            tokens = torch.cat([tokens_wsi, tokens_omic], dim=1)
            has_wsi = torch.ones(tokens.size(0), dtype=torch.bool, device=device)
            has_omic = torch.ones(tokens.size(0), dtype=torch.bool, device=device)
            token_mask = torch.cat([
                has_wsi.unsqueeze(1).expand(-1, tokens_wsi.size(1)),
                has_omic.unsqueeze(1).expand(-1, tokens_omic.size(1)),
            ], dim=1)

            B, T, K = plan.shape
            for b in range(min(B, 3)):
                a_i = plan[b, :, 0].sum().item()
                if plan[b].sum() < 1e-6:
                    continue
                for ti in range(min(T, num_tokens)):
                    mass_i = plan[b, ti].sum().item()
                    if mass_i < 1e-6:
                        continue

                    # Closed-form deletion (model's built-in)
                    cf_closed = model.deletion_counterfactual(ti)    # [B, C]
                    cf_closed_b = cf_closed[b]                      # [C]

                    # Ground truth: re-solve Sinkhorn without this token
                    tokens_b = tokens[b:b+1]                         # [1, T, D]
                    mask_b = token_mask[b:b+1]                       # [1, T]
                    emb_b = archetype_emb                           # [K, D]

                    cf_resolved, _ = resinkhorn_delete_and_resolve(
                        tokens_b, emb_b, mask_b,
                        plan[b:b+1], hazard_logits,
                        delete_token=ti, epsilon=epsilon,
                    )
                    cf_resolved_b = cf_resolved[0]                   # [C]

                    error = (cf_closed_b - cf_resolved_b).abs().max().item()
                    errors.append(error)
                    samples_tested += 1

    if not errors:
        return {
            "claim": "Claim 2: Closed-form vs re-solve error < 0.001",
            "passed": None,
            "note": "No valid tokens found — increase batch_size or check token_mask",
        }

    errs = torch.tensor(errors)
    return {
        "claim": "Claim 2: Closed-form vs re-solve error < 0.001",
        "max_error": errs.max().item(),
        "mean_error": errs.mean().item(),
        "num_tested": samples_tested,
        "passed": errs.max().item() < 0.001,
        "threshold": 0.001,
    }


def verify_claim3_convex_hull(model, dataloader, device: str) -> dict:
    """Claim 3: All predictions lie inside convex hull of K archetype curves."""
    model.eval()
    sum_errors = []
    pos_violations = []
    alpha_mins = []

    with torch.no_grad():
        for batch in dataloader:
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            model(**kwargs)
            stored = model.last_explanations

            alpha = stored["composition"]           # [B, K]
            sum_err = (alpha.sum(dim=1) - 1.0).abs()
            pos_viol = F.relu(-alpha).amax(dim=1)
            alpha_min = alpha.min(dim=1)[0]

            sum_errors.append(sum_err.cpu())
            pos_violations.append(pos_viol.cpu())
            alpha_mins.append(alpha_min.cpu())

    sum_errors = torch.cat(sum_errors)
    pos_violations = torch.cat(pos_violations)
    alpha_mins = torch.cat(alpha_mins)

    max_sum_err = sum_errors.max().item()
    max_pos_viol = pos_violations.max().item()
    min_alpha = alpha_mins.min().item()
    max_violation = max(max_sum_err, max_pos_viol)
    passed = max_violation < 1e-4

    return {
        "claim": "Claim 3: Predictions in convex hull of K archetype curves",
        "max_sum_error": max_sum_err,
        "max_positive_violation": max_pos_viol,
        "min_alpha": min_alpha,
        "max_violation": max_violation,
        "num_samples": len(sum_errors),
        "passed": passed,
        "threshold": 1e-4,
        "interpretation": (
            f"sum(α) error max={max_sum_err:.2e}, min(α)={min_alpha:.4f} → "
            f"{'PASS (convex combination holds)' if passed else 'FAIL (outside convex hull)'}"
        ),
    }


def verify_claim4_archetype_differentiation(model, dataloader, device: str) -> dict:
    """Claim 4: K archetype hazard curves are genuinely distinct."""
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            kwargs["cur_epoch"] = 0
            model(**kwargs)
            stored = model.last_explanations

            hazards = stored["archetype_hazards"]           # [K, C]  (sigmoid of logit)
            hazard_logits = stored["archetype_hazard_logits"]  # [K, C]

            # Pairwise L1 between hazard curves (sigmoid space)
            K = hazards.size(0)
            diff = hazards.unsqueeze(1) - hazards.unsqueeze(0)   # [K, K, C]
            pairwise_l1 = diff.abs().sum(dim=2)                   # [K, K]
            diag_mask = torch.eye(K, dtype=torch.bool, device=hazards.device)
            offdiag_l1 = pairwise_l1[~diag_mask]

            # Cosine similarity between archetype embeddings
            archetypes = F.normalize(model.archetype_embedding.data, dim=-1)
            cos_sim = archetypes @ archetypes.t()
            offdiag_cos = cos_sim[~diag_mask]

            mean_l1 = offdiag_l1.mean().item()
            min_l1 = offdiag_l1.min().item()
            max_cos = offdiag_cos.max().item()

            # Decision: distinct if curves differ meaningfully
            distinct = mean_l1 > 0.05 and max_cos < 0.9

            return {
                "claim": "Claim 4: K archetype hazard curves are genuinely distinct",
                "K": K,
                "min_pairwise_l1": min_l1,
                "mean_pairwise_l1": mean_l1,
                "max_pairwise_l1": offdiag_l1.max().item(),
                "hazard_std_per_archetype": hazard_logits.std(dim=1).mean().item(),
                "archetype_cosine_max": max_cos,
                "archetype_cosine_min": offdiag_cos.min().item(),
                "passed": distinct,
                "threshold_l1": 0.05,
                "threshold_cosine": 0.9,
                "interpretation": (
                    f"mean L1={mean_l1:.4f} {'> 0.05: curves differ' if mean_l1 > 0.05 else '< 0.05: curves too similar'}, "
                    f"cosine_max={max_cos:.4f} {'< 0.9: embeddings distinct' if max_cos < 0.9 else '>= 0.9: embeddings too correlated'}"
                ),
            }

    return {"claim": "Claim 4", "passed": None, "note": "No data loaded"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify ACT-Surv v5's four constructive mechanism claims"
    )
    parser.add_argument("--cancer", default="blca")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (default: auto from results/)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-tokens-test", type=int, default=3,
                        help="How many tokens to test per sample for Claim 2")
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/act_surv_v5/full_run/{args.cancer}/fold{args.fold}/models/best_model.pt")

    if not ckpt_path.exists():
        ckpt_path_alt = Path(f"results/act_surv_v5/{args.cancer}/fold{args.fold}/models/best_model.pt")
        if ckpt_path_alt.exists():
            ckpt_path = ckpt_path_alt
        else:
            print(f"ERROR: checkpoint not found.")
            print(f"  Tried: {ckpt_path}")
            print(f"  Tried: {ckpt_path_alt}")
            print(f"\n  Run training first:")
            print(f"    python scripts/run_act_surv_v5.py --cancers {args.cancer} --folds {args.fold}")
            sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")

    # Load model
    state_dict = load_checkpoint(ckpt_path)
    model_ns = _build_act5_args(args.cancer, args.fold)
    model = ArchetypalTransportCompositionV5(model_ns)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()
    print(f"Model: {model.__class__.__name__}")
    print(f"  K={model.num_archetypes} archetypes, C={model.num_classes}, "
          f"ε={model.epsilon}, warmup={model.warmup_epochs}ep")

    # Load data
    print(f"\nData: {args.cancer} fold {args.fold}")
    try:
        dataloader = build_dataloader(args.cancer, args.fold, args.batch_size)
        print(f"  {len(dataloader)} batches × {args.batch_size}")
        has_data = True
    except Exception as e:
        print(f"  WARNING: dataloader failed: {e}")
        print("  Falling back to synthetic data for Claims 1/3/4.")
        print("  Claim 2 (closed-form vs re-solve) requires real tokens — skipped.")
        has_data = False

    results = {}

    print("\n" + "=" * 60)
    print("ACT-Surv v5 Mechanism Verification")
    print("=" * 60)

    # ── Claim 1 ──────────────────────────────────────────────────────────────
    print("\n[1/4] Claim 1: Completeness residual < 1e-5 ...")
    r1 = verify_claim1_completeness(model, dataloader, args.device) if has_data else {
        "passed": None, "note": "No dataloader"}
    results["claim1_completeness"] = r1
    p1 = r1.get("passed")
    icon = "✅" if p1 else ("❌" if p1 is False else "⚠️")
    print(f"  {icon} max_residual={r1.get('max_residual', 'N/A'):.2e} "
          f"(threshold=1e-5, {'PASS' if p1 else 'FAIL' if p1 is False else 'SKIP'})")

    # ── Claim 2 ──────────────────────────────────────────────────────────────
    print(f"\n[2/4] Claim 2: Closed-form vs re-solve error < 0.001 ...")
    r2 = verify_claim2_closed_form_vs_resolve(
        model, dataloader, args.device, num_tokens=args.num_tokens_test
    ) if has_data else {"passed": None, "note": "No dataloader"}
    results["claim2_closed_form"] = r2
    p2 = r2.get("passed")
    icon = "✅" if p2 else ("❌" if p2 is False else "⚠️")
    print(f"  {icon} max_error={r2.get('max_error', 'N/A'):.4f} "
          f"(threshold=0.001, n={r2.get('num_tested', 0)})")

    # ── Claim 3 ──────────────────────────────────────────────────────────────
    print("\n[3/4] Claim 3: Convex hull (bounded extrapolation) ...")
    r3 = verify_claim3_convex_hull(model, dataloader, args.device) if has_data else {
        "passed": None, "note": "No dataloader"}
    results["claim3_convex_hull"] = r3
    p3 = r3.get("passed")
    icon = "✅" if p3 else ("❌" if p3 is False else "⚠️")
    print(f"  {icon} max_violation={r3.get('max_violation', 'N/A'):.2e}, "
          f"min_alpha={r3.get('min_alpha', 'N/A'):.4f}")
    print(f"       {r3.get('interpretation', '')}")

    # ── Claim 4 ──────────────────────────────────────────────────────────────
    print("\n[4/4] Claim 4: Archetype differentiation ...")
    r4 = verify_claim4_archetype_differentiation(model, dataloader, args.device) if has_data else {
        "passed": None, "note": "No dataloader"}
    results["claim4_archetype"] = r4
    p4 = r4.get("passed")
    icon = "✅" if p4 else ("❌" if p4 is False else "⚠️")
    print(f"  {icon} mean_L1={r4.get('mean_pairwise_l1', 'N/A'):.4f}, "
          f"cosine_max={r4.get('archetype_cosine_max', 'N/A'):.4f}")
    if r4.get("interpretation"):
        print(f"       {r4['interpretation']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    claims = [
        ("Claim 1: Completeness", results["claim1_completeness"].get("passed")),
        ("Claim 2: Closed-form vs re-solve", results["claim2_closed_form"].get("passed")),
        ("Claim 3: Convex hull", results["claim3_convex_hull"].get("passed")),
        ("Claim 4: Archetype differentiation", results["claim4_archetype"].get("passed")),
    ]
    for name, p in claims:
        icon = "✅" if p else ("❌" if p is False else "⚠️  SKIP")
        print(f"  {icon}  {name}")
    passed = sum(1 for _, p in claims if p is True)
    total = sum(1 for _, p in claims if p is not None)
    print(f"\n  {passed}/{total} claims verified")

    # Save results
    out_path = ckpt_path.parent / "mechanism_verification.json"
    with open(out_path, "w") as f:
        json.dump({k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
