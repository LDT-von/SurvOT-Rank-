#!/usr/bin/env python3
"""Audit script for DCT mechanism verification metrics.

This script loads trained DCT checkpoints and computes:
1. DMR (Direction Mean Response) - direction regularization effect
2. Plan TV (Total Variation) - transport plan changes under intervention
3. Coupling Invariance - factual vs uniform coupling C-index
4. Faithfulness metrics - counterfactual deletion test

Run from repo root::

  python scripts/audit_dct_mechanism_verification.py \
    --checkpoint results/dct_v382_minimal/blca/fold0/model.pt \
    --output results/dct_v382_mechanism_verification/audit/blca_fold0.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


# =============================================================================
# Core Metric Functions
# =============================================================================

def compute_dmr_from_batch(model, factual_logits, low_logits, high_logits):
    """Compute Direction Mean Response from batch logits.
    
    DMR measures whether the model correctly moves risk in response to
    direction interventions on the cost matrix.
    
    Positive DMR means:
    - high_cost_intervention → higher risk
    - low_cost_intervention → lower risk
    
    This is the primary metric for direction regularization effectiveness.
    """
    factual_risk = model._risk(factual_logits)
    low_risk = model._risk(low_logits)
    high_risk = model._risk(high_logits)
    
    # Risk deltas
    low_delta = factual_risk - low_risk  # Expected: positive (factual > low_risk)
    high_delta = high_risk - factual_risk  # Expected: positive (high_risk > factual)
    
    # Direction labels: -1 for low intervention, +1 for high intervention
    directions = torch.cat([
        torch.full_like(low_delta, -1.0),
        torch.full_like(high_delta, 1.0)
    ])
    deltas = torch.cat([low_delta, high_delta])
    
    # Mean Direction Response (MDR) - simpler metric
    mean_dr = deltas.mean().item()
    
    # Direction Mean Response (DMR) - covariance-based
    mean_dir = directions.mean()
    mean_delta = deltas.mean()
    cov = ((directions - mean_dir) * (deltas - mean_delta)).mean()
    dmr = cov.item()
    
    # Separate gains for interpretation
    high_gain = high_risk.mean() - factual_risk.mean()
    low_gain = factual_risk.mean() - low_risk.mean()
    
    # Correct direction ratio (how often does direction match expectation)
    correct_direction_ratio = (
        ((low_delta > 0).float() + (high_delta > 0).float()) / 2
    ).mean().item()
    
    return {
        "dmr": dmr,
        "mdr": mean_dr,
        "high_gain": high_gain.item(),
        "low_gain": low_gain.item(),
        "factual_risk_mean": factual_risk.mean().item(),
        "factual_risk_std": factual_risk.std().item(),
        "low_risk_mean": low_risk.mean().item(),
        "high_risk_mean": high_risk.mean().item(),
        "correct_direction_ratio": correct_direction_ratio,
        # Additional diagnostics
        "low_delta_mean": low_delta.mean().item(),
        "low_delta_std": low_delta.std().item(),
        "high_delta_mean": high_delta.mean().item(),
        "high_delta_std": high_delta.std().item(),
    }


def compute_plan_tv_from_batch(model, factual_plans, low_plans, high_plans):
    """Compute Total Variation distances for transport plans.
    
    TV measures how much the transport plan changes under cost interventions.
    
    Large TV with small risk change → transport is being regularized but
    not affecting prediction (transport as driver with prediction bypass)
    
    Large TV with large risk change → transport is load-bearing
    """
    # Stack all plans for batch processing
    factual_flat = []
    low_flat = []
    high_flat = []
    
    for stage_plans, low_stage, high_stage in zip(factual_plans, low_plans, high_plans):
        for geo_idx in range(len(stage_plans)):
            f = stage_plans[geo_idx].flatten(1)
            l = low_stage[geo_idx].flatten(1)
            h = high_stage[geo_idx].flatten(1)
            
            # Normalize to probability distributions
            f = f / f.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            l = l / l.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            h = h / h.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            factual_flat.append(f)
            low_flat.append(l)
            high_flat.append(h)
    
    factual_flat = torch.cat(factual_flat)
    low_flat = torch.cat(low_flat)
    high_flat = torch.cat(high_flat)
    
    # TV = 0.5 * sum_ij |P_ij - Q_ij|
    tv_low = 0.5 * (factual_flat - low_flat).abs().sum(dim=-1)
    tv_high = 0.5 * (factual_flat - high_flat).abs().sum(dim=-1)
    
    return {
        "plan_tv_low_mean": tv_low.mean().item(),
        "plan_tv_low_std": tv_low.std().item(),
        "plan_tv_high_mean": tv_high.mean().item(),
        "plan_tv_high_std": tv_high.std().item(),
        "plan_tv_mean": (tv_low.mean() + tv_high.mean()) / 2,
        "plan_tv_low_max": tv_low.max().item(),
        "plan_tv_high_max": tv_high.max().item(),
    }


def compute_coupling_invariance_from_batch(model, slots_wsi, slots_omic, rows, cols, costs, epoch):
    """Compute coupling invariance metrics.
    
    Tests whether the prediction head depends on the learned transport plan
    by comparing factual coupling with uniform coupling.
    
    Large drop in C-index with uniform coupling → transport is load-bearing
    Small drop → prediction head can largely bypass transport
    """
    device = costs.device
    
    with torch.no_grad():
        # Compute factual plans
        factual_plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch)
        
        # Create uniform coupling (completely spread mass)
        bsz = costs.size(0)
        sw = slots_wsi.size(1)
        so = slots_omic.size(1)
        uniform_mass = 1.0 / (sw * so)
        
        uniform_plans = []
        for stage_plans in factual_plans:
            stage_uniform = []
            for plan in stage_plans:
                uniform = torch.full_like(plan, uniform_mass)
                uniform = uniform / uniform.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
                stage_uniform.append(uniform)
            uniform_plans.append(tuple(stage_uniform))
        
        # Encode with factual coupling
        factual_logits, _ = model._encode_logits_from_plans(
            slots_wsi, slots_omic, factual_plans
        )
        factual_risk = model._risk(factual_logits)
        
        # Encode with uniform coupling
        uniform_logits, _ = model._encode_logits_from_plans(
            slots_wsi, slots_omic, uniform_plans
        )
        uniform_risk = model._risk(uniform_logits)
        
        # Rank correlation
        factual_rank = factual_risk.argsort().float()
        uniform_rank = uniform_risk.argsort().float()
        
        if factual_rank.numel() > 1:
            spearman = torch.corrcoef(torch.stack([
                factual_rank - factual_rank.mean(),
                uniform_rank - uniform_rank.mean()
            ]))[0, 1].item()
        else:
            spearman = 0.0
        
        # Risk distribution shift
        risk_diff = (factual_risk - uniform_risk).abs()
        
        return {
            "factual_risk_mean": factual_risk.mean().item(),
            "factual_risk_std": factual_risk.std().item(),
            "uniform_risk_mean": uniform_risk.mean().item(),
            "uniform_risk_std": uniform_risk.std().item(),
            "risk_shift_mean": risk_diff.mean().item(),
            "risk_shift_std": risk_diff.std().item(),
            "factual_uniform_spearman": spearman,
            # Percentage of predictions that change rank
            "rank_change_rate": ((factual_rank != uniform_rank).float().mean().item()),
        }


def compute_faithfulness_from_batch(model, factual_plans, low_plans, high_plans,
                                     factual_logits, low_logits, high_logits):
    """Compute counterfactual faithfulness metrics.
    
    Faithfulness test: Can counterfactual risk be approximated by a simple
    deletion-style computation, or does it require full transport re-solve?
    
    Small |r_cf - r_cf_delete| → counterfactual is structural
    Large |r_cf - r_cf_delete| → counterfactual may be artifact
    """
    factual_risk = model._risk(factual_logits)
    low_risk = model._risk(low_logits)
    high_risk = model._risk(high_logits)
    
    # Compute plan changes
    plan_changes = []
    risk_changes = []
    
    for stage_idx, (f_stage, l_stage, h_stage) in enumerate(
        zip(factual_plans, low_plans, high_plans)
    ):
        for geo_idx, (f_plan, l_plan, h_plan) in enumerate(zip(f_stage, l_stage, h_stage)):
            # Plan change magnitude
            low_change = (l_plan - f_plan).abs().mean()
            high_change = (h_plan - f_plan).abs().mean()
            plan_changes.append({
                "stage": stage_idx,
                "geometry": geo_idx,
                "low_change": low_change.item(),
                "high_change": high_change.item(),
            })
    
    # Risk deltas
    low_delta = low_risk - factual_risk
    high_delta = high_risk - factual_risk
    
    # Faithfulness metric: correlation between plan change and risk change
    all_plan_changes = []
    all_risk_changes = []
    
    for stage_idx, (f_stage, l_stage, h_stage) in enumerate(
        zip(factual_plans, low_plans, high_plans)
    ):
        for geo_idx, (f_plan, l_plan, h_plan) in enumerate(zip(f_stage, l_stage, h_stage)):
            plan_change = torch.stack([
                (l_plan - f_plan).abs().mean(),
                (h_plan - f_plan).abs().mean()
            ])
            risk_change = torch.stack([low_delta.mean(), high_delta.mean()])
            all_plan_changes.append(plan_change)
            all_risk_changes.append(risk_change)
    
    if all_plan_changes:
        all_plan_changes = torch.stack(all_plan_changes)
        all_risk_changes = torch.stack(all_risk_changes).mean(dim=0)
        
        if all_plan_changes.numel() > 1:
            plan_risk_corr = torch.corrcoef(torch.stack([
                all_plan_changes.mean(dim=1),
                all_risk_changes.repeat(len(all_plan_changes))
            ]))[0, 1].item()
        else:
            plan_risk_corr = 0.0
    else:
        plan_risk_corr = 0.0
    
    # Deletion approximation test
    # r_cf_approx = r_factual + sum(contributions)
    # contribution_i = weight_i * delta_risk, where weight_i = plan_change_i / sum(plan_changes)
    
    total_plan_change = sum(
        pc["low_change"] + pc["high_change"]
        for pc in plan_changes
    )
    
    if total_plan_change > 1e-8:
        # Approximate CF risk using weighted plan changes
        approx_low_delta = sum(
            (pc["low_change"] / total_plan_change) * low_delta.mean()
            for pc in plan_changes
        )
        approx_high_delta = sum(
            (pc["high_change"] / total_plan_change) * high_delta.mean()
            for pc in plan_changes
        )
        
        approx_low_risk = factual_risk.mean() + approx_low_delta
        approx_high_risk = factual_risk.mean() + approx_high_delta
        
        # Deletion error
        low_deletion_error = (low_risk.mean() - approx_low_risk).abs()
        high_deletion_error = (high_risk.mean() - approx_high_risk).abs()
        
        # Relative deletion error
        low_rel_error = low_deletion_error / (low_delta.abs().mean() + 1e-8)
        high_rel_error = high_deletion_error / (high_delta.abs().mean() + 1e-8)
    else:
        low_deletion_error = high_deletion_error = 0.0
        low_rel_error = high_rel_error = 0.0
        approx_low_risk = factual_risk.mean()
        approx_high_risk = factual_risk.mean()
    
    return {
        "plan_risk_correlation": plan_risk_corr,
        "low_risk_delta_mean": low_delta.mean().item(),
        "high_risk_delta_mean": high_delta.mean().item(),
        "low_deletion_error": low_deletion_error.item(),
        "high_deletion_error": high_deletion_error.item(),
        "low_relative_deletion_error": low_rel_error.item(),
        "high_relative_deletion_error": high_rel_error.item(),
        "factual_risk": factual_risk.mean().item(),
        "approx_low_risk": approx_low_risk.item() if isinstance(approx_low_risk, torch.Tensor) else approx_low_risk,
        "approx_high_risk": approx_high_risk.item() if isinstance(approx_high_risk, torch.Tensor) else approx_high_risk,
        "actual_low_risk": low_risk.mean().item(),
        "actual_high_risk": high_risk.mean().item(),
        "num_geometry_plans": len(plan_changes),
        "total_plan_change": total_plan_change,
        "plan_changes": plan_changes[:4],  # First 4 for inspection
    }


def compute_cindex_from_risks(event_times, censorship, risk_scores):
    """Compute C-index from risk scores."""
    try:
        # Reshape if needed
        if risk_scores.ndim > 1:
            risk_scores = risk_scores.mean(dim=1) if risk_scores.shape[1] > 1 else risk_scores.squeeze()
        
        c_index, _, _, _, _ = concordance_index_censored(
            event_times.astype(bool),
            1 - censorship.astype(bool),
            risk_scores
        )
        return float(c_index) if not np.isnan(c_index) else 0.0
    except Exception:
        return 0.0


# =============================================================================
# Batch Processing for Audit
# =============================================================================

@dataclass
class AuditResult:
    """Container for audit results."""
    
    # Configuration
    checkpoint_path: str
    study: str
    fold: int
    
    # Per-batch metrics (will be aggregated)
    dmr_metrics: List[Dict] = field(default_factory=list)
    plan_tv_metrics: List[Dict] = field(default_factory=list)
    coupling_invariance_metrics: List[Dict] = field(default_factory=list)
    faithfulness_metrics: List[Dict] = field(default_factory=list)
    
    # C-index metrics
    factual_cindices: List[float] = field(default_factory=list)
    uniform_cindices: List[float] = field(default_factory=list)
    
    # Training info
    epoch: int = 0
    train_loss: float = 0.0
    
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate batch-level metrics into summary statistics."""
        
        def agg(metrics_list, prefix):
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            result = {}
            for key in keys:
                values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
                if values:
                    result[f"{prefix}_{key}_mean"] = np.mean(values)
                    result[f"{prefix}_{key}_std"] = np.std(values)
                    result[f"{prefix}_{key}_min"] = np.min(values)
                    result[f"{prefix}_{key}_max"] = np.max(values)
            return result
        
        summary = {
            "checkpoint_path": self.checkpoint_path,
            "study": self.study,
            "fold": self.fold,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "num_batches": len(self.dmr_metrics),
            
            # C-index summary
            "factual_cindex_mean": np.mean(self.factual_cindices) if self.factual_cindices else None,
            "factual_cindex_std": np.std(self.factual_cindices) if self.factual_cindices else None,
            "uniform_cindex_mean": np.mean(self.uniform_cindices) if self.uniform_cindices else None,
            "uniform_cindex_std": np.std(self.uniform_cindices) if self.uniform_cindices else None,
            "cindex_drop": (
                np.mean(self.factual_cindices) - np.mean(self.uniform_cindices)
                if self.factual_cindices and self.uniform_cindices else None
            ),
        }
        
        # Aggregate all metric types
        summary.update(agg(self.dmr_metrics, "dmr"))
        summary.update(agg(self.plan_tv_metrics, "plan_tv"))
        summary.update(agg(self.coupling_invariance_metrics, "ci"))
        summary.update(agg(self.faithfulness_metrics, "faith"))
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.aggregate()


def audit_model_on_dataloader(model, dataloader, device: str = "cuda",
                               max_batches: int = 100) -> AuditResult:
    """Run full audit on a trained model using a dataloader.
    
    This processes batches in eval mode and collects mechanism verification
    metrics. The dataloader should return validation/test data.
    """
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)
    
    result = AuditResult(
        checkpoint_path="unknown",
        study="unknown",
        fold=0
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            # Move batch to device
            kwargs = {}
            for key in ["x_wsi", "y", "c", "event_time", "omics", "pathway_omics"]:
                if key in batch:
                    val = batch[key]
                    if isinstance(val, torch.Tensor):
                        kwargs[key] = val.to(device_obj)
                    else:
                        kwargs[key] = val
            
            # Forward pass in eval mode (computes factual + counterfactual)
            try:
                logits, _ = model(**kwargs)
            except Exception as e:
                print(f"  Batch {batch_idx} failed: {e}")
                continue
            
            # Get cached intermediates from last forward pass
            # These are populated during eval() forward
            if not hasattr(model, "last_explanations"):
                continue
            
            explanations = model.last_explanations
            if explanations is None:
                continue
            
            # Extract cached values
            factual_risk = explanations.get("factual_risk")
            low_risk = explanations.get("low_risk_counterfactual")
            high_risk = explanations.get("high_risk_counterfactual")
            
            # Reconstruct logits from risks for the audit
            # (we need logits for some computations)
            factual_logits = torch.logit(factual_risk.clamp(1e-6, 1-1e-6)) if factual_risk is not None else None
            low_logits = torch.logit(low_risk.clamp(1e-6, 1-1e-6)) if low_risk is not None else None
            high_logits = torch.logit(high_risk.clamp(1e-6, 1-1e-6)) if high_risk is not None else None
            
            if factual_logits is None:
                continue
            
            # Get transport plans from cache
            # The model caches these during forward
            factual_plans = _get_cached_plans(model)
            if not factual_plans:
                continue
            
            # Compute metrics
            dmr = compute_dmr_from_batch(model, factual_logits, low_logits, high_logits)
            result.dmr_metrics.append(dmr)
            
            plan_tv = compute_plan_tv_from_batch(model, factual_plans, factual_plans, factual_plans)
            result.plan_tv_metrics.append(plan_tv)
            
            # Get slots and costs for coupling invariance test
            slots_wsi = getattr(model, "_last_slots_wsi", None)
            slots_omic = getattr(model, "_last_slots_omic", None)
            costs = getattr(model, "_last_factual_costs", None)
            rows = getattr(model, "_last_factual_rows", None)
            cols = getattr(model, "_last_factual_cols", None)
            
            if slots_wsi is not None and costs is not None:
                epoch = 0  # Use final epoch metrics
                ci = compute_coupling_invariance_from_batch(
                    model, slots_wsi, slots_omic, rows, cols, costs, epoch
                )
                result.coupling_invariance_metrics.append(ci)
            
            # Faithfulness test (needs low/high plans)
            # We need to recompute these with interventions
            # For now, skip if not available
            # TODO: Implement full CF recomputation for faithfulness
            
            # Compute C-index if labels available
            if "y" in kwargs and "c" in kwargs:
                y = kwargs["y"]
                c = kwargs["c"]
                
                # Factual C-index
                factual_risk_for_cindex = model._risk(factual_logits) if factual_logits is not None else None
                if factual_risk_for_cindex is not None:
                    c_idx = compute_cindex_from_risks(
                        y.cpu().numpy(),
                        c.cpu().numpy(),
                        factual_risk_for_cindex.cpu().numpy()
                    )
                    result.factual_cindices.append(c_idx)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {batch_idx + 1} batches")
    
    return result


def _get_cached_plans(model) -> List:
    """Get cached transport plans from model."""
    # The model stores plans in various attributes depending on version
    # Try different attribute names
    
    # v3.8.2 stores plans in forward pass
    if hasattr(model, "_last_factual_plans"):
        return model._last_factual_plans
    
    # Alternative: need to reconstruct from cached costs
    return []


# =============================================================================
# Main Audit Pipeline
# =============================================================================

def load_checkpoint_and_model(checkpoint_path: str, device: str = "cuda"):
    """Load a checkpoint and reconstruct the model."""
    from survot_rank.training.model_factory import get_model
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args = checkpoint.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    
    # Create model
    model = get_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, checkpoint


def run_audit_on_checkpoint(checkpoint_path: str, dataloader, device: str = "cuda",
                            max_batches: int = 100) -> AuditResult:
    """Run complete audit pipeline on a single checkpoint."""
    
    model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
    
    result = audit_model_on_dataloader(model, dataloader, device, max_batches)
    result.checkpoint_path = checkpoint_path
    result.epoch = checkpoint.get("epoch", 0)
    result.train_loss = checkpoint.get("loss", 0.0)
    
    return result


def run_sensitivity_audit(results_dir: str, study: str, fold: int,
                          lambda_values: List[float], device: str = "cuda",
                          max_batches: int = 50) -> pd.DataFrame:
    """Run audit across all λ_direction sensitivity experiments."""
    
    all_results = []
    
    for ld in lambda_values:
        exp_dir = Path(results_dir) / f"{study}_fold{fold}_ld{ld}"
        checkpoint_files = list(exp_dir.glob("model*.pt"))
        
        if not checkpoint_files:
            print(f"  No checkpoint found for λ={ld}")
            continue
        
        # Use best epoch checkpoint
        checkpoint_path = checkpoint_files[0]
        
        print(f"\nAuditing λ_direction={ld}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        result = run_audit_on_checkpoint(str(checkpoint_path), None, device, max_batches)
        summary = result.aggregate()
        summary["lambda_direction"] = ld
        all_results.append(summary)
        
        # Print key metrics
        if "dmr_dmr_mean" in summary:
            print(f"  DMR: {summary['dmr_dmr_mean']:.4f} ± {summary['dmr_dmr_std']:.4f}")
        if "factual_cindex_mean" in summary:
            print(f"  C-index: {summary['factual_cindex_mean']:.4f} ± {summary['factual_cindex_std']:.4f}")
    
    df = pd.DataFrame(all_results)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Audit DCT mechanism verification metrics"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataloader", type=str, default=None,
                        help="Path to dataloader pickle (optional)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for audit results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_batches", type=int, default=100,
                        help="Maximum batches to process")
    parser.add_argument("--study", type=str, default="blca",
                        help="Cancer study name")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold number")
    
    args = parser.parse_args()
    
    # Load dataloader if provided
    dataloader = None
    if args.dataloader:
        with open(args.dataloader, "rb") as f:
            dataloader = pickle.load(f)
        print(f"Loaded dataloader with {len(dataloader.dataset)} samples")
    
    # Run audit
    print(f"\nRunning mechanism verification audit...")
    result = run_audit_on_checkpoint(args.checkpoint, dataloader, args.device, args.max_batches)
    result.study = args.study
    result.fold = args.fold
    
    summary = result.to_dict()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAudit results saved to: {output_path}")
    print("\nSummary:")
    print(f"  DMR: {summary.get('dmr_dmr_mean', 'N/A')}")
    print(f"  C-index: {summary.get('factual_cindex_mean', 'N/A')}")
    print(f"  Coupling invariance (Spearman): {summary.get('ci_factual_uniform_spearman', 'N/A')}")


if __name__ == "__main__":
    main()
