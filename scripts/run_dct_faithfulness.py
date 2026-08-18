#!/usr/bin/env python3
"""Counterfactual Faithfulness Test for DCT v3.8.2.

This experiment tests whether counterfactual explanations are faithful to the
model's actual computation process.

Key Question: Is the counterfactual risk computation (change cost → change
transport → change risk) a genuine structural property of the model, or is
it a numerical artifact?

Design:
  1. Compute factual risk r_f
  2. Compute counterfactual risk r_cf (after cost intervention)
  3. Approximate r_cf using a "deletion" style computation
  4. Compare deletion approximation to actual r_cf

Deletion Approximation:
  r_cf_delete = r_f + Σ(weight_i × Δrisk_i)
  
  where weight_i = plan_change_i / Σ(plan_changes)
        Δrisk_i ≈ contribution of geometry i to total risk change

Interpretation:
  - Small |r_cf - r_cf_delete|: CF is structural (deletion works)
  - Large |r_cf - r_cf_delete|: CF may be numerical noise (deletion fails)

Run from repo root::

  python scripts/run_dct_faithfulness.py plan
  python scripts/run_dct_faithfulness.py audit --checkpoint <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scripts import run_dct_v382_final_cross_cancer as base
except (ModuleNotFoundError, ImportError):
    import run_dct_v382_final_cross_cancer as base

REPO_ROOT = base.REPO_ROOT
RESULTS_BASE = Path("results/dct_v382_faithfulness")


# =============================================================================
# Faithfulness Metric Computation
# =============================================================================

@dataclass
class FaithfulnessMetrics:
    """Container for faithfulness test results."""
    
    # Basic metrics
    factual_risk_mean: float = 0.0
    low_risk_mean: float = 0.0
    high_risk_mean: float = 0.0
    
    # Risk deltas
    low_delta_mean: float = 0.0
    high_delta_mean: float = 0.0
    
    # Plan changes
    low_plan_change_mean: float = 0.0
    high_plan_change_mean: float = 0.0
    
    # Deletion approximation
    approx_low_risk_mean: float = 0.0
    approx_high_risk_mean: float = 0.0
    low_deletion_error: float = 0.0
    high_deletion_error: float = 0.0
    
    # Faithfulness score
    deletion_ratio_low: float = 0.0
    deletion_ratio_high: float = 0.0
    plan_risk_correlation: float = 0.0
    
    # Geometry breakdown
    num_geometries: int = 0
    geometry_contributions: List[Dict] = None
    
    def __post_init__(self):
        if self.geometry_contributions is None:
            self.geometry_contributions = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "factual_risk_mean": self.factual_risk_mean,
            "low_risk_mean": self.low_risk_mean,
            "high_risk_mean": self.high_risk_mean,
            "low_delta_mean": self.low_delta_mean,
            "high_delta_mean": self.high_delta_mean,
            "low_plan_change_mean": self.low_plan_change_mean,
            "high_plan_change_mean": self.high_plan_change_mean,
            "approx_low_risk_mean": self.approx_low_risk_mean,
            "approx_high_risk_mean": self.approx_high_risk_mean,
            "low_deletion_error": self.low_deletion_error,
            "high_deletion_error": self.high_deletion_error,
            "deletion_ratio_low": self.deletion_ratio_low,
            "deletion_ratio_high": self.deletion_ratio_high,
            "plan_risk_correlation": self.plan_risk_correlation,
            "num_geometries": self.num_geometries,
            "geometry_contributions": self.geometry_contributions,
        }


def compute_deletion_approximation(
    factual_plans: List,
    low_plans: List,
    high_plans: List,
    factual_risk: torch.Tensor,
    low_risk: torch.Tensor,
    high_risk: torch.Tensor,
) -> FaithfulnessMetrics:
    """Compute deletion-style approximation of counterfactual risk.
    
    The deletion approximation assumes that the counterfactual risk change
    can be decomposed as a weighted sum of per-geometry contributions,
    where the weight is proportional to the geometry's plan change.
    
    Math:
      Δr_cf ≈ Σ_i (w_i × Δr)
      where w_i = |P_cf^i - P_f^i| / Σ_j |P_cf^j - P_f^j|
    
    If this approximation works well (small deletion error), then the
    counterfactual is "structural" - it emerges from the additive
    contribution of each geometry's plan change.
    
    If this approximation fails (large deletion error), then the
    counterfactual may be a non-linear effect of the full transport
    re-solve, or potentially numerical noise.
    """
    metrics = FaithfulnessMetrics()
    
    # Basic risk stats
    metrics.factual_risk_mean = factual_risk.mean().item()
    metrics.low_risk_mean = low_risk.mean().item()
    metrics.high_risk_mean = high_risk.mean().item()
    
    # Risk deltas
    low_delta = low_risk - factual_risk
    high_delta = high_risk - factual_risk
    metrics.low_delta_mean = low_delta.mean().item()
    metrics.high_delta_mean = high_delta.mean().item()
    
    # Compute plan changes per geometry
    low_plan_changes = []
    high_plan_changes = []
    all_plan_changes = []
    
    for stage_idx, (f_stage, l_stage, h_stage) in enumerate(
        zip(factual_plans, low_plans, high_plans)
    ):
        for geo_idx, (f_plan, l_plan, h_plan) in enumerate(zip(f_stage, l_stage, h_stage)):
            # Total variation for each geometry
            low_change = (l_plan - f_plan).abs().mean()
            high_change = (h_plan - f_plan).abs().mean()
            
            low_plan_changes.append(low_change.item())
            high_plan_changes.append(high_change.item())
            all_plan_changes.append(low_change.item() + high_change.item())
            
            metrics.geometry_contributions.append({
                "stage": stage_idx,
                "geometry": geo_idx,
                "low_plan_change": low_change.item(),
                "high_plan_change": high_change.item(),
            })
    
    metrics.num_geometries = len(all_plan_changes)
    metrics.low_plan_change_mean = np.mean(low_plan_changes)
    metrics.high_plan_change_mean = np.mean(high_plan_changes)
    
    # Total plan change
    total_plan_change = sum(all_plan_changes)
    
    if total_plan_change < 1e-8:
        # No plan change, deletion approximation = factual
        metrics.approx_low_risk_mean = metrics.factual_risk_mean
        metrics.approx_high_risk_mean = metrics.factual_risk_mean
        metrics.low_deletion_error = 0.0
        metrics.high_deletion_error = 0.0
        metrics.deletion_ratio_low = 0.0
        metrics.deletion_ratio_high = 0.0
    else:
        # Deletion approximation:
        # r_cf_delete = r_f + Σ(w_i × Δr)
        # where w_i = plan_change_i / total_plan_change
        
        # For low intervention
        low_weights = [lc / total_plan_change for lc in low_plan_changes]
        low_weighted_delta = sum(
            w * low_delta.mean().item()
            for w in low_weights
        )
        metrics.approx_low_risk_mean = metrics.factual_risk_mean + low_weighted_delta
        
        # For high intervention
        high_weights = [hc / total_plan_change for hc in high_plan_changes]
        high_weighted_delta = sum(
            w * high_delta.mean().item()
            for w in high_weights
        )
        metrics.approx_high_risk_mean = metrics.factual_risk_mean + high_weighted_delta
        
        # Deletion errors
        metrics.low_deletion_error = abs(
            metrics.low_risk_mean - metrics.approx_low_risk_mean
        )
        metrics.high_deletion_error = abs(
            metrics.high_risk_mean - metrics.approx_high_risk_mean
        )
        
        # Deletion ratios (relative to total risk change)
        low_total_change = abs(metrics.low_delta_mean) + 1e-8
        high_total_change = abs(metrics.high_delta_mean) + 1e-8
        
        metrics.deletion_ratio_low = metrics.low_deletion_error / low_total_change
        metrics.deletion_ratio_high = metrics.high_deletion_error / high_total_change
    
    # Plan-risk correlation: correlation between plan changes and risk changes
    if len(low_plan_changes) >= 2 and len(high_plan_changes) >= 2:
        # Create paired observations
        plan_changes = np.array(low_plan_changes + high_plan_changes)
        risk_changes = np.array(
            [metrics.low_delta_mean] * len(low_plan_changes) +
            [metrics.high_delta_mean] * len(high_plan_changes)
        )
        
        # Pearson correlation
        plan_mean = plan_changes.mean()
        risk_mean = risk_changes.mean()
        cov = ((plan_changes - plan_mean) * (risk_changes - risk_mean)).mean()
        plan_std = np.sqrt(((plan_changes - plan_mean) ** 2).mean())
        risk_std = np.sqrt(((risk_changes - risk_mean) ** 2).mean())
        
        if plan_std > 1e-8 and risk_std > 1e-8:
            metrics.plan_risk_correlation = cov / (plan_std * risk_std)
        else:
            metrics.plan_risk_correlation = 0.0
    else:
        metrics.plan_risk_correlation = 0.0
    
    return metrics


def compute_geometry_isolation_test(
    model,
    slots_wsi: torch.Tensor,
    slots_omic: torch.Tensor,
    factual_plans: List,
    low_plans: List,
    high_plans: List,
    rows: torch.Tensor,
    cols: torch.Tensor,
    costs: torch.Tensor,
    epoch: int,
) -> Dict[str, float]:
    """Test each geometry's individual contribution to CF risk.
    
    This is a more rigorous test than deletion approximation:
    we isolate each geometry's transport plan and see how it affects
    the counterfactual risk prediction.
    
    For geometry g:
      - Replace all other geometries with uniform coupling
      - Keep geometry g's factual/counterfactual coupling
      - Compute risk with this "isolated" configuration
    """
    device = costs.device
    results = {}
    
    with torch.no_grad():
        # Get number of stages and geometries
        num_stages = len(factual_plans)
        num_geometries = len(factual_plans[0])
        
        # Create uniform coupling template
        sw = slots_wsi.size(1)
        so = slots_omic.size(1)
        uniform_mass = 1.0 / (sw * so)
        
        for iso_geo in range(num_geometries):
            # Create isolated plans for this geometry
            isolated_low = []
            isolated_high = []
            
            for stage_idx in range(num_stages):
                stage_low = []
                stage_high = []
                
                for geo_idx in range(num_geometries):
                    if geo_idx == iso_geo:
                        # Use actual counterfactual plans for isolated geometry
                        stage_low.append(low_plans[stage_idx][geo_idx])
                        stage_high.append(high_plans[stage_idx][geo_idx])
                    else:
                        # Use uniform for non-isolated geometries
                        uniform_plan = torch.full(
                            (costs.size(0), sw, so),
                            uniform_mass,
                            device=device
                        )
                        uniform_plan = uniform_plan / uniform_plan.sum(
                            dim=(-1, -2), keepdim=True
                        ).clamp_min(1e-8)
                        stage_low.append(uniform_plan)
                        stage_high.append(uniform_plan)
                
                isolated_low.append(tuple(stage_low))
                isolated_high.append(tuple(stage_high))
            
            # Encode with isolated plans
            isolated_low_logits, _ = model._encode_logits_from_plans(
                slots_wsi, slots_omic, isolated_low
            )
            isolated_high_logits, _ = model._encode_logits_from_plans(
                slots_wsi, slots_omic, isolated_high
            )
            
            # Get risk
            isolated_low_risk = model._risk(isolated_low_logits)
            isolated_high_risk = model._risk(isolated_high_logits)
            
            results[f"geo_{iso_geo}_low_risk"] = isolated_low_risk.mean().item()
            results[f"geo_{iso_geo}_high_risk"] = isolated_high_risk.mean().item()
            results[f"geo_{iso_geo}_low_delta"] = (
                isolated_low_risk.mean() - model._risk(
                    model._encode_logits_from_plans(
                        slots_wsi, slots_omic, factual_plans
                    )[0]
                ).mean()
            ).item()
            results[f"geo_{iso_geo}_high_delta"] = (
                isolated_high_risk.mean() - model._risk(
                    model._encode_logits_from_plans(
                        slots_wsi, slots_omic, factual_plans
                    )[0]
                ).mean()
            ).item()
    
    return results


# =============================================================================
# Batch Processing
# =============================================================================

def run_faithfulness_audit_on_batch(
    model,
    batch: Dict,
    device: torch.device,
) -> Optional[FaithfulnessMetrics]:
    """Run faithfulness test on a single batch."""
    
    with torch.no_grad():
        # Move batch to device
        kwargs = {}
        for key in ["x_wsi", "y", "c", "event_time", "omics", "pathway_omics"]:
            if key in batch:
                val = batch[key]
                kwargs[key] = val.to(device) if isinstance(val, torch.Tensor) else val
        
        # Forward pass (eval mode computes counterfactuals)
        logits, _ = model(**kwargs)
        
        # Get cached explanations
        if not hasattr(model, "last_explanations"):
            return None
        
        explanations = model.last_explanations
        if explanations is None:
            return None
        
        # Extract cached values
        factual_risk = explanations.get("factual_risk")
        low_risk = explanations.get("low_risk_counterfactual")
        high_risk = explanations.get("high_risk_counterfactual")
        
        if factual_risk is None:
            return None
        
        # Get cached plans (need to compute if not cached)
        # The model stores costs and slots, we need to reconstruct plans
        costs = getattr(model, "_last_factual_costs", None)
        rows = getattr(model, "_last_factual_rows", None)
        cols = getattr(model, "_last_factual_cols", None)
        slots_wsi = getattr(model, "_last_slots_wsi", None)
        slots_omic = getattr(model, "_last_slots_omic", None)
        
        if costs is None or slots_wsi is None:
            return None
        
        # Compute counterfactual costs and plans
        epoch = 0  # Final epoch
        
        # Factual plans
        factual_plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch)
        
        # Counterfactual plans
        low_costs, high_costs = model._counterfactual_costs(costs)
        low_plans, _ = model._plans_from_cost_tensor(low_costs, rows, cols, epoch)
        high_plans, _ = model._plans_from_cost_tensor(high_costs, rows, cols, epoch)
        
        # Compute faithfulness metrics
        metrics = compute_deletion_approximation(
            factual_plans, low_plans, high_plans,
            factual_risk, low_risk, high_risk
        )
        
        return metrics


def aggregate_faithfulness_metrics(
    batch_metrics: List[FaithfulnessMetrics]
) -> Dict[str, Any]:
    """Aggregate batch-level metrics into summary statistics."""
    
    if not batch_metrics:
        return {}
    
    # Collect all scalar metrics
    all_metrics = {}
    for m in batch_metrics:
        d = m.to_dict()
        for k, v in d.items():
            if k != "geometry_contributions" and isinstance(v, (int, float)):
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
    
    # Compute summary statistics
    summary = {}
    for k, values in all_metrics.items():
        if values:
            summary[f"{k}_mean"] = np.mean(values)
            summary[f"{k}_std"] = np.std(values)
            summary[f"{k}_min"] = np.min(values)
            summary[f"{k}_max"] = np.max(values)
    
    return summary


# =============================================================================
# Main Audit Pipeline
# =============================================================================

def run_faithfulness_audit(
    checkpoint_path: str,
    dataloader=None,
    device: str = "cuda",
    max_batches: int = 100,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete faithfulness audit on a checkpoint."""
    
    from survot_rank.training.model_factory import get_model
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args = checkpoint.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    
    device_obj = torch.device(device)
    
    model = get_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()
    
    print("Running faithfulness test...")
    
    all_metrics = []
    
    if dataloader is not None:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            metrics = run_faithfulness_audit_on_batch(model, batch, device_obj)
            if metrics is not None:
                all_metrics.append(metrics)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1} batches")
    
    # Aggregate results
    summary = aggregate_faithfulness_metrics(all_metrics)
    summary["checkpoint_path"] = checkpoint_path
    summary["num_batches"] = len(all_metrics)
    summary["epoch"] = checkpoint.get("epoch", "unknown")
    
    # Interpretation
    dmr = summary.get("dmr_mean", 0.0)
    del_low = summary.get("deletion_ratio_low_mean", 0.0)
    del_high = summary.get("deletion_ratio_high_mean", 0.0)
    plan_risk_corr = summary.get("plan_risk_correlation_mean", 0.0)
    
    if del_low < 0.2 and del_high < 0.2:
        summary["interpretation"] = (
            "HIGH FAITHFULNESS: Deletion approximation works well "
            f"(ratio={del_low:.3f}/{del_high:.3f}). "
            "Counterfactual is structural."
        )
    elif del_low < 0.5 and del_high < 0.5:
        summary["interpretation"] = (
            "MODERATE FAITHFULNESS: Deletion approximation partially works "
            f"(ratio={del_low:.3f}/{del_high:.3f}). "
            "Counterfactual has non-linear components."
        )
    else:
        summary["interpretation"] = (
            "LOW FAITHFULNESS: Deletion approximation fails "
            f"(ratio={del_low:.3f}/{del_high:.3f}). "
            "Counterfactual may be numerical artifact."
        )
    
    if plan_risk_corr > 0.5:
        summary["interpretation"] += (
            f" Strong plan-risk correlation ({plan_risk_corr:.3f})."
        )
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return summary


# =============================================================================
# CLI Interface
# =============================================================================

def plan_command(args):
    """Print experiment plan."""
    print("""
Counterfactual Faithfulness Test for DCT
======================================

Purpose:
  Test whether counterfactual explanations are faithful to the model.

Hypothesis:
  - H₀: CF is numerical artifact → deletion approximation fails
  - H₁: CF is structural → deletion approximation works

Design:
  1. Compute factual risk r_f
  2. Compute counterfactual risk r_cf (after cost intervention)
  3. Approximate r_cf using deletion-style computation
  4. Compare deletion approximation to actual r_cf

Deletion Approximation:
  r_cf_delete = r_f + Σ(weight_i × Δrisk_i)
  
  where weight_i = plan_change_i / Σ(plan_changes)
        Δrisk_i ≈ contribution of geometry i

Interpretation:
  ┌────────────────┬──────────────────────────────────┐
  │ Deletion Ratio │ Conclusion                       │
  ├────────────────┼──────────────────────────────────┤
  │ < 0.2          │ HIGH FAITHFULNESS                │
  │                │ CF is structural, deletion works │
  ├────────────────┼──────────────────────────────────┤
  │ 0.2 - 0.5      │ MODERATE FAITHFULNESS            │
  │                │ CF has non-linear components     │
  ├────────────────┼──────────────────────────────────┤
  │ > 0.5          │ LOW FAITHFULNESS                 │
  │                │ CF may be numerical artifact     │
  └────────────────┴──────────────────────────────────┘

This experiment directly addresses the reviewer's question:
  "Are your counterfactual explanations faithful to the model?"

Commands:
  python scripts/run_dct_faithfulness.py plan
  python scripts/run_dct_faithfulness.py audit --checkpoint <path>
""")


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual Faithfulness Test for DCT"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Show experiment plan")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit checkpoint")
    audit_parser.add_argument("--checkpoint", type=str, required=True,
                              help="Path to model checkpoint")
    audit_parser.add_argument("--dataloader", type=str, default=None,
                              help="Path to dataloader pickle (optional)")
    audit_parser.add_argument("--output", type=str, default=None,
                              help="Output JSON path")
    audit_parser.add_argument("--device", type=str, default="cuda")
    audit_parser.add_argument("--max_batches", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == "plan":
        plan_command(args)
    elif args.command == "audit":
        # Load dataloader if provided
        dataloader = None
        if args.dataloader:
            with open(args.dataloader, "rb") as f:
                dataloader = f
        
        output = args.output or f"results/dct_v382_faithfulness/{Path(args.checkpoint).stem}.json"
        
        summary = run_faithfulness_audit(
            args.checkpoint,
            dataloader=dataloader,
            device=args.device,
            max_batches=args.max_batches,
            output_path=output,
        )
        
        print("\n" + "=" * 60)
        print("FAITHFULNESS TEST RESULTS")
        print("=" * 60)
        print(f"\nDMR: {summary.get('dmr_mean', 'N/A')}")
        print(f"Deletion Ratio (Low): {summary.get('deletion_ratio_low_mean', 'N/A')}")
        print(f"Deletion Ratio (High): {summary.get('deletion_ratio_high_mean', 'N/A')}")
        print(f"Plan-Risk Correlation: {summary.get('plan_risk_correlation_mean', 'N/A')}")
        print(f"\nInterpretation: {summary.get('interpretation', 'N/A')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
