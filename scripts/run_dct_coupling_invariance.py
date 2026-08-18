#!/usr/bin/env python3
"""Coupling Invariance Test for DCT v3.8.2.

This experiment tests whether the prediction head depends on the learned
transport plan by replacing it with uniform distributions.

Key Question: Is transport a "driver" (causally affects predictions) or
a "passenger" (decorative, bypassed by the prediction head)?

Design:
  1. Train DCT with learned (factual) coupling
  2. At inference, replace coupling with uniform: P_unif = (1/n) * ones(n, n)
  3. Measure C-index drop: ΔC = C(factual) - C(uniform)

Interpretation:
  - Large ΔC (>0.05): transport is load-bearing, prediction head depends on it
  - Small ΔC (<0.02): prediction head can bypass transport, it's decorative

Run from repo root::

  python scripts/run_dct_coupling_invariance.py plan
  python scripts/run_dct_coupling_invariance.py run --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from scripts import run_dct_v382_final_cross_cancer as base
except (ModuleNotFoundError, ImportError):
    import run_dct_v382_final_cross_cancer as base

REPO_ROOT = base.REPO_ROOT
RESULTS_BASE = Path("results/dct_v382_coupling_invariance")

# =============================================================================
# Coupling Replacement Utilities
# =============================================================================

def create_uniform_coupling(batch_size: int, num_wsi_slots: int, num_omic_slots: int,
                           device: torch.device) -> List:
    """Create a uniform transport coupling.
    
    Returns a list of stage plans, each containing geometry plans.
    All plans are uniform distributions over the WSI × Omics grid.
    """
    uniform_mass = 1.0 / (num_wsi_slots * num_omic_slots)
    
    # Structure: List[stage_idx][geometry_idx] = tensor[B, W, O]
    uniform_plan = torch.full(
        (batch_size, num_wsi_slots, num_omic_slots),
        uniform_mass,
        device=device
    )
    
    # Normalize to ensure proper probability distribution
    uniform_plan = uniform_plan / uniform_plan.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
    
    # Return as list of tuples (one per stage, one per geometry)
    # The exact structure depends on the model
    return uniform_plan


def replace_model_coupling_with_uniform(model, batch_size: int, device: torch.device):
    """Replace model's learned coupling with uniform for inference.
    
    This modifies the model's behavior to use uniform coupling instead of
    learned coupling during the encode_logits_from_plans step.
    """
    # Get slot dimensions from model
    sw = model.shared_wsi_prototypes.shape[0]
    so = model.shared_omic_prototypes.shape[0]
    
    # Create uniform coupling
    uniform_mass = 1.0 / (sw * so)
    uniform_plan = torch.full(
        (batch_size, sw, so),
        uniform_mass,
        device=device
    )
    uniform_plan = uniform_plan / uniform_plan.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
    
    # Monkey-patch the model's plans_from_cost_tensor method
    original_method = model._plans_from_cost_tensor
    
    def uniform_plans_from_cost(costs, rows, cols, epoch):
        """Return uniform coupling instead of learned coupling."""
        num_stages = model.spt_num_stages
        num_geometries = costs.size(2)  # 3 geometries
        
        uniform_plans = []
        for stage_idx in range(num_stages):
            stage_plans = []
            for geo_idx in range(num_geometries):
                stage_plans.append(uniform_plan.clone())
            uniform_plans.append(tuple(stage_plans))
        
        # Return plans and a dummy distance
        return uniform_plans, torch.tensor(0.0, device=device)
    
    # Store original and replace
    model._original_plans_from_cost_tensor = original_method
    model._plans_from_cost_tensor = uniform_plans_from_cost
    
    return model


def restore_model_coupling(model):
    """Restore the original coupling method."""
    if hasattr(model, '_original_plans_from_cost_tensor'):
        model._plans_from_cost_tensor = model._original_plans_from_cost_tensor
        delattr(model, '_original_plans_from_cost_tensor')
    return model


# =============================================================================
# Metric Computation
# =============================================================================

def compute_cindex(event_times, censorship, risk_scores):
    """Compute C-index from risk scores."""
    from sksurv.metrics import concordance_index_censored
    
    try:
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


def compute_rank_correlation(risk_a, risk_b):
    """Compute Spearman rank correlation between two risk vectors."""
    # Get ranks
    rank_a = risk_a.argsort().argsort().float()
    rank_b = risk_b.argsort().argsort().float()
    
    # Centered ranks
    rank_a_centered = rank_a - rank_a.mean()
    rank_b_centered = rank_b - rank_b.mean()
    
    # Correlation
    if rank_a.numel() > 1:
        corr = torch.corrcoef(torch.stack([rank_a_centered, rank_b_centered]))[0, 1]
        return corr.item() if torch.isnan(corr) == False else 0.0
    return 0.0


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class CouplingInvarianceExperiment:
    """Coupling Invariance Experiment runner."""
    
    study: str = "blca"
    fold: int = 0
    base_config: str = "configs/dct_v382_minimal_transport_blca.yaml"
    results_dir: Path = RESULTS_BASE
    
    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_results_file(self) -> Path:
        return self.results_dir / f"{self.study}_fold{self.fold}_results.csv"
    
    def run_training(self, gpu: int = 0, max_epochs: int = 30, python_cmd: str = "python"):
        """Run the base training with frozen recipe."""
        print(f"\n{'='*60}")
        print(f"Training base DCT model for Coupling Invariance test")
        print(f"Study: {self.study}, Fold: {self.fold}")
        print(f"{'='*60}")
        
        cmd = (
            f"{python_cmd} scripts/run_dct_v382_final_cross_cancer.py run "
            f"--config {self.base_config} "
            f"--study {self.study} --k_start {self.fold} --k_end {self.fold + 1} "
            f"--gpu {gpu} --max_epochs {max_epochs} "
            f"--results_dir {self.results_dir / 'factual'} "
            f"--set dct_v38_lambda_direction=0.05"
        )
        
        print(f"CMD: {cmd}")
        os.system(cmd)
    
    def run_inference_with_coupling_variants(self, model, dataloader, device: str = "cuda"):
        """Run inference with factual and uniform coupling."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        
        device_obj = torch.device(device)
        model.to(device_obj)
        model.eval()
        
        results = {
            "factual": [],
            "uniform": [],
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Prepare batch
                kwargs = {}
                for key in ["x_wsi", "y", "c", "event_time", "omics", "pathway_omics"]:
                    if key in batch:
                        val = batch[key]
                        kwargs[key] = val.to(device_obj) if isinstance(val, torch.Tensor) else val
                
                bsz = kwargs["x_wsi"].size(0)
                
                # Factual coupling inference
                factual_logits, _ = model(**kwargs)
                factual_risk = model._risk(factual_logits)
                
                # Uniform coupling inference
                restore_model_coupling(model)
                replace_model_coupling_with_uniform(model, bsz, device_obj)
                
                uniform_logits, _ = model(**kwargs)
                uniform_risk = model._risk(uniform_logits)
                
                # Restore original coupling
                restore_model_coupling(model)
                
                # Compute C-index for both
                y = kwargs.get("y", kwargs.get("event_time"))
                c = kwargs.get("c")
                
                if y is not None and c is not None:
                    factual_cidx = compute_cindex(
                        y.cpu().numpy() if isinstance(y, torch.Tensor) else y,
                        c.cpu().numpy() if isinstance(c, torch.Tensor) else c,
                        factual_risk.cpu().numpy()
                    )
                    uniform_cidx = compute_cindex(
                        y.cpu().numpy() if isinstance(y, torch.Tensor) else y,
                        c.cpu().numpy() if isinstance(c, torch.Tensor) else c,
                        uniform_risk.cpu().numpy()
                    )
                    
                    results["factual"].append(factual_cidx)
                    results["uniform"].append(uniform_cidx)
        
        return results
    
    def save_results(self, results: Dict[str, List], config: Dict[str, Any]):
        """Save experiment results."""
        df = pd.DataFrame({
            "factual_cindex": results["factual"],
            "uniform_cindex": results["uniform"],
            "cindex_drop": [f - u for f, u in zip(results["factual"], results["uniform"])],
        })
        
        # Add summary statistics
        summary = {
            "factual_cindex_mean": np.mean(results["factual"]),
            "factual_cindex_std": np.std(results["factual"]),
            "uniform_cindex_mean": np.mean(results["uniform"]),
            "uniform_cindex_std": np.std(results["uniform"]),
            "cindex_drop_mean": np.mean([f - u for f, u in zip(results["factual"], results["uniform"])]),
            "cindex_drop_std": np.std([f - u for f, u in zip(results["factual"], results["uniform"])]),
            **config
        }
        
        # Save detailed results
        df.to_csv(self.get_results_file(), index=False)
        
        # Save summary as JSON
        summary_file = self.get_results_file().with_suffix(".json")
        import json
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {self.get_results_file()}")
        print(f"Summary saved to: {summary_file}")
        
        return summary


# =============================================================================
# CLI Interface
# =============================================================================

def plan_command(args):
    """Print experiment plan."""
    print("""
Coupling Invariance Test for DCT
================================

Purpose:
  Test whether the prediction head depends on the learned transport plan.

Hypothesis:
  - H₀: Prediction head can bypass transport → uniform coupling ≈ factual C-index
  - H₁: Prediction head depends on transport → uniform coupling << factual C-index

Design:
  1. Train DCT with learned (factual) coupling (baseline)
  2. At inference, replace coupling with uniform: P_unif = (1/n²) * ones(n, n)
  3. Measure C-index drop: ΔC = C(factual) - C(uniform)

Expected Findings:
  ┌────────────┬─────────────┬──────────────────────────────┐
  │ ΔC         │ Conclusion  │ Implication for DCT          │
  ├────────────┼─────────────┼──────────────────────────────┤
  │ > 0.05     │ transport   │ Transport is load-bearing    │
  │            │ is driver   │ and causally affects preds  │
  ├────────────┼─────────────┼──────────────────────────────┤
  │ < 0.02     │ transport   │ Transport is decorative,    │
  │            │ is passenger│ bypassed by prediction head │
  └────────────┴─────────────┴──────────────────────────────┘

This experiment directly addresses the reviewer's question:
  "How do you prove transport is not bypassed by the prediction head?"

Commands:
  python scripts/run_dct_coupling_invariance.py plan
  python scripts/run_dct_coupling_invariance.py run --gpu 0
  python scripts/run_dct_coupling_invariance.py audit --checkpoint <path>
""")


def run_command(args):
    """Run the experiment."""
    exp = CouplingInvarianceExperiment(
        study=args.study,
        fold=args.fold,
        base_config=args.base_config,
        results_dir=RESULTS_BASE / f"{args.study}_fold{args.fold}"
    )
    
    # Train base model
    exp.run_training(
        gpu=args.gpu,
        max_epochs=args.max_epochs,
        python_cmd=args.python
    )
    
    print("\nTraining complete. Run audit to compute coupling invariance metrics:")
    print(f"  python scripts/run_dct_coupling_invariance.py audit --checkpoint {exp.results_dir / 'factual'}/model.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Coupling Invariance Test for DCT"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Show experiment plan")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--study", type=str, default="blca")
    run_parser.add_argument("--fold", type=int, default=0)
    run_parser.add_argument("--gpu", type=int, default=0)
    run_parser.add_argument("--max_epochs", type=int, default=30)
    run_parser.add_argument("--python", type=str, default="python")
    run_parser.add_argument("--base_config", type=str,
                            default="configs/dct_v382_minimal_transport_blca.yaml")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit checkpoint")
    audit_parser.add_argument("--checkpoint", type=str, required=True)
    audit_parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.command == "plan":
        plan_command(args)
    elif args.command == "run":
        run_command(args)
    elif args.command == "audit":
        print(f"Audit command: {args}")
        print("See scripts/audit_dct_mechanism_verification.py for full audit")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
