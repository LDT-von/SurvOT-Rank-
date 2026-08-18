#!/usr/bin/env python3
"""Cross-fold Archetype Permutation Matching (Section 4.6).

Hungarian algorithm matches archetypes across folds by hazard logit trajectory
similarity, enabling valid cross-fold pooling of archetype-level statistics
(clinical enrichment, pathway attribution, etc.).

Run:
    python scripts/cross_fold_archetype_matching.py --cancer blca --all-folds
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
from scipy.optimize import linear_sum_assignment


def _find_ckpt_path(cancer: str, fold: int, variant: str = "v5_1") -> Path:
    candidates = [
        REPO_ROOT / f"results/act_surv_v5_{variant}/{cancer}",
    ]
    for parent in candidates:
        if not parent.exists():
            continue
        for d in parent.iterdir():
            if d.is_dir() and d.name.endswith(f"_fold{fold}"):
                ckpts = list(d.glob("model_best_s*.pth"))
                if ckpts:
                    return ckpts[0]
    raise FileNotFoundError(f"No checkpoint for {cancer} fold {fold} variant {variant}")


def load_archetype_hazards(ckpt_path: Path, device: str = "cpu") -> tuple[np.ndarray, int]:
    """Load archetype hazard logits [K, C] from a checkpoint."""
    state = load_checkpoint_pretrained_state(ckpt_path)
    state = {k: v.to(device) for k, v in state.items() if torch.is_tensor(v)}

    K = int(state["archetype_embedding"].shape[0])
    hazard_logits = state["_logit_hazard_raw"].cpu().numpy()  # [K, C]
    return hazard_logits, K


def cost_matrix_hazard(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """L1 cost matrix between two sets of archetype hazard logits.

    Args:
        h1: [K1, C] — reference archetypes (e.g., fold 0)
        h2: [K2, C] — target archetypes (e.g., fold n)

    Returns:
        [K1, K2] cost matrix. For K1 != K2, pad the smaller with dummy rows/cols.
    """
    K1, K2 = h1.shape[0], h2.shape[0]
    K_max = max(K1, K2)
    cost = np.zeros((K_max, K_max))

    for i in range(K1):
        for j in range(K2):
            cost[i, j] = np.abs(h1[i] - h2[j]).mean()

    return cost[:K1, :K2]  # [K1, K2]


def hungarian_match(hazard_logits_list: list[np.ndarray]) -> list[dict]:
    """Match archetypes across folds via Hungarian algorithm.

    Uses L1 distance on hazard logit trajectories as the matching cost.
    Fold 0 is the reference; all other folds are matched to it.

    Returns:
        List of dicts, one per fold beyond the first:
        {
            "fold": n,
            "K": K,
            "matches": [(ref_k, matched_k), ...],
            "costs": [cost_0, cost_1, ...],
            "mean_cost": float,
        }
    """
    ref = hazard_logits_list[0]
    K_ref = ref.shape[0]
    results = []

    for fold_idx, h in enumerate(hazard_logits_list[1:], start=1):
        K = h.shape[0]
        cost = cost_matrix_hazard(ref, h)  # [K_ref, K]

        # Hungarian: row=ref archetypes, col=matched archetypes
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
        costs = [float(cost[r, c]) for r, c in matches]
        mean_cost = float(np.mean(costs))

        results.append({
            "fold": fold_idx,
            "K": K,
            "K_ref": K_ref,
            "matches": matches,
            "costs": costs,
            "mean_cost": mean_cost,
        })

        print(f"  Fold {fold_idx}: K={K}, mean_match_cost={mean_cost:.4f}")
        for r, c in matches:
            print(f"    Ref A{r} → A{c}  cost={cost[r,c]:.4f}")

    return results


def run_cross_fold_matching(
    cancer: str,
    folds: list[int],
    variant: str = "v5_1",
    device: str = "cpu",
) -> dict:
    print(f"\n{'='*60}")
    print(f"Cross-Fold Archetype Matching: {cancer}")
    print(f"{'='*60}")

    hazard_logits_list = []
    K_list = []
    ckpt_paths = {}

    for fold in folds:
        try:
            ckpt = _find_ckpt_path(cancer, fold, variant)
            ckpt_paths[fold] = ckpt
            hazards, K = load_archetype_hazards(ckpt, device)
            hazard_logits_list.append(hazards)
            K_list.append(K)
            print(f"  Fold {fold}: K={K}, shape={hazards.shape}, ckpt={ckpt.name}")
        except Exception as e:
            print(f"  Fold {fold}: SKIPPED — {e}")
            hazard_logits_list.append(None)
            K_list.append(None)

    valid = [(i, h) for i, h in enumerate(hazard_logits_list) if h is not None]
    if len(valid) < 2:
        return {
            "experiment": "cross_fold_matching",
            "skipped": True,
            "reason": f"Only {len(valid)} valid checkpoints",
        }

    ref_idx = valid[0][0]
    hazard_logits_list_aligned = [h for h in hazard_logits_list if h is not None]

    K_unique = set(K for K in K_list if K is not None)
    if len(K_unique) > 1:
        print(f"  WARNING: K varies across folds: {K_unique}")
        print("  Hungarian matching handles this by padding smaller matrices.")

    print("\n[Hungarian Matching — Fold 0 as reference]")
    results = hungarian_match(hazard_logits_list_aligned)

    # Permutation invariance: check if mean cost is low (< 0.05 L1)
    all_costs = [c for r in results for c in r["costs"]]
    mean_cost = float(np.mean(all_costs))
    max_cost = float(np.max(all_costs))
    passed = mean_cost < 0.05 and max_cost < 0.10

    print(f"\n[Summary]")
    print(f"  Mean matching cost: {mean_cost:.4f}")
    print(f"  Max matching cost:  {max_cost:.4f}")
    print(f"  {'PASS' if passed else 'FAIL'}: permutation invariance "
          f"(threshold: mean<0.05, max<0.10)")

    return {
        "experiment": "cross_fold_matching",
        "cancer": cancer,
        "variant": variant,
        "folds": folds,
        "fold_checkpoint_paths": {str(k): str(v) for k, v in ckpt_paths.items()},
        "K_per_fold": {f: K_list[i] for i, f in enumerate(folds)},
        "reference_fold": folds[0],
        "matching_results": results,
        "mean_cost": mean_cost,
        "max_cost": max_cost,
        "passed": passed,
        "verdict": (
            f"Archetypes permutation-invariant across folds: "
            f"mean_cost={mean_cost:.4f} (threshold 0.05)"
            if passed
            else f"Archetypes NOT permutation-invariant: mean_cost={mean_cost:.4f}"
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Cross-fold archetype permutation matching")
    p.add_argument("--cancer", default="blca")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--variant", default="v5_1",
                   help="Training variant (default: v5_1)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",
                   default=str(REPO_ROOT / "results" / "act_surv_v5" / "cross_fold"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_cross_fold_matching(args.cancer, args.folds, args.variant, device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"cross_fold_matching_{args.cancer}_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    out_md = output_dir / f"cross_fold_matching_{args.cancer}_{stamp}.md"
    lines = [
        f"# Cross-Fold Archetype Matching — {args.cancer}",
        f"\n**Variant:** {args.variant}  |  **Reference fold:** {args.folds[0]}",
        f"\n**Mean cost:** {result.get('mean_cost', 'N/A'):.4f}  "
        f"| **Max cost:** {result.get('max_cost', 'N/A'):.4f}",
        f"\n**Verdict:** {result.get('verdict', 'N/A')}",
        "\n## Fold-level matches",
    ]
    for r in result.get("matching_results", []):
        lines.append(f"\n### Fold {r['fold']} (K={r['K']}, mean_cost={r['mean_cost']:.4f})")
        lines.append("| Ref | Matched | L1 Cost |")
        lines.append("|-----|---------|---------|")
        for (ref_k, matched_k), cost in zip(r["matches"], r["costs"]):
            lines.append(f"| A{ref_k} | A{matched_k} | {cost:.4f} |")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
