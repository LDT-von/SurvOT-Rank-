#!/usr/bin/env python3
"""ACT-Surv v5 Clinical Enrichment Analysis (Section 5.3 / idea.md Section 8).

Protocol:
1. Load v5.1 BLCA fold-N checkpoint → extract alpha = composition [B, K]
2. Match alpha to clinical labels (pathologic_stage, tumor_grade) from GDC CSV
3. For each archetype k × clinical endpoint:
   a. Binary split: G_k = {i | α_{i,k} >= median(α_{:,k})} (high vs low membership)
   b. 2×2 contingency table vs clinical category
   c. Fisher exact test → OR, 95% CI, raw p
   d. Benjamini-Hochberg across all K×S tests → q-values
4. Report discovery fold results; separately assess replication fold direction.

Usage:
    python scripts/clinical_enrichment_act_surv_v5.py --cancer blca --fold 0 --device cuda
    python scripts/clinical_enrichment_act_surv_v5.py --cancer blca --all-folds --device cuda
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_act_surv_v5_mechanism import (
    build_dataloader,
    detect_dims_from_state,
    load_checkpoint_pretrained_state,
)

# Clinical ordinal encoding for stage
STAGE_ORDINAL_MAP = {
    "Stage 0a": 0.0, "Stage 0is": 0.0,
    "Stage I": 1.0, "Stage IA": 1.0, "Stage IB": 1.0,
    "Stage II": 2.0, "Stage IIA": 2.0, "Stage IIB": 2.5, "Stage IIC": 2.7,
    "Stage III": 3.0, "Stage IIIA": 3.3, "Stage IIIB": 3.6, "Stage IIIC": 3.8,
    "Stage IV": 4.0, "Stage IVA": 4.0, "Stage IVB": 4.2,
    "Stage IVC": 4.3,
}

GRADE_MAP = {
    "Low Grade": 0.0,
    "High Grade": 1.0,
}


def _find_ckpt_path(cancer: str, fold: int) -> Path:
    """Find v5.1 model_best checkpoint for given cancer/fold."""
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
    raise FileNotFoundError(
        f"No v5.1 checkpoint for {cancer} fold {fold}. "
        f"Searched: {[str(p) for p in candidates]}"
    )


def _find_clinical_csv(cancer: str) -> Path:
    """Find clinical CSV with pathologic_stage and tumor_grade columns."""
    roots = [
        REPO_ROOT / "survot_rank/research/legacy/slotspe_runtime/dataset_csv/clinical/all",
        REPO_ROOT / "dataset_csv/clinical/all",
    ]
    for root in roots:
        if not root.exists():
            continue
        # Try _with_clinical_features first (has stage_ordinal / grade_ordinal)
        for fname in [f"{cancer}_with_clinical_features.csv", f"{cancer}.csv"]:
            path = root / fname
            if path.exists():
                return path
    raise FileNotFoundError(
        f"No clinical CSV for {cancer}. Searched: {[str(r / cancer) for r in roots]}"
    )


def _load_clinical_labels(cancer: str) -> pd.DataFrame:
    """Load clinical CSV and return case_id + stage/grade columns."""
    csv_path = _find_clinical_csv(cancer)
    df = pd.read_csv(csv_path)

    # Identify case_id column
    id_col = None
    for col in ["case id", "case_id", "Case ID"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        raise ValueError(f"No case ID column found in {csv_path}. Columns: {list(df.columns)}")

    # Build result with only relevant columns
    result = df[[id_col]].rename(columns={id_col: "case_id"}).copy()

    # pathologic_stage
    if "ajcc_pathologic_stage" in df.columns:
        result["stage_text"] = df["ajcc_pathologic_stage"]
        result["stage_ordinal"] = df["ajcc_pathologic_stage"].map(STAGE_ORDINAL_MAP)
    elif "pathologic_stage" in df.columns:
        result["stage_text"] = df["pathologic_stage"]
        result["stage_ordinal"] = df["pathologic_stage"].map(STAGE_ORDINAL_MAP)
    elif "stage_ordinal" in df.columns:
        result["stage_ordinal"] = df["stage_ordinal"]
        result["stage_text"] = df.get("ajcc_pathologic_stage", pd.Series(dtype=str))

    # tumor_grade
    if "tumor_grade" in df.columns:
        result["grade_text"] = df["tumor_grade"]
        result["grade_ordinal"] = df["tumor_grade"].map(GRADE_MAP)
    elif "grade_ordinal" in df.columns:
        result["grade_ordinal"] = df["grade_ordinal"]
        result["grade_text"] = df.get("tumor_grade", pd.Series(dtype=str))

    result = result.dropna(subset=["case_id"]).reset_index(drop=True)
    result["case_id"] = result["case_id"].astype(str).str.strip()
    print(f"[clinical] Loaded {len(result)} patients from {csv_path.name}")
    print(f"  stage: {result['stage_ordinal'].notna().sum()} non-null / {len(result)}")
    print(f"  grade: {result['grade_ordinal'].notna().sum()} non-null / {len(result)}")
    return result


def _benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (qvalues, reject)."""
    n = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = pvals[idx]
    thresholds = (np.arange(n) + 1) / n * alpha
    reject = np.zeros(n, dtype=bool)
    max_k = 0
    for k in range(n - 1, -1, -1):
        if sorted_p[k] <= thresholds[k]:
            max_k = k
            break
    reject[idx[:max_k + 1]] = True
    qvalues = np.zeros(n)
    for i in range(n):
        p_rank = idx[i]
        rank = np.sum(idx <= p_rank)
        qvalues[p_rank] = sorted_p[rank - 1] * n / rank
    qvalues = np.minimum.accumulate(qvalues[::-1])[::-1]
    qvalues = np.minimum(qvalues, 1.0)
    return qvalues, reject


def _run_fisher(aflag: np.ndarray, cflag: np.ndarray) -> dict | None:
    """Fisher exact test on 2×2 table: archetype high vs clinical category."""
    # [high_G, not_high] × [in_category, not_in]
    table = np.array([
        [aflag & cflag, aflag & ~cflag],
        [~aflag & cflag, ~aflag & ~cflag],
    ], dtype=int)
    if table.min() < 0 or table.sum() < 4:
        return None  # too few samples
    odds, pval = fisher_exact(table, alternative="two-sided")

    # 95% CI via median-unbiased OR (Woolf logit)
    n11, n12, n21, n22 = table.ravel()
    if n11 > 0 and n12 > 0 and n21 > 0 and n22 > 0:
        log_or = np.log(odds)
        se = np.sqrt(1/n11 + 1/n12 + 1/n21 + 1/n22)
        ci_lo = np.exp(log_or - 1.96 * se)
        ci_hi = np.exp(log_or + 1.96 * se)
    else:
        # Haldane-Anscombe: use 0.5 adjustment
        a, b, c, d = max(n11, 0.5), max(n12, 0.5), max(n21, 0.5), max(n22, 0.5)
        odds_raw = (a * d) / (b * c)
        log_or = np.log(max(odds_raw, 1e-10))
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo = np.exp(max(log_or - 1.96 * se, -10))
        ci_hi = np.exp(log_or + 1.96 * se)
    return {
        "table": table.tolist(),
        "odds_ratio": float(odds),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(pval),
        "n_total": int(table.sum()),
    }


def run_enrichment(
    checkpoint_path: Path,
    cancer: str,
    fold: int,
    device: str = "cpu",
    high_threshold: str = "median",
) -> dict:
    """Run clinical enrichment analysis for one fold."""
    print(f"\n{'='*60}")
    print(f"Clinical Enrichment: {cancer} fold {fold}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # ── 1. Load checkpoint ───────────────────────────────────────────────
    state = load_checkpoint_pretrained_state(checkpoint_path)
    dims = detect_dims_from_state(state)
    print(f"[ckpt] Detected dims: {dims}")

    # ── 2. Build model ──────────────────────────────────────────────────
    from survot_rank.research.methods.archetypal_transport_composition_v5.model import (
        ArchetypalTransportCompositionV5,
    )
    from types import SimpleNamespace

    model_args = SimpleNamespace(
        omic_sizes=dims.get("omic_sizes", [16] * 10),
        encoding_dim=dims.get("encoding_dim", 1536),
        wsi_projection_dim=dims.get("wsi_projection_dim", 256),
        act5_num_archetypes=dims.get("act5_num_archetypes", 6),
        act5_epsilon=0.10,
        act5_hazard_scale=1.0,
        act5_warmup_epochs=0,  # inference only
        act5_lambda_balance=0.01,
        act5_lambda_rank=0.10,
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

    # ── 3. Forward validation set ───────────────────────────────────────
    try:
        val_loader = build_dataloader(cancer, fold, batch_size=4, device=device)
    except Exception as e:
        return {"experiment": "clinical_enrichment", "skipped": True,
                "reason": f"build_dataloader failed: {e}", "passed": None}

    print(f"[loader] {len(val_loader)} batches")

    alpha_rows: list[dict] = []  # {case_id, alpha_k...}
    for batch in val_loader:
        kwargs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        kwargs["cur_epoch"] = 0
        with torch.no_grad():
            model(**kwargs)
        comp = model.last_explanations["composition"].cpu()  # [B, K]

        # Try to get case_id from batch
        case_ids = batch.get("case_id", [f"patient_{i}" for i in range(comp.size(0))])
        for i, cid in enumerate(case_ids):
            row = {"case_id": str(cid)}
            for k in range(comp.size(1)):
                row[f"alpha_{k}"] = float(comp[i, k].item())
            alpha_rows.append(row)

    if not alpha_rows:
        return {"experiment": "clinical_enrichment", "skipped": True,
                "reason": "No samples extracted from dataloader", "passed": None}

    alpha_df = pd.DataFrame(alpha_rows)
    print(f"[alpha] {len(alpha_df)} patients, K={model.num_archetypes}")

    # ── 4. Merge with clinical labels ───────────────────────────────────
    clinical_df = _load_clinical_labels(cancer)
    merged = alpha_df.merge(clinical_df, on="case_id", how="left")
    print(f"[merged] {len(merged)} patients after join")

    K = model.num_archetypes
    n_with_stage = merged["stage_ordinal"].notna().sum()
    n_with_grade = merged["grade_ordinal"].notna().sum()
    print(f"[merged] {n_with_stage} with stage, {n_with_grade} with grade")

    # ── 5. Run Fisher exact tests ────────────────────────────────────────
    all_tests: list[dict] = []

    for k in range(K):
        col = f"alpha_{k}"
        if col not in merged.columns:
            continue

        # Binary split: high vs low membership
        vals = merged[col].dropna()
        if high_threshold == "median":
            threshold = vals.median()
        else:
            threshold = vals.quantile(0.75)
        aflag = (merged[col] >= threshold).values
        n_high = int(aflag.sum())

        # Stage categories (binary: high-stage ≥ Stage III vs low-stage)
        if n_with_stage >= 10:
            stage_binary = (merged["stage_ordinal"] >= 3.0).values  # Stage III+
            mask = merged["stage_ordinal"].notna().values
            if mask.sum() >= 10:
                res = _run_fisher(aflag[mask], stage_binary[mask])
                if res:
                    res["archetype"] = k
                    res["clinical_endpoint"] = "stage_high"
                    res["threshold_label"] = f"≥StageIII (α_{k}≥{threshold:.3f})"
                    res["n_high_archetype"] = n_high
                    all_tests.append(res)

            # Ordinal: Stage I/II vs III/IV (low vs high)
            stage_ord = merged["stage_ordinal"].values
            mask_ord = merged["stage_ordinal"].notna().values
            stage_binary2 = np.where(mask_ord, stage_ord >= 3.0, False)
            res2 = _run_fisher(aflag[mask_ord], stage_binary2[mask_ord])
            if res2:
                res2["archetype"] = k
                res2["clinical_endpoint"] = "stage_ordinal"
                res2["threshold_label"] = f"α_{k}≥{threshold:.3f}"
                res2["n_high_archetype"] = n_high
                all_tests.append(res2)

        # Grade: High Grade vs Low Grade
        if n_with_grade >= 10:
            grade_binary = (merged["grade_ordinal"] >= 0.5).values  # High Grade
            mask = merged["grade_ordinal"].notna().values
            if mask.sum() >= 10:
                res = _run_fisher(aflag[mask], grade_binary[mask])
                if res:
                    res["archetype"] = k
                    res["clinical_endpoint"] = "grade_high"
                    res["threshold_label"] = f"HighGrade (α_{k}≥{threshold:.3f})"
                    res["n_high_archetype"] = n_high
                    all_tests.append(res)

    if not all_tests:
        return {
            "experiment": "clinical_enrichment",
            "fold": fold,
            "skipped": True,
            "reason": "Fewer than 10 patients per endpoint — cannot run Fisher exact test",
            "passed": None,
        }

    # ── 6. Benjamini-Hochberg correction ──────────────────────────────────
    pvals = np.array([t["p_value"] for t in all_tests])
    qvals, reject = _benjamini_hochberg(pvals, alpha=0.05)
    for t, q in zip(all_tests, qvals):
        t["q_value_bh"] = float(q)
        t["significant_q005"] = bool(q < 0.05)

    # ── 7. Summary ───────────────────────────────────────────────────────
    print(f"\n[results] {len(all_tests)} tests, {sum(1 for t in all_tests if t['significant_q005'])} significant at q<0.05")
    for t in sorted(all_tests, key=lambda x: x["q_value_bh"]):
        sig = "✓" if t["significant_q005"] else " "
        print(f"  {sig} Archetype{t['archetype']} × {t['clinical_endpoint']:15s}: "
              f"OR={t['odds_ratio']:.2f} 95%CI=[{t['ci_lo']:.2f},{t['ci_hi']:.2f}]  "
              f"p={t['p_value']:.3f}  q={t['q_value_bh']:.3f}")

    # Discovery fold: significant results
    discovery_results = [t for t in all_tests if t["significant_q005"]]

    return {
        "experiment": "clinical_enrichment",
        "cancer": cancer,
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "n_patients": int(len(merged)),
        "n_with_stage": int(n_with_stage),
        "n_with_grade": int(n_with_grade),
        "n_archetypes": K,
        "high_threshold": high_threshold,
        "all_tests": all_tests,
        "n_tests": len(all_tests),
        "n_significant_q005": sum(1 for t in all_tests if t["significant_q005"]),
        "discovery_results": discovery_results,
        "passed": bool(discovery_results),
        "verdict": (
            f"{len(discovery_results)} significant associations at BH q<0.05"
            if discovery_results
            else "No significant associations at BH q<0.05"
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACT-Surv v5 Clinical Enrichment")
    p.add_argument("--cancer", default="blca")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true",
                   help="Run all 5 folds and aggregate")
    p.add_argument("--checkpoint", default="",
                   help="Explicit checkpoint path (default: auto from results/)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "act_surv_v5" / "clinical"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu")
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_folds_results: list[dict] = []

    if args.all_folds:
        folds = list(range(5))
    else:
        folds = [args.fold]

    for fold in folds:
        try:
            if args.checkpoint:
                ckpt = Path(args.checkpoint)
            else:
                ckpt = _find_ckpt_path(args.cancer, fold)
            result = run_enrichment(ckpt, args.cancer, fold, device)
            all_folds_results.append(result)
        except Exception as e:
            all_folds_results.append({
                "experiment": "clinical_enrichment",
                "cancer": args.cancer,
                "fold": fold,
                "skipped": True,
                "reason": str(e),
                "passed": None,
            })

    # ── Aggregate across folds ──────────────────────────────────────────
    agg = {
        "timestamp": stamp,
        "device": device,
        "cancer": args.cancer,
        "per_fold": all_folds_results,
    }

    # Count significant results across folds
    total_sig = sum(
        r.get("n_significant_q005", 0) for r in all_folds_results
        if not r.get("skipped")
    )
    print(f"\n{'='*60}")
    print(f"Aggregate: {total_sig} significant associations across {len(folds)} folds")
    print(f"{'='*60}")

    # Save JSON
    out_json = output_dir / f"clinical_enrichment_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2, default=str)

    # Save readable summary
    out_md = output_dir / f"clinical_enrichment_{stamp}.md"
    lines = [f"# ACT-Surv v5 Clinical Enrichment — {args.cancer}\n"]
    lines.append(f"**Timestamp:** {stamp}  |  **Device:** {device}\n")
    for r in all_folds_results:
        fold = r.get("fold", "?")
        if r.get("skipped"):
            lines.append(f"\n## Fold {fold} — SKIPPED: {r.get('reason', 'n/a')}\n")
            continue
        lines.append(f"\n## Fold {fold} — {r['n_patients']} patients, K={r['n_archetypes']}\n")
        lines.append(f"| Arch | Endpoint | OR | 95%CI | p | q(BH) | Sig |")
        lines.append("|------|----------|----|-----------|---|--------|-----|")
        for t in sorted(r["all_tests"], key=lambda x: x["q_value_bh"]):
            sig = "✓" if t["significant_q005"] else ""
            lines.append(
                f"| A{t['archetype']} | {t['clinical_endpoint']} | "
                f"{t['odds_ratio']:.2f} | [{t['ci_lo']:.2f},{t['ci_hi']:.2f}] | "
                f"{t['p_value']:.4f} | {t['q_value_bh']:.4f} | {sig} |"
            )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nResults saved:")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
