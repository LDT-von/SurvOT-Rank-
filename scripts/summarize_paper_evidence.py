#!/usr/bin/env python3
"""Summarize paper-evidence ablations and audits into one ledger.

Reads:

* ``results/dct_v3.8.2_paper_evidence/robust/<variant>/<cancer>/split_<fold>_results_final.pkl``
* ``results/dct_v3.8.2_paper_evidence/audit/<cancer>_fold<fold>/audit_metrics_fold<fold>.json``
* ``results/dct_v3.8.2_paper_evidence/audit/<cancer>_fold<fold>/sweep_metrics_fold<fold>.json``
* ``results/dct_v3.8.2_cross_cancer_prototype/robust/<pair>/fold<fold>/split_<fold>_results_final.pkl``

Produces one Markdown ledger (``PAPER_EVIDENCE_LEDGER.md``) plus a JSON file
(``paper_evidence_summary.json``) at the project root.

Usage::

  python scripts/summarize_paper_evidence.py
  python scripts/summarize_paper_evidence.py --root results/dct_v3.8.2_paper_evidence --out PAPER_EVIDENCE_LEDGER.md
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class EvidenceRow:
    variant: str
    cancer: str
    fold: int
    c_index: float | None = None
    c_index_ipcw: float | None = None
    n: int | None = None
    direction_rate: float | None = None
    direction_chance_gap: float | None = None
    reconfiguration_above_margin: float | None = None
    dose_monotone_rate: float | None = None
    notes: str = ""


def _load_pickle_c_index(path: Path) -> tuple[float | None, float | None, int | None]:
    if not path.exists():
        return None, None, None
    try:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
    except Exception as error:
        return None, None, None
    # split_1_results.pkl has structure {'case_id': {...}} or a list of metrics.
    if isinstance(data, dict):
        if "c_index" in data:
            return float(data["c_index"]), float(data.get("c_index_ipcw", 0.0)), int(data.get("n", 0))
        # patient_results: case_id -> {risk, censor, time}
        return None, None, len(data)
    return None, None, None


def collect_evidence(root: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    if not root.exists():
        return rows
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for cancer_dir in sorted(variant_dir.iterdir()):
            if not cancer_dir.is_dir():
                continue
            cancer = cancer_dir.name
            for fold_dir in sorted(cancer_dir.glob("split_*_results_final.pkl")):
                fold = int(fold_dir.stem.split("_")[1])
                c_index, c_index_ipcw, n = _load_pickle_c_index(fold_dir)
                rows.append(EvidenceRow(
                    variant=variant,
                    cancer=cancer,
                    fold=fold,
                    c_index=c_index,
                    c_index_ipcw=c_index_ipcw,
                    n=n,
                ))
    return rows


def collect_audits(audit_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Return {(variant, cancer, fold): {...audit metrics...}}."""
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not audit_root.exists():
        return out
    for audit_dir in sorted(audit_root.iterdir()):
        if not audit_dir.is_dir():
            continue
        for metrics_path in sorted(audit_dir.glob("audit_metrics_fold*.json")):
            fold = int(metrics_path.stem.split("fold")[-1])
            cancer_tag = audit_dir.name
            # The audit directory is named ``<cancer>_fold<fold>`` by the
            # launcher; ``<cancer>_fold<fold>`` maps to (variant="fixed", cancer, fold).
            # We treat every audit as belonging to ``fixed_full`` until paired
            # with its source variant.
            cancer = cancer_tag.split("_fold")[0]
            try:
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
            except Exception:
                continue
            sweep_path = metrics_path.parent / f"sweep_metrics_fold{fold}.json"
            if sweep_path.exists():
                try:
                    with open(sweep_path, "r", encoding="utf-8") as handle:
                        metrics["dose_monotonicity"] = json.load(handle).get("dose_monotonicity", {})
                except Exception:
                    pass
            out[("fixed_full", cancer, fold)] = metrics
    return out


def collect_cross_cancer(root: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    if not root.exists():
        return rows
    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir():
            continue
        if not pair_dir.name.startswith("src_"):
            continue
        for fold_dir in sorted(pair_dir.glob("fold*")):
            ckpt = fold_dir / "split_1_results_final.pkl"
            fold = int(fold_dir.name.replace("fold", ""))
            c_index, c_index_ipcw, n = _load_pickle_c_index(ckpt)
            rows.append(EvidenceRow(
                variant="cross_cancer_source",
                cancer=pair_dir.name.replace("src_", ""),
                fold=fold,
                c_index=c_index,
                c_index_ipcw=c_index_ipcw,
                n=n,
            ))
    target_root = root
    for pair_dir in sorted(target_root.iterdir()):
        if not pair_dir.is_dir() or pair_dir.name.startswith("src_"):
            continue
        if "_to_" not in pair_dir.name:
            continue
        for fold_dir in sorted(pair_dir.glob("fold*")):
            ckpt = fold_dir / "split_1_results_final.pkl"
            fold = int(fold_dir.name.replace("fold", ""))
            c_index, c_index_ipcw, n = _load_pickle_c_index(ckpt)
            source, _, target = pair_dir.name.partition("_to_")
            rows.append(EvidenceRow(
                variant=f"cross_cancer_{source}_to_{target}",
                cancer=target,
                fold=fold,
                c_index=c_index,
                c_index_ipcw=c_index_ipcw,
                n=n,
            ))
    return rows


def render_markdown(
    rows: list[EvidenceRow],
    audit_map: dict[tuple[str, str, int], dict[str, Any]],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# DCT v3.8.2 paper evidence ledger")
    lines.append("")
    lines.append(
        "This ledger collects the C-index numbers and audit metrics produced by the "
        "paper-evidence ablation launchers and the audit loader. It is the canonical "
        "machine-readable snapshot for the v3.8.2 paper. Numbers are pulled from the "
        "raw ``split_<fold>_results_final.pkl`` files and the audit JSON outputs; "
        "if a cell is empty the run has not completed yet."
    )
    lines.append("")
    lines.append("## A. Score-first ablation C-index (3 cancers × fold 1)")
    lines.append("")
    lines.append("| Variant | Cancer | Fold | C-index | C-index IPCW | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        if row.variant == "cross_cancer_source" or row.variant.startswith("cross_cancer_") and "_to_" in row.variant:
            continue
        c_index = f"{row.c_index:.4f}" if row.c_index is not None else "—"
        c_index_ipcw = f"{row.c_index_ipcw:.4f}" if row.c_index_ipcw is not None else "—"
        n = str(row.n) if row.n is not None else "—"
        lines.append(
            f"| `{row.variant}` | {row.cancer.upper()} | {row.fold} | "
            f"{c_index} | {c_index_ipcw} | {n} |"
        )
    lines.append("")
    lines.append("## B. Audit metrics (factual risk vs counterfactual risk)")
    lines.append("")
    if not audit_map:
        lines.append("_No audit results available yet. Run::")
        lines.append("")
        lines.append("    python scripts/audit_dct_v382.py audit --config <cfg> --checkpoint <ckpt> --fold <f> --output-dir <out>_")
    else:
        lines.append("| Cancer | Fold | direction rate | chance gap | reconfiguration > margin | dose monotone |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for (variant, cancer, fold), metrics in sorted(audit_map.items()):
            direction = metrics.get("direction_consistency", {})
            rate = direction.get("correct_rate")
            chance = direction.get("chance_gap")
            reconfig = metrics.get("reconfiguration", {}).get("above_margin_rate")
            dose = metrics.get("dose_monotonicity", {}).get("monotone_rate") if metrics.get("dose_monotonicity") else None
            rate_str = f"{rate:.4f}" if isinstance(rate, float) else "—"
            chance_str = f"{chance:.4f}" if isinstance(chance, float) else "—"
            reconfig_str = f"{reconfig:.4f}" if isinstance(reconfig, float) else "—"
            dose_str = f"{dose:.4f}" if isinstance(dose, float) else "—"
            lines.append(
                f"| {cancer.upper()} | {fold} | {rate_str} | {chance_str} | {reconfig_str} | {dose_str} |"
            )
    lines.append("")
    lines.append("## C. Cross-cancer shared prototype transfer (BLCA source → ?)")
    lines.append("")
    lines.append("| Pair | Target cancer | Fold | C-index (target) |")
    lines.append("|---|---|---:|---:|")
    for row in rows:
        if row.variant.startswith("cross_cancer_") and "_to_" in row.variant:
            c_index = f"{row.c_index:.4f}" if row.c_index is not None else "—"
            lines.append(
                f"| `{row.variant.replace('cross_cancer_', '')}` | {row.cancer.upper()} | {row.fold} | {c_index} |"
            )
    lines.append("")
    lines.append("## D. Pass criteria (paper-facing)")
    lines.append("")
    lines.append(
        "- Ablation 1 (`fixed_coupling`): C-index should drop ≥ 0.04 from fixed-full. "
        "Audit ``reconfiguration > margin`` should drop below 0.20."
    )
    lines.append(
        "- Ablation 2 (`random_anchors`): C-index should drop ≥ 0.03 from fixed-full. "
        "Audit ``direction rate`` should fall back to chance (≤ 0.55)."
    )
    lines.append(
        "- Ablation 4 (`null_calibration`): audit metrics should match the "
        "random-signal baseline (direction ≈ 0.50, dose monotone ≈ chance)."
    )
    lines.append(
        "- Ablation 5 (`stage_randomization`): C-index should drop ≥ 0.02 if exact "
        "edge placement matters."
    )
    lines.append(
        "- Cross-cancer transfer: target C-index ≥ 0.60 of source for the prototype "
        "to count as cross-cancer semantic."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize paper evidence")
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=REPO_ROOT / "results" / "dct_v3.8.2_paper_evidence" / "robust",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=REPO_ROOT / "results" / "dct_v3.8.2_paper_evidence" / "audit",
    )
    parser.add_argument(
        "--cross-cancer-root",
        type=Path,
        default=REPO_ROOT / "results" / "dct_v3.8.2_cross_cancer_prototype" / "robust",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "PAPER_EVIDENCE_LEDGER.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "paper_evidence_summary.json",
    )
    args = parser.parse_args()

    rows = collect_evidence(args.ablation_root)
    rows.extend(collect_cross_cancer(args.cross_cancer_root))
    audit_map = collect_audits(args.audit_root)
    render_markdown(rows, audit_map, args.out_md)

    summary = {
        "rows": [row.__dict__ for row in rows],
        "audits": {
            f"{variant}|{cancer}|{fold}": metrics
            for (variant, cancer, fold), metrics in audit_map.items()
        },
    }
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"[summary] markdown -> {args.out_md}")
    print(f"[summary] json    -> {args.out_json}")
    print(f"[summary] ablation rows: {len(rows)}")
    print(f"[summary] audit keys:   {len(audit_map)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
