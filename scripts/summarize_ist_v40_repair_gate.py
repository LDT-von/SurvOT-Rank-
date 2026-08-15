#!/usr/bin/env python3
"""Summarize the pre-registered IST v4.0 BLCA repair gate.

The decision rule is fixed before repaired results are inspected: the repaired
recipe passes only when its three-fold mean best validation C-index improves
over factual A by at least 0.005 and it improves at least two of folds 1/2/4.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean

try:
    from scripts.run_ist_v40_final_cross_cancer import REPAIRED_RESULT_ROOT
    from scripts.summarize_dct_v3_score_diagnostics import read_curve, summarize_curve
except (ModuleNotFoundError, ImportError):
    from run_ist_v40_final_cross_cancer import REPAIRED_RESULT_ROOT
    from summarize_dct_v3_score_diagnostics import read_curve, summarize_curve


GATE_FOLDS = (1, 2, 4)
FACTUAL_A = {1: 0.6884, 2: 0.6632, 4: 0.7701}
MIN_MEAN_GAIN = 0.005
MIN_IMPROVED_FOLDS = 2
CURVE_PATTERN = re.compile(r"epoch_curve_fold(\d+)\.csv$")


def discover_curves(root: Path) -> dict[int, Path]:
    curves: dict[int, Path] = {}
    for path in root.rglob("epoch_curve_fold*.csv"):
        match = CURVE_PATTERN.search(path.name)
        if not match:
            continue
        fold = int(match.group(1))
        if fold not in GATE_FOLDS:
            continue
        if fold in curves:
            raise ValueError(
                f"Duplicate repaired curves for fold {fold}: {curves[fold]}, {path}"
            )
        curves[fold] = path
    return curves


def evaluate_gate(curves: dict[int, Path]) -> dict[str, object]:
    missing = [fold for fold in GATE_FOLDS if fold not in curves]
    rows: list[dict[str, object]] = []
    for fold in GATE_FOLDS:
        if fold not in curves:
            continue
        curve_summary = summarize_curve(read_curve(curves[fold]))
        repaired = float(curve_summary["best_val_cindex"])
        baseline = FACTUAL_A[fold]
        rows.append(
            {
                "fold": fold,
                "factual_a": baseline,
                "repaired_best": repaired,
                "delta": repaired - baseline,
                "best_epoch": curve_summary["best_epoch"],
                "repaired_best3": curve_summary["best3_val_cindex"],
                "repaired_last5": curve_summary["last5_val_cindex"],
            }
        )

    factual_mean = mean(FACTUAL_A.values())
    repaired_mean = (
        mean(float(row["repaired_best"]) for row in rows) if not missing else None
    )
    improved_folds = sum(float(row["delta"]) > 0.0 for row in rows)
    threshold = factual_mean + MIN_MEAN_GAIN
    passed = bool(
        not missing
        and repaired_mean is not None
        and repaired_mean >= threshold
        and improved_folds >= MIN_IMPROVED_FOLDS
    )
    return {
        "rows": rows,
        "missing": missing,
        "factual_mean": factual_mean,
        "repaired_mean": repaired_mean,
        "threshold": threshold,
        "improved_folds": improved_folds,
        "passed": passed,
    }


def write_csv(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "fold",
                "factual_a",
                "repaired_best",
                "delta",
                "best_epoch",
                "repaired_best3",
                "repaired_last5",
            ),
        )
        writer.writeheader()
        writer.writerows(report["rows"])


def print_report(report: dict[str, object]) -> None:
    print("| fold | factual A | repaired best | delta | epoch | best3 | last5 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["rows"]:
        print(
            f"| {row['fold']} | {row['factual_a']:.4f} | "
            f"{row['repaired_best']:.4f} | {row['delta']:+.4f} | "
            f"{row['best_epoch']} | {row['repaired_best3']:.4f} | "
            f"{row['repaired_last5']:.4f} |"
        )
    if report["missing"]:
        missing = ", ".join(f"fold{fold}" for fold in report["missing"])
        print(f"\n[INCOMPLETE] Missing repaired curves: {missing}")
        return
    print(
        f"\nFactual A mean={report['factual_mean']:.4f}; "
        f"repaired mean={report['repaired_mean']:.4f}; "
        f"required mean>={report['threshold']:.4f}; "
        f"improved folds={report['improved_folds']}/{len(GATE_FOLDS)}."
    )
    if report["passed"]:
        print("[PASS] Complete BLCA folds 0/3, then run the locked cross-cancer recipe.")
    else:
        print("[STOP] Do not expand IST; retain it as a negative ablation/appendix result.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPAIRED_RESULT_ROOT / "blca",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Results root does not exist: {args.root}")
    report = evaluate_gate(discover_curves(args.root))
    output = args.output or args.root / "repair_gate_summary.csv"
    write_csv(report, output)
    print_report(report)
    print(f"Wrote {output}")
    return 2 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
