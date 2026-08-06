#!/usr/bin/env python3
"""Show progress for the frozen IST-Surv v4.0 cross-cancer queue."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.monitor_dct_v382_final_cross_cancer import fold_text
    from scripts.run_ist_v40_final_cross_cancer import (
        DEFAULT_CANCERS,
        RESULT_ROOT,
        parse_cancers,
    )
except ModuleNotFoundError:
    from monitor_dct_v382_final_cross_cancer import fold_text
    from run_ist_v40_final_cross_cancer import (
        DEFAULT_CANCERS,
        RESULT_ROOT,
        parse_cancers,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancers", type=parse_cancers, default=list(DEFAULT_CANCERS))
    args = parser.parse_args()
    root = REPO_ROOT / RESULT_ROOT
    print("IST-Surv v4.0 staged cost-feedback-only cross-cancer progress")
    print("Cancer    Fold0          Fold1          Fold2          Fold3          Fold4          Mean")
    print("-" * 100)
    for cancer in args.cancers:
        cells: list[str] = []
        scores: list[float] = []
        for fold in range(5):
            cell, score = fold_text(root / cancer, fold)
            cells.append(f"{cell:14s}")
            if score is not None:
                scores.append(score)
        mean = sum(scores) / len(scores) if scores else None
        mean_text = f"{mean:.4f} ({len(scores)}/5)" if mean is not None else "---"
        print(f"{cancer.upper():8s}  {''.join(cells)}{mean_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
