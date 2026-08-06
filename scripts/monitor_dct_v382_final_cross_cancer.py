#!/usr/bin/env python3
"""Show fold progress for the frozen DCT v3.8.2 cross-cancer queue."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.monitor_priority_queue import (
        _find_epoch_csv,
        _find_final_pkl,
        _parse_epoch_csv,
        _parse_final_pkl,
    )
    from scripts.run_dct_v382_final_cross_cancer import (
        DEFAULT_CANCERS,
        RESULT_ROOT,
        parse_cancers,
    )
except ModuleNotFoundError:
    from monitor_priority_queue import (
        _find_epoch_csv,
        _find_final_pkl,
        _parse_epoch_csv,
        _parse_final_pkl,
    )
    from run_dct_v382_final_cross_cancer import (
        DEFAULT_CANCERS,
        RESULT_ROOT,
        parse_cancers,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def fold_text(base: Path, fold: int) -> tuple[str, float | None]:
    final = _find_final_pkl(base, fold)
    if final:
        info = _parse_final_pkl(final)
        if info.get("done") and info.get("cindex") is not None:
            epoch = info.get("epoch")
            suffix = f"@{epoch}" if epoch is not None else ""
            score = float(info["cindex"])
            return f"{score:.4f}{suffix}", score
    curve = _find_epoch_csv(base, fold)
    if curve:
        info = _parse_epoch_csv(curve)
        if info.get("running"):
            score = info.get("best_cindex")
            epoch = info.get("best_epoch")
            last = info.get("last_epoch")
            if score is not None:
                return f"~{float(score):.4f}@{epoch}(e{last})", None
            return f"running(e{last})", None
    return "---", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cancers",
        type=parse_cancers,
        default=["blca", *DEFAULT_CANCERS],
    )
    args = parser.parse_args()
    root = REPO_ROOT / RESULT_ROOT

    print("DCT v3.8.2 fixed-full final cross-cancer progress")
    print("Cancer    Fold0          Fold1          Fold2          Fold3          Fold4          Mean")
    print("-" * 100)
    for cancer in args.cancers:
        cells: list[str] = []
        scores: list[float] = []
        base = root / cancer
        for fold in range(5):
            cell, score = fold_text(base, fold)
            cells.append(f"{cell:14s}")
            if score is not None:
                scores.append(score)
        mean = sum(scores) / len(scores) if scores else None
        mean_text = f"{mean:.4f} ({len(scores)}/5)" if mean is not None else "---"
        print(f"{cancer.upper():8s}  {''.join(cells)}{mean_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
