#!/usr/bin/env python3
"""Summarise DCT v3.8.2 C-index and learned auxiliary weights."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LOSS_NAMES = ("ipcw_rank", "direction", "dose", "reconfiguration", "mgptr")


def _csv_row(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                row["epoch"] = int(row["epoch"])
                row["val_cindex"] = float(row["val_cindex"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    if not rows:
        return None, None
    return max(rows, key=lambda item: item["val_cindex"]), rows[-1]


def _weight(row, name):
    if row is None:
        return None
    value = row.get(f"train_v382_adaptive_weight_{name}")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scan(root: Path, cancers, variants, folds):
    records = []
    for variant in variants:
        for cancer in cancers:
            directory = root / variant / cancer
            for fold in folds:
                paths = list(directory.rglob(f"epoch_curve_fold{fold}.csv"))
                best = last = None
                if paths:
                    best, last = _csv_row(paths[0])
                records.append(
                    {
                        "variant": variant,
                        "cancer": cancer,
                        "fold": fold,
                        "best": best,
                        "last": last,
                        "weights": {
                            name: _weight(best, name) for name in LOSS_NAMES
                        },
                    }
                )
    return records


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/dct_v3.8.2_20ep/robust"),
    )
    parser.add_argument("--cancers", default="blca,brca")
    parser.add_argument("--variants", default="base,fixed_full,adaptive_full")
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    cancers = [item.strip().lower() for item in args.cancers.split(",") if item]
    variants = [item.strip() for item in args.variants.split(",") if item]
    folds = [int(item) for item in args.folds.split(",") if item.strip()]
    records = scan(args.root, cancers, variants, folds)

    print(f"DCT v3.8.2 results: {args.root}")
    for cancer in cancers:
        print(f"\n[{cancer.upper()}]")
        for variant in variants:
            selected = [
                row
                for row in records
                if row["cancer"] == cancer and row["variant"] == variant
            ]
            complete = [row for row in selected if row["best"] is not None]
            scores = [row["best"]["val_cindex"] for row in complete]
            score = _mean(scores)
            rendered = "pending" if score is None else f"mean={score:.4f} ({len(scores)} folds)"
            print(f"  {variant:<16} {rendered}")
            if variant == "adaptive_full" and complete:
                means = {
                    name: _mean([row["weights"][name] for row in complete])
                    for name in LOSS_NAMES
                }
                weight_text = " ".join(
                    f"{name}={value:.4f}"
                    for name, value in means.items()
                    if value is not None
                )
                if weight_text:
                    print(f"    best-epoch weights: {weight_text}")


if __name__ == "__main__":
    main()
