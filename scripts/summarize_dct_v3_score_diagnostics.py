"""Summarize reproducible DCT v3 score-diagnostic epoch curves."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

CURVE_PATTERN = re.compile(r"epoch_curve_fold(\d+)\.csv$")
METRIC_COLUMNS = ("val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc")


def read_curve(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            try:
                row = {"epoch": int(raw["epoch"])}
                for column in METRIC_COLUMNS:
                    row[column] = float(raw[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid epoch curve row in {path}: {raw}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty epoch curve: {path}")
    return rows


def summarize_curve(rows: list[dict[str, float]]) -> dict[str, float | int | bool]:
    best_index = max(range(len(rows)), key=lambda index: rows[index]["val_cindex"])
    best = rows[best_index]
    last5 = rows[-5:]
    best3 = rows[max(0, best_index - 1): min(len(rows), best_index + 2)]
    last5_mean = mean(row["val_cindex"] for row in last5)
    return {
        "n_epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_val_cindex": best["val_cindex"],
        "best3_val_cindex": mean(row["val_cindex"] for row in best3),
        "last5_val_cindex": last5_mean,
        "best_gap": best["val_cindex"] - last5_mean,
        "val_cindex_ipcw_at_best": best["val_cindex_ipcw"],
        "val_IBS_at_best": best["val_IBS"],
        "val_iauc_at_best": best["val_iauc"],
        "best_near_end": best_index >= len(rows) - 3,
    }


def discover_variant_curves(variant_dir: Path) -> dict[int, Path]:
    curves: dict[int, Path] = {}
    for path in variant_dir.rglob("epoch_curve_fold*.csv"):
        match = CURVE_PATTERN.search(path.name)
        if not match:
            continue
        fold = int(match.group(1))
        if fold in curves:
            raise ValueError(f"Duplicate fold {fold} curves for {variant_dir}: {curves[fold]}, {path}")
        curves[fold] = path
    return curves


def collect_rows(root: Path, expected_folds: Iterable[int]) -> list[dict[str, object]]:
    expected = list(expected_folds)
    output: list[dict[str, object]] = []
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        curves = discover_variant_curves(variant_dir)
        for fold in expected:
            base: dict[str, object] = {"variant": variant_dir.name, "fold": fold}
            if fold not in curves:
                output.append({**base, "status": "missing"})
            else:
                output.append({**base, **summarize_curve(read_curve(curves[fold])), "status": "ok"})
    return output


FIELDNAMES = (
    "variant", "fold", "n_epochs", "best_epoch", "best_val_cindex", "best3_val_cindex",
    "last5_val_cindex", "best_gap", "val_cindex_ipcw_at_best", "val_IBS_at_best",
    "val_iauc_at_best", "best_near_end", "status",
)


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows: list[dict[str, object]]) -> None:
    print("| variant | folds | best mean | best3 mean | last5 mean | gap mean |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for variant in sorted({str(row["variant"]) for row in rows}):
        ok_rows = [row for row in rows if row["variant"] == variant and row["status"] == "ok"]
        if not ok_rows:
            print(f"| {variant} | 0 | - | - | - | - |")
            continue
        values = lambda name: mean(float(row[name]) for row in ok_rows)
        print(f"| {variant} | {len(ok_rows)} | {values('best_val_cindex'):.4f} | {values('best3_val_cindex'):.4f} | {values('last5_val_cindex'):.4f} | {values('best_gap'):.4f} |")
    missing = [f"{row['variant']}:fold{row['fold']}" for row in rows if row["status"] == "missing"]
    if missing:
        print("Missing curves: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/dct_v3_score_first_diagnostics"))
    parser.add_argument("--expected-folds", default="0,2,3")
    parser.add_argument("--output", default="dct_v3_score_summary.csv")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Results root does not exist: {args.root}")
    expected_folds = [int(value.strip()) for value in args.expected_folds.split(",") if value.strip()]
    rows = collect_rows(args.root, expected_folds)
    if not rows:
        raise SystemExit(f"No variant directories found below: {args.root}")
    output = args.root / args.output
    write_summary(rows, output)
    print_report(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
