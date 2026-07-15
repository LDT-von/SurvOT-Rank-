#!/usr/bin/env python3
"""
Extract per-fold best-epoch metrics from epoch_curve CSV files,
generate unified per-fold and per-method summary tables.

Usage: python scripts/generate_final_results.py
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/ubuntu/SurvOT-Rank")
DATA1 = Path("/data1/sweep_results_30ep")

METHODS = [
    # (method, config, seed, split_protocol, source_dirs)
    # SurvOT-Rank fix methods (binning B)
    ("dct_fix", "configs/fix/distributional_counterfactual_transport_fix_blca.yaml", 14623, "B global qcut fixed",
     list((REPO / "results/distributional_counterfactual_transport_fix_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    ("catet_fix", "configs/fix/censoring_aware_temporal_evidence_transport_fix_blca.yaml", 27785, "B global qcut fixed",
     list((REPO / "results/censoring_aware_temporal_evidence_transport_fix_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    ("faithful_fix", "configs/fix/faithful_evidence_transport_fix_blca.yaml", 23541, "B global qcut fixed",
     list((REPO / "results/faithful_evidence_transport_fix_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    ("rg_et_fix", "configs/fix/rank_guided_event_transport_fix_blca.yaml", 19552, "B global qcut fixed",
     list((REPO / "results/rank_guided_event_transport_fix_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    # SurvOT-Rank norank methods (binning B)
    ("v50_norank", "configs/fix/v50_norank_blca.yaml", 22646, "B global qcut fixed",
     list((REPO / "results/v50_norank_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    ("v45_norank", "configs/fix/v45_norank_blca.yaml", 6792, "B global qcut fixed",
     list((REPO / "results/v45_norank_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    ("v45v2_norank", "configs/fix/v45v2_norank_blca.yaml", 323, "B global qcut fixed",
     list((REPO / "results/v45v2_norank_5fold_noseed/blca").glob("**/epoch_curve_fold*.csv"))),
    # V60 (binning B)
    ("V60", "configs/v60_ot_event_rank_blca.yaml", 3, "B global qcut fixed",
     list((REPO / "results/v60_ot_event_rank/blca").glob("**/epoch_curve_fold*.csv"))),
    # V51 SlimBridge from newSlotSPE (binning B)
    ("V51", "configs/v51_slimbridge.yaml", 3, "B global qcut fixed",
     list((DATA1 / "v51_slimbridge_seed3/blca").glob("**/epoch_curve_fold*.csv"))),
    ("V51", "configs/v51_slimbridge.yaml", 5, "B global qcut fixed",
     list((DATA1 / "v51_slimbridge_seed5/blca").glob("**/epoch_curve_fold*.csv"))),
    # Stagewise (binning A*, old fold-aware)
    ("stagewise_prognostic_transport", "configs/stagewise_prognostic_transport_blca.yaml", 3,
     "A* old fold-aware qcut",
     list((REPO / "results/stagewise_prognostic_transport_blca/blca").glob("**/epoch_curve_fold*.csv"))),
]


def extract_fold(filename):
    """Extract fold number from epoch_curve_fold{N}.csv"""
    name = Path(filename).stem  # epoch_curve_fold0
    return int(name.replace("epoch_curve_fold", ""))


def read_epoch_curve(filepath):
    """Read epoch_curve CSV and return list of dicts."""
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_best_epoch(rows):
    """Find the epoch with max val_cindex. Returns (epoch_idx, row_dict)."""
    best = None
    best_cidx = -1.0
    for i, row in enumerate(rows):
        cidx = float(row["val_cindex"])
        if cidx > best_cidx:
            best_cidx = cidx
            best = (i, row)
    return best


def compute_last5(rows):
    """Compute mean val_cindex of last 5 epochs."""
    last5 = rows[-5:]
    vals = [float(r["val_cindex"]) for r in last5]
    return sum(vals) / len(vals)


def main():
    os.makedirs(REPO / "results", exist_ok=True)

    per_fold_rows = []
    per_method_data = defaultdict(list)  # key=(method, seed) → list of fold dicts

    for method, config, seed, split_protocol, csv_files in METHODS:
        # Group by fold
        fold_files = defaultdict(list)
        for f in csv_files:
            fold = extract_fold(f)
            fold_files[fold].append(f)

        available_folds = sorted(fold_files.keys())
        if not available_folds:
            if method == "V51":
                # Distinguish seed3/seed5 in display name
                display_method = f"V51 seed{seed}"
            else:
                display_method = method
            per_fold_rows.append({
                "method": display_method,
                "config": config,
                "seed": seed,
                "fold": "N/A",
                "best_epoch": "N/A",
                "val_cindex": "N/A",
                "val_cindex_ipcw": "N/A",
                "val_IBS": "N/A",
                "val_iauc": "N/A",
                "last5_val_cindex": "N/A",
                "train_cindex_last": "N/A",
                "split_protocol": split_protocol,
                "notes": "missing_raw_curve",
            })
            continue

        partial = len(available_folds) < 5
        expected_folds = set(range(5))
        missing_folds = expected_folds - set(available_folds)

        for fold in available_folds:
            filepath = fold_files[fold][0]  # take first if multiple
            rows = read_epoch_curve(filepath)

            if not rows:
                continue

            best_idx, best_row = find_best_epoch(rows)
            last5 = compute_last5(rows)

            # train_cindex may or may not be present
            train_cindex_last = "N/A"
            if "train_cindex" in rows[-1]:
                train_cindex_last = rows[-1]["train_cindex"]
            elif "train_cindex" in best_row:
                train_cindex_last = best_row.get("train_cindex", "N/A")
            else:
                # Check last row for any train column
                for key in rows[-1]:
                    if "train" in key.lower() and "cindex" in key.lower():
                        train_cindex_last = rows[-1][key]
                        break

            if method == "V51":
                display_method = f"V51 seed{seed}"
            else:
                display_method = method

            notes = ""
            if partial:
                missing_str = ",".join([str(f) for f in sorted(missing_folds)])
                notes = f"partial_folds_present={'fold'+','.join([str(f) for f in available_folds])}_missing={missing_str}"

            row_data = {
                "method": display_method,
                "config": config,
                "seed": seed,
                "fold": fold,
                "best_epoch": best_idx,
                "val_cindex": float(best_row["val_cindex"]),
                "val_cindex_ipcw": float(best_row["val_cindex_ipcw"]),
                "val_IBS": float(best_row["val_IBS"]),
                "val_iauc": float(best_row["val_iauc"]),
                "last5_val_cindex": last5,
                "train_cindex_last": train_cindex_last,
                "split_protocol": split_protocol,
                "notes": notes,
            }
            per_fold_rows.append(row_data)
            per_method_data[(display_method, seed)].append(row_data)

    # --- Write per_fold_best_metrics.csv ---
    fieldnames = [
        "method", "config", "seed", "fold", "best_epoch",
        "val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc",
        "last5_val_cindex", "train_cindex_last", "split_protocol", "notes",
    ]
    out_path = REPO / "results" / "per_fold_best_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in per_fold_rows:
            writer.writerow(row)
    print(f"Wrote {len(per_fold_rows)} rows → {out_path}")

    # --- Write per_method_summary.csv ---
    summary_rows = []
    for (method, seed), fold_list in per_method_data.items():
        n = len(fold_list)
        cidx_vals = [r["val_cindex"] for r in fold_list if isinstance(r["val_cindex"], (int, float))]
        last5_vals = [r["last5_val_cindex"] for r in fold_list if isinstance(r["last5_val_cindex"], (int, float))]
        ipcw_vals = [r["val_cindex_ipcw"] for r in fold_list if isinstance(r["val_cindex_ipcw"], (int, float))]
        ibs_vals = [r["val_IBS"] for r in fold_list if isinstance(r["val_IBS"], (int, float))]
        iauc_vals = [r["val_iauc"] for r in fold_list if isinstance(r["val_iauc"], (int, float))]

        def mean_std(vals):
            if not vals:
                return "N/A", "N/A"
            m = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
                s = var ** 0.5
            else:
                s = 0.0
            return m, s

        cidx_m, cidx_s = mean_std(cidx_vals)
        last5_m, last5_s = mean_std(last5_vals)
        ipcw_m, ipcw_s = mean_std(ipcw_vals)
        ibs_m, ibs_s = mean_std(ibs_vals)
        iauc_m, iauc_s = mean_std(iauc_vals)

        gap = cidx_m - last5_m if isinstance(cidx_m, float) and isinstance(last5_m, float) else "N/A"

        summary_rows.append({
            "method": method,
            "config": fold_list[0]["config"],
            "seed": seed,
            "n_folds": n,
            "expected_folds": 5,
            "split_protocol": fold_list[0]["split_protocol"],
            "best_val_cindex_mean": f"{cidx_m:.4f}" if isinstance(cidx_m, float) else "N/A",
            "best_val_cindex_std": f"{cidx_s:.4f}" if isinstance(cidx_s, float) else "N/A",
            "last5_val_cindex_mean": f"{last5_m:.4f}" if isinstance(last5_m, float) else "N/A",
            "last5_val_cindex_std": f"{last5_s:.4f}" if isinstance(last5_s, float) else "N/A",
            "best_gap": f"{gap:.4f}" if isinstance(gap, float) else "N/A",
            "val_cindex_ipcw_mean": f"{ipcw_m:.4f}" if isinstance(ipcw_m, float) else "N/A",
            "val_IBS_mean": f"{ibs_m:.4f}" if isinstance(ibs_m, float) else "N/A",
            "val_iauc_mean": f"{iauc_m:.4f}" if isinstance(iauc_m, float) else "N/A",
            "notes": fold_list[0].get("notes", ""),
        })

    fieldnames_summary = [
        "method", "config", "seed", "n_folds", "expected_folds", "split_protocol",
        "best_val_cindex_mean", "best_val_cindex_std",
        "last5_val_cindex_mean", "last5_val_cindex_std", "best_gap",
        "val_cindex_ipcw_mean", "val_IBS_mean", "val_iauc_mean",
        "notes",
    ]
    out_path2 = REPO / "results" / "per_method_summary.csv"
    with open(out_path2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_summary, extrasaction='ignore')
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote {len(summary_rows)} rows → {out_path2}")

    # --- Generate markdown table for EXPERIMENT_SUMMARY.md ---
    print("\n=== Markdown Table for EXPERIMENT_SUMMARY.md ===")
    # Build fold0..fold4 column for each method
    methods_order = [
        ("dct_fix", None), ("catet_fix", None), ("faithful_fix", None),
        ("rg_et_fix", None), ("stagewise_prognostic_transport", None),
        ("v50_norank", None), ("v45_norank", None), ("v45v2_norank", None),
        ("V60", None),
        ("V51 seed3", None), ("V51 seed5", None),
    ]
    method_fold_map = {}
    for row in per_fold_rows:
        key = (row["method"], row["fold"])
        method_fold_map[key] = row

    header = "| method | fold0 | fold1 | fold2 | fold3 | fold4 |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for mname, _ in methods_order:
        vals = []
        for f in range(5):
            row = method_fold_map.get((mname, f))
            if row and isinstance(row.get("val_cindex"), (int, float)):
                epoch = row["best_epoch"]
                cidx = row["val_cindex"]
                vals.append(f"{cidx:.4f}@{epoch}")
            else:
                vals.append("—")
        lines.append(f"| {mname} | " + " | ".join(vals) + " |")

    md_table = "\n".join(lines)
    print(md_table)

    # Export as JSON for later use
    json_path = REPO / "results" / "per_fold_best_metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "per_fold": [{k: str(v) if isinstance(v, float) else v for k, v in row.items()}
                         for row in per_fold_rows],
            "per_method": summary_rows,
            "md_table": md_table,
        }, f, indent=2)
    print(f"\nJSON export → {json_path}")

    return per_fold_rows, summary_rows, md_table


if __name__ == "__main__":
    main()
