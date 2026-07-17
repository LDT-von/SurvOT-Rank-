#!/usr/bin/env python3
"""Watch DCT v3 score diagnostics training progress.
Usage: python scripts/watch_progress.py [refresh_seconds]
"""
import csv
import glob
import os
import sys
import time
from datetime import datetime

# Always resolve to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dct_v3_score_diagnostics")

def read_curves():
    pattern = os.path.join(RESULTS_DIR, "*/*/SurvOTRank_distributional_counterfactual_transport/*/epoch_curve_fold*.csv")
    files = sorted(glob.glob(pattern))
    stats = {}
    for f in files:
        # path: results/dct_v3_score_diagnostics/{variant}/blca/.../{run_dir}/epoch_curve_fold{N}.csv
        rel = os.path.relpath(f, RESULTS_DIR)
        parts = rel.split(os.sep)
        variant = parts[0]
        fname = os.path.basename(f)
        fold = fname.replace("epoch_curve_fold", "").replace(".csv", "")

        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        n = len(rows)
        last = rows[-1]
        best = max(rows, key=lambda r: float(r.get("val_cindex", 0)))

        sorted_ci = sorted(rows, key=lambda r: float(r.get("val_cindex", 0)), reverse=True)
        best3 = sum(float(r["val_cindex"]) for r in sorted_ci[:3]) / min(3, n)
        last5_rows = rows[-5:] if n >= 5 else rows
        last5 = sum(float(r["val_cindex"]) for r in last5_rows) / len(last5_rows)

        stats[f"{variant}/f{fold}"] = {
            "variant": variant, "fold": fold, "ep": n,
            "best_ep": int(best["epoch"]),
            "best_c": float(best["val_cindex"]),
            "best_ipcw": float(best["val_cindex_ipcw"]),
            "best_ibs": float(best["val_IBS"]),
            "best_iauc": float(best["val_iauc"]),
            "best_loss": float(best["val_loss"]),
            "last_c": float(last["val_cindex"]),
            "last_loss": float(last["val_loss"]),
            "best3": best3, "last5": last5,
            "stage_risk": float(last.get("train_stage_risk", 0)),
            "active_stage": float(last.get("train_active_stage_fraction", 0)),
            "anchor_cov": float(last.get("train_anchor_coverage", 0)),
        }
    return stats

def print_table(stats):
    import shutil
    w = shutil.get_terminal_size((120, 40)).columns
    now = datetime.now().strftime("%H:%M:%S")

    print(f"\n{'=' * min(w, 120)}")
    print(f"  DCT v3 Score Diagnostics  —  {now}")
    print(f"{'=' * min(w, 120)}")

    hdr = (f"{'Variant':<16s} {'F':>2s} {'Ep':>3s} {'Best@':>5s} {'C-idx':>7s} "
           f"{'IPCW':>7s} {'IBS':>6s} {'iAUC':>7s} {'Loss':>6s} | "
           f"{'LastC':>7s} {'L.Loss':>6s} | {'B3':>6s} {'L5':>6s} | "
           f"{'StgR':>6s} {'Act%':>5s} {'Anc':>5s}")
    print(hdr)
    print("-" * min(w, 120))

    for variant in ["full", "no_anchor", "no_stage_risk", "evidence_cost"]:
        folds = {k: v for k, v in stats.items() if v["variant"] == variant}
        if not folds:
            print(f"  {variant:<14s}  (no data)")
            continue
        for fold in ["0", "2", "3"]:
            key = f"{variant}/f{fold}"
            if key in folds:
                s = folds[key]
                done = "✓" if s["ep"] >= 50 else ""
                loss_flag = " !" if s["last_loss"] > 2.5 else (" ~" if s["last_loss"] > 1.5 else "")
                line = (f"  {variant:<14s} {fold:>2s} {s['ep']:>3d}{done:<1s} "
                        f"@{s['best_ep']:>3d}  {s['best_c']:>6.4f} {s['best_ipcw']:>7.4f} "
                        f"{s['best_ibs']:>6.4f} {s['best_iauc']:>7.4f} {s['best_loss']:>6.2f} | "
                        f"{s['last_c']:>6.4f} {s['last_loss']:>6.2f}{loss_flag} | "
                        f"{s['best3']:>6.4f} {s['last5']:>6.4f} | "
                        f"{s['stage_risk']:>6.4f} {s['active_stage']:>4.0%} {s['anchor_cov']:>4.0%}")
                print(line)
            else:
                print(f"  {variant:<14s} {fold:>2s}   —  missing")

    print(f"{'=' * min(w, 120)}")

    # Per-variant summary
    print(f"\n{'Variant':<16s} {'Folds':>5s} {'Best μ':>7s} {'Best3 μ':>7s} {'Last5 μ':>7s} {'Gap':>7s}")
    print("-" * 56)
    for variant in ["full", "no_anchor", "no_stage_risk", "evidence_cost"]:
        folds = [v for v in stats.values() if v["variant"] == variant]
        if not folds:
            print(f"  {variant:<14s}  {'0':>5s}")
            continue
        bm = sum(v["best_c"] for v in folds) / len(folds)
        b3 = sum(v["best3"] for v in folds) / len(folds)
        l5 = sum(v["last5"] for v in folds) / len(folds)
        gap = bm - l5
        gap_flag = " !!!" if gap > 0.10 else ("  !" if gap > 0.06 else "")
        print(f"  {variant:<14s} {len(folds):>5d} {bm:>7.4f} {b3:>7.4f} {l5:>7.4f} {gap:>7.4f}{gap_flag}")

    # Missing
    expected = {"full": {"0", "2", "3"}, "no_anchor": {"0", "2", "3"},
                "no_stage_risk": {"0", "2", "3"}, "evidence_cost": {"0", "2", "3"}}
    missing = []
    for variant, exp_folds in expected.items():
        actual = {v["fold"] for k, v in stats.items() if v["variant"] == variant}
        for f in exp_folds - actual:
            missing.append(f"{variant}:f{f}")
        for k, v in stats.items():
            if v["variant"] == variant and v["ep"] < 50 and f"{variant}:f{v['fold']}" not in missing:
                missing.append(f"{variant}:f{v['fold']} ({v['ep']}ep)")

    if missing:
        print(f"\n  Missing/Incomplete: {', '.join(sorted(missing))}")
    else:
        print(f"\n  All 12 runs complete.")

def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    while True:
        stats = read_curves()
        os.system("clear")
        print_table(stats)
        time.sleep(interval)

if __name__ == "__main__":
    main()
