#!/usr/bin/env python3
"""监控 dct_v3.8_transport_consistency_20ep 训练进度"""

import csv
from pathlib import Path
from datetime import datetime

RESULTS = Path("results/dct_v3.8_transport_consistency_20ep/highscore")


def scan_results():
    rows = []
    for variant in ["full", "base"]:
        for cancer in ["brca", "blca"]:
            for fold in range(5):
                d = RESULTS / variant / cancer
                csv_files = list(d.rglob(f"epoch_curve_fold{fold}.csv"))
                row = {
                    "cancer": cancer.upper(),
                    "variant": variant,
                    "fold": fold,
                    "status": "pending",
                    "cindex": None,
                    "epoch": None,
                    "running_epoch": None,
                    "running_cindex": None,
                    "best_running_epoch": None,
                    "best_running_cindex": None,
                }
                if csv_files:
                    best_epoch, best_cindex, last_epoch_info = _parse_epoch_csv(csv_files[0])
                    row["running_epoch"] = last_epoch_info[0]
                    row["running_cindex"] = last_epoch_info[1]
                    row["best_running_epoch"] = best_epoch
                    row["best_running_cindex"] = best_cindex
                    if best_cindex is not None:
                        if last_epoch_info[0] >= 19:
                            row["status"] = "done"
                            row["cindex"] = best_cindex
                            row["epoch"] = best_epoch
                rows.append(row)
    return rows


def _parse_epoch_csv(path: Path):
    """Return (best_epoch, best_cindex, (last_epoch, last_cindex))."""
    best_epoch, best_cindex = -1, -1.0
    last_epoch, last_cindex = -1, -1.0
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                ep = int(r["epoch"])
                c = float(r["val_cindex"])
                last_epoch = ep
                last_cindex = c
                if c > best_cindex:
                    best_cindex = c
                    best_epoch = ep
    except Exception:
        pass
    if last_epoch < 19:
        return best_epoch, round(best_cindex, 4) if best_cindex > 0 else None, (last_epoch, last_cindex)
    return best_epoch, round(best_cindex, 4), (last_epoch, last_cindex)


def main():
    rows = scan_results()
    done = sum(1 for r in rows if r["status"] == "done")
    running = sum(1 for r in rows if r["status"] == "pending" and r["running_epoch"] is not None)
    total = len(rows)

    print(f"=== DCT v3.8 Highscore 20ep 进度  ({datetime.now().strftime('%H:%M:%S')}) ===")
    print(f"    完成: {done}/{total}  进行中: {running}")
    print()

    for variant in ["full", "base"]:
        print(f"--- {variant} ---")
        print(f"{'Cancer':<6} {'Fold 0':>10} {'Fold 1':>10} {'Fold 2':>10} {'Fold 3':>10} {'Fold 4':>10}  {'Avg C-Index':>12}")
        print("-" * 76)
        for cancer in ["BRCA", "BLCA"]:
            cindices = []
            parts = [f"{cancer:<6}"]
            for fold in range(5):
                r = next((r for r in rows if r["cancer"] == cancer and r["variant"] == variant and r["fold"] == fold), None)
                if r and r["cindex"] is not None:
                    cindices.append(r["cindex"])
                    parts.append(f"{r['cindex']:.4f}({r['epoch']:>2})".center(10))
                elif r and r["running_epoch"] is not None:
                    best = r.get("best_running_cindex", r["running_cindex"])
                    best_ep = r.get("best_running_epoch", r["running_epoch"])
                    if best and best != -1.0:
                        parts.append(f"{best:.4f}@{best_ep}".center(10))
                    else:
                        parts.append(f"e{r['running_epoch']}/{r['running_cindex']:.3f}".center(10))
                else:
                    parts.append("   ...    ".center(10))
            avg = sum(cindices) / len(cindices) if cindices else float("nan")
            parts.append(f"{avg:.4f}" if cindices else "  -")
            print("".join(parts))
        print()

    print("--- 汇总 ---")
    for cancer in ["BRCA", "BLCA"]:
        for variant in ["full", "base"]:
            folds = [r for r in rows if r["cancer"] == cancer and r["variant"] == variant]
            done_folds = [r for r in folds if r["cindex"] is not None]
            running_folds = [r for r in folds if r["cindex"] is None and r["running_epoch"] is not None]
            if done_folds:
                avg = sum(r["cindex"] for r in done_folds) / len(done_folds)
                msg = f"  {cancer} {variant}: avg={avg:.4f}  ({len(done_folds)}/5 folds)"
                if running_folds:
                    r = running_folds[0]
                    msg += f"  [fold{r['fold']} in epoch {r['running_epoch']}: {r['running_cindex']:.4f}]"
                print(msg)
            elif running_folds:
                r = running_folds[0]
                best = r.get("best_running_cindex")
                best_ep = r.get("best_running_epoch")
                msg = f"  {cancer} {variant}: fold{r['fold']} running (epoch {r['running_epoch']})"
                if best and best != -1.0:
                    msg += f" best={best:.4f}@{best_ep}"
                print(msg)
            else:
                print(f"  {cancer} {variant}: 尚未开始")


if __name__ == "__main__":
    main()
