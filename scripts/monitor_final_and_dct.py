#!/usr/bin/env python3
"""监控三个 Final 方法(BLCA fold0) + DCT v3.8.2 提分闸门。"""

import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINAL_ROOT = REPO / "results/three_method_final"
GATE_ROOT = REPO / "results/dct_v3.8.2_score_gate"

FINAL_METHODS = {
    "capsa_final": "CA-PSA Final",
    "arcsurv_final": "ArcSurv Final",
    "catet_final": "CATET Final",
}
GATE_VARIANTS = {
    "patches4096": "patches 2048->4096",
    "grad_accum4": "grad_accum 1->4",
    "slot_iters5": "slot_iters 3->5",
    "lr2e4": "lr 5e-4->2e-4",
}
GATE_CANCERS = ("blca", "kirc", "skcm")
GATE_FOLDS = (1, 2, 4)


def parse_epoch_csv(path: Path) -> dict:
    best_ep, best_ci, last_ep = None, -1.0, None
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                last_ep = int(r["epoch"])
                ci = float(r.get("val_cindex", r.get("val_c_index", 0.0)))
                if ci > best_ci:
                    best_ci, best_ep = ci, last_ep
    except Exception:
        return {}
    return {"best_ep": best_ep, "best_ci": best_ci, "last_ep": last_ep}


def fmt_cell(info: dict, max_ep: int = 50) -> str:
    if not info:
        return "···"
    best_ci = info["best_ci"]
    best_ep = info["best_ep"]
    last_ep = info["last_ep"]
    if last_ep is not None and last_ep >= max_ep - 1:
        return f"{best_ci:.4f}@{best_ep}"
    return f"{best_ci:.4f}@{best_ep}/{last_ep}"


def scan_final():
    rows = []
    for method, label in FINAL_METHODS.items():
        base = FINAL_ROOT / method / "blca"
        cell = "···"
        done = False
        csvs = list(base.rglob("epoch_curve_fold0.csv"))
        if csvs:
            info = parse_epoch_csv(csvs[0])
            cell = fmt_cell(info)
            done = info.get("last_ep", 0) and info["last_ep"] >= 49
        rows.append((label, cell, done))
    return rows


def scan_gate():
    rows = []
    for variant, label in GATE_VARIANTS.items():
        for cancer in GATE_CANCERS:
            base = GATE_ROOT / variant / cancer
            cells = []
            done_count = 0
            for fold in GATE_FOLDS:
                csvs = list(base.rglob(f"epoch_curve_fold{fold}.csv"))
                if not csvs:
                    cells.append("···")
                    continue
                info = parse_epoch_csv(csvs[0])
                cells.append(fmt_cell(info))
                if info.get("last_ep", 0) and info["last_ep"] >= 49:
                    done_count += 1
            rows.append((label, cancer.upper(), cells, done_count))
    return rows


def main():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*84}")
    print(f"  三个 Final (BLCA fold0) + DCT 提分闸门监控  {now}")
    print(f"{'='*84}")

    print("\n  ── 三个 Final 方法 (BLCA fold0) ──")
    print(f"  {'方法':<18} {'fold0 (best@ep/进度)':<24} 状态")
    print(f"  {'-'*60}")
    final_rows = scan_final()
    for label, cell, done in final_rows:
        status = "✅" if done else ("🔄" if cell != "···" else "⏳")
        print(f"  {label:<18} {cell:<24} {status}")

    print("\n  ── DCT v3.8.2 提分闸门 (fold 1/2/4) ──")
    print(f"  {'变体':<22} {'癌症':<7} {'F1':<18} {'F2':<18} {'F4':<18} 完成")
    print(f"  {'-'*80}")
    gate_rows = scan_gate()
    for label, cancer, cells, done_count in gate_rows:
        f1, f2, f4 = cells
        print(f"  {label:<22} {cancer:<7} {f1:<18} {f2:<18} {f4:<18} {done_count}/3")

    # 进程
    r = os.popen("ps aux | grep 'survot_rank.cli' | grep -v grep").read().strip()
    print(f"\n  {'*'*18} 进程{'运行中' if r else '已结束'} {'*'*18}")
    # 日志尾部
    log = Path("/tmp/final_plus_dct.log")
    if log.exists():
        out = subprocess.run(["tail", "-3", str(log)], capture_output=True, text=True)
        if out.stdout.strip():
            print(f"  [log] {out.stdout.strip()[-140:]}")


if __name__ == "__main__":
    main()
