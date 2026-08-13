#!/usr/bin/env python3
"""Monitor the three-method final cross-cancer queues."""

import csv
import os
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent

QUEUES = {
    "CA-PSA": {
        "root": "results/capsa_full_final",
    },
    "ArcSurv": {
        "root": "results/arcsurv_staged_final",
    },
    "CATET": {
        "root": "results/catet_repaired_final",
    },
}
CANCERS = ["blca", "skcm", "hnsc", "lusc", "kirc", "ucec"]
FOLDS = range(5)


def scan(root: str) -> dict:
    base = REPO / root
    results = {}
    for cancer in CANCERS:
        results[cancer] = {}
        for f in FOLDS:
            csvs = list((base / cancer).rglob(f"epoch_curve_fold{f}.csv"))
            if not csvs:
                results[cancer][f] = None
                continue
            best_ep, best_ci, last_ep = None, -1.0, None
            with open(csvs[0]) as fh:
                for r in csv.DictReader(fh):
                    last_ep = int(r["epoch"])
                    ci = float(r.get("val_cindex", 0))
                    if ci > best_ci:
                        best_ci, best_ep = ci, last_ep
            results[cancer][f] = (best_ci, best_ep, last_ep >= 49 if last_ep is not None else False)
    return results


def main():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"  {'='*72}")
    print(f"  三方法跨癌种队列监控  {now}")
    print(f"  {'='*72}")

    for method, cfg in QUEUES.items():
        data = scan(cfg["root"])
        total_done = sum(1 for c in CANCERS for f in FOLDS if data[c][f] and data[c][f][2])
        total = len(CANCERS) * len(FOLDS)
        print(f"\n  ── {method}  [{total_done}/{total}] ──")
        print(f"  {'Cancer':<8} {'F0':>10} {'F1':>10} {'F2':>10} {'F3':>10} {'F4':>10}  {'Mean':>10}")
        print(f"  {'-'*68}")
        for cancer in CANCERS:
            parts = []
            cis = []
            for f in FOLDS:
                v = data[cancer][f]
                if v is None:
                    parts.append("      ···")
                elif v[2]:
                    parts.append(f"{v[0]:10.4f}")
                    cis.append(v[0])
                else:
                    parts.append(f"{v[0]:.4f}@e{v[1]:<2}")
            mean = f"{sum(cis)/len(cis):.4f}" if cis else "  ···"
            print(f"  {cancer.upper():<8} {' '.join(parts)}  {mean:>10}")

    r = os.popen("ps aux | grep 'survot_rank.cli' | grep -v grep").read().strip()
    print(f'\n  {"*"*18} 进程{"运行中" if r else "已结束"} {"*"*18}')


if __name__ == "__main__":
    main()
