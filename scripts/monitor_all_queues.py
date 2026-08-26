#!/usr/bin/env python3
"""全队列监控：DCT v3.8.2 + IST v4.0 + 修复诊断 Gates"""

import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

dct_dir = Path("results/dct_v3.8.2/robust/fixed_full")
ist_dir = Path("results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only")
arc_dir = Path("results/archetypal_risk_composition_repaired")
v41_dir = Path("results/dct_v4.1_survival_evidence_ledger_repaired")

dct_cancers = ["blca", "skcm", "hnsc", "lusc", "kirc", "ucec"]
ist_cancers = ["blca", "skcm", "hnsc", "lusc", "kirc", "ucec"]


def scan(d: Path, cancers: list, folds: range):
    rows = []
    for c in cancers:
        for f in folds:
            csvs = list((d / c).rglob(f"epoch_curve_fold{f}.csv"))
            if not csvs:
                rows.append((c, f, "...", "-"))
                continue
            best_ep, best_ci, last_ep = None, -1.0, None
            with open(csvs[0]) as fh:
                for r in csv.DictReader(fh):
                    last_ep = int(r["epoch"])
                    ci = float(r.get("val_cindex", 0))
                    if ci > best_ci:
                        best_ci, best_ep = ci, last_ep
            done = last_ep is not None and last_ep >= 49
            if done:
                rows.append((c, f, "done", f"{best_ci:.4f}@{best_ep}"))
            else:
                rows.append((c, f, f"e{last_ep}", f"{best_ci:.4f}@{best_ep}"))
    return rows


now = datetime.now().strftime("%H:%M:%S")
print(f'  {"="*72}')
print(f"  全队列监控  {now}")
print(f'  {"="*72}')

# ── DCT ──
print("\n  ── DCT v3.8.2 fixed-full ──")
dct_rows = scan(dct_dir, dct_cancers, range(5))
for c in dct_cancers:
    parts = [r for r in dct_rows if r[0] == c]
    done_n = sum(1 for r in parts if r[2] == "done")
    run = [r for r in parts if r[2] not in ("done", "...")]
    pends = [r[1] for r in parts if r[2] == "..."]
    cis = [float(r[3].split("@")[0]) for r in parts if r[2] == "done" and "@" in r[3]]
    mean_str = f" mean={sum(cis)/len(cis):.4f}" if cis else ""
    print(f"  {c.upper():<8} {done_n}/{len(parts)}{mean_str}")
    for r in run:
        print(f'         F{r[1]} {r[2]:<10} best={r[3]}')
    if pends:
        pend_str = ",".join(str(x) for x in pends)
        print(f"         F{pend_str} pending")

# ── IST ──
print("\n  ── IST v4.0 abl_b cost-only ──")
ist_rows = scan(ist_dir, ist_cancers, range(5))
for c in ist_cancers:
    parts = [r for r in ist_rows if r[0] == c]
    done_n = sum(1 for r in parts if r[2] == "done")
    run = [r for r in parts if r[2] not in ("done", "...")]
    cis = [float(r[3].split("@")[0]) for r in parts if r[2] == "done" and "@" in r[3]]
    mean_str = f" mean={sum(cis)/len(cis):.4f}" if cis else ""
    print(f"  {c.upper():<8} {done_n}/{len(parts)}{mean_str}")
    for r in run:
        print(f'         F{r[1]} {r[2]:<10} best={r[3]}')

# ── Gates ──
print("\n  ── 修复诊断 Gates ──")
for label, d in [("ArcSurv repaired", arc_dir), ("v4.1 repaired", v41_dir)]:
    csvs = list(d.rglob("epoch_curve_fold*.csv"))
    if not csvs:
        print(f"  {label:<24} pending")
        continue
    best_ep, best_ci, last_ep = None, -1.0, None
    with open(csvs[0]) as fh:
        for r in csv.DictReader(fh):
            last_ep = int(r["epoch"])
            ci = float(r.get("val_cindex", 0))
            if ci > best_ci:
                best_ci, best_ep = ci, last_ep
    done = last_ep is not None and last_ep >= 49
    status = "done" if done else f"e{last_ep}/50"
    print(f"  {label:<24} {status}  best={best_ci:.4f}@{best_ep}")

# 进程
r = os.popen("ps aux | grep 'survot_rank.cli' | grep -v grep").read().strip()
is_running = "运行中" if r else "已结束"
print(f'\n  {"*"*18} 进程{is_running} {"*"*18}')

# 日志尾部
log = Path("/tmp/all_queues.log")
if log.exists():
    out = subprocess.run(["tail", "-4", "/tmp/all_queues.log"], capture_output=True, text=True)
    if out.stdout.strip():
        print(f"  [log] {out.stdout.strip()[-120:]}")
