#!/usr/bin/env python3
"""Compact fold-event tracker: print only fold start and last val-epoch per fold.

Runs forever; refreshes every REFRESH seconds.
"""

import os
import re
import time
import subprocess
import sys
import shutil
from datetime import datetime

LOG = "/home/ubuntu/SurvOT-Rank/logs/v45v2_blca_clinical.log"
REFRESH = 15

RE_FOLD = re.compile(r"^\[Fold (\d+)\] start", re.M)
RE_VAL = re.compile(
    r"^\[Epoch (\d+)\]\s+val cindex=([0-9.\-]+)\s+ipcw=([0-9.\-]+)\s+IBS=([0-9.\-]+)\s+iauc=([0-9.\-]+)",
    re.M,
)
RE_BATCH = re.compile(
    r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+(\d+)%\|"
)


def read_log():
    if not os.path.exists(LOG):
        return ""
    try:
        with open(LOG, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def fold_running_fold(content):
    """Walk through content and remember fold index for each val epoch."""
    fold_idx = {}
    current_fold = 0
    for line in content.splitlines():
        m = RE_FOLD.search(line)
        if m:
            current_fold = int(m.group(1))
        fold_idx.setdefault(current_fold, []).append(line)
    return fold_idx


def find_pid():
    try:
        out = subprocess.run(
            ["pgrep", "-f", "-a", "python.*survot_rank.cli train"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()
        for line in out:
            if "/bin/bash" in line or "shopt" in line:
                continue
            parts = line.split(None, 1)
            if parts:
                try:
                    return int(parts[0])
                except ValueError:
                    pass
    except Exception:
        pass
    return None


def main():
    last_size = 0
    last_best = None
    started_at = time.time()
    print(f"=== fold-event tracker (log={LOG}) ===", flush=True)
    while True:
        content = read_log()
        if len(content) > last_size:
            last_size = len(content)
        elif len(content) < last_size:
            last_size = len(content)
            print("---- log rotated/truncated, resetting ----", flush=True)

        # partition val by fold
        per_fold = {}
        current_fold = 0
        for line in content.splitlines():
            m = RE_FOLD.search(line)
            if m:
                current_fold = int(m.group(1))
            m = RE_VAL.search(line)
            if m:
                per_fold.setdefault(current_fold, []).append(m.groups())

        pid = find_pid()
        alive = pid is not None
        free_gb = shutil.disk_usage(os.path.dirname(LOG)).free / (1024 ** 3)

        # Determine current fold from latest batch progress.
        m = list(RE_BATCH.finditer(content))
        cur_fold = int(m[-1].group(1)) if m else max(per_fold.keys()) if per_fold else 0
        cur_e = int(m[-1].group(2)) if m else 0
        cur_tot = int(m[-1].group(3)) if m else 0
        cur_pct = int(m[-1].group(4)) if m else 0

        out_lines = []
        out_lines.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] pid={pid} alive={alive} "
            f"free={free_gb:.1f}GB now=Fold{cur_fold} Ep{cur_e}/{cur_tot} {cur_pct}% "
            f"uptime={int(time.time()-started_at)}s"
        )
        for k in sorted(per_fold.keys()):
            runs = per_fold[k]
            if not runs:
                continue
            epochs = [int(r[0]) for r in runs]
            cindex = [float(r[1]) for r in runs]
            best_i = max(range(len(cindex)), key=lambda i: cindex[i])
            mark = "*" if k == cur_fold else " "
            status = "running" if k == cur_fold and k == max(per_fold.keys()) and cur_e > 0 else (
                "running" if k == cur_fold and len(epochs) < cur_tot else "done "
            )
            out_lines.append(
                f"   {mark} Fold {k}: {status:<7} epochs_done={len(epochs):2d}/{cur_tot or 30:2d}  "
                f"last_cindex={cindex[-1]:.4f} (ep{epochs[-1]:2d})  "
                f"best={cindex[best_i]:.4f} (ep{epochs[best_i]:2d})"
            )
        out_lines.append("-" * 78)
        sys.stdout.write("\n".join(out_lines) + "\n")
        sys.stdout.flush()
        time.sleep(REFRESH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
