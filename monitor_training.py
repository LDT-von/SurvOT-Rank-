#!/usr/bin/env python3
"""Monitor SurvOT-Rank training in real time.

Usage:
    python monitor_training.py [--log PATH] [--refresh SECONDS] [--once]

Defaults:
    log = /home/ubuntu/SurvOT-Rank/logs/v45v2_blca_clinical.log
    refresh = 5 seconds
"""
import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta


DEFAULT_LOG = "/home/ubuntu/SurvOT-Rank/logs/v45v2_blca_clinical.log"
RESULT_DIR = "/home/ubuntu/SurvOT-Rank/results/v45v2_blca_clinical/blca/SurvOTRank_otehv2_rankevent_v2"


# ---------- regex matchers ----------
RE_START = re.compile(r"^\[Fold (\d+)\] start", re.M)
RE_VAL = re.compile(
    r"^\[Epoch (\d+)\]\s+val cindex=([0-9.\-]+)\s+ipcw=([0-9.\-]+)\s+IBS=([0-9.\-]+)\s+iauc=([0-9.\-]+)",
    re.M,
)
RE_TRAIN_LOSS = re.compile(r"^\[Epoch (\d+)\]\s+train_loss=([0-9.\-]+)\s+train_cindex=([0-9.\-]+)", re.M)
RE_BATCH_PROGRESS = re.compile(
    r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+(\d+)%\|[^|]+\|\s+(\d+)/(\d+)\s+\[<?(\d+):(\d+)<"
)
RE_PID_LINE = re.compile(r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+\d+%\|")


def humanize_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def disk_free_gb(path: str) -> float:
    try:
        u = shutil.disk_usage(path)
        return u.free / (1024 ** 3)
    except Exception:
        return -1.0


def parse_progress(progress_text: str):
    """Return (fold, epoch, total_epoch, pct, batch, total_batch, eta_sec) or None.

    Accepts both fresh entries like
        "[Fold 0] Epoch 10/30:  41%|...| 41/75 [00:29<..."
    and stale truncated ones emitted across tqdm rewrites (the trailing
    part may be missing). The first pattern still works because tqdm
    overwrites the bar in place via \r, so the file actually contains
    the LATEST bar on its own line.
    """
    # 1) Preferred: full bar with closing ] then timing.
    pattern_full = re.compile(
        r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+)\s+\[<?(\d+):(\d+)<(\d+):(\d+)"
    )
    m = pattern_full.search(progress_text)
    if m:
        fold, epoch, tot_epoch, pct, batch, tot_batch, mm_cur, ss_cur, mm_tot, ss_tot = m.groups()
        cur = int(mm_cur) * 60 + int(ss_cur)
        tot = int(mm_tot) * 60 + int(ss_tot)
        eta = max(tot - cur, 0)
        return int(fold), int(epoch), int(tot_epoch), int(pct), int(batch), int(tot_batch), eta

    # 2) Fallback: just look at the last "[Fold X] Epoch Y/Z: NN%|..."
    pattern_simple = re.compile(
        r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+(\d+)%\|"
    )
    ms = list(pattern_simple.finditer(progress_text))
    if ms:
        fold, epoch, tot_epoch, pct = ms[-1].groups()
        return int(fold), int(epoch), int(tot_epoch), int(pct), -1, -1, -1

    return None


def read_full_log(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def best_so_far(content: str):
    """Return best (val_cindex, epoch) over all finished val epochs in log."""
    best = (None, None)
    best_per_fold = {}
    current_fold = 0
    # Walk line by line to track fold.
    lines = content.splitlines()
    for i, line in enumerate(lines):
        m = RE_START.search(line)
        if m:
            current_fold = int(m.group(1))
        m = RE_VAL.search(line)
        if m:
            epoch = int(m.group(1))
            cidx = float(m.group(2))
            if best[0] is None or cidx > best[0]:
                best = (cidx, epoch)
            prev = best_per_fold.get(current_fold)
            if prev is None or cidx > prev[0]:
                best_per_fold[current_fold] = (cidx, epoch)
    return best, best_per_fold, current_fold


def result_csv_for_fold(run_dir: str, fold: int):
    p = os.path.join(run_dir, f"epoch_curve_fold{fold}.csv")
    return p if os.path.exists(p) else None


def find_run_dir(results_root: str) -> str | None:
    if not os.path.isdir(results_root):
        return None
    sub = os.listdir(results_root)
    if not sub:
        return None
    # pick latest modified leaf
    full = [os.path.join(results_root, s) for s in sub]
    full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    leaf = full[0]
    sub2 = [os.path.join(leaf, s) for s in os.listdir(leaf)] if os.path.isdir(leaf) else []
    if sub2:
        sub2.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        leaf = sub2[0]
    return leaf if os.path.isdir(leaf) else None


def bar(pct, width=30):
    n = int(round(width * pct / 100))
    return "[" + "#" * n + " " * (width - n) + "]"


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_training_pid() -> int | None:
    """Look up the actual python training PID, skipping nested shells."""
    import subprocess
    try:
        out = subprocess.run(
            ["pgrep", "-f", "-a", "python.*survot_rank.cli train"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    candidates = []
    for line in out:
        # Format: "PID ..." (with -a we get the full command).
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        # Skip shells that merely wrap the python invocation.
        if cmd.startswith("/bin/bash") or cmd.startswith("bash") or "shopt" in cmd:
            continue
        if "python" not in cmd:
            continue
        candidates.append((pid, cmd))
    if not candidates:
        return None
    # Prefer the lowest PID (the real python, parents will be shells).
    candidates.sort(key=lambda x: x[0])
    return candidates[0][0]


def proc_status_str(pid: int) -> str:
    """Return a short status: 'running', 'zombie', or 'gone'."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split()[1].rstrip(")").lstrip("(")
    except FileNotFoundError:
        return "gone"
    return "?"


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render(args, content: str, log_path: str, start_ts: float, last_pids: set[int]):
    pid = find_training_pid()
    alive = pid is not None and is_pid_alive(pid)
    state = proc_status_str(pid) if alive else "—"

    # parse progress from tail
    tail = content[-4000:]  # last few KB
    progress = parse_progress(tail)
    last_fold_for_progress = -1
    if progress:
        last_fold_for_progress = progress[0]  # fold index

    best, best_per_fold, last_fold_seen = best_so_far(content)
    run_dir = find_run_dir(RESULT_DIR)
    csv_files = []
    if run_dir:
        for f in sorted(os.listdir(run_dir)):
            if f.startswith("epoch_curve_fold"):
                csv_files.append(f)

    elapsed = time.time() - start_ts
    free_gb = disk_free_gb(os.path.dirname(log_path))

    clear_screen()
    print("=" * 78)
    print(f" SurvOT-Rank Training Monitor    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print(f" Log file:  {log_path}")
    print(f" Free disk: {free_gb:.2f} GB at {os.path.dirname(log_path)}")
    print(f" PID:       {pid if pid else 'N/A'} ({state})    alive: {alive}     monitor uptime: {humanize_seconds(elapsed)}")
    print("-" * 78)

    if progress:
        fold, epoch, total_epoch, pct, batch, total_batch, eta_sec = progress
        fold_str = f"Fold {fold}"
        epi_str = f"Epoch {epoch}/{total_epoch}"
        bat_str = f"batch {batch}/{total_batch}"
        eta_str = humanize_seconds(eta_sec)
        print(f" Current :  {fold_str:<8} {epi_str:<14} {bat_str:<14} ETA {eta_str}")
        print(f"            {bar(pct)} {pct}%")
    else:
        print(" Current :  (no progress line yet — trainer may still be initializing)")

    print("-" * 78)
    print(" Best val C-index so far")
    if best[0] is not None:
        print(f"   global best = {best[0]:.4f}  @ epoch {best[1]}")
    else:
        print("   (none yet)")
    if best_per_fold:
        print("   per-fold best:")
        for k in sorted(best_per_fold.keys()):
            v, e = best_per_fold[k]
            print(f"     fold {k}: {v:.4f}  @ epoch {e}")
    print("-" * 78)

    print(" Last 8 validation epochs (across all folds):")
    matches = list(RE_VAL.finditer(content))
    for m in matches[-8:]:
        epoch_n, cidx, ipcw, ibs, iauc = m.groups()
        print(f"   epoch {epoch_n:>3}  cindex={cidx}  ipcw={ipcw}  IBS={ibs}  iauc={iauc}")
    if not matches:
        print("   (none yet)")

    print("-" * 78)
    print(" Result CSVs written:")
    if csv_files:
        for f in csv_files:
            full = os.path.join(run_dir, f)
            sz = os.path.getsize(full)
            mt = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%H:%M:%S")
            print(f"   {f}   ({sz} bytes, mtime {mt})")
    else:
        print("   (none yet)")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--refresh", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="Print once and exit")
    args = ap.parse_args()

    log_path = args.log
    start_ts = time.time()
    if not os.path.exists(log_path):
        print(f"[monitor] log not found yet: {log_path} — waiting...")

    while True:
        content = read_full_log(log_path)
        try:
            render(args, content, log_path, start_ts, set())
        except Exception as e:
            print(f"[monitor] render error: {e}", flush=True)
        if args.once:
            return
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
