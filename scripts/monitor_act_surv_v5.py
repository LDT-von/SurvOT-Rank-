#!/usr/bin/env python3
"""Watch the ACT-Surv v5 5-fold (BLCA + HNSC) background run.

Reads /tmp/full_run.log (the launcher stdout/stderr log), watches
results/act_surv_v5/full_run/, and prints a compact table that updates
once per refresh interval. Designed to run in a separate terminal while
the launcher is doing its 10 jobs in the background.

Usage:
    python scripts/monitor_act_surv_v5.py               # 30s refresh, forever
    python scripts/monitor_act_surv_v5.py --interval 60 # 60s refresh
    python scripts/monitor_act_surv_v5.py --once        # print once, exit

What it shows (each refresh):
    * job state for each of the 10 jobs (BLCA/HNSC fold 0..4):
        - status:   running | done | failed | pending
        - best epoch val_cindex (extracted from epoch_curve_foldN.csv)
        - progress: current epoch / 50 (extracted from launcher log)
    * launcher-level: process alive?, which fold is running now
    * GPU snapshot (memory, util)
    * any ERROR lines since the last refresh
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path("/tmp/full_run.log")
RESULT_ROOT = REPO_ROOT / "results" / "act_surv_v5" / "full_run"
EXPECTED = [("blca", n) for n in range(5)] + [("hnsc", n) for n in range(5)]
TOTAL_EPOCHS = 50


@dataclass
class JobState:
    cancer: str
    fold: int
    status: str = "pending"
    epoch: int = 0
    best_cindex: float | None = None
    best_cindex_ipcw: float | None = None
    best_epoch: int | None = None
    final_metrics: dict | None = None
    last_error: str | None = None


def gpu_snapshot() -> tuple[str, str]:
    """Return (memory_used, util_pct). Empty if nvidia-smi missing."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2.0, stderr=subprocess.DEVNULL,
        ).decode().strip()
        mem, util = out.split(",")
        return f"{mem.strip()} MiB", f"{util.strip()} %"
    except Exception:
        return "?", "?"


def launcher_alive() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "scripts/run_act_surv_v5.py"],
            timeout=2.0, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def active_train_fold() -> str:
    """Return the cancer+fold (and epoch) currently being trained."""
    if not LOG_PATH.exists():
        return "?"
    try:
        text = LOG_PATH.read_text(errors="replace")
    except Exception:
        return "?"
    starts = list(re.finditer(r"\[Fold (\d+)\] start", text))
    if not starts:
        return "?"
    last_start = starts[-1]
    pre = text[: last_start.start()]
    run_match = list(re.finditer(r"Running:\s+(\w+)\s+fold\s+(\d+)", pre))
    cancer = run_match[-1].group(1) if run_match else "?"
    tail = text[last_start.end():]
    next_marker = re.search(r"\[Fold \d+\]|Running:", tail)
    section = tail[: next_marker.start()] if next_marker else tail
    epoch_match = re.findall(r"Epoch (\d+)/50", section)
    if not epoch_match:
        return f"{cancer} fold {last_start.group(1)} (starting)"
    return f"{cancer} fold {last_start.group(1)} epoch {epoch_match[-1]}/50"


def read_epoch_curve(path: Path) -> dict:
    """Parse epoch_curve_foldN.csv. Return {epoch: {val_cindex, val_cindex_ipcw}}."""
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    try:
        with path.open() as fh:
            header = fh.readline().strip().split(",")
            try:
                i_epoch = header.index("epoch")
                i_cindex = header.index("val_cindex")
                i_ipcw = header.index("val_cindex_ipcw")
            except ValueError:
                return {}
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) <= max(i_epoch, i_cindex, i_ipcw):
                    continue
                try:
                    epoch = int(parts[i_epoch])
                    cindex = float(parts[i_cindex])
                    ipcw = float(parts[i_ipcw])
                except ValueError:
                    continue
                rows[epoch] = {"val_cindex": cindex, "val_cindex_ipcw": ipcw}
    except Exception:
        return {}
    return rows


def best_metric(curve: dict) -> tuple[float, float, int] | None:
    if not curve:
        return None
    best_epoch = max(curve, key=lambda e: curve[e]["val_cindex"])
    return (
        curve[best_epoch]["val_cindex"],
        curve[best_epoch]["val_cindex_ipcw"],
        best_epoch,
    )


def latest_summary(path: Path) -> dict | None:
    candidates = sorted(path.glob("summary_partial_*.csv"))
    if not candidates:
        candidates = sorted(path.glob("summary.csv"))
    if not candidates:
        return None
    fp = candidates[-1]
    rows: dict = {}
    try:
        with fp.open() as fh:
            header = fh.readline().strip().split(",")
            try:
                i_fold = header.index("fold")
            except ValueError:
                return None
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) <= i_fold:
                    continue
                row = dict(zip(header, parts))
                fold_key = row["fold"]
                try:
                    rows[int(fold_key)] = row
                except ValueError:
                    rows[fold_key] = row
    except Exception:
        return None
    return rows


def scan_errors(text: str, since_offset: int) -> tuple[list[str], int]:
    new = text[since_offset:]
    errs = [line for line in new.splitlines()
            if "Traceback" in line or "[ERROR]" in line]
    return errs, since_offset + len(new)


def _find_fold_dir(root: Path, fold_n: int) -> Path | None:
    """Walk root recursively for the deepest dir whose name embeds foldN."""
    tag = f"fold{fold_n}"
    for p in root.rglob("*"):
        if p.is_dir() and tag in p.name:
            return p
    return None


def gather_states() -> tuple[dict[tuple[str, int], JobState], list[str]]:
    """Walk the result tree and produce a state object per expected job."""
    states = {(c, n): JobState(c, n) for c, n in EXPECTED}
    errors: list[str] = []

    for (cancer, fold), st in states.items():
        run_dir = RESULT_ROOT / cancer
        if not run_dir.exists():
            continue
        # Layout may be results/<cancer>/<method_class>/<run_signature>/
        # where the deepest folder embeds the fold tag.
        run_dir = _find_fold_dir(run_dir, fold) or run_dir

        curve_files = list(run_dir.glob(f"epoch_curve_fold{fold}.csv"))
        if curve_files:
            curve = read_epoch_curve(curve_files[0])
            best = best_metric(curve)
            if best:
                st.best_cindex, st.best_cindex_ipcw, st.best_epoch = best
            epochs_completed = max(curve.keys()) + 1 if curve else 0
            st.epoch = epochs_completed
            if epochs_completed >= TOTAL_EPOCHS:
                st.status = "done"
            else:
                st.status = "running"
        else:
            st.status = "pending"

        if list(run_dir.glob(f"split_{fold}_results_final.pkl")):
            st.status = "done"

        summary = latest_summary(run_dir)
        if summary and fold in summary:
            row = summary[fold]
            try:
                st.final_metrics = {
                    "val_cindex": float(row.get("val_cindex", "nan")),
                    "val_cindex_ipcw": float(row.get("val_cindex_ipcw", "nan")),
                    "val_IBS": float(row.get("val_IBS", "nan")),
                    "val_iauc": float(row.get("val_iauc", "nan")),
                }
            except ValueError:
                pass

    if LOG_PATH.exists():
        text = LOG_PATH.read_text(errors="replace")
        for m in re.finditer(r"Running:\s+(\w+)\s+fold\s+(\d+)", text):
            cancer = m.group(1).lower()
            fold = int(m.group(2))
            st = states.get((cancer, fold))
            if st and st.status not in {"done"}:
                st.status = "running"
        for m in re.finditer(r"FAILED:\s+(\w+)\s+fold\s+(\d+)", text):
            cancer = m.group(1).lower()
            fold = int(m.group(2))
            st = states.get((cancer, fold))
            if st:
                st.status = "failed"

    return states, errors


def render(states: dict[tuple[str, int], JobState],
           active_fold: str,
           alive: bool,
           mem: str,
           util: str,
           errors: list[str]) -> str:
    lines = []
    lines.append("─" * 95)
    lines.append(f" ACT-Surv v5 5-fold (BLCA + HNSC)  •  launcher: "
                 f"{'ALIVE' if alive else 'DEAD'}  •  "
                 f"now: {active_fold}  •  GPU mem/util: {mem} / {util}")
    lines.append("─" * 95)
    header = (f"{'cancer':<5} {'fold':<4} {'status':<9} "
              f"{'epoch':<5} {'best_cidx':<10} {'best_epoch':<10} "
              f"{'best_ipcw':<10} {'summary_metrics':<30}")
    lines.append(header)
    lines.append("─" * 95)
    for (cancer, fold), st in states.items():
        best = f"{st.best_cindex:.4f}" if st.best_cindex is not None else "—"
        best_ep = str(st.best_epoch) if st.best_epoch is not None else "—"
        best_ipcw = (f"{st.best_cindex_ipcw:.4f}"
                     if st.best_cindex_ipcw is not None else "—")
        if st.final_metrics:
            m = st.final_metrics
            metr = (f"c={m['val_cindex']:.4f} ipcw={m['val_cindex_ipcw']:.4f} "
                    f"iauc={m['val_iauc']:.4f}")
        else:
            metr = "—"
        lines.append(
            f"{cancer:<5} {fold:<4} {st.status:<9} {st.epoch:<5} "
            f"{best:<10} {best_ep:<10} {best_ipcw:<10} {metr:<30}"
        )
    lines.append("─" * 95)
    done = sum(1 for s in states.values() if s.status == "done")
    failed = sum(1 for s in states.values() if s.status == "failed")
    running = sum(1 for s in states.values() if s.status == "running")
    pending = sum(1 for s in states.values() if s.status == "pending")
    lines.append(f" total: {done} done, {running} running, "
                 f"{pending} pending, {failed} failed  "
                 f"({done + failed}/{len(states)})")
    if errors:
        lines.append("─" * 95)
        lines.append(" recent errors:")
        for e in errors[-5:]:
            lines.append(f"   {e[:120]}")
    lines.append("─" * 95)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=float, default=30.0,
                   help="Refresh interval in seconds (default 30)")
    p.add_argument("--once", action="store_true",
                   help="Print once and exit")
    args = p.parse_args()

    offset = 0
    if LOG_PATH.exists():
        offset = LOG_PATH.stat().st_size
    last_render = ""
    try:
        while True:
            states, errs = gather_states()
            text = LOG_PATH.read_text(errors="replace") if LOG_PATH.exists() else ""
            new_errs, offset = scan_errors(text, offset)
            active_fold = active_train_fold()
            alive = launcher_alive()
            mem, util = gpu_snapshot()
            rendered = render(states, active_fold, alive, mem, util,
                              errs + new_errs)
            if rendered != last_render:
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(rendered + "\n")
                sys.stdout.flush()
                last_render = rendered
            if args.once:
                return 0
            if not alive and all(s.status in {"done", "failed"} for s in states.values()):
                sys.stdout.write("\n[monitor] launcher exited and all jobs "
                                 "have reached a terminal state. stopping.\n")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())