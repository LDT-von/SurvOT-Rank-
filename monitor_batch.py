#!/usr/bin/env python3
"""SurvOT-Rank 16-实验批量监控脚本

用法:
    python monitor_batch.py                   # 每5秒刷新
    python monitor_batch.py --refresh 10      # 每10秒刷新
    python monitor_batch.py --once            # 仅打印一次

自动检测最新的 batch_runs 目录并监控其中所有实验。
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = "/home/ubuntu/SurvOT-Rank"
BATCH_ROOT = os.path.join(PROJECT_DIR, "results", "batch_runs")

# ── 与 run_batch.sh 保持一致的实验列表 ──
ALL_EXPS = [
    ("v45_blca",             "configs/v45_blca.yaml"),
    ("v45_brca",             "configs/v45_brca.yaml"),
    ("v45_stad",             "configs/v45_stad.yaml"),
    ("v45_coadread",         "configs/v45_coadread.yaml"),
    ("v45_hnsc",             "configs/v45_hnsc.yaml"),
    ("v45_best_blca",        "configs/v45_best_blca.yaml"),
    ("abl_00_baseline",      "configs/ablation/abl_00_baseline.yaml"),
    ("abl_01_clinical",      "configs/ablation/abl_01_clinical.yaml"),
    ("abl_02_unified",       "configs/ablation/abl_02_unified.yaml"),
    ("abl_03_disentangle",   "configs/ablation/abl_03_disentangle.yaml"),
    ("abl_04_sinkhorn",      "configs/ablation/abl_04_sinkhorn.yaml"),
    ("abl_05_crossmodal",    "configs/ablation/abl_05_crossmodal.yaml"),
    ("abl_06_adaptive_iters",    "configs/ablation/abl_06_adaptive_iters.yaml"),
    ("abl_07_learnable_weights", "configs/ablation/abl_07_learnable_weights.yaml"),
    ("abl_08_all_on",            "configs/ablation/abl_08_all_on.yaml"),
    ("abl_09_all_on_learnable",  "configs/ablation/abl_09_all_on_learnable.yaml"),
]


def find_latest_batch_dir() -> str | None:
    if not os.path.isdir(BATCH_ROOT):
        return None
    dirs = sorted(
        [d for d in os.listdir(BATCH_ROOT) if os.path.isdir(os.path.join(BATCH_ROOT, d))],
        reverse=True,
    )
    return os.path.join(BATCH_ROOT, dirs[0]) if dirs else None


def read_log(path: str, tail_kb: int = 8) -> str:
    """读取日志尾部（用于解析进度条），以及全文（用于解析 val cindex）"""
    if not os.path.exists(path):
        return "", ""
    try:
        with open(path, "r", errors="replace") as f:
            full = f.read()
    except Exception:
        return "", ""
    # 尾部用于进度条，全文用于 val 指标
    tail = full[-tail_kb * 1024:] if len(full) > tail_kb * 1024 else full
    return full, tail


# ── 正则 ──
RE_FOLD_START = re.compile(r"^\[Fold (\d+)\] start", re.M)
RE_VAL = re.compile(
    r"^\[Epoch\s+(\d+)\]\s+val\s+cindex=([0-9.\-]+)\s+ipcw=([0-9.\-]+)\s+IBS=([0-9.\-]+)\s+iauc=([0-9.\-]+)",
    re.M,
)
RE_PROGRESS = re.compile(
    r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+(\d+)%"
)
RE_PROGRESS_FULL = re.compile(
    r"\[Fold (\d+)\] Epoch (\d+)/(\d+):\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+)\s+\[<?(\d+):(\d+)<(\d+):(\d+)"
)


def parse_best(content: str):
    """从全文解析：当前 fold 编号 + 每 fold 最佳 C-index"""
    best_per_fold = {}
    current_fold = 0
    for line in content.splitlines():
        m = RE_FOLD_START.search(line)
        if m:
            current_fold = int(m.group(1))
            continue
        m = RE_VAL.search(line)
        if m:
            epoch = int(m.group(1))
            cidx = float(m.group(2))
            if current_fold not in best_per_fold or cidx > best_per_fold[current_fold][0]:
                best_per_fold[current_fold] = (cidx, epoch)
    return best_per_fold, current_fold


def parse_progress(tail: str):
    """从日志尾部解析当前进度"""
    m = RE_PROGRESS_FULL.search(tail)
    if m:
        fold = int(m.group(1))
        epoch = int(m.group(2))
        total_epoch = int(m.group(3))
        pct = int(m.group(4))
        batch = int(m.group(5))
        total_batch = int(m.group(6))
        cur_sec = int(m.group(7)) * 60 + int(m.group(8))
        tot_sec = int(m.group(9)) * 60 + int(m.group(10))
        eta_sec = max(tot_sec - cur_sec, 0)
        return fold, epoch, total_epoch, pct, batch, total_batch, eta_sec
    # 简化版：只有百分比没有时间
    m = RE_PROGRESS.search(tail)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), -1, -1, -1
    return None


def fmt_time(seconds: int) -> str:
    if seconds < 0:
        return "?"
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def detect_status_phase(name: str, full: str, tail: str, has_log: bool, was_skipped: bool = False):
    """判断实验状态"""
    if not has_log:
        if was_skipped:
            return "⏭ SKIPPED", "(before start-from)", "—"
        return "⏳ 排队中", "—", "—"

    # 1. failed
    if "AssertionError" in full or "KeyError" in full or "Traceback" in full:
        # 提取具体错误
        if "assert os.path.isdir(args.split_dir)" in full:
            return "❌ FAILED", "缺少分片目录", "—"
        if "KeyError" in full:
            m = re.search(r"KeyError.*", full)
            err = m.group(0)[:60] if m else "KeyError"
            return "❌ FAILED", err, "—"
        # 检查最后一行是否有 exit
        return "❌ FAILED", "运行异常", "—"

    # 2. skipped (log 中有 SKIPPED 字样)
    if "SKIPPED" in full:
        return "⏭ SKIPPED", "—", "—"

    # 3. completed (fold 4 done)
    if "[Fold 4] start" in full and "[Epoch 29] val " in full:
        best, _ = parse_best(full)
        vals = [v for v, _ in best.values()]
        if vals:
            mean_cindex = sum(vals) / len(vals)
            return "✅ 完成", f"{mean_cindex:.4f}", f"best per fold: { {k: f'{v:.4f}' for k, (v, _) in best.items()} }"
        return "✅ 完成", "—", "—"

    # 4. running
    prog = parse_progress(tail)
    if prog:
        fold, epoch, total_epoch, pct, batch, total_batch, eta = prog
        status = f"🔄 Fold {fold}"
        detail = f"Epoch {epoch}/{total_epoch} ({pct}%)"
        eta_str = fmt_time(eta) if eta > 0 else "—"
        return status, detail, f"ETA epoch: {eta_str}"

    # 5. 有日志但没有明确状态标记 → 可能刚启动
    return "🔄 初始化", "—", "—"


def calc_remaining_estimate(name: str, status: str, detail: str, progress):
    """估算该实验剩余时间（秒）"""
    if not progress:
        return None
    fold, epoch, total_epoch, _, _, _, _ = progress
    epochs_left_in_fold = total_epoch - epoch
    folds_left = 4 - fold  # folds 0-4, so remaining folds after current
    total_epochs_left = epochs_left_in_fold + folds_left * total_epoch
    # ~55s per epoch
    return total_epochs_left * 55


def clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render(batch_dir: str, start_ts: float, completed_count: dict):
    clear()
    elapsed = time.time() - start_ts
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 105)
    print(f"  SurvOT-Rank 批量实验监控    {now}    运行时间: {fmt_time(int(elapsed))}")
    print(f"  Batch dir: {batch_dir}")
    print("=" * 105)

    # 统计
    n_done = 0
    n_failed = 0
    n_running = 0
    n_pending = 0
    n_skipped = 0

    total_eta_sec = 0
    running_name = ""

    # 构建 log 存在情况的索引，用于判断 SKIPPED vs PENDING
    log_exists = [os.path.exists(os.path.join(batch_dir, f"{name}.log")) for name, _ in ALL_EXPS]
    first_log_idx = next((i for i, e in enumerate(log_exists) if e), len(ALL_EXPS))

    for idx, (name, config) in enumerate(ALL_EXPS):
        log_path = os.path.join(batch_dir, f"{name}.log")
        has_log = log_exists[idx]
        full, tail = read_log(log_path) if has_log else ("", "")

        # 无 log 但在后续有 log → 被 start-from 跳过
        was_skipped = (not has_log) and (first_log_idx < len(ALL_EXPS)) and (idx < first_log_idx)
        status, detail, extra = detect_status_phase(name, full, tail, has_log, was_skipped)
        idx_str = f"[{idx+1:2d}/16]"

        line = f"  {idx_str}  {name:<28s}  {status:<14s}  {detail:<28s}  {extra}"

        # 颜色标记
        if "FAILED" in status:
            line = f"\033[31m{line}\033[0m"
            n_failed += 1
        elif "完成" in status:
            line = f"\033[32m{line}\033[0m"
            n_done += 1
        elif "排队" in status:
            n_pending += 1
        elif "SKIPPED" in status:
            n_skipped += 1
        elif "Fold" in status or "初始化" in status:
            line = f"\033[33m{line}\033[0m"
            n_running += 1
            running_name = name

        print(line)

    # ── 进度条 ──
    total_processed = n_done + n_failed + n_skipped
    pct = total_processed * 100 // 16
    bar_width = 40
    filled = int(bar_width * total_processed / 16)
    bar_str = "[" + "#" * filled + "·" * (bar_width - filled) + "]"

    print()
    print(f"  进度: {bar_str}  {total_processed}/16 ({pct}%)")
    print(f"  ✅={n_done}  ❌={n_failed}  ⏭={n_skipped}  🔄={n_running}  ⏳={n_pending}")
    print("-" * 105)

    # ── 当前运行中的实验：展示详细进度 ──
    if running_name:
        log_path = os.path.join(batch_dir, f"{running_name}.log")
        full, tail = read_log(log_path)
        prog = parse_progress(tail)
        best, cur_fold = parse_best(full)

        print(f"  正在运行: {running_name}")
        if prog:
            fold, epoch, total_epoch, pct, batch, total_batch, eta = prog
            print(f"    Fold {fold}  |  Epoch {epoch}/{total_epoch}  |  Batch {batch}/{total_batch if total_batch > 0 else '?'}  |  {pct}%")
            if eta > 0:
                print(f"    当前 epoch 剩余: {fmt_time(eta)}")
            # 估算该实验剩余
            remaining = calc_remaining_estimate(running_name, "", "", prog)
            if remaining:
                print(f"    本实验剩余 (est): {fmt_time(remaining)}")

        if best:
            print(f"    各折最佳 C-index:")
            for k in sorted(best.keys()):
                v, ep = best[k]
                done_mark = "✅" if k < cur_fold or (k == cur_fold and "Epoch 29" in tail) else "  "
                print(f"      {done_mark} Fold {k}: {v:.4f} @ epoch {ep}")

        # 整体剩余时间估算
        if prog:
            # 当前实验剩余
            remaining_this = calc_remaining_estimate(running_name, "", "", prog)
            if remaining_this:
                total_eta_sec += remaining_this

        # 剩余排队实验
        found_current = False
        for name, _ in ALL_EXPS:
            if name == running_name:
                found_current = True
                continue
            if found_current:
                log_path_p = os.path.join(batch_dir, f"{name}.log")
                if not os.path.exists(log_path_p):
                    # 每个约 3 小时
                    total_eta_sec += 3 * 3600

        print(f"    ─────────────────────────────")
        if total_eta_sec > 0:
            print(f"    总计预估剩余: {fmt_time(total_eta_sec)} (约 {total_eta_sec/3600:.1f} 小时)")

    print("-" * 105)

    # ── 最近 val 输出 ──
    if running_name:
        log_path = os.path.join(batch_dir, f"{running_name}.log")
        full, _ = read_log(log_path)
        vals = list(RE_VAL.finditer(full))
        if vals:
            print("  最近 5 条 val 记录:")
            for m in vals[-5:]:
                ep, c, ip, ib, ia = m.groups()
                print(f"    Epoch {ep:>3s}  cidx={c}  ipcw={ip}  IBS={ib}  iauc={ia}")

    print("=" * 105)


def main():
    ap = argparse.ArgumentParser(description="SurvOT-Rank 批量实验监控")
    ap.add_argument("--refresh", type=float, default=5.0, help="刷新间隔(秒)")
    ap.add_argument("--once", action="store_true", help="仅打印一次")
    ap.add_argument("--batch-dir", type=str, default=None, help="指定 batch 目录")
    args = ap.parse_args()

    batch_dir = args.batch_dir or find_latest_batch_dir()
    if not batch_dir:
        print("[monitor] 未找到 batch_runs 目录", file=sys.stderr)
        sys.exit(1)

    print(f"[monitor] 监控目录: {batch_dir}")
    print(f"[monitor] 刷新间隔: {args.refresh}s")
    print()

    start_ts = time.time()
    completed = {}

    while True:
        try:
            render(batch_dir, start_ts, completed)
        except Exception as e:
            print(f"[monitor] render error: {e}", flush=True)

        if args.once:
            return

        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
