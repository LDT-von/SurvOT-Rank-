#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 V45 系列实验的 summary.csv，按 val_cindex 排名并对照 V45 基准 0.7105。

扫描 result_dir 下各子目录（递归）里的 summary.csv，抽取 mean 行的
val_cindex / std / IBS / iAUC / Loss，排序打印，并标注相对 0.7105 与 baseline 的增量。

用法:
  python collect_results.py --result_dir /data1/sweep_results_30ep
  python collect_results.py --result_dir /data1/sweep_results_30ep --filter v45
"""

import argparse
import glob
import os

import pandas as pd

V45_REF = 0.7105
V9_REF = 0.7078
BASELINE_REF = 0.7014


def _extract_mean_row(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    # summary.csv 结构：包含 fold 行 + mean/std 行（index 被 reset 成一列）
    idx_col = df.columns[0]
    mean_row = df[df[idx_col].astype(str).str.lower() == "mean"]
    std_row = df[df[idx_col].astype(str).str.lower() == "std"]
    if mean_row.empty:
        return None

    def _get(row, col):
        if col in row.columns and not row.empty:
            try:
                return float(row.iloc[0][col])
            except Exception:
                return None
        return None

    return {
        "val_cindex": _get(mean_row, "val_cindex"),
        "std": _get(std_row, "val_cindex"),
        "ibs": _get(mean_row, "val_IBS"),
        "iauc": _get(mean_row, "val_iauc"),
        "loss": _get(mean_row, "val_loss"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--filter", default="", help="只显示目录名包含该子串的实验")
    args = ap.parse_args()

    rows = []
    seen = set()
    patterns = [
        os.path.join(args.result_dir, "**", "summary.csv"),
        os.path.join(args.result_dir, "**", "summary_partial_*.csv"),
    ]
    for pat in patterns:
        for csv_path in glob.glob(pat, recursive=True):
            # 用 summary.csv 所在目录相对 result_dir 的顶层名作为实验名
            rel = os.path.relpath(csv_path, args.result_dir)
            exp_name = rel.split(os.sep)[0]
            key = (exp_name, os.path.basename(csv_path))
            if key in seen:
                continue
            seen.add(key)
            if args.filter and args.filter not in exp_name:
                continue
            info = _extract_mean_row(csv_path)
            if info is None or info["val_cindex"] is None:
                continue
            info["exp"] = exp_name
            info["partial"] = "partial" in os.path.basename(csv_path)
            rows.append(info)

    if not rows:
        print(f"[collect] 在 {args.result_dir} 未找到可解析的 summary.csv"
              f"{'（filter=' + args.filter + '）' if args.filter else ''}")
        return

    rows.sort(key=lambda r: r["val_cindex"], reverse=True)

    print("=" * 96)
    print(f"{'rank':<4} {'exp':<28} {'cindex':>8} {'std':>7} {'IBS':>7} "
          f"{'iAUC':>7} {'Loss':>7} {'Δv45':>8} {'Δbase':>8}")
    print("=" * 96)
    for i, r in enumerate(rows, 1):
        c = r["val_cindex"]
        dv45 = c - V45_REF
        dbase = c - BASELINE_REF
        flag = ""
        if c > V45_REF:
            flag = " *超V45"
        elif c > V9_REF:
            flag = " ~超v9"
        std = f"{r['std']:.4f}" if r["std"] is not None else "  -  "
        ibs = f"{r['ibs']:.4f}" if r["ibs"] is not None else "  -  "
        iauc = f"{r['iauc']:.4f}" if r["iauc"] is not None else "  -  "
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "  -  "
        part = " (partial)" if r["partial"] else ""
        print(f"{i:<4} {r['exp']:<28} {c:>8.4f} {std:>7} {ibs:>7} "
              f"{iauc:>7} {loss:>7} {dv45:>+8.4f} {dbase:>+8.4f}{flag}{part}")

    print("=" * 96)
    print(f"参照: V45={V45_REF}  v9={V9_REF}  baseline={BASELINE_REF}  目标≥0.72")


if __name__ == "__main__":
    main()
