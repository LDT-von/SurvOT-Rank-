#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诚实汇总与消融/校准报告。

作用
----
扫描一个或多个结果目录，找到训练器落盘的 ``epoch_curve_fold*.csv``
（支持嵌套在 ``seed*/`` 或 ``<study>/<method>/`` 子目录下），然后：

1. 用 ``epoch_curve_selection`` 的稳健策略重新计算 mean ± std，
   替代原始「验证峰值」上报，得到可写进论文的诚实数字。
2. 量化乐观偏差：并排给出 best（泄漏）vs robust（稳健）以及二者差值。
3. 校准告警：检查 val_IBS。IBS 越小越好、随机基线约 0.25；若某 fold IBS
   过高或 fold 间方差过大，说明风险概率没校准好，会被审稿人抓。
4. 消融对比：传入多个目录时，并排列出各方法/配置的稳健指标，方便回答
   「去掉某模块掉多少分」。

用法
----
    # 单次实验的诚实汇总
    python robust_eval/honest_report.py --dirs results/v45v2_blca_clinical

    # 多 seed 目录（stable_train_launcher 产出）
    python robust_eval/honest_report.py --dirs results/v45v2_blca_clinical \
        --strategy last_k_mean --k 5

    # 消融对比多个方法目录
    python robust_eval/honest_report.py \
        --dirs results/full results/no_ot results/no_rank \
        --labels full no_ot no_rank --out report.md
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from robust_eval.epoch_curve_selection import (  # noqa: E402
    METRIC_HIGHER_IS_BETTER,
    aggregate_folds,
    select_fold_score,
)

REPORT_METRICS = ["val_cindex", "val_cindex_ipcw", "val_iauc", "val_IBS"]
IBS_RANDOM_BASELINE = 0.25


def find_epoch_curves(root: str) -> list[str]:
    """在 root 下递归查找所有 epoch_curve_fold*.csv。"""
    pattern = os.path.join(root, "**", "epoch_curve_fold*.csv")
    return sorted(glob.glob(pattern, recursive=True))


def _load_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fold" not in df.columns:
        # 从文件名 epoch_curve_fold<K>.csv 推断 fold 号
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        df["fold"] = int(digits) if digits else -1
    return df


def _dedupe_curves_by_fold(paths: list[str]) -> dict[tuple[str, int], pd.DataFrame]:
    """把 (seed 目录, fold) 作为键收集曲线，保留多 seed 的每条独立曲线。"""
    curves: dict[tuple[str, int], pd.DataFrame] = {}
    for p in paths:
        df = _load_curve(p)
        fold = int(df["fold"].iloc[0])
        # 用 seed 目录名（若有）区分同 fold 的不同 seed
        seed_tag = ""
        parts = Path(p).parts
        for part in parts:
            if part.lower().startswith("seed"):
                seed_tag = part
                break
        curves[(seed_tag, fold)] = df
    return curves


def summarize_dir(
    root: str, strategy: str, k: int, smooth_window: int
) -> dict:
    """对一个结果目录做稳健 vs 乐观汇总。"""
    paths = find_epoch_curves(root)
    if not paths:
        return {"root": root, "found": 0, "error": "未找到 epoch_curve_fold*.csv"}

    curves = _dedupe_curves_by_fold(paths)
    result: dict = {"root": root, "found": len(curves), "metrics": {}}

    for metric in REPORT_METRICS:
        usable = [c for c in curves.values() if metric in c.columns]
        if not usable:
            continue
        robust_scores, best_scores = [], []
        for c in usable:
            robust_scores.append(
                select_fold_score(c, metric, strategy, k=k, smooth_window=smooth_window)
            )
            best_scores.append(select_fold_score(c, metric, "best"))
        robust_agg = aggregate_folds(robust_scores)
        best_agg = aggregate_folds(best_scores)
        higher = METRIC_HIGHER_IS_BETTER[metric]
        gap = (best_agg.mean - robust_agg.mean) if higher else (robust_agg.mean - best_agg.mean)
        result["metrics"][metric] = {
            "robust_mean": robust_agg.mean,
            "robust_std": robust_agg.std,
            "best_mean": best_agg.mean,
            "best_std": best_agg.std,
            "optimism_gap": gap,
            "per_fold_robust": robust_agg.values,
            "higher_is_better": higher,
        }
    return result


def _calibration_warnings(summary: dict) -> list[str]:
    warns: list[str] = []
    ibs = summary.get("metrics", {}).get("val_IBS")
    if ibs is None:
        return warns
    if ibs["robust_mean"] > IBS_RANDOM_BASELINE:
        warns.append(
            f"IBS 稳健均值 {ibs['robust_mean']:.3f} 高于随机基线 {IBS_RANDOM_BASELINE:.2f}"
            "：风险概率未校准，NLL/离散 hazard 头训练不充分。"
        )
    if ibs["robust_std"] > 0.15:
        warns.append(
            f"IBS 跨 fold 标准差 {ibs['robust_std']:.3f} 过大：校准在不同 fold 之间极不稳定。"
        )
    return warns


def _fmt_metric_line(name: str, m: dict) -> str:
    arrow = "↑" if m["higher_is_better"] else "↓"
    return (
        f"  {name:16s}{arrow}  robust={m['robust_mean']:.4f} ± {m['robust_std']:.4f}"
        f"   |  best(泄漏)={m['best_mean']:.4f} ± {m['best_std']:.4f}"
        f"   |  乐观偏差={m['optimism_gap']:+.4f}"
    )


def render_report(summaries: list[dict], labels: list[str], strategy: str) -> str:
    lines: list[str] = []
    lines.append("# SurvOT-Rank 诚实评测报告\n")
    lines.append(f"稳健选择策略: `{strategy}`  |  ↑=越大越好, ↓=越小越好\n")
    lines.append(
        "> best(泄漏) = 逐 epoch 取验证峰值（原训练器上报方式，乐观偏差）；"
        "robust = 稳健策略。乐观偏差越大，原始上报越不可信。\n"
    )

    for label, summary in zip(labels, summaries):
        lines.append(f"\n## {label}")
        lines.append(f"- 目录: `{summary['root']}`")
        if summary.get("error"):
            lines.append(f"- **{summary['error']}**")
            continue
        lines.append(f"- 找到曲线数 (seed×fold): {summary['found']}\n")
        for name in REPORT_METRICS:
            m = summary["metrics"].get(name)
            if m:
                lines.append(_fmt_metric_line(name, m))
        warns = _calibration_warnings(summary)
        if warns:
            lines.append("\n  校准告警:")
            for w in warns:
                lines.append(f"    - {w}")

    # 消融并排表（仅在多目录时输出）
    valid = [(l, s) for l, s in zip(labels, summaries) if not s.get("error")]
    if len(valid) > 1:
        lines.append("\n## 消融对比（robust C-index）\n")
        lines.append("| 方法 | C-index↑ | IPCW↑ | iAUC↑ | IBS↓ |")
        lines.append("|------|----------|-------|-------|------|")
        ref = None
        for label, s in valid:
            def cell(metric: str) -> str:
                m = s["metrics"].get(metric)
                return f"{m['robust_mean']:.4f}±{m['robust_std']:.4f}" if m else "-"
            row = f"| {label} | {cell('val_cindex')} | {cell('val_cindex_ipcw')} | {cell('val_iauc')} | {cell('val_IBS')} |"
            lines.append(row)
        # 相对第一个（通常是 full）的 C-index 差
        base_m = valid[0][1]["metrics"].get("val_cindex")
        if base_m:
            ref = base_m["robust_mean"]
            lines.append("\n相对 `%s` 的 C-index 变化（负号=该模块被移除后掉分，说明其有用）:" % valid[0][0])
            for label, s in valid[1:]:
                m = s["metrics"].get("val_cindex")
                if m:
                    lines.append(f"- {label}: {m['robust_mean'] - ref:+.4f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SurvOT-Rank 诚实汇总/消融/校准报告")
    parser.add_argument("--dirs", nargs="+", required=True, help="一个或多个结果目录")
    parser.add_argument("--labels", nargs="*", default=None, help="各目录的显示名（可选）")
    parser.add_argument(
        "--strategy", default="last_k_mean",
        choices=["last", "last_k_mean", "smoothed_peak", "plateau"],
        help="稳健选择策略（默认 last_k_mean，不含泄漏的 best）",
    )
    parser.add_argument("--k", type=int, default=5, help="last_k_mean 的 k")
    parser.add_argument("--smooth-window", type=int, default=3, help="平滑窗口")
    parser.add_argument("--out", default=None, help="输出 markdown 路径（可选）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise SystemExit("--labels 数量需与 --dirs 一致")

    summaries = [
        summarize_dir(d, args.strategy, args.k, args.smooth_window) for d in args.dirs
    ]
    report = render_report(summaries, labels, args.strategy)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[report] 已写入 {args.out}")


if __name__ == "__main__":
    main()
