#!/usr/bin/env python3
"""
aggregate_cross_cancer.py —— 跨癌种（blca/brca/coadread/hnsc/stad）验证结果汇总工具

改造自仓库已有的 `tools/aggregate_multicancer.py`（后者硬编码了 Linux 绝对路径与仅 2 个癌种，
本文件不修改原文件，是一个独立的、更通用/更健壮的新脚本），对应
`.kiro/specs/survot-rank-enhancements/requirements.md` 需求 2（多癌种验证配置与跨癌种结果汇总）
的 AC3-AC7。

用法示例：
    python tools/aggregate_cross_cancer.py --results-root /data1/sweep_results_30ep/multicancer_v2

行为要点（对应各 AC）：
- `--results-root` 与 `--studies`（默认 5 个癌种，字母序）两个命令行参数（AC7）。
- 对每个 study，在 `{results_root}/{study}/` 下用 `rglob("summary.csv")` 查找结果文件；
  目录不存在或找不到文件时标记 `status=missing`；找到但缺少 `"mean"` 索引行时标记
  `status=invalid, reason=missing_mean_row`，跳过该文件的均值/标准差计算但不中断其余 study
  的处理（AC6）。
- 汇总数据列固定为 `val_cindex, val_cindex_ipcw, val_IBS, val_iauc`（AC3），通过读取
  `mean`/`std` 索引行获取并输出各指标的均值与标准差（AC4）。
- 输出按字母序排列各癌种所在行的顺序（AC7），同时写出 CSV 与 Markdown 两种格式的汇总文件，
  两种格式均包含各癌种原始指标值以及 `mean`、`std` 汇总行，并标注缺失/无效的癌种（AC5）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# 汇总数据列固定为这四项指标（需求2 AC3）
METRIC_COLUMNS = ["val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc"]

# 默认覆盖的 5 个癌种，始终按字母序排列（需求2 AC7）
DEFAULT_STUDIES = sorted(["blca", "brca", "coadread", "hnsc", "stad"])


def find_summary(study_dir: Path) -> Optional[Path]:
    """在 study 结果目录下递归查找 summary.csv（与原脚本 find_summary 逻辑一致）。

    找到多个候选时取排序后的第一个，保证结果可复现（不依赖文件系统遍历顺序）。
    """
    candidates = sorted(study_dir.rglob("summary.csv"))
    return candidates[0] if candidates else None


def process_study(results_root: Path, study: str) -> dict:
    """处理单个癌种，返回一行结果字典。

    该函数内部捕获所有异常，保证单个 study 的问题不会中断其余 study 的处理
    （需求2 AC6："跳过该文件的均值/标准差计算但不中断其余 study 的处理"）。
    """
    row: dict = {"study": study, "status": "missing", "reason": "", "summary_csv": ""}
    try:
        study_dir = results_root / study
        if not study_dir.is_dir():
            row["reason"] = "directory_not_found"
            return row

        summary_path = find_summary(study_dir)
        if summary_path is None:
            row["reason"] = "summary_csv_not_found"
            return row

        row["summary_csv"] = str(summary_path)

        try:
            df = pd.read_csv(summary_path, index_col=0)
        except Exception as e:  # noqa: BLE001 —— 读取失败也不能中断其他 study
            row["status"] = "invalid"
            row["reason"] = f"read_error: {e}"
            return row

        if "mean" not in df.index:
            row["status"] = "invalid"
            row["reason"] = "missing_mean_row"
            return row

        mean_row = df.loc["mean"]
        std_row = df.loc["std"] if "std" in df.index else None

        row["status"] = "ok"
        row["reason"] = ""
        for col in METRIC_COLUMNS:
            row[col] = float(mean_row.get(col, float("nan")))
            row[f"{col}_std"] = float(std_row.get(col, float("nan"))) if std_row is not None else float("nan")
        return row
    except Exception as e:  # noqa: BLE001 —— 兜底：任何未预料到的异常也标记为 invalid 而不是崩溃
        row["status"] = "invalid"
        row["reason"] = f"unexpected_error: {e}"
        return row


def build_summary_rows(results_root: Path, studies: list[str]) -> list[dict]:
    """按字母序依次处理每个 study，返回结果行列表（顺序即输出顺序，需求2 AC7）。"""
    ordered_studies = sorted(set(studies))
    return [process_study(results_root, s) for s in ordered_studies]


def append_mean_std_rows(rows: list[dict]) -> list[dict]:
    """在各癌种行之后追加跨癌种 `mean`/`std` 汇总行（需求2 AC4/AC5）。

    仅使用 status=ok 的 study 参与均值/标准差计算；若不存在任何有效 study，
    汇总行的指标值填 NaN，但仍会输出该行以便在报告中明确看到"无有效结果"。
    """
    ok_rows = [r for r in rows if r["status"] == "ok"]

    mean_row = {"study": "mean", "status": "summary", "reason": "", "summary_csv": ""}
    std_row = {"study": "std", "status": "summary", "reason": "", "summary_csv": ""}

    if ok_rows:
        metrics_df = pd.DataFrame([{col: r[col] for col in METRIC_COLUMNS} for r in ok_rows])
        means = metrics_df.mean()
        stds = metrics_df.std()  # 样本标准差（ddof=1）；仅 1 个有效 study 时为 NaN，属预期行为
        for col in METRIC_COLUMNS:
            mean_row[col] = float(means[col])
            mean_row[f"{col}_std"] = float("nan")
            std_row[col] = float(stds[col])
            std_row[f"{col}_std"] = float("nan")
    else:
        mean_row["reason"] = "no_valid_studies"
        std_row["reason"] = "no_valid_studies"
        for col in METRIC_COLUMNS:
            mean_row[col] = float("nan")
            mean_row[f"{col}_std"] = float("nan")
            std_row[col] = float("nan")
            std_row[f"{col}_std"] = float("nan")

    return rows + [mean_row, std_row]


def write_csv(all_rows: list[dict], output_csv: Path) -> None:
    """写出 CSV 格式汇总文件（需求2 AC5）。"""
    columns = ["study", "status", "reason"]
    for col in METRIC_COLUMNS:
        columns.append(col)
        columns.append(f"{col}_std")
    columns.append("summary_csv")

    df = pd.DataFrame(all_rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def format_metric(row: dict, col: str) -> str:
    """把某一行某个指标格式化为 `mean±std` 字符串；缺失/无效的行直接标注状态原因。"""
    if row.get("status") not in ("ok", "summary"):
        return "-"
    value = row.get(col, float("nan"))
    std = row.get(f"{col}_std", float("nan"))
    if value != value:  # NaN 检测（NaN != NaN 为真）
        return "-"
    if std == std:  # std 非 NaN
        return f"{value:.4f}±{std:.4f}"
    return f"{value:.4f}"


def write_markdown(all_rows: list[dict], output_md: Path) -> None:
    """写出 Markdown 格式汇总文件，缺失/无效癌种用 `status`/`reason` 列明确标注（需求2 AC5/AC6）。"""
    header = "| study | status | " + " | ".join(METRIC_COLUMNS) + " | reason |"
    sep = "|-------|--------|" + "|".join(["--------"] * len(METRIC_COLUMNS)) + "|--------|"
    lines = [header, sep]
    for row in all_rows:
        metric_cells = " | ".join(format_metric(row, col) for col in METRIC_COLUMNS)
        lines.append(f"| {row['study']} | {row['status']} | {metric_cells} | {row.get('reason', '') or '-'} |")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总多个癌种（blca/brca/coadread/hnsc/stad）的验证结果为 CSV 与 Markdown 报告。"
    )
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="结果根目录，其下应有各癌种子目录，如 {results-root}/{study}/.../summary.csv",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=DEFAULT_STUDIES,
        help="要汇总的癌种列表（输出时总是按字母序排列），默认: blca brca coadread hnsc stad",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出文件前缀（不含扩展名），默认写到 {results-root}/cross_cancer_summary.{csv,md}",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)

    output_prefix = Path(args.output_prefix) if args.output_prefix else results_root / "cross_cancer_summary"
    output_csv = output_prefix.with_suffix(".csv")
    output_md = output_prefix.with_suffix(".md")

    per_study_rows = build_summary_rows(results_root, args.studies)
    all_rows = append_mean_std_rows(per_study_rows)

    write_csv(all_rows, output_csv)
    write_markdown(all_rows, output_md)

    print("=" * 60)
    print("跨癌种验证结果汇总")
    print("=" * 60)
    for row in per_study_rows:
        status = row["status"]
        if status == "ok":
            print(
                f"[{row['study']}] status=ok "
                f"val_cindex={row['val_cindex']:.4f} "
                f"val_cindex_ipcw={row['val_cindex_ipcw']:.4f} "
                f"val_IBS={row['val_IBS']:.4f} "
                f"val_iauc={row['val_iauc']:.4f}"
            )
        else:
            print(f"[{row['study']}] status={status} reason={row['reason']}")
    print("-" * 60)
    print(f"CSV 已保存: {output_csv}")
    print(f"Markdown 已保存: {output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
