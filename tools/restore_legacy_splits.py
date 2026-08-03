#!/usr/bin/env python3
"""从 git 历史恢复 2026-07-30 重划之前的 5 折划分，仅用于 v3.3 历史复现。

背景
----
提交 ``bee66a2`` (2026-07-30, "rebalance splits") 在**新路径**创建了
``survot_rank/.../dataset_csv/splits/5fold/``，旧划分保留在初始提交
``77833de`` (2026-07-01) 的**根级路径** ``dataset_csv/splits/5fold/``。

本脚本把旧划分恢复到 ``splits/5fold_legacy/<cancer>/fold_*.csv``。

范围
----
默认只恢复 **BLCA**：当前唯一需要旧划分的任务是 v3.3 复现历史分数 (0.7311)。
所有其他方法与阶段一律使用 ``5fold`` / ``5fold_uni2h``。

⚠️ 旧划分存在已知缺陷，**不得用于新实验**（BLCA 实测）：

- fold1 含 1 个缺 DSS 标签的患者（标签未定义，本不应入折）
- 每折验证事件数 ``[27, 26, 27, 23, 25]``，极差 4
  （新划分 ``[26, 26, 25, 26, 25]``，极差 1）
- fold0 验证集 77 人，其余 76 人（不均）

用法
----
    python tools/restore_legacy_splits.py           # 只恢复 BLCA
    python tools/restore_legacy_splits.py --check   # 只比较，不写文件
    python tools/restore_legacy_splits.py --studies blca,brca
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "77833de"
LEGACY_PATH = "dataset_csv/splits/5fold"
SPLIT_ROOT = (
    REPO_ROOT
    / "survot_rank"
    / "research"
    / "legacy"
    / "slotspe_runtime"
    / "dataset_csv"
    / "splits"
)
TARGET_DIR_NAME = "5fold_legacy"
STUDIES = (
    "blca",
    "brca",
    "coadread",
    "hnsc",
    "kirc",
    "luad",
    "lusc",
    "skcm",
    "stad",
    "ucec",
)
# 只有 v3.3 需要旧划分，因此默认范围仅 BLCA。
DEFAULT_STUDIES = ("blca",)
N_FOLDS = 5


def read_legacy_fold(study: str, fold: int) -> pd.DataFrame:
    """从 git 对象读取旧划分的一个折，并去掉无名行索引列。"""
    completed = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:{LEGACY_PATH}/{study}/fold_{fold}.csv"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    frame = pd.read_csv(io.BytesIO(completed.stdout))
    drop = [column for column in frame.columns if str(column).startswith("Unnamed")]
    return frame.drop(columns=drop)


def val_ids(frame: pd.DataFrame) -> set[str]:
    column = "val" if "val" in frame.columns else frame.columns[-1]
    return {
        str(value).strip().upper()
        for value in frame[column].dropna()
        if str(value).strip()
    }


def current_val_ids(study: str, fold: int) -> set[str] | None:
    path = SPLIT_ROOT / "5fold" / study / f"fold_{fold}.csv"
    if not path.is_file():
        return None
    return val_ids(pd.read_csv(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--studies",
        default=",".join(DEFAULT_STUDIES),
        help="逗号分隔的癌种，或 all。默认仅 blca",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只比较旧/新划分差异，不写任何文件",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.studies.strip().lower() == "all":
        studies = list(STUDIES)
    else:
        studies = [item.strip().lower() for item in args.studies.split(",") if item.strip()]
        unknown = sorted(set(studies) - set(STUDIES))
        if unknown:
            print(f"[ERROR] 未知癌种: {', '.join(unknown)}", file=sys.stderr)
            return 2

    header = f"{'study':<10} {'fold':>4} {'legacy':>7} {'current':>8} {'overlap':>8} {'same':>6}"
    print(header)
    print("-" * len(header))
    written = 0
    for study in studies:
        target = SPLIT_ROOT / TARGET_DIR_NAME / study
        for fold in range(N_FOLDS):
            frame = read_legacy_fold(study, fold)
            legacy = val_ids(frame)
            current = current_val_ids(study, fold)
            if current is None:
                current_text, overlap_text, same_text = "-", "-", "-"
            else:
                current_text = str(len(current))
                overlap_text = str(len(legacy & current))
                same_text = str(legacy == current)
            print(
                f"{study:<10} {fold:>4} {len(legacy):>7} {current_text:>8} "
                f"{overlap_text:>8} {same_text:>6}"
            )
            if args.check:
                continue
            target.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target / f"fold_{fold}.csv", index=False)
            written += 1

    if args.check:
        print("\n[check] 未写入任何文件。")
    else:
        print(f"\n[ok] 已恢复 {written} 个折文件到 {SPLIT_ROOT / TARGET_DIR_NAME}")
        print("[warn] 该划分仅用于 v3.3 历史复现；其他方法必须用 5fold / 5fold_uni2h。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
