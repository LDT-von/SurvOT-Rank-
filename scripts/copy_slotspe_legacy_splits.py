#!/usr/bin/env python3
"""Copy SlotSPE-main 5-fold splits into the project 5fold_legacy directory.

Mirrors the format already used for ``5fold_legacy/blca``:
- Drop pandas' default index column (`Unnamed: 0`) so the file has only
  `train,val` columns.
- Drop empty train rows (some folds have a few trailing rows where train is
  empty because patient count is odd).

Source: C:/Users/<user>/Desktop/<paper>/SlotSPE-main/SlotSPE-main/dataset_csv/splits/5fold/
Target: <repo>/survot_rank/research/legacy/slotspe_runtime/dataset_csv/splits/5fold_legacy/
"""
from pathlib import Path
import shutil
import sys

import pandas as pd

SRC_ROOT = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main\dataset_csv\splits\5fold")
DST_ROOT = Path(r"E:\SurvOT-Rank\survot_rank\research\legacy\slotspe_runtime\dataset_csv\splits\5fold_legacy")

CANCERS = ["blca", "brca", "coadread", "hnsc", "kirc", "luad", "lusc", "skcm", "stad", "ucec"]


def process_one(cancer: str) -> list[str]:
    src_dir = SRC_ROOT / cancer
    dst_dir = DST_ROOT / cancer
    dst_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    for fold_idx in range(5):
        src = src_dir / f"fold_{fold_idx}.csv"
        dst = dst_dir / f"fold_{fold_idx}.csv"
        df = pd.read_csv(src)
        # drop pandas default index column if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # ensure exactly the canonical schema `train,val`
        if not {"train", "val"}.issubset(df.columns):
            raise RuntimeError(f"{src}: unexpected columns {list(df.columns)}")
        df = df[["train", "val"]]
        df.to_csv(dst, index=False)
        log.append(
            f"  {cancer}/fold_{fold_idx}.csv  shape={df.shape}  "
            f"paired_rows={df[['train','val']].dropna(how='any').shape[0]}"
        )
    return log


def main() -> int:
    overall: list[str] = []
    for cancer in CANCERS:
        print(f"== {cancer} ==")
        try:
            overall.extend(process_one(cancer))
        except FileNotFoundError as e:
            print(f"  MISSING: {e}", file=sys.stderr)
            return 1
        for line in overall[-5:]:
            print(line)
    print(f"\nWrote {len(CANCERS)*5} files under {DST_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
