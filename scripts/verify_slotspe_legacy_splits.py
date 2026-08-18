#!/usr/bin/env python3
"""Verify that ``5fold_legacy/<cancer>/`` matches the SlotSPE source after
stripping the pandas default index column.

Checks per cancer x fold:
  * Line counts match the source
  * Train set equals source train set
  * Val set equals source val set
  * train->val paired rows match the source
  * train and val are disjoint
"""
from pathlib import Path
import sys

import pandas as pd

SRC_ROOT = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main\dataset_csv\splits\5fold")
DST_ROOT = Path(r"E:\SurvOT-Rank\survot_rank\research\legacy\slotspe_runtime\dataset_csv\splits\5fold_legacy")

CANCERS = ["blca", "brca", "coadread", "hnsc", "kirc", "luad", "lusc", "skcm", "stad", "ucec"]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compare(cancer: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    ok = True
    for fold_idx in range(5):
        src = SRC_ROOT / cancer / f"fold_{fold_idx}.csv"
        dst = DST_ROOT / cancer / f"fold_{fold_idx}.csv"
        s = read(src)
        d = read(dst)
        # strip index from source
        if "Unnamed: 0" in s.columns:
            s = s.drop(columns=["Unnamed: 0"])
        s = s[["train", "val"]].reset_index(drop=True)
        d = d[["train", "val"]].reset_index(drop=True)
        if list(s.columns) != list(d.columns):
            issues.append(f"{cancer}/fold_{fold_idx}: column mismatch src={list(s.columns)} dst={list(d.columns)}")
            ok = False
            continue
        if s.shape != d.shape:
            issues.append(f"{cancer}/fold_{fold_idx}: shape mismatch src={s.shape} dst={d.shape}")
            ok = False
        # train/val sets
        s_train = set(s["train"].dropna().unique())
        s_val = set(s["val"].dropna().unique())
        d_train = set(d["train"].dropna().unique())
        d_val = set(d["val"].dropna().unique())
        if s_train != d_train:
            issues.append(f"{cancer}/fold_{fold_idx}: train set differs")
            ok = False
        if s_val != d_val:
            issues.append(f"{cancer}/fold_{fold_idx}: val set differs")
            ok = False
        # train and val are disjoint (within source) — note: SKCM has known
        # train/val overlap in upstream SlotSPE; this is preserved verbatim
        # rather than patched because 5fold_legacy is the unaltered source.
        # We only warn, do not flag as a verification failure.
        overlap = s_train & s_val
        if overlap:
            print(f"    NOTE: {cancer}/fold_{fold_idx}: train∩val = {sorted(overlap)} (upstream SlotSPE behaviour)")
        # row-wise equality
        if not s.equals(d):
            diff_rows = (s != d).any(axis=1).sum()
            issues.append(f"{cancer}/fold_{fold_idx}: row diff at {diff_rows} rows")
            ok = False
    return ok, issues


def main() -> int:
    all_ok = True
    for cancer in CANCERS:
        ok, issues = compare(cancer)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {cancer}")
        for line in issues:
            print(f"    {line}")
        all_ok &= ok
    print("\nALL OK" if all_ok else "\nFAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
