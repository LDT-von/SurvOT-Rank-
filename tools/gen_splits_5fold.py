#!/usr/bin/env python3
"""
Generate 5-fold stratified CV splits for any TCGA study.

Usage:
    python gen_splits_5fold.py --study brca
    python gen_splits_5fold.py --study brca --label_col survival_months_os
    python gen_splits_5fold.py --study luad --n_folds 5 --seed 42

Output format:
    ,train,val
    0,TCGA-XX-XXXX,TCGA-YY-YYYY
    1,...

Stratification key: 4 buckets = (event, time_quartile) on label_col
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset_csv"


def make_strat_key(df: pd.DataFrame, label_col: str, censor_col: str):
    """Return (strat_key_array, sub_df_with_case_id_and_strat)."""
    sub = df[['case id', label_col, censor_col]].dropna().copy()
    sub['event'] = (sub[censor_col] == 0).astype(int)
    sub['time_q'] = pd.qcut(sub[label_col], q=4, labels=False, duplicates='drop')
    sub['time_q'] = sub['time_q'].fillna(0).astype(int)
    sub['strat'] = sub['event'].astype(str) + '_' + sub['time_q'].astype(str)
    return sub['strat'].values, sub


def gen(study: str,
        data_path: str,
        label_col: str,
        censor_col: str,
        n_folds: int,
        seed: int,
        out_dir: str):
    csv = os.path.join(data_path, "clinical", "all", f"{study}.csv")
    assert os.path.isfile(csv), f"missing clinical csv: {csv}"

    df = pd.read_csv(csv, index_col=0)
    print(f"[{study}] loaded {len(df)} rows from {csv}")

    # rows with valid label
    valid_mask = df[label_col].notna() & df[censor_col].notna()
    valid_cases = df.loc[valid_mask, "case id"].tolist()
    print(f"[{study}] {len(valid_cases)} cases have valid {label_col}")

    sub = df.loc[valid_mask].copy()
    strat_key, sub2 = make_strat_key(sub, label_col, censor_col)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assign = np.zeros(len(sub2), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(sub2, strat_key)):
        fold_assign[val_idx] = fold_idx

    sub2['fold'] = fold_assign

    # write per fold (each row = one train/val case pair, indexing from 0)
    target_dir = os.path.join(out_dir, study)
    os.makedirs(target_dir, exist_ok=True)
    n_total = len(sub2)
    print(f"[{study}] writing {n_folds} fold CSVs to {target_dir}")

    for k in range(n_folds):
        train = sub2.loc[sub2['fold'] != k, 'case id'].tolist()
        val = sub2.loc[sub2['fold'] == k, 'case id'].tolist()
        # interleave to match BLCA format: index, train, val
        # (BLCA shows train[i] paired with val[i] but they're independent lists)
        rows = list(zip(train, val))
        out = pd.DataFrame(rows, columns=['train', 'val'])
        out.index.name = ''
        out_path = os.path.join(target_dir, f"fold_{k}.csv")
        out.to_csv(out_path)
        print(f"  fold_{k}.csv : train={len(train)} val={len(val)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--label_col", default="survival_months_dss")
    ap.add_argument("--censor_col", default="censorship_dss")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(DEFAULT_DATA_PATH / "splits" / "5fold"))
    args = ap.parse_args()
    gen(args.study, args.data_path, args.label_col, args.censor_col,
        args.n_folds, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
