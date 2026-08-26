#!/usr/bin/env python3
"""
Generate 5-fold splits for missing TCGA cancers.

Output: {data_path}/splits/5fold/{study}/fold_{0..4}.csv
Format: column headers `,train,val`, each row pairs a train patient with a val patient.
"""

import os
import sys
import csv
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_PATH = 'survot_rank/research/legacy/slotspe_runtime/dataset_csv'
CLINICAL_DIR = os.path.join(DATA_PATH, 'clinical', 'all')
SPLITS_BASE = os.path.join(DATA_PATH, 'splits', '5fold')

# 6 cancers missing 5-fold splits
MISSING = ['coadread', 'kirc', 'lusc', 'hnsc', 'skcm', 'stad']


def generate_splits(study: str, n_splits: int = 5, random_state: int = 42):
    clinical_csv = os.path.join(CLINICAL_DIR, f'{study}.csv')
    if not os.path.exists(clinical_csv):
        print(f"  SKIP {study}: no clinical CSV at {clinical_csv}")
        return

    df = pd.read_csv(clinical_csv)
    if 'case id' not in df.columns:
        print(f"  SKIP {study}: no 'case id' column")
        return

    # Use DSS event indicator for stratification
    event_col = 'censorship_dss'
    if event_col not in df.columns:
        print(f"  SKIP {study}: no '{event_col}' column for stratification")
        return

    patient_ids = df['case id'].tolist()
    events = df[event_col].fillna(0).astype(int).tolist()
    n_patients = len(patient_ids)
    print(f"  {study}: {n_patients} patients, events={sum(events)}/{n_patients}")

    # Create output dir
    out_dir = os.path.join(SPLITS_BASE, study)
    os.makedirs(out_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_idx = 0

    for train_idx, val_idx in skf.split(patient_ids, events):
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        train_events = sum(events[i] for i in train_idx)
        val_events = sum(events[i] for i in val_idx)

        # Create train-val pairs (cycling through val patients)
        n_pairs = len(train_patients)
        rows = []
        for i in range(n_pairs):
            train_p = train_patients[i % len(train_patients)]
            val_p = val_patients[i % len(val_patients)]
            rows.append((train_p, val_p))

        out_path = os.path.join(out_dir, f'fold_{fold_idx}.csv')
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'train', 'val'])
            for idx, (tp, vp) in enumerate(rows):
                writer.writerow([idx, tp, vp])

        # Strip duplicates to count actual unique patients
        unique_train = len(set(r[0] for r in rows))
        unique_val = len(set(r[1] for r in rows))

        print(f"    fold_{fold_idx}: {n_pairs} pairs, "
              f"train={unique_train}(events={train_events}), "
              f"val={unique_val}(events={val_events})")
        fold_idx += 1

    print(f"  -> Done: {out_dir}/fold_0..{n_splits-1}.csv")


def main():
    print("Generating 5-fold splits for missing cancers...\n")
    for study in MISSING:
        try:
            generate_splits(study)
        except Exception as e:
            print(f"  ERROR {study}: {e}")
        print()

    # Verify
    print("=" * 50)
    print("Verification:")
    for study in MISSING:
        d = os.path.join(SPLITS_BASE, study)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            print(f"  {study}: {len(files)} files: {files}")
        else:
            print(f"  {study}: MISSING")

    # Also verify existing splits
    for study in ['brca', 'blca', 'ucec', 'luad']:
        d = os.path.join(SPLITS_BASE, study)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            print(f"  {study}: {len(files)} files: {files}")


if __name__ == '__main__':
    main()
