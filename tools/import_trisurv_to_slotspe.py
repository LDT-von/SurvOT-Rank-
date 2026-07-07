#!/usr/bin/env python3
"""
Import Tri-Surv/DeReF data into the packaged SurvOT-Rank data layout.

Input:  /data1/Tri-Surv/csv/tcga_{study}_all_clean.csv   (case_id as index, ~20K cols)
        /data1/Tri-Surv/splits/5foldcv/{study}/splits_*.csv  (5 files, train/val 2 cols)

Output: survot_rank/research/legacy/slotspe_runtime/dataset_csv/clinical/all/{study}.csv
        (cols: case id, survival_months_dss, censorship_dss, wsi)
        survot_rank/research/legacy/slotspe_runtime/dataset_csv/raw_rna_data_inter/{study}_rna_inter.csv
        (rows: gene names, cols: case_id)
        survot_rank/research/legacy/slotspe_runtime/dataset_csv/splits/5fold/{study}/fold_*.csv
        (renamed from splits_*.csv; cols: ,train,val)

Note: Tri-Surv uses OS (survival_months), we map it to survival_months_dss/censorship_dss
for compatibility with the legacy dataset loader. The split files already match
the expected (,train,val) format.
"""
import argparse
import os
import shutil
from pathlib import Path
import pandas as pd

TRISURV = Path("/data1/Tri-Surv")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLOTSPE = PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset_csv"

STUDIES = ["blca", "brca", "gbmlgg", "luad", "ucec"]


def import_study(study: str, force: bool = False, verbose: bool = True):
    src_csv = TRISURV / "csv" / f"tcga_{study}_all_clean.csv"
    src_split_dir = TRISURV / "splits" / "5foldcv" / f"tcga_{study}"

    if not src_csv.exists():
        print(f"[{study}] missing {src_csv}")
        return False

    # output paths
    out_clinical = SLOTSPE / "clinical" / "all" / f"{study}.csv"
    out_rna = SLOTSPE / "raw_rna_data_inter" / f"{study}_rna_inter.csv"
    out_split_dir = SLOTSPE / "splits" / "5fold" / study

    if (out_clinical.exists() or out_rna.exists()) and not force:
        print(f"[{study}] clinical/rna already exist (use --force to overwrite)")
        # only force-split copy
        if not out_split_dir.exists():
            out_split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                src = src_split_dir / f"splits_{i}.csv"
                dst = out_split_dir / f"fold_{i}.csv"
                shutil.copy(src, dst)
                print(f"  copied {src.name} -> {dst.name}")
        return True

    print(f"[{study}] reading {src_csv} ...")
    df = pd.read_csv(src_csv, index_col=0)
    print(f"  loaded {len(df)} rows, {len(df.columns)} cols")

    # ----- clinical.csv -----
    clin = pd.DataFrame({
        "case id": df.index,
        "survival_months_dss": df["survival_months"].values,
        "censorship_dss": df["censorship"].values,
        "wsi": df["slide_id"].values,
    })
    clin = clin.dropna(subset=["survival_months_dss", "censorship_dss"])
    out_clinical.parent.mkdir(parents=True, exist_ok=True)
    clin.to_csv(out_clinical, index=False)
    print(f"  -> wrote {out_clinical} ({len(clin)} rows)")

    # ----- rna_inter.csv -----
    # Tri-Surv: rows=case, cols={gene}_rnaseq | {gene}_cnv | mutation_name
    # Legacy dataset loader: rows=gene, cols=case_id
    rna_cols = [c for c in df.columns if "_rnaseq" in c]
    if not rna_cols:
        print(f"  [{study}] no _rnaseq cols; using all non-metadata cols")
        # fallback: take columns from 'train' onwards (metadata ends at train)
        meta = {"slide_id", "site", "is_female", "oncotree_code", "age",
                "survival_months", "censorship", "train"}
        rna_cols = [c for c in df.columns if c not in meta]
    rna = df[rna_cols].T  # rows=gene, cols=case
    # strip '_rnaseq' suffix for clean gene names
    rna.index = [c.replace("_rnaseq", "") for c in rna.index]
    out_rna.parent.mkdir(parents=True, exist_ok=True)
    rna.to_csv(out_rna)
    print(f"  -> wrote {out_rna} ({rna.shape[0]} genes x {rna.shape[1]} cases)")

    # ----- splits -----
    if not src_split_dir.exists():
        print(f"  [{study}] missing {src_split_dir}, skipping splits")
        return True
    out_split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        src = src_split_dir / f"splits_{i}.csv"
        if not src.exists():
            print(f"  [{study}] missing {src}")
            continue
        dst = out_split_dir / f"fold_{i}.csv"
        shutil.copy(src, dst)
        if verbose:
            print(f"  copied {src.name} -> {dst.name}")
    print(f"[{study}] DONE")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", default=STUDIES)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing clinical/rna CSV")
    args = ap.parse_args()
    for s in args.studies:
        import_study(s, force=args.force)


if __name__ == "__main__":
    main()
