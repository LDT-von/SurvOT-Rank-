#!/usr/bin/env python3
"""Merge age/gender/race/stage/grade from /data1/TCGA_clinical_Data into the
project's clinical CSV per case id.

Source:      /data1/TCGA_clinical_Data/{STUDY}/clinical.csv
             columns: case_id, submitter_id, age, gender, race, vital_status,
                      primary_diagnosis, tumor_stage, grade, days_to_death
Target:      survot_rank/.../dataset_csv/clinical/all/{study}.csv
             adds: patient_id, age, gender, race, tumor_stage, grade

Usage:  python tools/enrich_clinical_with_trisurv.py --study brca
        python tools/enrich_clinical_with_trisurv.py --studies brca luad
"""
import argparse
from pathlib import Path
import pandas as pd

TRISURV = Path("/data1/TCGA_clinical_Data")
PROJECT = Path(__file__).resolve().parents[1]
SLOTSPE = PROJECT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset_csv"

# Source rows have either:
#  - case_id = TCGA-XX-XXXX style (12 char), submitter_id == case_id
#  - case_id = UUID, submitter_id = TCGA-XX-XXXX
# We always join on TCGA-style submitter_id (the canonical patient id).
SOURCE_KEY = "submitter_id"
TARGET_KEY = "case id"
COL_MAP = {
    "submitter_id": "case id",
    "age": "age",
    "gender": "gender",
    "race": "race",
    "tumor_stage": "tumor_stage",
    "grade": "grade",
}


def enrich(study: str):
    src = TRISURV / study.upper() / "clinical.csv"
    dst = SLOTSPE / "clinical" / "all" / f"{study}.csv"

    if not src.exists():
        print(f"[{study}] missing source {src}")
        return False
    if not dst.exists():
        print(f"[{study}] missing target {dst}")
        return False

    src_df = pd.read_csv(src)
    src_df.columns = [c.lower() for c in src_df.columns]
    if SOURCE_KEY not in src_df.columns:
        print(f"[{study}] missing column {SOURCE_KEY} in {src}")
        return False
    keep = [c for c in COL_MAP if c in src_df.columns]
    src_df = src_df[keep].rename(columns=COL_MAP).drop_duplicates(subset=[TARGET_KEY])

    dst_df = pd.read_csv(dst)
    # drop any existing merged columns to make this idempotent
    for col in ("age", "gender", "race", "tumor_stage", "grade"):
        if col in dst_df.columns and col != TARGET_KEY:
            dst_df = dst_df.drop(columns=[col])

    merged = dst_df.merge(src_df, on=TARGET_KEY, how="left")
    merged.to_csv(dst, index=False)
    print(f"[{study}] enriched: {len(merged)} rows  "
          f"(age non-null: {merged['age'].notna().sum()}, "
          f"gender non-null: {merged['gender'].notna().sum()})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", default=["brca", "luad"])
    args = ap.parse_args()
    for s in args.studies:
        enrich(s)


if __name__ == "__main__":
    main()
