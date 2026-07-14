#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 GDC 下载的原始临床字段（tools/download_gdc_clinical.py 的输出）编码成
数值特征列，并与现有 clinical/all/{study}.csv 按 case id 左连接合并，另存为
一份新的 CSV（不覆盖/不修改原有 clinical/all/{study}.csv）。

编码规则：
- age_at_diagnosis_days -> age_at_diagnosis_years（除以 365.25），缺失保持 NaN
- ajcc_pathologic_stage -> stage_ordinal（有序编码：0a/I/II/IIB/IIC/III/IIIA/IV
  映射为单调递增的数值），缺失或非标准取值保持 NaN
- tumor_grade -> grade_ordinal（Low Grade=0, High Grade=1），"Not Reported"/
  "Unknown"/缺失一律视为缺失（NaN）

缺失值全部保持 NaN（不做均值填充/众数填充），交给 ClinicalEncoder 的可学习
填充参数在训练时处理，避免在数据预处理阶段引入人为假设。

用法：
    python tools/build_clinical_features.py --study blca
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLINICAL_CSV_ROOT = Path("survot_rank/research/legacy/slotspe_runtime/dataset_csv/clinical/all")

# AJCC 分期有序编码：数值越大分期越晚。非标准/罕见取值（如 "Stage 0a"）单独
# 处理为 0（视为分期最早），未出现在映射表中的取值编码为 NaN。
STAGE_ORDINAL_MAP = {
    "Stage 0a": 0.0,
    "Stage 0is": 0.0,
    "Stage I": 1.0,
    "Stage IA": 1.0,
    "Stage IB": 1.0,
    "Stage II": 2.0,
    "Stage IIA": 2.0,
    "Stage IIB": 2.5,
    "Stage IIC": 2.7,
    "Stage III": 3.0,
    "Stage IIIA": 3.3,
    "Stage IIIB": 3.6,
    "Stage IIIC": 3.8,
    "Stage IV": 4.0,
    "Stage IVA": 4.0,
    "Stage IVB": 4.2,
}

GRADE_ORDINAL_MAP = {
    "Low Grade": 0.0,
    "High Grade": 1.0,
    # "Not Reported" / "Unknown" 均不在映射表中 -> 编码为 NaN
}

# 本次接入训练管线的三个数值特征列，顺序即 Clinical_Encoder 输入张量的列顺序。
CLINICAL_FEATURE_COLS = ["age_at_diagnosis_years", "stage_ordinal", "grade_ordinal"]


def encode_clinical_extra(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"case id": df["case id"]})
    out["age_at_diagnosis_years"] = df["age_at_diagnosis_days"] / 365.25
    out["stage_ordinal"] = df["ajcc_pathologic_stage"].map(STAGE_ORDINAL_MAP)
    out["grade_ordinal"] = df["tumor_grade"].map(GRADE_ORDINAL_MAP)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="把 GDC 原始临床字段编码为数值特征并合并进现有临床 CSV")
    parser.add_argument("--study", type=str, required=True)
    parser.add_argument(
        "--extra-csv",
        type=str,
        default=None,
        help="tools/download_gdc_clinical.py 的输出路径，默认 E:/tcga_clinical/{study}_clinical_extra.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出合并后的 CSV 路径，默认写到 clinical/all/{study}_with_clinical_features.csv"
        "（新文件，不覆盖原 clinical/all/{study}.csv）",
    )
    args = parser.parse_args()

    extra_csv = Path(args.extra_csv) if args.extra_csv else Path("E:/tcga_clinical") / f"{args.study}_clinical_extra.csv"
    if not extra_csv.exists():
        print(f"错误：找不到 {extra_csv}，请先运行 tools/download_gdc_clinical.py", file=sys.stderr)
        return 1

    base_csv = CLINICAL_CSV_ROOT / f"{args.study}.csv"
    if not base_csv.exists():
        print(f"错误：找不到现有临床 CSV {base_csv}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else CLINICAL_CSV_ROOT / f"{args.study}_with_clinical_features.csv"
    )

    base_df = pd.read_csv(base_csv)
    extra_raw_df = pd.read_csv(extra_csv)
    encoded_df = encode_clinical_extra(extra_raw_df)

    merged = base_df.merge(encoded_df, on="case id", how="left")

    n_total = len(merged)
    print(f"[{args.study}] 合并后总行数: {n_total}（应与原 clinical/all/{args.study}.csv 行数一致）")
    for col in CLINICAL_FEATURE_COLS:
        n_missing = merged[col].isna().sum()
        print(f"  {col}: 缺失 {n_missing}/{n_total} ({100*n_missing/n_total:.1f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[done] 已保存: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
