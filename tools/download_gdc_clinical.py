#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 GDC (Genomic Data Commons) 官方 API 下载 TCGA 临床变量数据。

GDC API 是公开的，不需要账号/token（下载的是 Open Access 层级的临床元数据，
不是受控的基因型数据）。本脚本按现有 clinical/all/{study}.csv 里的 case id
列表，批量查询年龄/AJCC 分期/肿瘤分级等字段，另存为一份新的 CSV，供后续
接入 Clinical_Modality 使用（不覆盖/修改现有 clinical/all/{study}.csv）。

用法：
    python tools/download_gdc_clinical.py --study blca
    python tools/download_gdc_clinical.py --study blca --output E:/tcga_clinical/blca_clinical_extra.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

GDC_API_URL = "https://api.gdc.cancer.gov/cases"

# 与本仓库 study 简称到 GDC project_id 的映射
STUDY_TO_PROJECT = {
    "blca": "TCGA-BLCA",
    "brca": "TCGA-BRCA",
    "coadread": "TCGA-COADREAD",  # GDC 里 COAD 与 READ 是分开的两个 project
    "hnsc": "TCGA-HNSC",
    "stad": "TCGA-STAD",
}
# COADREAD 是 SlotSPE/SurvOT-Rank 里的合并癌种命名，GDC 没有这个 project，
# 需要展开成两个真实 project 一起查询。
STUDY_TO_PROJECTS = {
    "blca": ["TCGA-BLCA"],
    "brca": ["TCGA-BRCA"],
    "coadread": ["TCGA-COAD", "TCGA-READ"],
    "hnsc": ["TCGA-HNSC"],
    "stad": ["TCGA-STAD"],
}

FIELDS = [
    "submitter_id",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.tumor_grade",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
]

CLINICAL_CSV_ROOT = Path("survot_rank/research/legacy/slotspe_runtime/dataset_csv/clinical/all")


def load_case_ids(study: str) -> list[str]:
    csv_path = CLINICAL_CSV_ROOT / f"{study}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到现有临床 CSV: {csv_path}，无法确定要查询哪些 case id")
    df = pd.read_csv(csv_path)
    if "case id" not in df.columns:
        raise ValueError(f"{csv_path} 中没有 'case id' 列")
    case_ids = df["case id"].dropna().unique().tolist()
    return case_ids


def query_gdc_batch(project_ids: list[str], case_ids_batch: list[str]) -> list[dict]:
    """查询一批 case id（GDC 用 submitter_id 表示 TCGA barcode）的临床字段。"""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": project_ids}},
            {"op": "in", "content": {"field": "submitter_id", "value": case_ids_batch}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(FIELDS),
        "format": "JSON",
        "size": str(len(case_ids_batch) * 2 + 10),
    }
    url = GDC_API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "survot-rank-clinical-fetch/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["data"]["hits"]
        except urllib.error.URLError as e:
            print(f"[warn] 请求失败（第 {attempt + 1} 次）: {e}", file=sys.stderr)
            time.sleep(2)
    raise RuntimeError(f"GDC API 请求多次失败，case_ids_batch={case_ids_batch[:3]}...")


def flatten_hit(hit: dict) -> dict:
    """把一个 case 的 GDC JSON 记录展平成单行字典（取第一个 diagnosis 记录为主）。"""
    row = {"case id": hit.get("submitter_id")}

    demographic = hit.get("demographic") or {}
    row["gender"] = demographic.get("gender")
    row["race"] = demographic.get("race")
    row["ethnicity"] = demographic.get("ethnicity")
    row["vital_status"] = demographic.get("vital_status")

    diagnoses = hit.get("diagnoses") or []
    if diagnoses:
        # 优先取标记为原发肿瘤（primary）的诊断记录；否则取第一条。
        primary = next(
            (d for d in diagnoses if d.get("classification_of_tumor") == "primary"),
            diagnoses[0],
        )
        row["age_at_diagnosis_days"] = primary.get("age_at_diagnosis")
        row["ajcc_pathologic_stage"] = primary.get("ajcc_pathologic_stage")
        row["ajcc_pathologic_t"] = primary.get("ajcc_pathologic_t")
        row["ajcc_pathologic_n"] = primary.get("ajcc_pathologic_n")
        row["ajcc_pathologic_m"] = primary.get("ajcc_pathologic_m")
        row["tumor_grade"] = primary.get("tumor_grade")
        row["primary_diagnosis"] = primary.get("primary_diagnosis")
        row["morphology"] = primary.get("morphology")
        row["tissue_or_organ_of_origin"] = primary.get("tissue_or_organ_of_origin")
    else:
        for field in [
            "age_at_diagnosis_days",
            "ajcc_pathologic_stage",
            "ajcc_pathologic_t",
            "ajcc_pathologic_n",
            "ajcc_pathologic_m",
            "tumor_grade",
            "primary_diagnosis",
            "morphology",
            "tissue_or_organ_of_origin",
        ]:
            row[field] = None

    return row


def download_clinical(study: str, batch_size: int = 50) -> pd.DataFrame:
    if study not in STUDY_TO_PROJECTS:
        raise ValueError(f"未知 study: {study}，目前支持 {list(STUDY_TO_PROJECTS)}")

    project_ids = STUDY_TO_PROJECTS[study]
    case_ids = load_case_ids(study)
    print(f"[{study}] 共 {len(case_ids)} 个 case id，对应 GDC project(s): {project_ids}")

    rows = []
    seen = set()
    for i in range(0, len(case_ids), batch_size):
        batch = case_ids[i : i + batch_size]
        hits = query_gdc_batch(project_ids, batch)
        for hit in hits:
            row = flatten_hit(hit)
            cid = row["case id"]
            if cid in seen:
                continue
            seen.add(cid)
            rows.append(row)
        print(f"  已查询 {min(i + batch_size, len(case_ids))}/{len(case_ids)}，累计获取 {len(rows)} 条")
        time.sleep(0.3)  # 温和限速，避免触发 GDC 速率限制

    df = pd.DataFrame(rows)
    missing = set(case_ids) - seen
    if missing:
        print(f"[warn] 有 {len(missing)} 个 case id 在 GDC 未查询到记录: {sorted(missing)[:10]}...")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="从 GDC API 下载 TCGA 临床变量（年龄/分期/分级等）")
    parser.add_argument("--study", type=str, required=True, choices=list(STUDY_TO_PROJECTS))
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径，默认写到 E:/tcga_clinical/{study}_clinical_extra.csv",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path("E:/tcga_clinical") / f"{args.study}_clinical_extra.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = download_clinical(args.study, batch_size=args.batch_size)
    df.to_csv(output_path, index=False)
    print(f"[done] 已保存 {len(df)} 行到: {output_path}")
    print(df.head(5).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
