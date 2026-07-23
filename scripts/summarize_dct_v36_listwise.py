#!/usr/bin/env python3
"""Summarize DCT v3.6 fold curves and apply the pre-registered promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VARIANTS = ("nll", "ipcw", "etar", "ipcw_etar", "gpl", "tcl")


def _csv_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _fold_list(value: str) -> list[int]:
    return [int(item) for item in _csv_list(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/dct_v3.6_listwise")
    parser.add_argument("--variants", type=_csv_list, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--cancers", type=_csv_list, default=["blca", "brca"])
    parser.add_argument("--folds", type=_fold_list, default=[0, 2])
    parser.add_argument("--output", default="results/dct_v3.6_listwise/report.csv")
    return parser


def _fold_record(root: Path, variant: str, cancer: str, fold: int) -> dict:
    curve_path = root / variant / cancer / f"epoch_curve_fold{fold}.csv"
    record = {
        "variant": variant,
        "cancer": cancer,
        "fold": fold,
        "status": "missing",
    }
    if not curve_path.exists():
        return record
    curve = pd.read_csv(curve_path)
    required = {"epoch", "val_cindex"}
    if not required.issubset(curve.columns) or curve.empty:
        record["status"] = "invalid_curve"
        return record

    values = pd.to_numeric(curve["val_cindex"], errors="coerce")
    finite = np.isfinite(values.to_numpy())
    if not finite.any():
        record["status"] = "nonfinite"
        return record
    best_position = int(np.nanargmax(values.to_numpy()))
    best = float(values.iloc[best_position])
    last5 = float(values.iloc[-5:].mean())
    record.update(
        {
            "status": "ok" if finite.all() else "partial_nonfinite",
            "epochs": int(len(curve)),
            "best_cindex": best,
            "best_epoch": int(curve["epoch"].iloc[best_position]),
            "last5_cindex": last5,
            "best_last_gap": best - last5,
        }
    )
    optional = (
        "val_cindex_ipcw",
        "val_IBS",
        "val_iauc",
        "train_ipcw_pairs",
        "train_etar_pairs",
        "train_etar_evidence",
        "train_etar_uncertainty",
        "train_listwise_lists",
        "train_listwise_avg_risk_set",
        "train_listwise_stage_coverage",
        "train_listwise_finite_scores",
        "train_listwise_finite_gradients",
    )
    for column in optional:
        if column in curve:
            record[column] = float(
                pd.to_numeric(curve[column], errors="coerce").iloc[-5:].mean()
            )
    return record


def _promotion_gate(report: pd.DataFrame, cancers: list[str], folds: list[int]) -> dict:
    required = report[
        report["variant"].isin(["ipcw", "gpl", "tcl"])
        & report["cancer"].isin(cancers)
        & report["fold"].isin(folds)
    ]
    expected = len(cancers) * len(folds) * 3
    complete = (
        len(required) == expected
        and required["status"].eq("ok").all()
        and np.isfinite(required["best_cindex"]).all()
        and np.isfinite(required["last5_cindex"]).all()
    )
    decision = {
        "promote": False,
        "complete": bool(complete),
        "criteria": {},
    }
    if not complete:
        decision["reason"] = "matched IPCW/GPL/TCL fold curves are incomplete or non-finite"
        return decision

    indexed = required.set_index(["variant", "cancer", "fold"]).sort_index()
    cancer_noninferior = {}
    for cancer in cancers:
        tcl_mean = indexed.loc[("tcl", cancer), "best_cindex"].mean()
        ipcw_mean = indexed.loc[("ipcw", cancer), "best_cindex"].mean()
        cancer_noninferior[cancer] = bool(tcl_mean >= ipcw_mean - 0.005)

    tcl = indexed.loc["tcl"]
    ipcw = indexed.loc["ipcw"]
    gpl = indexed.loc["gpl"]
    last5_gain = float(tcl["last5_cindex"].mean() - ipcw["last5_cindex"].mean())
    gap_reduction = float(
        ipcw["best_last_gap"].mean() - tcl["best_last_gap"].mean()
    )
    stability = last5_gain >= 0.01 or gap_reduction >= 0.02
    matched = tcl[["best_cindex"]].join(
        gpl[["best_cindex"]],
        lsuffix="_tcl",
        rsuffix="_gpl",
    )
    tcl_wins = int(
        (matched["best_cindex_tcl"] > matched["best_cindex_gpl"]).sum()
    )
    mean_tcl_over_gpl = float(
        matched["best_cindex_tcl"].mean() - matched["best_cindex_gpl"].mean()
    )
    transport_specific = tcl_wins >= 3 or mean_tcl_over_gpl >= 0.005

    decision["criteria"] = {
        "per_cancer_best_noninferior": cancer_noninferior,
        "last5_gain": last5_gain,
        "gap_reduction": gap_reduction,
        "stability_pass": bool(stability),
        "tcl_fold_wins_over_gpl": tcl_wins,
        "mean_tcl_over_gpl": mean_tcl_over_gpl,
        "transport_specific_pass": bool(transport_specific),
    }
    decision["promote"] = bool(
        all(cancer_noninferior.values()) and stability and transport_specific
    )
    decision["reason"] = (
        "all screening gates passed"
        if decision["promote"]
        else "one or more pre-registered screening gates failed"
    )
    return decision


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    records = [
        _fold_record(root, variant, cancer, fold)
        for variant in args.variants
        for cancer in args.cancers
        for fold in args.folds
    ]
    report = pd.DataFrame(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    decision = _promotion_gate(report, args.cancers, args.folds)
    decision_path = output.with_name(f"{output.stem}_promotion.json")
    with open(decision_path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(report.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"[report] {output}")
    print(f"[promotion] {decision_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
