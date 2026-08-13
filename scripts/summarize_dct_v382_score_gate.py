#!/usr/bin/env python3
"""Summarize DCT score-gate curves and apply a pre-registered promotion rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.run_dct_v382_score_gate import (
        DEFAULT_CANCERS,
        DEFAULT_FOLDS,
        VARIANTS,
    )
except ModuleNotFoundError:
    from run_dct_v382_score_gate import DEFAULT_CANCERS, DEFAULT_FOLDS, VARIANTS


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _folds(value: str) -> list[int]:
    return [int(item) for item in _csv(value)]


def _curve(root: Path, cancer: str, fold: int) -> Path | None:
    matches = sorted((root / cancer).rglob(f"epoch_curve_fold{fold}.csv"))
    if len(matches) > 1:
        raise ValueError(f"duplicate curves for {cancer} fold{fold}: {matches}")
    return matches[0] if matches else None


def fold_record(root: Path, variant: str, cancer: str, fold: int) -> dict[str, object]:
    path = _curve(root, cancer, fold)
    record: dict[str, object] = {
        "variant": variant,
        "cancer": cancer,
        "fold": fold,
        "status": "missing",
    }
    if path is None:
        return record
    curve = pd.read_csv(path)
    if curve.empty or not {"epoch", "val_cindex"}.issubset(curve.columns):
        record["status"] = "invalid_curve"
        return record
    values = pd.to_numeric(curve["val_cindex"], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        record["status"] = "nonfinite"
        return record
    best_pos = int(np.argmax(values))
    last5 = float(np.mean(values[-5:]))
    record.update(
        {
            "status": "ok",
            "epochs": len(values),
            "best_cindex": float(values[best_pos]),
            "best_epoch": int(curve["epoch"].iloc[best_pos]),
            "last5_cindex": last5,
            "best_last_gap": float(values[best_pos] - last5),
            "curve": str(path),
        }
    )
    return record


def promotion(candidate: pd.DataFrame, control: pd.DataFrame) -> dict[str, object]:
    keys = ["cancer", "fold"]
    if not candidate["status"].eq("ok").all() or not control["status"].eq("ok").all():
        return {"promote": False, "complete": False, "reason": "missing or invalid matched curves"}
    joined = candidate.merge(control, on=keys, suffixes=("_candidate", "_control"))
    expected = len(candidate)
    if len(joined) != expected:
        return {"promote": False, "complete": False, "reason": "unmatched cancer/fold rows"}
    joined["best_gain"] = joined["best_cindex_candidate"] - joined["best_cindex_control"]
    joined["last5_gain"] = joined["last5_cindex_candidate"] - joined["last5_cindex_control"]
    cancer_gain = joined.groupby("cancer")["best_gain"].mean()
    macro_gain = float(joined["best_gain"].mean())
    wins = int((cancer_gain > 0.0).sum())
    noninferior = bool((cancer_gain >= -0.005).all())
    skcm_gain = float(cancer_gain.get("skcm", np.nan))
    stability_gain = float(joined["last5_gain"].mean())
    passed = (
        macro_gain >= 0.005
        and wins >= 2
        and noninferior
        and np.isfinite(skcm_gain)
        and skcm_gain >= 0.005
        and stability_gain >= 0.0
    )
    return {
        "promote": bool(passed),
        "complete": True,
        "criteria": {
            "macro_best_gain_ge_0.005": macro_gain,
            "cancer_wins_ge_2": wins,
            "every_cancer_noninferior_minus_0.005": noninferior,
            "skcm_gain_ge_0.005": skcm_gain,
            "macro_last5_gain_ge_0": stability_gain,
            "per_cancer_best_gain": {key: float(value) for key, value in cancer_gain.items()},
        },
        "reason": "all pre-registered gates passed" if passed else "one or more gates failed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=Path("results/dct_v3.8.2_score_gate"))
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("results/dct_v3.8.2/robust/fixed_full"),
    )
    parser.add_argument("--variants", type=_csv, default=list(VARIANTS))
    parser.add_argument("--cancers", type=_csv, default=list(DEFAULT_CANCERS))
    parser.add_argument("--folds", type=_folds, default=list(DEFAULT_FOLDS))
    parser.add_argument("--output", type=Path, default=Path("results/dct_v3.8.2_score_gate/report.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    control_rows = [
        fold_record(args.control_root, "fixed_full", cancer, fold)
        for cancer in args.cancers
        for fold in args.folds
    ]
    rows: list[dict[str, object]] = []
    decisions: dict[str, object] = {}
    control = pd.DataFrame(control_rows)
    for variant in args.variants:
        candidate_rows = [
            fold_record(args.candidate_root / variant, variant, cancer, fold)
            for cancer in args.cancers
            for fold in args.folds
        ]
        rows.extend(candidate_rows)
        decisions[variant] = promotion(pd.DataFrame(candidate_rows), control)
    report = pd.DataFrame(control_rows + rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    decision_path = args.output.with_name(f"{args.output.stem}_promotion.json")
    decision_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.to_string(index=False))
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    print(f"[report] {args.output}")
    print(f"[promotion] {decision_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
