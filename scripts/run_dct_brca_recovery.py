#!/usr/bin/env python3
"""Isolated BRCA causal audit and recovery screening for DCT.

This runner never edits or reuses another cancer's result directory.  The
protocol variants form a causal ladder from the published v3.3 recipe to the
v3.5R recipe, followed by conservative BRCA-only recovery candidates.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE = {
    "batch_size": 8,
    "max_epochs": 50,
    "grad_accum_steps": 1,
    "warmup_epochs": 5,
    "grad_clip_norm": 1.0,
    "early_stop_patience": 0,
    "fit_bins_on_train": False,
    "event_sampling_fraction": 0.0,
    "event_stratified_batches": False,
    "seed": 3,
    "lr": 0.0005,
    "opt": "adamW",
    "reg": 0.0005,
    "scheduler": "cosine",
    "eta_min": 0.000001,
    "bag_loss": "nll_surv",
    "alpha_surv": 0.15,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_memory_size": 0,
    "dct_lambda_ot": 0.0,
    "dct_lambda_rank": 0.0,
    "dct_lambda_anchor": 0.0,
    "dct_lambda_stage_risk": 0.0,
    "dct_lambda_coordinate": 0.0,
    "dct_evidence_marginal_strength": 1.0,
    "dct_geometry_reliability_strength": 0.0,
    "dct_slot_eval_seed": 1729,
}

# ref -> det -> bin -> strat is a one-change-at-a-time causal chain.
# a30/norank/reg are BRCA-only recovery candidates relative to det.
VARIANTS = {
    "ref": {
        "label": "exact v3.3 protocol control (stochastic validation slots)",
        "dct_slot_init_mode": "gaussian",
    },
    "det": {
        "label": "v3.3 with deterministic validation slots",
        "dct_slot_init_mode": "deterministic",
    },
    "det_legacy": {
        "label": "det with legacy SlotSPE equal-width binning (pd.cut bins=4)",
        "dct_slot_init_mode": "deterministic",
        "binning_mode": "legacy_equal_width",
    },
    "bin": {
        "label": "det + train-fold survival bins",
        "dct_slot_init_mode": "deterministic",
        "fit_bins_on_train": True,
    },
    "bin_legacy": {
        "label": "bin with legacy SlotSPE equal-width binning",
        "dct_slot_init_mode": "deterministic",
        "fit_bins_on_train": True,
        "binning_mode": "legacy_equal_width",
    },
    "strat": {
        "label": "bin + patient-complete event-stratified batches (v3.5R)",
        "dct_slot_init_mode": "deterministic",
        "fit_bins_on_train": True,
        "event_stratified_batches": True,
    },
    "strat_legacy": {
        "label": "strat with legacy SlotSPE equal-width binning",
        "dct_slot_init_mode": "deterministic",
        "fit_bins_on_train": True,
        "event_stratified_batches": True,
        "binning_mode": "legacy_equal_width",
    },
    "a30": {
        "label": "det + moderate uncensored-event emphasis",
        "dct_slot_init_mode": "deterministic",
        "alpha_surv": 0.30,
    },
    "a30_legacy": {
        "label": "a30 with legacy SlotSPE equal-width binning",
        "dct_slot_init_mode": "deterministic",
        "alpha_surv": 0.30,
        "binning_mode": "legacy_equal_width",
    },
    "norank": {
        "label": "det + NLL-only control for sparse BRCA events",
        "dct_slot_init_mode": "deterministic",
        "dct_lambda_ipcw_rank": 0.0,
    },
    "norank_legacy": {
        "label": "norank with legacy SlotSPE equal-width binning",
        "dct_slot_init_mode": "deterministic",
        "dct_lambda_ipcw_rank": 0.0,
        "binning_mode": "legacy_equal_width",
    },
    "reg": {
        "label": "det + conservative BRCA optimizer",
        "dct_slot_init_mode": "deterministic",
        "lr": 0.0002,
        "reg": 0.001,
    },
    "reg_legacy": {
        "label": "reg with legacy SlotSPE equal-width binning",
        "dct_slot_init_mode": "deterministic",
        "lr": 0.0002,
        "reg": 0.001,
        "binning_mode": "legacy_equal_width",
    },
}


def parse_csv(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid {name}: {unknown or values}; choose from {', '.join(allowed)}"
        )
    return values


def parse_variants(value: str) -> list[str]:
    return parse_csv(value, tuple(VARIANTS), "variant")


def parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from error
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be selected from 0,1,2,3,4")
    return folds


def override_args(values: dict[str, object]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        result.extend(("--set", f"{key}={value}"))
    return result


def build_command(
    python_bin: str, variant: str, fold: int, gpu: str, num_workers: str, *, smoke: bool
) -> tuple[list[str], Path]:
    result_root = Path("results") / ("dct_brca_recovery_smoke" if smoke else "dct_brca_recovery")
    result_dir = result_root / variant
    values = dict(BASE)
    values.update(VARIANTS[variant])
    values.pop("label", None)
    values.update({
        "k_start": fold,
        "k_end": fold + 1,
        "gpu": gpu,
        "num_workers": num_workers,
        "results_dir": result_dir.as_posix(),
        "specific_simple": f"dct_brca_recovery_{variant}{'_smoke' if smoke else ''}",
    })
    if smoke:
        values.update({"max_epochs": 1, "max_smoke_batches": 2})
    command = [
        python_bin, "-m", "survot_rank.cli", "train", "--config",
        "configs/distributional_counterfactual_transport_brca.yaml",
        *override_args(values),
    ]
    return command, result_dir


def audit_results(variants: list[str], folds: list[int], *, smoke: bool = False) -> int:
    root = Path("results") / ("dct_brca_recovery_smoke" if smoke else "dct_brca_recovery")
    rows: list[dict[str, object]] = []
    failures = 0
    required = ("val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc", "val_loss")
    for variant in variants:
        for fold in folds:
            result_dir = root / variant
            curve_path = result_dir / f"epoch_curve_fold{fold}.csv"
            row: dict[str, object] = {"variant": variant, "fold": fold, "curve": str(curve_path)}
            problems: list[str] = []
            if not curve_path.exists():
                problems.append("missing_curve")
            else:
                curve = pd.read_csv(curve_path)
                if curve.empty:
                    problems.append("empty_curve")
                missing = [name for name in required if name not in curve.columns]
                if missing:
                    problems.append("missing_columns:" + ",".join(missing))
                if not curve.empty and not missing:
                    numeric = curve.loc[:, required].apply(pd.to_numeric, errors="coerce")
                    for name in required:
                        finite_ratio = float(numeric[name].map(math.isfinite).mean())
                        row[f"{name}_finite_ratio"] = finite_ratio
                        if finite_ratio < 1.0:
                            problems.append(f"nonfinite_{name}")
                    row["epochs"] = len(curve)
                    if numeric["val_cindex"].map(math.isfinite).any():
                        best_pos = numeric["val_cindex"].idxmax()
                        row.update({
                            "best_epoch": int(curve.loc[best_pos, "epoch"]),
                            "best_cindex": float(numeric.loc[best_pos, "val_cindex"]),
                            "last_cindex": float(numeric.iloc[-1]["val_cindex"]),
                        })
                    else:
                        problems.append("no_finite_cindex")
            if not (result_dir / f"model_best_s{fold}.pth").exists():
                problems.append("missing_best_checkpoint")
            if not smoke and not (result_dir / f"split_{fold}_results_final.pkl").exists():
                problems.append("missing_final_results")
            row["status"] = "ok" if not problems else "FAIL"
            row["problems"] = ";".join(problems)
            failures += bool(problems)
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    report = root / "integrity_audit.csv"
    pd.DataFrame(rows).to_csv(report, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[audit] report={report} failures={failures}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("plan", "smoke", "run", "audit", "deep"), nargs="?", default="plan"
    )
    parser.add_argument("--variants", type=parse_variants, default=parse_variants("ref,det,bin,strat"))
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,2"))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--force", action="store_true")
    return parser


def run_jobs(args, *, smoke: bool) -> int:
    for variant in args.variants:
        for fold in args.folds:
            command, result_dir = build_command(
                args.python_bin, variant, fold, args.gpu, args.num_workers, smoke=smoke
            )
            final_path = result_dir / f"split_{fold}_results_final.pkl"
            if final_path.exists() and not args.force and not smoke:
                print(f"[skip] BRCA {variant} fold{fold}: {final_path}")
                continue
            print(f"[BRCA/{variant}/fold{fold}] {VARIANTS[variant]['label']}")
            print("$ " + " ".join(command))
            if args.mode == "plan":
                continue
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                print(f"[ERROR] BRCA {variant} fold{fold} returned {completed.returncode}")
                return completed.returncode
    return 0


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    if args.mode == "audit":
        return audit_results(args.variants, args.folds)
    if args.mode == "deep":
        checks = [
            [args.python_bin, "-m", "survot_rank.cli", "doctor"],
            [args.python_bin, "-m", "pytest", "-q"],
        ]
        for command in checks:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
        result = run_jobs(args, smoke=True)
        if result:
            return result
        result = run_jobs(args, smoke=False)
        audit_result = audit_results(args.variants, args.folds)
        return result or audit_result
    result = run_jobs(args, smoke=args.mode == "smoke")
    if args.mode == "run" and result == 0:
        return audit_results(args.variants, args.folds)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
