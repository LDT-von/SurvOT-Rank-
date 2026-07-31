#!/usr/bin/env python3
"""Screen fixed and adaptive DCT v3.8.2 objectives on matched folds."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.run_dct_v38_transport_consistency import (
        CANCERS,
        COMMON_OVERRIDES as V38_COMMON_OVERRIDES,
        DEFAULT_DATA_ROOT,
        PROTOCOLS,
        VARIANTS as V38_VARIANTS,
        _override_args,
        inspect_feature_directory,
        inspect_split_directory,
        parse_cancers,
        parse_folds,
        parse_positive_int,
        parse_protocols,
        verify_child_cuda,
    )
    from scripts.task_lock import (
        ActiveRunError,
        acquire_run_lock,
        release_run_lock,
    )
except ModuleNotFoundError:
    from run_dct_v38_transport_consistency import (
        CANCERS,
        COMMON_OVERRIDES as V38_COMMON_OVERRIDES,
        DEFAULT_DATA_ROOT,
        PROTOCOLS,
        VARIANTS as V38_VARIANTS,
        _override_args,
        inspect_feature_directory,
        inspect_split_directory,
        parse_cancers,
        parse_folds,
        parse_positive_int,
        parse_protocols,
        verify_child_cuda,
    )
    from task_lock import ActiveRunError, acquire_run_lock, release_run_lock


REPO_ROOT = Path(__file__).resolve().parent.parent
SCREEN_VARIANTS = (
    "base",
    "mgptr",
    "selected",
    "selected_mgptr",
    "fixed_full",
    "adaptive_full",
)
DEFAULT_MGPTR_WEIGHT = 0.05

COMMON_OVERRIDES = dict(V38_COMMON_OVERRIDES)
COMMON_OVERRIDES.update(
    {
        "survot_method": "dct_v382_prognostic_transport_reconstruction",
        "dct_v382_lambda_mgptr": DEFAULT_MGPTR_WEIGHT,
        "dct_v382_distill_weight": 0.50,
        "dct_v382_warmup_epochs": 1,
        "dct_v382_ramp_epochs": 4,
        "dct_v382_adaptive_aux_weights": False,
        "dct_v382_adaptive_prior_fraction": 0.25,
        "dct_v382_adaptive_temperature": 1.0,
        "dct_v382_adaptive_kl_strength": 0.01,
    }
)


def parse_screen_variants(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(SCREEN_VARIANTS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SCREEN_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown v3.8.2 variant: "
            f"{', '.join(unknown)}; choose from {', '.join(SCREEN_VARIANTS)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("at least one v3.8.2 variant is required")
    return selected


def _old_loss_settings(name: str | None) -> dict[str, float]:
    if name is None:
        raise ValueError(
            "--selected-v38-variant is required for selected or selected_mgptr"
        )
    values = V38_VARIANTS[name]
    return {
        "dct_v38_lambda_direction": values["dct_v38_lambda_direction"],
        "dct_v38_lambda_dose": values["dct_v38_lambda_dose"],
        "dct_v38_lambda_reconfiguration": values[
            "dct_v38_lambda_reconfiguration"
        ],
    }


def variant_settings(
    variant: str, selected_v38_variant: str | None
) -> tuple[str, dict[str, object]]:
    zero_old = {
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
    }
    if variant == "base":
        return "base", {
            **zero_old,
            "dct_v382_lambda_mgptr": 0.0,
            "dct_v382_adaptive_aux_weights": False,
        }
    if variant == "mgptr":
        return "mgptr", {
            **zero_old,
            "dct_v382_lambda_mgptr": DEFAULT_MGPTR_WEIGHT,
            "dct_v382_adaptive_aux_weights": False,
        }

    full = _old_loss_settings("full")
    if variant == "fixed_full":
        return "fixed_full", {
            **full,
            "dct_v382_lambda_mgptr": DEFAULT_MGPTR_WEIGHT,
            "dct_v382_adaptive_aux_weights": False,
        }
    if variant == "adaptive_full":
        return "adaptive_full", {
            **full,
            "dct_v382_lambda_mgptr": DEFAULT_MGPTR_WEIGHT,
            "dct_v382_adaptive_aux_weights": True,
            # Every enabled objective must be observed on each eligible epoch
            # so the adaptive controller never mistakes a skipped dose term
            # for an easy zero-loss task.
            "dct_v38_dose_every": 1,
        }

    selected = _old_loss_settings(selected_v38_variant)
    label = f"selected_{selected_v38_variant}"
    if variant == "selected":
        return label, {
            **selected,
            "dct_v382_lambda_mgptr": 0.0,
            "dct_v382_adaptive_aux_weights": False,
        }
    if variant == "selected_mgptr":
        return f"{label}_mgptr", {
            **selected,
            "dct_v382_lambda_mgptr": DEFAULT_MGPTR_WEIGHT,
            "dct_v382_adaptive_aux_weights": False,
        }
    raise ValueError(f"unknown v3.8.2 variant: {variant}")


def _result_root(max_epochs: int, smoke: bool) -> str:
    if smoke:
        return "dct_v3.8.2_smoke"
    if max_epochs == 50:
        return "dct_v3.8.2"
    return f"dct_v3.8.2_{max_epochs}ep"


def build_train_command(
    python_bin: str,
    cancer: str,
    protocol: str,
    variant: str,
    fold: int,
    gpu: str,
    num_workers: str,
    data_root: str,
    *,
    selected_v38_variant: str | None = None,
    max_epochs: int = 20,
    smoke: bool = False,
) -> tuple[list[str], Path]:
    config = Path("configs") / f"distributional_counterfactual_transport_{cancer}.yaml"
    variant_label, settings = variant_settings(variant, selected_v38_variant)
    result_dir = (
        Path("results")
        / _result_root(max_epochs, smoke)
        / protocol
        / variant_label
        / cancer
    )

    overrides = dict(COMMON_OVERRIDES)
    overrides.update(PROTOCOLS[protocol])
    overrides.update(settings)
    overrides.pop("label", None)
    overrides.update(
        {
            "data_root_dir": data_root,
            "max_epochs": 2 if smoke else max_epochs,
            "max_smoke_batches": 2 if smoke else 0,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": gpu,
            "num_workers": num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": (
                f"dct_v382_{protocol}_{variant_label}_{cancer}_{max_epochs}ep"
            ),
        }
    )
    command = [
        python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config.as_posix(),
        *_override_args(overrides),
    ]
    return command, result_dir


def scheduler_lock_path(smoke: bool, gpu: str) -> Path:
    safe_gpu = "".join(
        character if character.isalnum() else "_" for character in gpu
    )
    root = (
        "dct_v3.8.2_smoke"
        if smoke
        else "dct_v3.8.2"
    )
    return Path("results") / root / f".scheduler_gpu_{safe_gpu}.lock"


def task_lock_path(result_dir: Path, fold: int) -> Path:
    return result_dir / f".split_{fold}.run.lock"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("doctor", "plan", "smoke", "run"), nargs="?", default="plan"
    )
    parser.add_argument(
        "--cancers", type=parse_cancers, default=parse_cancers("blca,brca")
    )
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0"))
    parser.add_argument("--protocols", type=parse_protocols, default=["robust"])
    parser.add_argument(
        "--variants", type=parse_screen_variants, default=["adaptive_full"]
    )
    parser.add_argument(
        "--selected-v38-variant",
        choices=tuple(V38_VARIANTS),
        default=None,
        help=(
            "Winner of the old 2^3 screen. Required only by selected and "
            "selected_mgptr; for example direction_dose."
        ),
    )
    parser.add_argument(
        "--data-root", default=os.environ.get("UNI2H_ROOT", DEFAULT_DATA_ROOT)
    )
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--max-epochs", type=parse_positive_int, default=20)
    parser.add_argument(
        "--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable)
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _validate_selection(parser, args) -> None:
    needs_selected = {"selected", "selected_mgptr"}.intersection(args.variants)
    if needs_selected and args.selected_v38_variant is None:
        parser.error(
            "--selected-v38-variant is required when --variants contains "
            "selected or selected_mgptr"
        )


def _doctor(args) -> int:
    failed = False
    for cancer in args.cancers:
        feature_report = inspect_feature_directory(args.data_root, cancer)
        feature_status = "OK" if feature_report["ok"] else "MISSING"
        print(
            f"{feature_status:8s} {cancer.upper():8s} "
            f"files={feature_report['count']:<5} "
            f"shape={feature_report['shape']} "
            f"path={feature_report['directory']}"
        )
        if feature_report["error"]:
            print(f"         {feature_report['error']}")
        failed = failed or not feature_report["ok"]

        split_report = inspect_split_directory(cancer)
        split_status = "OK" if split_report["ok"] else "INVALID"
        print(
            f"{split_status:8s} {cancer.upper():8s} "
            f"eligible={split_report['eligible_cases']:<5} "
            f"events={split_report['observed_events']:<4} "
            f"val_events={split_report['validation_event_counts']}"
        )
        for error in split_report["errors"]:
            print(f"         {error}")
        failed = failed or not split_report["ok"]
    return int(failed)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_selection(parser, args)
    os.chdir(REPO_ROOT)

    if args.mode == "doctor":
        return _doctor(args)

    if args.mode in ("smoke", "run") and "robust" in args.protocols:
        for cancer in args.cancers:
            split_report = inspect_split_directory(cancer)
            if not split_report["ok"]:
                print(f"[ERROR] {cancer.upper()} split audit failed.")
                for error in split_report["errors"]:
                    print(f"        {error}")
                return 2

    scheduler_lock = None
    if args.mode in ("smoke", "run"):
        try:
            scheduler_lock = acquire_run_lock(
                scheduler_lock_path(args.mode == "smoke", args.gpu),
                label=f"DCT v3.8.2 scheduler on GPU {args.gpu}",
            )
        except ActiveRunError as error:
            print(f"[already-running] {error}")
            return 3

    try:
        for protocol in args.protocols:
            for variant in args.variants:
                variant_label, _ = variant_settings(
                    variant, args.selected_v38_variant
                )
                for cancer in args.cancers:
                    config = (
                        Path("configs")
                        / f"distributional_counterfactual_transport_{cancer}.yaml"
                    )
                    if not config.exists():
                        print(f"[ERROR] missing config: {config}")
                        return 2
                    for fold in args.folds:
                        command, result_dir = build_train_command(
                            args.python_bin,
                            cancer,
                            protocol,
                            variant,
                            fold,
                            args.gpu,
                            args.num_workers,
                            args.data_root,
                            selected_v38_variant=args.selected_v38_variant,
                            max_epochs=args.max_epochs,
                            smoke=args.mode == "smoke",
                        )
                        completed = list(
                            result_dir.rglob(f"split_{fold}_results_final.pkl")
                        )
                        if completed and not args.force and args.mode == "run":
                            print(
                                f"[skip] {protocol}/{variant_label} "
                                f"{cancer.upper()} fold{fold}: {completed[0]}"
                            )
                            continue

                        print("\n" + "=" * 76)
                        print(
                            f"[DCT v3.8.2/{protocol}/{variant_label}] "
                            f"{cancer.upper()} fold{fold}"
                        )
                        print(" ".join(command))
                        if args.mode not in ("smoke", "run"):
                            continue

                        task_lock = None
                        try:
                            task_lock = acquire_run_lock(
                                task_lock_path(result_dir, fold),
                                label=(
                                    f"DCT v3.8.2 {protocol}/{variant_label} "
                                    f"{cancer.upper()} fold{fold}"
                                ),
                            )
                        except ActiveRunError as error:
                            print(f"[skip-running] {error}")
                            continue

                        try:
                            completed = list(
                                result_dir.rglob(
                                    f"split_{fold}_results_final.pkl"
                                )
                            )
                            if completed and not args.force and args.mode == "run":
                                print(
                                    f"[skip] {protocol}/{variant_label} "
                                    f"{cancer.upper()} fold{fold}: {completed[0]}"
                                )
                                continue

                            environment = os.environ.copy()
                            environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                            environment.setdefault("PYTHONUNBUFFERED", "1")
                            if not verify_child_cuda(
                                args.python_bin, environment
                            ):
                                return 1
                            completed_process = subprocess.run(
                                command,
                                check=False,
                                env=environment,
                            )
                            if completed_process.returncode != 0:
                                return completed_process.returncode
                        finally:
                            release_run_lock(task_lock)
        return 0
    finally:
        release_run_lock(scheduler_lock)


if __name__ == "__main__":
    raise SystemExit(main())
