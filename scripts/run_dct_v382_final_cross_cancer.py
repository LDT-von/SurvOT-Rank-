#!/usr/bin/env python3
"""Run the frozen DCT v3.8.2 fixed-full recipe across feature-complete cancers.

This launcher intentionally exposes no method-weight or epoch overrides.  Its
purpose is to evaluate one paper-facing method under one protocol, rather than
to tune a different recipe for every cancer.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.run_dct_v38_transport_consistency import (
        DATASET_CSV_ROOT,
        DEFAULT_DATA_ROOT,
        _override_args,
        inspect_feature_directory,
        inspect_split_directory,
        parse_folds,
        verify_child_cuda,
    )
    from scripts.task_lock import ActiveRunError, acquire_run_lock, release_run_lock
except ModuleNotFoundError:
    from run_dct_v38_transport_consistency import (
        DATASET_CSV_ROOT,
        DEFAULT_DATA_ROOT,
        _override_args,
        inspect_feature_directory,
        inspect_split_directory,
        parse_folds,
        verify_child_cuda,
    )
    from task_lock import ActiveRunError, acquire_run_lock, release_run_lock


REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_CANCERS = (
    "blca",
    "brca",
    "coadread",
    "hnsc",
    "kirc",
    "luad",
    "lusc",
    "skcm",
    "stad",
    "ucec",
)

# These five cancers had complete UNI2-h coverage in the 2026-08-06 audit.
# BLCA is omitted because its five folds are completed by the priority queue.
# The doctor still re-audits current server data before every formal run.
DEFAULT_CANCERS = ("skcm", "hnsc", "lusc", "kirc", "ucec")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
RESULT_ROOT = Path("results/dct_v3.8.2/robust/fixed_full")
WHICH_SPLITS = "5fold_uni2h"

# Paper-facing frozen recipe. Do not add cancer-specific branches here.
FINAL_OVERRIDES: dict[str, object] = {
    "survot_method": "dct_v382_prognostic_transport_reconstruction",
    "max_epochs": 50,
    "dct_v382_warmup_epochs": 5,
    "dct_v382_ramp_epochs": 10,
    "dct_v382_lambda_mgptr": 0.05,
    "dct_v382_distill_weight": 0.50,
    "dct_v382_adaptive_aux_weights": False,
    "dct_v38_lambda_direction": 0.05,
    "dct_v38_lambda_dose": 0.03,
    "dct_v38_lambda_reconfiguration": 0.02,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_slot_init_mode": "deterministic",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_memory_size": 64,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_mix_ratio": 1.0,
    "num_patches": 2048,
    "batch_size": 8,
    "which_splits": WHICH_SPLITS,
    "on_missing_wsi": "error",
    "wsi_encoder": "uni2-h",
    "encoding_dim": 1536,
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_cancers(value: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SUPPORTED_CANCERS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown cancer: {', '.join(unknown)}; choose from "
            f"{', '.join(SUPPORTED_CANCERS)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("at least one cancer is required")
    return selected


def build_job(args: argparse.Namespace, cancer: str, fold: int, *, smoke: bool) -> Job:
    config = Path("configs") / f"distributional_counterfactual_transport_{cancer}.yaml"
    result_dir = RESULT_ROOT / cancer
    values = dict(FINAL_OVERRIDES)
    values.update(
        {
            "data_root_dir": args.data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": args.gpu,
            "num_workers": args.num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": f"dct_v382_robust_fixed_full_{cancer}_50ep",
        }
    )
    if smoke:
        values.update({"max_epochs": 2, "max_smoke_batches": 2})
        result_dir = Path("results/dct_v3.8.2_final_smoke/fixed_full") / cancer
        values["results_dir"] = result_dir.as_posix()
        values["specific_simple"] = f"dct_v382_final_smoke_{cancer}"
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config.as_posix(),
        *_override_args(values),
    )
    return Job(cancer, fold, command, result_dir, config)


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[Job]:
    folds = args.folds[:1] if smoke else args.folds
    return [
        build_job(args, cancer, fold, smoke=smoke)
        for cancer in args.cancers
        for fold in folds
    ]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def _safe_gpu_name(gpu: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in gpu)


def scheduler_lock_path(gpu: str, smoke: bool) -> Path:
    # Deliberately identical to the priority queue lock: two schedulers cannot
    # silently train on the same GPU at once.
    kind = "smoke" if smoke else "run"
    return Path("results/priority_experiment_queue") / (
        f".{kind}_gpu_{_safe_gpu_name(gpu)}.lock"
    )


def task_lock_path(job: Job) -> Path:
    # Deliberately identical to the priority queue's per-fold lock.
    return job.result_dir / f".split_{job.fold}.priority_queue.lock"


def doctor(args: argparse.Namespace) -> int:
    failed = False
    for cancer in args.cancers:
        config = REPO_ROOT / "configs" / f"distributional_counterfactual_transport_{cancer}.yaml"
        config_ok = config.is_file()
        print(f"{'OK' if config_ok else 'MISSING':8s} config {config}")
        failed = failed or not config_ok

        feature = inspect_feature_directory(args.data_root, cancer)
        print(
            f"{'OK' if feature['ok'] else 'MISSING':8s} feature {cancer.upper()} "
            f"files={feature['count']} shape={feature['shape']} path={feature['directory']}"
        )
        if feature["error"]:
            print(f"         {feature['error']}")
        failed = failed or not bool(feature["ok"])

        try:
            split = inspect_split_directory(
                cancer,
                data_root=args.data_root,
                which_splits=WHICH_SPLITS,
            )
        except Exception as error:
            print(f"INVALID  split {cancer.upper()} {WHICH_SPLITS}: {error}")
            failed = True
            continue
        print(
            f"{'OK' if split['ok'] else 'INVALID':8s} split {cancer.upper()} "
            f"eligible={split['eligible_cases']} "
            f"val_events={split['validation_event_counts']}"
        )
        for error in split["errors"]:
            print(f"         {error}")
        failed = failed or not bool(split["ok"])
    if failed:
        print(
            "[BLOCKED] Complete UNI2-h features and prepare valid 5fold_uni2h "
            "splits before training. Zero-filled WSI bags are not allowed."
        )
    return int(failed)


def prepare_splits(args: argparse.Namespace) -> int:
    """Create only missing feature-complete splits; never overwrite an invalid one."""
    from tools.gen_splits_5fold import collect_matching_feature_case_ids, gen

    split_root = DATASET_CSV_ROOT / "splits" / WHICH_SPLITS
    failed = False
    for cancer in args.cancers:
        feature = inspect_feature_directory(args.data_root, cancer)
        if not feature["ok"]:
            print(f"[BLOCKED] {cancer.upper()}: {feature['error']}")
            failed = True
            continue
        report = inspect_split_directory(
            cancer,
            data_root=args.data_root,
            which_splits=WHICH_SPLITS,
        )
        if report["ok"]:
            print(f"[skip] {cancer.upper()}: existing {WHICH_SPLITS} audit passed")
            continue
        cancer_dir = split_root / cancer
        if cancer_dir.exists():
            print(
                f"[BLOCKED] {cancer.upper()}: an invalid split directory already exists; "
                "refusing to overwrite it"
            )
            for error in report["errors"]:
                print(f"          {error}")
            failed = True
            continue
        if report.get("clinical_without_features"):
            print(
                f"[BLOCKED] {cancer.upper()}: "
                f"{report['clinical_without_features']} eligible patients lack features"
            )
            failed = True
            continue
        print(f"[prepare] {cancer.upper()} deterministic seed=42 -> {cancer_dir}")
        clinical_csv = DATASET_CSV_ROOT / "clinical" / "all" / f"{cancer}.csv"
        feature_dir = Path(args.data_root) / cancer / "uni2-h" / "pt_files"
        eligible_case_ids = collect_matching_feature_case_ids(
            clinical_csv,
            feature_dir,
        )
        gen(
            study=cancer,
            data_path=str(DATASET_CSV_ROOT),
            label_col="survival_months_dss",
            censor_col="censorship_dss",
            n_folds=5,
            seed=42,
            out_dir=str(split_root),
            eligible_case_ids=eligible_case_ids,
        )
    return int(failed)


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("FINAL METHOD: DCT v3.8.2 fixed-full (frozen, non-adaptive)")
    print(f"Queue: {len(jobs)} jobs; cancers={len({job.cancer for job in jobs})}")
    current_cancer = None
    for index, job in enumerate(jobs, start=1):
        if job.cancer != current_cancer:
            current_cancer = job.cancer
            print(f"\n[{job.cancer.upper()}]")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} fold{job.fold}")
        print("    " + shlex.join(job.command))
        if completion:
            print(f"    completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not verify_child_cuda(args.python_bin, environment):
        return 1

    scheduler_lock = None
    try:
        scheduler_lock = acquire_run_lock(
            scheduler_lock_path(args.gpu, smoke),
            label=f"final DCT cross-cancer queue on GPU {args.gpu}",
        )
    except ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3

    try:
        for index, job in enumerate(jobs, start=1):
            completion = _completion(job)
            if completion and not args.force and not smoke:
                print(
                    f"[{index:02d}/{len(jobs):02d}] [skip] "
                    f"{job.cancer.upper()} fold{job.fold}: {completion}"
                )
                continue
            task_lock = None
            try:
                task_lock = acquire_run_lock(
                    task_lock_path(job),
                    label=f"final DCT {job.cancer.upper()} fold{job.fold}",
                )
            except ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"DCT v3.8.2 fixed-full {job.cancer.upper()} fold{job.fold}"
                )
                print(shlex.join(job.command))
                completed_process = subprocess.run(
                    job.command,
                    check=False,
                    env=environment,
                )
                if completed_process.returncode != 0:
                    print(
                        f"[ERROR] job failed with code {completed_process.returncode}; "
                        "queue stopped"
                    )
                    return completed_process.returncode
            finally:
                release_run_lock(task_lock)
        return 0
    finally:
        release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("plan", "prepare", "doctor", "smoke", "run"),
        nargs="?",
        default="plan",
    )
    parser.add_argument("--cancers", type=parse_cancers, default=list(DEFAULT_CANCERS))
    parser.add_argument("--folds", type=parse_folds, default=list(DEFAULT_FOLDS))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("UNI2H_ROOT", DEFAULT_DATA_ROOT),
    )
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    if args.mode == "prepare":
        return prepare_splits(args)
    if args.mode == "doctor":
        return doctor(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    print_plan(jobs, force=args.force, run_mode=args.mode == "run")
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
