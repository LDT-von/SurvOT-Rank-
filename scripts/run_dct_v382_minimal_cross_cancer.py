#!/usr/bin/env python3
"""Run the frozen DCT v3.8.2 Monotone Dose-Response recipe across feature-complete cancers.

This launcher evaluates the smallest recipe that still answers the
monotone dose-response claim of the DCT family:

* factual v3.8 path (Sinkhorn + shared prototypes + IPCW anchors)
* IPCW ranking loss (λ=0.10, memory size 64)
* direction loss (λ=0.05)
* MGPTR / adaptive / dose / reconfiguration all forced to zero

The launcher intentionally exposes no method-weight or epoch overrides.
Its purpose is to evaluate one paper-facing method under one protocol,
rather than to tune a different recipe for every cancer.
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
    from scripts.run_dct_v382_final_cross_cancer import (
        SUPPORTED_CANCERS,
        WHICH_SPLITS,
        prepare_splits,
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
    from run_dct_v382_final_cross_cancer import (
        SUPPORTED_CANCERS,
        WHICH_SPLITS,
        prepare_splits,
    )
    from task_lock import ActiveRunError, acquire_run_lock, release_run_lock


REPO_ROOT = Path(__file__).resolve().parent.parent

# Same five cancers as the v3.8.2 fixed-full queue.  BLCA is intentionally
# left to the priority queue because its five folds must follow the
# 2026-07-30 re-split protocol.
DEFAULT_CANCERS = ("skcm", "hnsc", "lusc", "kirc", "ucec")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
RESULT_ROOT = Path("results/dct_v382_minimal/robust/fixed_min")

# Frozen monotone dose-response recipe.  Differs from the v3.8.2 fixed-full recipe only
# in the auxiliary-loss set: MGPTR / adaptive / dose / reconfiguration are
# set to zero.  The class itself also re-forces these to zero inside
# __init__ so a CLI override cannot accidentally re-enable a harmful term.
FROZEN_MINIMAL_OVERRIDES: dict[str, object] = {
    "survot_method": "dct_v382_minimal_transport",
    "max_epochs": 50,
    "dct_v38_lambda_direction": 0.05,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_slot_init_mode": "deterministic",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_memory_size": 64,
    "dct_lambda_etar": 0.0,
    "dct_mix_ratio": 1.0,
    "dct_v382_lambda_mgptr": 0.0,
    "dct_v382_adaptive_aux_weights": False,
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
    config = REPO_ROOT / "configs" / f"dct_v382_minimal_transport_{cancer}.yaml"
    if not config.is_file():
        # Fall back to the BLCA minimal template so other cancers still work
        # until per-cancer minimal yaml files are added.
        config = REPO_ROOT / "configs" / "dct_v382_minimal_transport_blca.yaml"
    result_dir = RESULT_ROOT / cancer
    values = dict(FROZEN_MINIMAL_OVERRIDES)
    values.update(
        {
            "data_root_dir": args.data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": args.gpu,
            "num_workers": args.num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": f"dct_v382_minimal_robust_fixed_min_{cancer}_50ep",
        }
    )
    if smoke:
        values.update({"max_epochs": 2, "max_smoke_batches": 2})
        result_dir = Path("results/dct_v382_minimal_smoke/fixed_min") / cancer
        values["results_dir"] = result_dir.as_posix()
        values["specific_simple"] = f"dct_v382_minimal_smoke_{cancer}"
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
    kind = "smoke" if smoke else "run"
    return Path("results/priority_experiment_queue") / (
        f".{kind}_gpu_{_safe_gpu_name(gpu)}.lock"
    )


def task_lock_path(job: Job) -> Path:
    return job.result_dir / f".split_{job.fold}.priority_queue.lock"


def doctor(args: argparse.Namespace) -> int:
    failed = False
    for cancer in args.cancers:
        config = REPO_ROOT / "configs" / f"dct_v382_minimal_transport_{cancer}.yaml"
        config_ok = config.is_file() or (
            REPO_ROOT / "configs" / "dct_v382_minimal_transport_blca.yaml"
        ).is_file()
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


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("FINAL METHOD: DCT v3.8.2 Monotone Dose-Response (IPCW + direction only)")
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
            label=f"DCT v3.8.2 Monotone Dose-Response cross-cancer queue on GPU {args.gpu}",
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
                    label=f"DCT v3.8.2 Monotone Dose-Response {job.cancer.upper()} fold{job.fold}",
                )
            except ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"DCT v3.8.2 minimal fixed-min {job.cancer.upper()} fold{job.fold}"
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