#!/usr/bin/env python3
"""Run the three frozen paper-facing final methods under one protocol."""

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
    "blca", "brca", "coadread", "hnsc", "kirc",
    "luad", "lusc", "skcm", "stad", "ucec",
)
DEFAULT_CANCERS = ("blca", "ucec", "kirc", "skcm", "hnsc", "lusc")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
METHOD_ORDER = ("capsa_final", "arcsurv_final", "catet_final")
WHICH_SPLITS = "5fold_uni2h"
RESULT_ROOT = Path("results/three_method_final")

COMMON_OVERRIDES: dict[str, object] = {
    "max_epochs": 50,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "num_patches": 2048,
    "which_splits": WHICH_SPLITS,
    "on_missing_wsi": "error",
    "wsi_encoder": "uni2-h",
    "encoding_dim": 1536,
}

METHODS: dict[str, dict[str, object]] = {
    "capsa_final": {
        "config": "configs/cohort_anchored_adaptive_prognostic_slot_attention_blca.yaml",
        "survot_method": "cohort_anchored_adaptive_prognostic_slot_attention",
        "batch_size": 8,
        "capsa_max_slots": 16,
        "capsa_slot_iters": 3,
        "capsa_heads": 4,
        "capsa_lambda_budget": 0.01,
        "capsa_lambda_identity": 0.02,
        "capsa_target_active_ratio": 0.25,
        "capsa_identity_temperature": 0.10,
        "capsa_anchor_cosine_margin": 0.20,
        "capsa_anchor_scale": 0.50,
    },
    "arcsurv_final": {
        "config": "configs/archetypal_risk_composition_blca.yaml",
        "survot_method": "archetypal_risk_composition",
        "batch_size": 8,
        "arc_num_archetypes": 6,
        "arc_bank_size": 256,
        "arc_temperature": 0.25,
        "arc_warmup_epochs": 5,
        "arc_ramp_epochs": 10,
        "arc_bank_update_epochs": -1,
        "arc_freeze_state_encoder": 1,
        "arc_seed_anchors": 0,
        "arc_lambda_sharpness": 0.0,
    },
    "catet_final": {
        "config": "configs/censoring_aware_temporal_evidence_transport_blca.yaml",
        "survot_method": "censoring_aware_temporal_evidence_transport",
        "batch_size": 16,
        "catet_num_stages": 4,
        "catet_lambda_ot": 0.04,
        "catet_lambda_rank": 0.08,
        "catet_lambda_stage": 0.04,
        "catet_lambda_intervention": 0.05,
        "catet_intervention_cost": 1.0,
        "catet_plan_diversity_margin": 0.01,
        "catet_rank_temperature": 0.50,
        "catet_ipcw_max_weight": 10.0,
    },
}


@dataclass(frozen=True)
class Job:
    method: str
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label}: {', '.join(unknown)}; choose from {', '.join(allowed)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    return selected


def build_job(args, method: str, cancer: str, fold: int, *, smoke: bool) -> Job:
    recipe = METHODS[method]
    config = Path(str(recipe["config"]))
    result_dir = RESULT_ROOT / method / cancer
    values = dict(COMMON_OVERRIDES)
    values.update({key: value for key, value in recipe.items() if key != "config"})
    values.update(
        {
            "study": cancer,
            "data_root_dir": args.data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": args.gpu,
            "num_workers": args.num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": f"{method}_{cancer}_50ep",
        }
    )
    if smoke:
        result_dir = Path("results/three_method_final_smoke") / method / cancer
        values.update(
            {
                "max_epochs": 2,
                "max_smoke_batches": 2,
                "results_dir": result_dir.as_posix(),
                "specific_simple": f"{method}_{cancer}_smoke",
            }
        )
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config.as_posix(),
        *_override_args(values),
    )
    return Job(method, cancer, fold, command, result_dir, config)


def build_jobs(args, *, smoke: bool = False) -> list[Job]:
    folds = args.folds[:1] if smoke else args.folds
    return [
        build_job(args, method, cancer, fold, smoke=smoke)
        for method in args.methods
        for cancer in args.cancers
        for fold in folds
    ]


def completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def doctor(args) -> int:
    failed = False
    for method in args.methods:
        config = REPO_ROOT / str(METHODS[method]["config"])
        exists = config.is_file()
        print(f"{'OK' if exists else 'MISSING':8s} {method} config={config}")
        failed = failed or not exists
    for cancer in args.cancers:
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
            print(f"INVALID  split {cancer.upper()}: {error}")
            failed = True
            continue
        print(
            f"{'OK' if split['ok'] else 'INVALID':8s} split {cancer.upper()} "
            f"eligible={split['eligible_cases']} val_events={split['validation_event_counts']}"
        )
        for error in split["errors"]:
            print(f"         {error}")
        failed = failed or not bool(split["ok"])
    if failed:
        print("[BLOCKED] Final runs require complete UNI2-h features and valid 5fold_uni2h splits.")
    return int(failed)


def print_plan(jobs: list[Job], *, force: bool = False) -> None:
    print("FINAL METHODS: CA-PSA -> ArcSurv -> CATET")
    print(f"Queue: {len(jobs)} jobs")
    for index, job in enumerate(jobs, start=1):
        done = completion(job) if not force else None
        print(
            f"{index:03d}. {'SKIP' if done else 'RUN '} "
            f"{job.method} {job.cancer.upper()} fold{job.fold}"
        )
        print("     " + shlex.join(job.command))


def _lock_path(gpu: str, smoke: bool) -> Path:
    safe = "".join(character if character.isalnum() else "_" for character in gpu)
    kind = "smoke" if smoke else "run"
    return Path("results/priority_experiment_queue") / f".{kind}_gpu_{safe}.lock"


def run_queue(args, jobs: list[Job], *, smoke: bool) -> int:
    if doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not verify_child_cuda(args.python_bin, environment):
        return 1
    try:
        scheduler_lock = acquire_run_lock(
            _lock_path(args.gpu, smoke),
            label=f"three-method final queue on GPU {args.gpu}",
        )
    except ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3
    try:
        for index, job in enumerate(jobs, start=1):
            done = completion(job)
            if done and not args.force and not smoke:
                print(f"[{index}/{len(jobs)}] [skip] {done}")
                continue
            task_lock = None
            try:
                task_lock = acquire_run_lock(
                    job.result_dir / f".split_{job.fold}.priority_queue.lock",
                    label=f"{job.method} {job.cancer.upper()} fold{job.fold}",
                )
            except ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"[{index}/{len(jobs)}] {job.method} "
                    f"{job.cancer.upper()} fold{job.fold}"
                )
                completed = subprocess.run(job.command, check=False, env=environment)
                if completed.returncode != 0:
                    print(f"[ERROR] code={completed.returncode}; queue stopped")
                    return completed.returncode
            finally:
                release_run_lock(task_lock)
        return 0
    finally:
        release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("plan", "doctor", "smoke", "run"), nargs="?", default="plan"
    )
    parser.add_argument(
        "--methods",
        type=lambda value: parse_csv(value, METHOD_ORDER, "method"),
        default=list(METHOD_ORDER),
    )
    parser.add_argument(
        "--cancers",
        type=lambda value: parse_csv(value, SUPPORTED_CANCERS, "cancer"),
        default=list(DEFAULT_CANCERS),
    )
    parser.add_argument("--folds", type=parse_folds, default=list(DEFAULT_FOLDS))
    parser.add_argument("--data-root", default=os.environ.get("UNI2H_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument(
        "--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable)
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    if args.mode == "doctor":
        return doctor(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs, force=args.force)
        return 0
    print_plan(jobs, force=args.force)
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
