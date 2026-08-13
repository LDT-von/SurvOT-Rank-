#!/usr/bin/env python3
"""Run the three v2 methods under one protocol (BLCA fold0 first).

This launcher mirrors ``run_three_method_final_cross_cancer.py`` but loads
the ``*_v2_blca.yaml`` configs and pushes the v2 hyperparameters (ArchetypeBank,
hard top-K gate, CohortAnchoredRouter, etc.).  All result directories live
under ``results/three_method_v2`` to keep them isolated from the existing
``three_method_final`` runs.

Usage
-----
    python scripts/run_three_method_v2_cross_cancer.py smoke --gpu 0
    python scripts/run_three_method_v2_cross_cancer.py run --gpu 0 --methods capsa_v2
    python scripts/run_three_method_v2_cross_cancer.py run --gpu 0 --cancers blca --folds 0,1,2,3,4
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
METHOD_ORDER = ("capsa_v2", "arcsurv_v2", "catet_v2")
WHICH_SPLITS = "5fold_uni2h"
RESULT_ROOT = Path("results/three_method_v2")

# Shared by all three v2 methods.  These overrides match the v2 configs
# exactly; the launcher only needs to set per-method values that differ.
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
    "capsa_v2": {
        "config": "configs/capsa_v2_blca.yaml",
        "survot_method": "cohort_anchored_adaptive_prognostic_slot_attention",
        "batch_size": 8,
        "capsa_max_slots": 16,
        "capsa_slot_iters": 3,
        "capsa_heads": 4,
        # v2: cohort archetypal anchors.
        "capsa_archetype_bank_size": 256,
        "capsa_archetype_beta_init_scale": 1.5,
        "capsa_lambda_archetypal_recon": 0.02,
    },
    "arcsurv_v2": {
        "config": "configs/arcsurv_v2_blca.yaml",
        "survot_method": "archetypal_risk_composition",
        "batch_size": 8,
        "arc_num_archetypes": 6,
        "arc_bank_size": 256,
        "arc_temperature": 0.25,
        # v2: balanced re-transport + hard top-K gate.
        "arc_lambda_ot": 0.04,
        "arc_lambda_gate": 0.01,
        "arc_topk_active": 3,
        "arc_ot_eps": 0.05,
        "arc_ot_iters": 25,
        # Build bank in epoch 0 only (staged_final semantics).
        "arc_bank_update_epochs": 0,
    },
    "catet_v2": {
        "config": "configs/catet_v2_blca.yaml",
        "survot_method": "censoring_aware_temporal_evidence_transport",
        "batch_size": 16,
        "catet_num_stages": 4,
        "catet_lambda_ot": 0.04,
        "catet_lambda_rank": 0.08,
        "catet_lambda_intervention": 0.05,
        # v2: cohort-anchored pre-routing + lazy archetype prior.
        "catet_cohort_routes": 4,
        "catet_cohort_topk": 2,
        "catet_lambda_route": 0.02,
        "catet_use_archetype_prior": 1,
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
        result_dir = Path("results/three_method_v2_smoke") / method / cancer
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
        print("[BLOCKED] v2 runs require complete UNI2-h features and valid 5fold_uni2h splits.")
    return int(failed)


def print_plan(jobs: list[Job], *, force: bool = False) -> None:
    print("V2 METHODS: capsa_v2 -> arcsurv_v2 -> catet_v2")
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
    return Path("results/priority_experiment_queue") / f".v2_{kind}_gpu_{safe}.lock"


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
            label=f"three-method v2 queue on GPU {args.gpu}",
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
        "mode",
        choices=("run", "smoke", "doctor", "plan"),
        help=(
            "smoke = 1 fold, 2 epochs, 2 batches (verify wiring); "
            "run = full protocol (skips finished jobs); "
            "doctor = feature/split check; "
            "plan = print the queue without launching"
        ),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--methods",
        type=lambda value: parse_csv(value, METHOD_ORDER, "method"),
        default=METHOD_ORDER,
    )
    parser.add_argument(
        "--cancers",
        type=lambda value: parse_csv(value, SUPPORTED_CANCERS, "cancer"),
        default=("blca",),
    )
    parser.add_argument(
        "--folds",
        type=lambda value: parse_folds(value),
        default=DEFAULT_FOLDS,
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs even if their result pickle already exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "plan":
        jobs = build_jobs(args)
        print_plan(jobs, force=args.force)
        return 0
    if args.mode == "doctor":
        return doctor(args)
    jobs = build_jobs(args, smoke=(args.mode == "smoke"))
    print_plan(jobs, force=args.force)
    return run_queue(args, jobs, smoke=(args.mode == "smoke"))


if __name__ == "__main__":
    raise SystemExit(main())