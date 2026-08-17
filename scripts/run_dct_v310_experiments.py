#!/usr/bin/env python3
"""Matched objective and mechanism experiments for DCT v3.10.

Core 2x2 objective ablation:

* nll_only:       NLL
* ipcw_only:      NLL + 0.10 IPCW-rank
* direction_only: NLL + 0.05 direction
* full:           NLL + 0.10 IPCW-rank + 0.05 direction

Mechanism controls reuse the same v3.10 architecture and protocol but are
registered through the v3.8 parent so the frozen final class remains immutable.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts import run_dct_v310_final_cross_cancer as base
except (ModuleNotFoundError, ImportError):
    import run_dct_v310_final_cross_cancer as base


PARENT_METHOD = "dct_transport_intervention_consistency"
VARIANTS: dict[str, dict[str, object]] = {
    "nll_only": {
        "survot_method": PARENT_METHOD,
        "dct_lambda_ipcw_rank": 0.0,
        "dct_v38_lambda_direction": 0.0,
    },
    "ipcw_only": {
        "survot_method": PARENT_METHOD,
        "dct_lambda_ipcw_rank": 0.10,
        "dct_v38_lambda_direction": 0.0,
    },
    "direction_only": {
        "survot_method": PARENT_METHOD,
        "dct_lambda_ipcw_rank": 0.0,
        "dct_v38_lambda_direction": 0.05,
    },
    "full": {},
    "fixed_coupling": {
        "survot_method": PARENT_METHOD,
        "dct_fixed_coupling": True,
    },
    "noisy_batch_mean_anchors": {
        "survot_method": PARENT_METHOD,
        "dct_random_anchors": True,
    },
    "permuted_reference": {
        "survot_method": PARENT_METHOD,
        "dct_perm_labels_seed": 1,
    },
    "stage_jitter": {
        "survot_method": PARENT_METHOD,
        "dct_stage_jitter_fraction": 0.30,
    },
}

DESCRIPTIONS = {
    "nll_only": "Prediction-only baseline.",
    "ipcw_only": "Isolates the independent contribution of IPCW ranking.",
    "direction_only": "Isolates direction regularization without IPCW ranking.",
    "full": "Frozen DCT v3.10 final objective.",
    "fixed_coupling": "Replays the current factual coupling under intervention.",
    "noisy_batch_mean_anchors": (
        "Replaces prognostic risk-set anchors with noisy batch-mean cost anchors."
    ),
    "permuted_reference": "Permutes train-reference times for null calibration.",
    "stage_jitter": "Jitters train-fold stage edges by 30 percent.",
}

DEFAULT_VARIANTS = ("nll_only", "ipcw_only", "direction_only", "full")
RESULT_ROOT = Path("results/dct_v3.10_experiments/robust")
SMOKE_ROOT = Path("results/dct_v3.10_experiments_smoke")


@dataclass(frozen=True)
class Job:
    variant: str
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_variants(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(VARIANTS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variant: {', '.join(unknown)}; choose from "
            f"{', '.join(VARIANTS)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("at least one variant is required")
    return selected


def _replace_override(command: list[str], key: str, value: object) -> None:
    prefix = f"{key}="
    for index, item in enumerate(command[:-1]):
        if item == "--set" and command[index + 1].startswith(prefix):
            command[index + 1] = f"{key}={value}"
            return
    command.extend(("--set", f"{key}={value}"))


def build_job(
    args: argparse.Namespace,
    variant: str,
    cancer: str,
    fold: int,
    *,
    smoke: bool,
) -> Job:
    frozen = base.build_job(args, cancer, fold, smoke=smoke)
    result_dir = (SMOKE_ROOT if smoke else RESULT_ROOT) / variant / cancer
    command = list(frozen.command)
    for key, value in VARIANTS[variant].items():
        _replace_override(command, key, value)
    _replace_override(command, "results_dir", result_dir.as_posix())
    _replace_override(
        command,
        "specific_simple",
        f"dct_v310_{variant}_{cancer}_{'smoke' if smoke else '50ep'}",
    )
    return Job(variant, cancer, fold, tuple(command), result_dir, frozen.config)


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[Job]:
    folds = args.folds[:1] if smoke else args.folds
    return [
        build_job(args, variant, cancer, fold, smoke=smoke)
        for variant in args.variants
        for cancer in args.cancers
        for fold in folds
    ]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("DCT v3.10 MATCHED EXPERIMENTS")
    print(
        f"Queue: {len(jobs)} jobs; variants={len({job.variant for job in jobs})}; "
        f"cancers={len({job.cancer for job in jobs})}"
    )
    current = None
    for index, job in enumerate(jobs, start=1):
        if job.variant != current:
            current = job.variant
            print(f"\n[{job.variant}] {DESCRIPTIONS[job.variant]}")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} {job.cancer.upper()} fold{job.fold}")
        print("    " + shlex.join(job.command))
        if completion:
            print(f"    completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if base.legacy.doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not base.legacy.verify_child_cuda(args.python_bin, environment):
        return 1

    scheduler_lock = None
    try:
        scheduler_lock = base.legacy.acquire_run_lock(
            base.legacy.scheduler_lock_path(args.gpu, smoke),
            label=f"DCT v3.10 matched experiments on GPU {args.gpu}",
        )
    except base.legacy.ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3

    try:
        for index, job in enumerate(jobs, start=1):
            completion = _completion(job)
            if completion and not args.force and not smoke:
                print(
                    f"[{index:02d}/{len(jobs):02d}] [skip] {job.variant} "
                    f"{job.cancer.upper()} fold{job.fold}: {completion}"
                )
                continue
            task_lock = None
            try:
                task_lock = base.legacy.acquire_run_lock(
                    base.legacy.task_lock_path(job),
                    label=(
                        f"DCT v3.10 {job.variant} "
                        f"{job.cancer.upper()} fold{job.fold}"
                    ),
                )
            except base.legacy.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] DCT v3.10 "
                    f"{job.variant} {job.cancer.upper()} fold{job.fold}"
                )
                print(shlex.join(job.command))
                completed = subprocess.run(job.command, check=False, env=environment)
                if completed.returncode != 0:
                    return completed.returncode
            finally:
                base.legacy.release_run_lock(task_lock)
        return 0
    finally:
        base.legacy.release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(cancers=["blca"], folds=list(base.DEFAULT_FOLDS))
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=list(DEFAULT_VARIANTS),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(base.REPO_ROOT)
    if args.mode == "prepare":
        return base.legacy.prepare_splits(args)
    if args.mode == "doctor":
        return base.legacy.doctor(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    print_plan(jobs, force=args.force, run_mode=args.mode == "run")
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
