#!/usr/bin/env python3
"""Run paper-facing leave-one-out ablations for frozen DCT v3.8.2.

The completed fixed-full result and all-aux-off control are reused.  This queue
changes exactly one objective term at a time on BLCA folds 1/2/4 and writes to
an isolated directory.  It must not be used for cancer-specific tuning.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts import run_dct_v382_final_cross_cancer as base
except (ModuleNotFoundError, ImportError):
    import run_dct_v382_final_cross_cancer as base


ABLATIONS: dict[str, dict[str, object]] = {
    "no_mgptr": {"dct_v382_lambda_mgptr": 0.0},
    "no_direction": {"dct_v38_lambda_direction": 0.0},
    "no_dose": {"dct_v38_lambda_dose": 0.0},
    "no_reconfiguration": {"dct_v38_lambda_reconfiguration": 0.0},
    "no_ipcw_rank": {"dct_lambda_ipcw_rank": 0.0},
}
DEFAULT_ABLATIONS = tuple(ABLATIONS)
RESULT_ROOT = Path("results/dct_v3.8.2_paper_ablations/robust")
SMOKE_ROOT = Path("results/dct_v3.8.2_paper_ablations_smoke")


@dataclass(frozen=True)
class Job:
    variant: str
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_ablations(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(DEFAULT_ABLATIONS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ABLATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown ablation: {', '.join(unknown)}; choose from "
            f"{', '.join(DEFAULT_ABLATIONS)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("at least one ablation is required")
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
    for key, value in ABLATIONS[variant].items():
        _replace_override(command, key, value)
    _replace_override(command, "results_dir", result_dir.as_posix())
    _replace_override(
        command,
        "specific_simple",
        f"dct_v382_fixed_full_{variant}_{cancer}_{'smoke' if smoke else '50ep'}",
    )
    return Job(variant, cancer, fold, tuple(command), result_dir, frozen.config)


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[Job]:
    folds = args.folds[:1] if smoke else args.folds
    return [
        build_job(args, variant, cancer, fold, smoke=smoke)
        for variant in args.ablations
        for cancer in args.cancers
        for fold in folds
    ]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("DCT v3.8.2 PAPER ABLATIONS: one-term leave-one-out")
    print(f"Queue: {len(jobs)} jobs")
    current = None
    for index, job in enumerate(jobs, start=1):
        if current != (job.variant, job.cancer):
            current = (job.variant, job.cancer)
            print(f"\n[{job.variant} / {job.cancer.upper()}]")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} fold{job.fold}")
        print("    " + shlex.join(job.command))
        if completion:
            print(f"    completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if base.doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not base.verify_child_cuda(args.python_bin, environment):
        return 1

    scheduler_lock = None
    try:
        scheduler_lock = base.acquire_run_lock(
            base.scheduler_lock_path(args.gpu, smoke),
            label=f"DCT v3.8.2 paper ablations on GPU {args.gpu}",
        )
    except base.ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3

    try:
        for index, job in enumerate(jobs, start=1):
            completion = _completion(job)
            if completion and not args.force and not smoke:
                print(
                    f"[{index:02d}/{len(jobs):02d}] [skip] "
                    f"{job.variant} {job.cancer.upper()} fold{job.fold}: {completion}"
                )
                continue
            task_lock = None
            try:
                task_lock = base.acquire_run_lock(
                    base.task_lock_path(job),
                    label=(
                        f"DCT v3.8.2 {job.variant} "
                        f"{job.cancer.upper()} fold{job.fold}"
                    ),
                )
            except base.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] DCT v3.8.2 "
                    f"{job.variant} {job.cancer.upper()} fold{job.fold}"
                )
                print(shlex.join(job.command))
                completed = subprocess.run(job.command, check=False, env=environment)
                if completed.returncode != 0:
                    print(
                        f"[ERROR] job failed with code {completed.returncode}; "
                        "queue stopped"
                    )
                    return completed.returncode
            finally:
                base.release_run_lock(task_lock)
        return 0
    finally:
        base.release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.set_defaults(cancers=["blca"], folds=[1, 2, 4])
    parser.add_argument(
        "--ablations",
        type=parse_ablations,
        default=list(DEFAULT_ABLATIONS),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(base.REPO_ROOT)
    if args.mode == "prepare":
        return base.prepare_splits(args)
    if args.mode == "doctor":
        return base.doctor(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    print_plan(jobs, force=args.force, run_mode=args.mode == "run")
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
