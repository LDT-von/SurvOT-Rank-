#!/usr/bin/env python3
"""Cross-cancer shared prototype transfer: freeze WSI/omic prototypes, retrain rest.

Paper-evidence ablation 3. Train the frozen v3.8.2 recipe on a source cancer,
save the best checkpoint, then load it on a target cancer with the two shared
prototype tensors frozen and all other parameters trainable.

If the target-cancer C-index stays close to the un-ablated value, the shared
prototypes carry cross-cancer semantic content rather than cancer-specific
overfitting. If the target-cancer C-index collapses, the prototypes are
cancer-specific and the "global coordinate" claim is weakened.

Pairs default to BLCA→KIRC, BLCA→UCEC, BLCA→LUSC. Each pair is two 50-epoch
training jobs (source + target), so a full run is 6 jobs per pair set.

Run from repo root::

  python scripts/run_dct_v382_cross_cancer_prototype.py plan --python python
  python scripts/run_dct_v382_cross_cancer_prototype.py run --python python --gpu 0
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


REPO_ROOT = base.REPO_ROOT
RESULT_ROOT = Path("results/dct_v3.8.2_cross_cancer_prototype/robust")
SMOKE_ROOT = Path("results/dct_v3.8.2_cross_cancer_prototype_smoke")

# Source → target pairs. Each pair tests whether shared prototypes trained on
# one cancer still serve another. BLCA→* spans high/mid/low target events.
DEFAULT_PAIRS = (
    ("blca", "kirc"),
    ("blca", "ucec"),
    ("blca", "lusc"),
)
DEFAULT_FOLDS = (1,)


@dataclass(frozen=True)
class Job:
    phase: str  # "source" or "target"
    source: str
    target: str
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path
    source_ckpt: Path | None  # only set for target-phase jobs


def build_source_job(args: argparse.Namespace, source: str, fold: int, *, smoke: bool) -> Job:
    """Train the frozen v3.8.2 recipe on ``source`` and store its best checkpoint."""
    frozen = base.build_job(args, source, fold, smoke=smoke)
    result_dir = (SMOKE_ROOT if smoke else RESULT_ROOT) / f"src_{source}" / f"fold{fold}"
    command = list(frozen.command)
    _replace_override(command, "results_dir", result_dir.as_posix())
    _replace_override(
        command,
        "specific_simple",
        f"dct_v382_cross_cancer_src_{source}_fold{fold}_"
        f"{'smoke' if smoke else '50ep'}",
    )
    return Job(
        phase="source",
        source=source,
        target=source,
        cancer=source,
        fold=fold,
        command=tuple(command),
        result_dir=result_dir,
        config=frozen.config,
        source_ckpt=None,
    )


def build_target_job(
    args: argparse.Namespace,
    source: str,
    target: str,
    fold: int,
    source_ckpt: Path,
    *,
    smoke: bool,
) -> Job:
    """Train ``target`` while freezing the shared prototype tensors from ``source``."""
    frozen = base.build_job(args, target, fold, smoke=smoke)
    result_dir = (SMOKE_ROOT if smoke else RESULT_ROOT) / f"{source}_to_{target}" / f"fold{fold}"
    command = list(frozen.command)
    _replace_override(command, "results_dir", result_dir.as_posix())
    _replace_override(
        command,
        "specific_simple",
        f"dct_v382_cross_cancer_{source}_to_{target}_fold{fold}_"
        f"{'smoke' if smoke else '50ep'}",
    )
    _replace_override(command, "dct_freeze_source_prototype", str(source_ckpt))
    return Job(
        phase="target",
        source=source,
        target=target,
        cancer=target,
        fold=fold,
        command=tuple(command),
        result_dir=result_dir,
        config=frozen.config,
        source_ckpt=source_ckpt,
    )


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[Job]:
    jobs: list[Job] = []
    for source, target in args.pairs:
        for fold in args.folds:
            source_job = build_source_job(args, source, fold, smoke=smoke)
            jobs.append(source_job)
            ckpt = source_job.result_dir / f"model_best_s{fold}.pth"
            jobs.append(build_target_job(args, source, target, fold, ckpt, smoke=smoke))
    return jobs


def task_lock_path(result_dir: Path, fold: int, suffix: str) -> Path:
    return result_dir / f".split_{fold}.{suffix}.priority_queue.lock"


def _replace_override(command: list[str], key: str, value: object) -> None:
    prefix = f"{key}="
    for index, item in enumerate(command[:-1]):
        if item == "--set" and command[index + 1].startswith(prefix):
            command[index + 1] = f"{key}={value}"
            return
    command.extend(("--set", f"{key}={value}"))


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def parse_pairs(value: str) -> list[tuple[str, str]]:
    pairs = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        parts = token.split("->")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"pair must look like SRC->TGT, got {token!r}"
            )
        if not all(part in base.SUPPORTED_CANCERS for part in parts):
            raise argparse.ArgumentTypeError(
                f"unknown cancer in {token!r}; choose from "
                f"{', '.join(base.SUPPORTED_CANCERS)}"
            )
        pairs.append((parts[0], parts[1]))
    if not pairs:
        raise argparse.ArgumentTypeError("at least one pair is required")
    return pairs


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("DCT v3.8.2 CROSS-CANCER SHARED PROTOTYPE TRANSFER")
    print("Phase 1 trains a source cancer to convergence; phase 2 freezes the shared")
    print("WSI/omic prototype tensors and retrains everything else on the target.")
    print(f"Queue: {len(jobs)} jobs ({len(jobs) // 2} pairs × 2 phases)")
    for index, job in enumerate(jobs, start=1):
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        header = f"{index:02d}. {state} [{job.phase.upper()}] {job.cancer.upper()} fold{job.fold}"
        if job.phase == "target":
            header += f" (← {job.source.upper()} ckpt)"
        print(header)
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
            label=f"DCT v3.8.2 cross-cancer prototype on GPU {args.gpu}",
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
                    f"{job.phase} {job.cancer.upper()} fold{job.fold}: {completion}"
                )
                continue
            if job.phase == "target" and not (job.source_ckpt and job.source_ckpt.exists()):
                print(
                    f"[{index:02d}/{len(jobs):02d}] [skip-missing-source-ckpt] "
                    f"{job.source}->{job.target} fold{job.fold}: {job.source_ckpt}"
                )
                continue
            task_lock = None
            try:
                task_lock = base.acquire_run_lock(
                    task_lock_path(job.result_dir, job.fold, "cross_cancer_proto"),
                    label=(
                        f"DCT v3.8.2 {job.phase} {job.cancer.upper()} "
                        f"fold{job.fold}"
                    ),
                )
            except base.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] DCT v3.8.2 "
                    f"{job.phase} {job.cancer.upper()} fold{job.fold}"
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
    parser.set_defaults(cancers=None, folds=list(DEFAULT_FOLDS))
    parser.add_argument(
        "--pairs",
        type=parse_pairs,
        default=list(DEFAULT_PAIRS),
        help=(
            "Comma-separated source->target pairs. "
            f"Default: {','.join(f'{s}->{t}' for s, t in DEFAULT_PAIRS)}"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cancers is not None:
        # Without --cancers, the queue is determined entirely by --pairs.
        args.cancers = sorted({cancer for pair in args.pairs for cancer in pair})
    os.chdir(REPO_ROOT)
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
