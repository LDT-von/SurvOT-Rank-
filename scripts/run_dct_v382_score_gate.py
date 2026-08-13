#!/usr/bin/env python3
"""Run pre-registered DCT v3.8.2 score-improvement gates.

Scientific role
---------------
This launcher does not create a new DCT paper method for every cancer.  It
tests cancer-agnostic training/representation changes against the frozen
``fixed_full`` control on the same UNI2-h, clean 5-fold protocol.

Default screen: BLCA (regression guard), KIRC (near SlotSPE), and SKCM (largest
observed deficit to SlotSPE), folds 1/2/4.  A candidate may be promoted only by
``summarize_dct_v382_score_gate.py``; looking at a single best fold is not a
promotion rule.

What each registered version is for is stored in ``VARIANTS`` and printed in
every plan.  In particular, the grouped phase-2 variants can establish the net
effect of a bundle, but cannot attribute gains to an individual component.
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


RESULT_ROOT = Path("results/dct_v3.8.2_score_gate")
SMOKE_ROOT = Path("results/dct_v3.8.2_score_gate_smoke")
DEFAULT_CANCERS = ("blca", "kirc", "skcm")
DEFAULT_FOLDS = (1, 2, 4)


@dataclass(frozen=True)
class VariantSpec:
    phase: int
    purpose: str
    proves: str
    cannot_prove: str
    overrides: dict[str, object]


# Phase 1 entries change exactly one training/configuration variable relative
# to frozen fixed_full.  They diagnose *where* headroom comes from without
# changing DCT's scientific mechanism.  Phase 2 entries are deliberately
# grouped confirmation recipes and must not be used for component attribution.
VARIANTS: dict[str, VariantSpec] = {
    "patches4096": VariantSpec(
        phase=1,
        purpose="Test whether DCT is input-information limited at 2048 WSI patches.",
        proves="A matched gain supports a larger pathology sampling budget.",
        cannot_prove="It does not prove that the counterfactual transport mechanism improved.",
        overrides={"num_patches": 4096},
    ),
    "grad_accum4": VariantSpec(
        phase=1,
        purpose="Test optimizer stability with four micro-batches per parameter update.",
        proves="A matched gain supports lower-variance parameter updates.",
        cannot_prove=(
            "It does not increase pairwise IPCW comparisons inside each micro-batch and "
            "does not prove better censoring supervision."
        ),
        overrides={"grad_accum_steps": 4},
    ),
    "slot_iters5": VariantSpec(
        phase=1,
        purpose="Test whether three slot-refinement iterations underfit shared semantic coordinates.",
        proves="A matched gain supports deeper iterative slot refinement.",
        cannot_prove="It does not isolate transport or intervention losses.",
        overrides={"slot_iters": 5},
    ),
    "lr2e4": VariantSpec(
        phase=1,
        purpose="Test whether the frozen 5e-4 learning rate causes unstable validation peaks.",
        proves="A gain with a smaller best-to-last gap supports more stable optimization.",
        cannot_prove="It does not establish a new model contribution.",
        overrides={"lr": 2e-4},
    ),
    "predictive_core": VariantSpec(
        phase=2,
        purpose="Test the net prediction effect of removing the unproven auxiliary bundle.",
        proves=(
            "A matched gain shows that MGPTR plus direction/dose/reconfiguration is "
            "not needed by the prediction recipe as a bundle."
        ),
        cannot_prove="It cannot attribute the change to any one of the four removed terms.",
        overrides={
            "dct_v382_lambda_mgptr": 0.0,
            "dct_v38_lambda_direction": 0.0,
            "dct_v38_lambda_dose": 0.0,
            "dct_v38_lambda_reconfiguration": 0.0,
        },
    ),
    "capacity_stable": VariantSpec(
        phase=2,
        purpose="Confirm the joint recipe suggested by the four phase-1 capacity/stability tests.",
        proves="A matched gain supports the combined training recipe before six-cancer expansion.",
        cannot_prove="It cannot attribute a gain to one constituent and must run only after phase 1.",
        overrides={
            "num_patches": 4096,
            "grad_accum_steps": 4,
            "slot_iters": 5,
            "lr": 2e-4,
        },
    ),
}
PHASE1_VARIANTS = tuple(name for name, spec in VARIANTS.items() if spec.phase == 1)
PHASE2_VARIANTS = tuple(name for name, spec in VARIANTS.items() if spec.phase == 2)


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
    aliases = {
        "screen": PHASE1_VARIANTS,
        "phase1": PHASE1_VARIANTS,
        "phase2": PHASE2_VARIANTS,
        "all": tuple(VARIANTS),
    }
    if value in aliases:
        return list(aliases[value])
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variant: {', '.join(unknown)}; choose from {', '.join(VARIANTS)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("at least one variant is required")
    return selected


def _replace_override(command: list[str], key: str, value: object) -> None:
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    prefix = f"{key}="
    for index, item in enumerate(command[:-1]):
        if item == "--set" and command[index + 1].startswith(prefix):
            command[index + 1] = f"{key}={rendered}"
            return
    command.extend(("--set", f"{key}={rendered}"))


def build_job(
    args: argparse.Namespace,
    variant: str,
    cancer: str,
    fold: int,
    *,
    smoke: bool,
) -> Job:
    frozen = base.build_job(args, cancer, fold, smoke=smoke)
    root = SMOKE_ROOT if smoke else RESULT_ROOT
    result_dir = root / variant / cancer
    command = list(frozen.command)
    for key, value in VARIANTS[variant].overrides.items():
        _replace_override(command, key, value)
    _replace_override(command, "results_dir", result_dir.as_posix())
    _replace_override(
        command,
        "specific_simple",
        f"dct_v382_score_gate_{variant}_{cancer}_{'smoke' if smoke else '50ep'}",
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


def completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("DCT v3.8.2 SCORE-IMPROVEMENT GATE")
    print("Control: frozen fixed_full; same UNI2-h / 5fold_uni2h / clean / 50ep protocol")
    print(f"Queue: {len(jobs)} jobs")
    current_variant = None
    for index, job in enumerate(jobs, start=1):
        if job.variant != current_variant:
            current_variant = job.variant
            spec = VARIANTS[job.variant]
            print(f"\n[{job.variant}] phase={spec.phase}")
            print(f"  USE: {spec.purpose}")
            print(f"  CAN SUPPORT: {spec.proves}")
            print(f"  CANNOT SUPPORT: {spec.cannot_prove}")
            print(f"  CHANGES: {spec.overrides}")
        done = completion(job) if run_mode and not force else None
        print(
            f"{index:03d}. {'SKIP' if done else 'RUN '} "
            f"{job.cancer.upper()} fold{job.fold}"
        )
        print("     " + shlex.join(job.command))
        if done:
            print(f"     completed: {done}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if base.doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not base.verify_child_cuda(args.python_bin, environment):
        return 1
    try:
        scheduler_lock = base.acquire_run_lock(
            base.scheduler_lock_path(args.gpu, smoke),
            label=f"DCT v3.8.2 score gate on GPU {args.gpu}",
        )
    except base.ActiveRunError as error:
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
                task_lock = base.acquire_run_lock(
                    job.result_dir / f".split_{job.fold}.priority_queue.lock",
                    label=f"DCT score gate {job.variant} {job.cancer} fold{job.fold}",
                )
            except base.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"[{index}/{len(jobs)}] {job.variant} "
                    f"{job.cancer.upper()} fold{job.fold}"
                )
                completed = subprocess.run(job.command, check=False, env=environment)
                if completed.returncode != 0:
                    print(f"[ERROR] code={completed.returncode}; queue stopped")
                    return completed.returncode
            finally:
                base.release_run_lock(task_lock)
        return 0
    finally:
        base.release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.set_defaults(cancers=list(DEFAULT_CANCERS), folds=list(DEFAULT_FOLDS))
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=list(PHASE1_VARIANTS),
        help="screen/phase1 (default), phase2, all, or comma-separated names",
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
