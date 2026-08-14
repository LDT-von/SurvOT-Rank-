#!/usr/bin/env python3
"""Paper-facing evidence ablations for frozen DCT v3.8.2.

Five ablations, each changing exactly one mechanism from the frozen recipe:

  fixed_coupling        — ablation 1 (OT intervention chain evidence).
                          Replay cached factual plans instead of re-solving
                          Sinkhorn under each cost intervention. If C-index
                          and audit metrics do not collapse, the OT re-solve
                          chain is decorative.

  random_anchors        — ablation 2 (anchor evidence).
                          Replace IPCW-aggregated risk-set anchors with
                          randomly-perturbed versions. If the model still
                          improves, the IPCW anchors carry no signal.

  cross_cancer_prototype — ablation 3 (shared semantic coordinate evidence).
                          Train on cancer A, freeze shared prototype weights,
                          train other parameters on cancer B. If frozen
                          prototypes remain competitive, the shared coordinates
                          are not cancer-specific.

  null_calibration      — ablation 4 (audit specificity evidence).
                          Permute train labels at multiple seeds. Audit
                          metrics under label permutation should approach
                          their chance values; large gaps prove the audit
                          measures real signal rather than numerical artefact.

  stage_randomization   — ablation 5 (stage-edge evidence).
                          Jitter stage quantile edges by ± fraction while
                          keeping them monotonic. If C-index is unaffected,
                          the exact edge placement carries no information.

Cancers and folds default to UCEC fold 1, BLCA fold 1, LUSC fold 1 — one
high-, one mid-, one low-performing cancer, single fold each (per project
agreement on Friday 2026-08-14).

Run from repo root::

  python scripts/run_dct_v382_paper_evidence.py plan --python python
  python scripts/run_dct_v382_paper_evidence.py run --python python --gpu 0
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

# ---------------------------------------------------------------------------
# Ablation table. Each entry names the CLI flag(s) added on top of the frozen
# FINAL_OVERRIDES. ``description`` is printed in plan output so the operator
# sees what the ablation changes without needing the paper draft.
# ---------------------------------------------------------------------------

ABLATIONS: dict[str, dict[str, object]] = {
    "fixed_coupling": {
        "dct_fixed_coupling": True,
    },
    "random_anchors": {
        "dct_random_anchors": True,
    },
    "null_calibration": {
        "dct_perm_labels_seed": 1,
    },
    "stage_randomization": {
        "dct_stage_jitter_fraction": 0.30,
    },
}

DESCRIPTIONS: dict[str, str] = {
    "fixed_coupling": (
        "Replay cached factual plans; do not re-solve Sinkhorn on each "
        "cost intervention. Probes whether OT re-solve is load-bearing."
    ),
    "random_anchors": (
        "Replace IPCW-aggregated risk-set anchors with random perturbations. "
        "Probes whether IPCW anchor carries the prognostic signal."
    ),
    "null_calibration": (
        "Permute train labels (seed=1) so censoring reference and stage edges "
        "see shuffled ordering. Probes whether audit metrics are chance-level "
        "under label shuffling."
    ),
    "stage_randomization": (
        "Jitter stage quantile edges by ±30% of the edge spread. Probes "
        "whether the exact edge placement carries the IPCW signal."
    ),
}

DEFAULT_ABLATIONS = tuple(ABLATIONS)
RESULT_ROOT = Path("results/dct_v3.8.2_paper_evidence/robust")
SMOKE_ROOT = Path("results/dct_v3.8.2_paper_evidence_smoke")

# Single fold per cancer by agreement with user on 2026-08-14:
# keep total runs manageable while preserving high/mid/low C-index spread.
DEFAULT_CANCERS = ("ucec", "blca", "lusc")
DEFAULT_FOLDS = (1,)


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
        f"dct_v382_evidence_{variant}_{cancer}_{'smoke' if smoke else '50ep'}",
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
    print("DCT v3.8.2 PAPER EVIDENCE ABLATIONS (mechanism, not score)")
    print(f"Queue: {len(jobs)} jobs across {len({job.variant for job in jobs})} variants "
          f"× {len({job.cancer for job in jobs})} cancers × {len({job.fold for job in jobs})} folds")
    seen_descriptions: set[str] = set()
    current = None
    for index, job in enumerate(jobs, start=1):
        if job.variant != current:
            current = job.variant
            if job.variant not in seen_descriptions:
                print(f"\n[{job.variant}] {DESCRIPTIONS[job.variant]}")
                seen_descriptions.add(job.variant)
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} {job.cancer.upper()} fold{job.fold}")
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
            label=f"DCT v3.8.2 paper evidence on GPU {args.gpu}",
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
    parser.set_defaults(cancers=list(DEFAULT_CANCERS), folds=list(DEFAULT_FOLDS))
    parser.add_argument(
        "--ablations",
        type=parse_ablations,
        default=list(DEFAULT_ABLATIONS),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
