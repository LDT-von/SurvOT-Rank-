#!/usr/bin/env python3
"""Run DCT v3.10 on the old (pre-2026-07-30) splits for legacy comparison.

This queue mirrors the DCT v3.10 paper objective:

    NLL + 0.10 * IPCW-rank

but uses `5fold_legacy` splits so results are comparable with v3.3/v3.6/v3.7
experiments that were run before the split re-generation in commit bee66a2.

Queue structure:
  Stage 0: smoke  — BLCA fold0, 2 epochs, 2 batches (sanity check)
  Stage 1: proof  — BLCA fold0, 50 epochs  (proves it works on old split)
  Stage 2: main   — BLCA + SKCM + HNSC + LUSC + KIRC + UCEC, all 5 folds
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
    from scripts import run_dct_v382_final_cross_cancer as legacy
except (ModuleNotFoundError, ImportError):
    import run_dct_v382_final_cross_cancer as legacy


REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_CANCERS = legacy.SUPPORTED_CANCERS
LEGACY_WHICH_SPLITS = "5fold_legacy"
LEGACY_DATA_PATH = "survot_rank/research/legacy/slotspe_runtime/dataset_csv"

# Old split uses UNI v1 (1024d), not UNI2-h (1536d)
DEFAULT_CANCERS = ("blca", "skcm", "hnsc", "lusc", "kirc", "ucec")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
RESULT_ROOT = Path("results/dct_v310_legacy/final")
SMOKE_ROOT = Path("results/dct_v310_legacy_smoke/final")
PROOF_ROOT = Path("results/dct_v310_legacy/proof")

FROZEN_LEGACY_OVERRIDES: dict[str, object] = {
    "survot_method": "dct_v310_directional_regularized_transport",
    "bag_loss": "nll_surv",
    "max_epochs": 50,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    "dct_v38_warmup_epochs": 0,
    "dct_v38_ramp_epochs": 0,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_v382_lambda_mgptr": 0.0,
    "dct_v382_adaptive_aux_weights": False,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_slot_init_mode": "deterministic",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "dct_ipcw_rank_memory_size": 64,
    "dct_mix_ratio": 1.0,
    "num_patches": 2048,
    "batch_size": 8,
    # Legacy split
    "which_splits": LEGACY_WHICH_SPLITS,
    "data_path": LEGACY_DATA_PATH,
    "on_missing_wsi": "error",
    # UNI v1 (1024d) for old split
    "wsi_encoder": "uni",
    "encoding_dim": 1024,
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_cancers(value: str) -> list[str]:
    return legacy.parse_cancers(value)


def _build_job(
    args: argparse.Namespace,
    cancer: str,
    fold: int,
    *,
    result_dir: Path,
    max_epochs: int = 50,
    specific_suffix: str = "dct_v310_legacy",
) -> Job:
    config = Path("configs/dct_v310_directional_regularized_transport.yaml")
    values = dict(FROZEN_LEGACY_OVERRIDES)
    values.update(
        {
            "study": cancer,
            "data_root_dir": args.data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": args.gpu,
            "num_workers": args.num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": f"{specific_suffix}_{cancer}_f{fold}",
            "max_epochs": max_epochs,
        }
    )
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config.as_posix(),
        *legacy._override_args(values),
    )
    return Job(cancer, fold, command, result_dir, config)


def build_smoke_jobs(args: argparse.Namespace) -> list[Job]:
    """Stage 0: smoke test — BLCA fold0, 2 epochs."""
    return [
        _build_job(
            args,
            cancer="blca",
            fold=0,
            result_dir=SMOKE_ROOT / "blca",
            max_epochs=2,
            specific_suffix="dct_v310_legacy_smoke",
        )
    ]


def build_proof_jobs(args: argparse.Namespace) -> list[Job]:
    """Stage 1: proof — BLCA fold0, 50 epochs."""
    return [
        _build_job(
            args,
            cancer="blca",
            fold=0,
            result_dir=PROOF_ROOT / "blca",
            max_epochs=50,
            specific_suffix="dct_v310_legacy_proof",
        )
    ]


def build_main_jobs(args: argparse.Namespace) -> list[Job]:
    """Stage 2: main — 6 cancers × 5 folds."""
    jobs = []
    for cancer in args.cancers:
        for fold in args.folds:
            jobs.append(
                _build_job(
                    args,
                    cancer=cancer,
                    fold=fold,
                    result_dir=RESULT_ROOT / cancer,
                    max_epochs=50,
                    specific_suffix="dct_v310_legacy",
                )
            )
    return jobs


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], label: str, *, force: bool = False, run_mode: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"  DCT v3.10 (legacy splits) — {label}")
    print(f"{'=' * 60}")
    print(f"OBJECTIVE: NLL + 0.10*IPCW-rank")
    print(f"SPLIT: {LEGACY_WHICH_SPLITS} (old, pre-2026-07-30)")
    print(f"ENCODER: UNI v1 (1024d)")
    print(f"Queue: {len(jobs)} jobs")
    current_cancer = None
    for index, job in enumerate(jobs, start=1):
        if job.cancer != current_cancer:
            current_cancer = job.cancer
            print(f"\n[{job.cancer.upper()}]")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"  {index:02d}. {state} fold{job.fold}")
        print(f"      " + shlex.join(job.command))
        if completion:
            print(f"      completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, label: str) -> int:
    if legacy.doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not legacy.verify_child_cuda(args.python_bin, environment):
        return 1

    scheduler_lock = None
    try:
        scheduler_lock = legacy.acquire_run_lock(
            legacy.scheduler_lock_path(args.gpu, smoke=False),
            label=f"DCT v3.10 legacy {label} on GPU {args.gpu}",
        )
    except legacy.ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3

    try:
        for index, job in enumerate(jobs, start=1):
            completion = _completion(job)
            if completion and not args.force:
                print(
                    f"[{index:02d}/{len(jobs):02d}] [skip] "
                    f"{job.cancer.upper()} fold{job.fold}: {completion}"
                )
                continue
            task_lock = None
            try:
                task_lock = legacy.acquire_run_lock(
                    legacy.task_lock_path(job),
                    label=f"DCT v3.10 legacy {job.cancer.upper()} fold{job.fold}",
                )
            except legacy.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"DCT v3.10 legacy {label} — {job.cancer.upper()} fold{job.fold}"
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
                legacy.release_run_lock(task_lock)
        return 0
    finally:
        legacy.release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = legacy.build_parser()
    parser.description = __doc__
    parser.set_defaults(cancers=list(DEFAULT_CANCERS), folds=list(DEFAULT_FOLDS))
    # Add 'proof' subcommand
    mode_arg = next(a for a in parser._actions if a.dest == "mode")
    mode_arg.choices = ("plan", "prepare", "doctor", "smoke", "proof", "run")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)

    if args.mode == "prepare":
        return legacy.prepare_splits(args)
    if args.mode == "doctor":
        return legacy.doctor(args)

    # Smoke / proof / main all support plan mode via --plan flag
    is_smoke = args.mode == "smoke"
    is_proof = args.mode == "proof"
    is_plan = args.mode == "plan"
    is_main = args.mode in ("run", "plan") or not (is_smoke or is_proof)

    if is_smoke:
        jobs = build_smoke_jobs(args)
        print_plan(jobs, "SMOKE (2ep/2batch)", run_mode=not is_plan)
        if is_plan:
            return 0
        return run_queue(args, jobs, label="smoke")

    if is_proof:
        jobs = build_proof_jobs(args)
        print_plan(jobs, "PROOF (BLCA fold0, 50ep)", run_mode=not is_plan)
        if is_plan:
            return 0
        return run_queue(args, jobs, label="proof")

    # Main
    jobs = build_main_jobs(args)
    print_plan(jobs, "MAIN (6 cancers × 5 folds)", force=args.force, run_mode=not is_plan)
    if is_plan:
        return 0
    return run_queue(args, jobs, label="main")


if __name__ == "__main__":
    raise SystemExit(main())
