#!/usr/bin/env python3
"""Run one frozen, non-trivial IST-Surv v4.0 recipe across cancers.

The retained mechanism is staged intervention-stability feedback into the OT
cost. Auxiliary plan/attribution/risk losses are fixed at zero because the
matched BLCA ablation found no measurable gain from them.
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
    from scripts.run_dct_v382_final_cross_cancer import (
        DEFAULT_FOLDS,
        SUPPORTED_CANCERS,
        _safe_gpu_name,
        prepare_splits,
    )
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
    from run_dct_v382_final_cross_cancer import (
        DEFAULT_FOLDS,
        SUPPORTED_CANCERS,
        _safe_gpu_name,
        prepare_splits,
    )
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
CONFIG = Path("configs/intervention_stable_survival_transport.yaml")
WHICH_SPLITS = "5fold_uni2h"
RESULT_ROOT = Path("results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only")
REPAIRED_RESULT_ROOT = Path(
    "results/ist_surv_v4.0_repaired_50ep/clean/importance_instability"
)

# BLCA needs fold0/3 to complete its existing A/B/C comparison. The other five
# cancers were feature-complete in the 2026-08-06 audit. Current server state
# is always rechecked by doctor before training.
DEFAULT_CANCERS = ("blca", "skcm", "hnsc", "lusc", "kirc", "ucec")

# Frozen B-stage recipe: the simplest configuration that still contains the
# defining IST intervention-stability mechanism.
FINAL_OVERRIDES: dict[str, object] = {
    "survot_method": "intervention_stable_survival_transport",
    "max_epochs": 50,
    "num_patches": 2048,
    "batch_size": 8,
    "grad_accum_steps": 1,
    "warmup_epochs": 5,
    "grad_clip_norm": 1.0,
    "early_stop_patience": 0,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "event_sampling_fraction": 0.0,
    "event_stratified_batches": False,
    "seed": 3,
    "lr": 0.0005,
    "opt": "adamW",
    "reg": 0.0005,
    "scheduler": "cosine",
    "eta_min": 0.000001,
    "bag_loss": "nll_surv",
    "alpha_surv": 0.15,
    "wsi_projection_dim": 256,
    "ist_eps": 0.05,
    "ist_sinkhorn_iters": 30,
    "ist_num_interventions": 2,
    "ist_keep_ratio": 0.75,
    "ist_stability_beta": 1.0,
    "ist_stability_strength": 0.10,
    "ist_stability_normalization": "raw_mass",
    "ist_feedback_mode": "legacy_product",
    "ist_lambda_plan": 0.0,
    "ist_lambda_attribution": 0.0,
    "ist_lambda_risk": 0.0,
    "ist_edge_value_scale": 4.0,
    "ist_eval_seed": 20260725,
    "ist_deletion_penalty": 8.0,
    "ist_warmup_epochs": 5,
    "ist_ramp_epochs": 10,
    "which_splits": WHICH_SPLITS,
    "on_missing_wsi": "error",
    "wsi_encoder": "uni2-h",
    "encoding_dim": 1536,
}

# Corrected feedback semantics.  This is deliberately opt-in so the completed
# v4.0 results remain exactly reproducible.  The repair gate must pass on the
# pre-registered BLCA folds before any cross-cancer expansion.
REPAIRED_OVERRIDES: dict[str, object] = {
    "ist_stability_normalization": "independence_lift",
    "ist_feedback_mode": "importance_weighted_instability",
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path


def parse_cancers(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(SUPPORTED_CANCERS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
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
    recipe = getattr(args, "recipe", "legacy")
    repaired = recipe == "repaired"
    result_root = REPAIRED_RESULT_ROOT if repaired else RESULT_ROOT
    result_dir = result_root / cancer
    values = dict(FINAL_OVERRIDES)
    if repaired:
        values.update(REPAIRED_OVERRIDES)
    values.update(
        {
            "study": cancer,
            "data_root_dir": args.data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": args.gpu,
            "num_workers": args.num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": (
                f"ist_v40_repaired_importance_instability_{cancer}_50ep"
                if repaired
                else f"ist_v40_abl_b_cost_only_{cancer}_50ep"
            ),
        }
    )
    if smoke:
        smoke_variant = "repaired" if repaired else "cost_only"
        result_dir = Path("results/ist_surv_v4.0_final_smoke") / smoke_variant / cancer
        values.update(
            {
                "max_epochs": 1,
                "max_smoke_batches": 2,
                "results_dir": result_dir.as_posix(),
                "specific_simple": f"ist_v40_{smoke_variant}_smoke_{cancer}",
            }
        )
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        CONFIG.as_posix(),
        *_override_args(values),
    )
    return Job(cancer=cancer, fold=fold, command=command, result_dir=result_dir)


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


def scheduler_lock_path(gpu: str, smoke: bool) -> Path:
    # Shared with every current long-running queue in this repository.
    kind = "smoke" if smoke else "run"
    return Path("results/priority_experiment_queue") / (
        f".{kind}_gpu_{_safe_gpu_name(gpu)}.lock"
    )


def task_lock_path(job: Job) -> Path:
    return job.result_dir / f".split_{job.fold}.priority_queue.lock"


def doctor(args: argparse.Namespace) -> int:
    config_ok = (REPO_ROOT / CONFIG).is_file()
    print(f"{'OK' if config_ok else 'MISSING':8s} config {REPO_ROOT / CONFIG}")
    failed = not config_ok
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
            "[BLOCKED] Formal IST training requires complete UNI2-h coverage "
            "and an audited 5fold_uni2h split for every selected cancer."
        )
    return int(failed)


def print_plan(
    jobs: list[Job],
    *,
    recipe: str = "legacy",
    force: bool = False,
    run_mode: bool = False,
) -> None:
    if recipe == "repaired":
        print("IST REPAIR GATE: support-normalized importance-instability feedback")
    else:
        print("FINAL IST: v4.0 staged stability-cost feedback only (B stage)")
    print(f"Queue: {len(jobs)} jobs; cancers={len({job.cancer for job in jobs})}")
    current_cancer = None
    for index, job in enumerate(jobs, start=1):
        if current_cancer != job.cancer:
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
            label=f"final IST cross-cancer queue on GPU {args.gpu}",
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
                    label=f"final IST {job.cancer.upper()} fold{job.fold}",
                )
            except ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"IST {getattr(args, 'recipe', 'legacy')} "
                    f"{job.cancer.upper()} fold{job.fold}"
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
    parser.add_argument(
        "--recipe",
        choices=("legacy", "repaired"),
        default="legacy",
        help="legacy reproduces completed v4.0; repaired is the pre-registered repair gate",
    )
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
        print_plan(jobs, recipe=args.recipe)
        return 0
    print_plan(
        jobs,
        recipe=args.recipe,
        force=args.force,
        run_mode=args.mode == "run",
    )
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
