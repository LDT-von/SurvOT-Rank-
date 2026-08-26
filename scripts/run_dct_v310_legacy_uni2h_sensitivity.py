#!/usr/bin/env python3
"""DCT v3.10 sensitivity queue: lambda_dir in {0.0, 0.05} on the legacy split.

This is a one-off launcher reserved for the direction-regularization sensitivity
sweep described by the user:

* Protocol: UNI2-h (1536d), 5fold_legacy splits, clean binning, 50 epochs.
* Cancers: the six UNI2-h-complete set (BLCA, SKCM, HNSC, LUSC, KIRC, UCEC).
* Folds: 1 and 2 only, to keep the budget at 6 cancers × 2 folds × 2 lambdas
  = 24 jobs (~6 GPU-hours on the production partition).
* Objective: NLL + 0.10 * IPCW-rank + lambda_dir * direction.

The frozen v3.10 paper objective is NLL + 0.10 * IPCW-rank with direction at
zero. The model class honours an opt-in environment variable
``SURVOT_V310_DIR_WEIGHT`` that lets this launcher re-enable a non-zero
direction coefficient without modifying the frozen recipe in source. Passing
``--lambda-dir 0.05`` to this launcher sets that environment variable on the
child process before the trainer imports the model class.

Run from the repo root::

    python scripts/run_dct_v310_legacy_uni2h_sensitivity.py plan
    python scripts/run_dct_v310_legacy_uni2h_sensitivity.py run --gpu 0
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


def _trisurv_python() -> str:
    """Pick the Python interpreter that ships with the ``trisurv`` env.

    The CUDA probe this launcher runs as a child must use the same interpreter
    as the trainer; that interpreter lives in the ``trisurv`` conda environment
    on this host.  We honour ``CONDA_PREFIX`` if already inside trisurv and
    fall back to the canonical env path so the launcher keeps working on
    other machines.
    """

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and conda_prefix.endswith("trisurv"):
        candidate = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    default_trisurv = "/home/ubuntu/.conda/envs/trisurv/bin/python"
    if os.path.exists(default_trisurv):
        return default_trisurv
    return os.environ.get("PYTHON_BIN", sys.executable)

# Cancers that have a complete UNI2-h feature matrix; mirrors
# run_dct_v310_final_cross_cancer.DEFAULT_CANCERS so sensitivity runs are
# directly comparable to the cross-cancer results.
DEFAULT_CANCERS = ("blca", "skcm", "hnsc", "lusc", "kirc", "ucec")
# Folds 1 and 2 only — the user-requested budget cap.
DEFAULT_FOLDS = (1, 2)
# Two lambda values only.
DEFAULT_LAMBDAS = (0.0, 0.05)

RESULT_ROOT = Path("results/dct_v310_legacy_uni2h_sensitivity")


FROZEN_SENSITIVITY_OVERRIDES: dict[str, object] = {
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
    # Sensitivity target: legacy fold set on the UNI2-h encoding path.
    "which_splits": LEGACY_WHICH_SPLITS,
    "data_path": LEGACY_DATA_PATH,
    "on_missing_wsi": "error",
    "wsi_encoder": "uni2-h",
    "encoding_dim": 1536,
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    lambda_dir: float
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_cancers(value: str) -> list[str]:
    return legacy.parse_cancers(value)


def parse_folds(value: str) -> list[int]:
    return legacy.parse_folds(value)


def parse_lambdas(value: str) -> list[float]:
    out: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(float(raw))
    if not out:
        raise argparse.ArgumentTypeError("at least one --lambda-dir value is required")
    return out


def _build_job(
    args: argparse.Namespace,
    cancer: str,
    fold: int,
    lambda_dir: float,
) -> Job:
    config = Path("configs/dct_v310_directional_regularized_transport.yaml")
    lambda_tag = f"d{lambda_dir:.2f}".rstrip("0").rstrip(".") or "d0"
    result_dir = RESULT_ROOT / f"lambda_{lambda_tag}" / cancer
    values = dict(FROZEN_SENSITIVITY_OVERRIDES)
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
                f"dct_v310_legacy_uni2h_d{int(round(lambda_dir * 100)):02d}_"
                f"{cancer}_f{fold}"
            ),
            "max_epochs": 50,
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
    return Job(cancer, fold, lambda_dir, command, result_dir, config)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        _build_job(args, cancer, fold, lambda_dir)
        for cancer in args.cancers
        for fold in args.folds
        for lambda_dir in args.lambda_dirs
    ]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(
    jobs: list[Job], *, force: bool = False, run_mode: bool = False
) -> None:
    print(f"{'=' * 70}")
    print("  DCT v3.10 sensitivity — legacy split, UNI2-h, direction only")
    print(f"{'=' * 70}")
    print("OBJECTIVE: NLL + 0.10 * IPCW-rank + lambda_dir * direction")
    print(f"SPLIT: {LEGACY_WHICH_SPLITS} (legacy / pre-2026-07-30 fold set)")
    print(f"ENCODER: UNI2-h (1536d)")
    print(
        f"Queue: {len(jobs)} jobs; "
        f"cancers={len({job.cancer for job in jobs})}; "
        f"folds={sorted({job.fold for job in jobs})}; "
        f"lambdas={sorted({job.lambda_dir for job in jobs})}"
    )
    last_bucket = None
    for index, job in enumerate(jobs, start=1):
        bucket = (job.lambda_dir, job.cancer)
        if bucket != last_bucket:
            last_bucket = bucket
            lambda_tag = f"{job.lambda_dir:g}"
            print(f"\n[lambda_dir={lambda_tag}] [{job.cancer.upper()}]")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"  {index:02d}. {state} fold{job.fold}")
        print(f"      " + shlex.join(job.command))
        if completion:
            print(f"      completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job]) -> int:
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
            label=f"DCT v3.10 legacy UNI2-h sensitivity on GPU {args.gpu}",
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
                    f"lambda={job.lambda_dir} {job.cancer.upper()} "
                    f"fold{job.fold}: {completion}"
                )
                continue

            task_lock = None
            try:
                task_lock = legacy.acquire_run_lock(
                    legacy.task_lock_path(job),
                    label=(
                        f"DCT v3.10 sensitivity lambda={job.lambda_dir} "
                        f"{job.cancer.upper()} fold{job.fold}"
                    ),
                )
            except legacy.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue

            child_env = dict(environment)
            child_env["SURVOT_V310_DIR_WEIGHT"] = f"{job.lambda_dir:g}"

            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"DCT v3.10 sensitivity "
                    f"lambda_dir={job.lambda_dir:g} {job.cancer.upper()} "
                    f"fold{job.fold}"
                )
                print(f"SURVOT_V310_DIR_WEIGHT={child_env['SURVOT_V310_DIR_WEIGHT']}")
                print(shlex.join(job.command))
                completed = subprocess.run(
                    job.command, check=False, env=child_env
                )
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


def _trisurv_python() -> str:
    """Locate the Python interpreter that ships with the ``trisurv`` env.

    The doctor step in this launcher assumes the legacy split lives on a
    Python with ``torch`` available; that interpreter lives in the ``trisurv``
    conda environment on this host.  We do not hard-code the absolute path
    so the launcher keeps working on other machines, but we honour
    ``CONDA_PREFIX`` and the obvious env-relative lookup.
    """

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and conda_prefix.endswith("trisurv"):
        candidate = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    default_trisurv = "/home/ubuntu/.conda/envs/trisurv/bin/python"
    if os.path.exists(default_trisurv):
        return default_trisurv
    return os.environ.get("PYTHON_BIN", sys.executable)


def build_parser() -> argparse.ArgumentParser:
    parser = legacy.build_parser()
    parser.description = __doc__
    # Drop the smoke/proof/default-mode machinery inherited from the parent
    # launcher; this script is a one-off sensitivity queue.
    mode_arg = next(a for a in parser._actions if a.dest == "mode")
    mode_arg.choices = ("plan", "doctor", "run")
    parser.set_defaults(
        cancers=list(DEFAULT_CANCERS),
        folds=list(DEFAULT_FOLDS),
        mode="plan",
        python_bin=_trisurv_python(),
    )
    parser.add_argument(
        "--lambda-dir",
        dest="lambda_dirs",
        type=parse_lambdas,
        default=list(DEFAULT_LAMBDAS),
        help=(
            "Comma-separated direction regularization coefficients. "
            "Default: %(default)s. The flag is propagated to the child "
            "process via SURVOT_V310_DIR_WEIGHT so the v3.10 paper-objective "
            "invariant in source code stays untouched."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)

    if args.mode == "doctor":
        return legacy.doctor(args)

    jobs = build_jobs(args)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    print_plan(jobs, force=args.force, run_mode=True)
    return run_queue(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
