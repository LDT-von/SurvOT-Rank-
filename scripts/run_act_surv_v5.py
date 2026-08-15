#!/usr/bin/env python3
"""Run ACT-Surv v5 (archetypal_transport_composition_v5) on BLCA + cross-cancer.

This launcher is intentionally minimal — it mirrors the frozen DCT v3.8.2 fixed-full
launcher structure but launches only ACT-Surv v5 under the same protocol so results
are directly comparable.

Usage:
    # BLCA 3-fold quick validation (before committing to full 5-fold)
    python scripts/run_act_surv_v5.py --cancers blca --folds 0 1 2

    # Full BLCA 5-fold
    python scripts/run_act_surv_v5.py --cancers blca --folds 0 1 2 3 4

    # 6-cancer cross-cancer
    python scripts/run_act_surv_v5.py --cancers blca,kirc,ucec,hnsc,lusc,skcm

    # Dry-run (print commands only)
    python scripts/run_act_surv_v5.py --cancers blca --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Shared infrastructure (mirrors run_dct_v382_final_cross_cancer.py) ────────

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.run_dct_v38_transport_consistency import (
        DATASET_CSV_ROOT,
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
        DATASET_CSV_ROOT,
        DEFAULT_DATA_ROOT,
        _override_args,
        inspect_feature_directory,
        inspect_split_directory,
        parse_folds,
        verify_child_cuda,
    )
    from task_lock import ActiveRunError, acquire_run_lock, release_run_lock


SUPPORTED_CANCERS = (
    "blca", "brca", "coadread", "hnsc", "kirc",
    "luad", "lusc", "skcm", "stad", "ucec",
)
DEFAULT_CANCERS = ("blca", "kirc", "ucec", "hnsc", "lusc", "skcm")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
RESULT_ROOT = Path("results/act_surv_v5")
WHICH_SPLITS = "5fold_uni2h"
DATA_ROOT = DEFAULT_DATA_ROOT  # mutable; main() may overwrite via --data-root

# Frozen v5 recipe — do not add cancer-specific tuning here.
FINAL_OVERRIDES: dict[str, object] = {
    "data_root_dir": "/data1/TCGA-UNI2-h-features",
    "data_path": "survot_rank/research/legacy/slotspe_runtime/dataset_csv",
    "survot_method": "archetypal_transport_composition_v5",
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
    # ACT-Surv v5 core
    "act5_num_archetypes": 6,
    "act5_epsilon": 0.10,
    "act5_hazard_scale": 1.0,
    "act5_warmup_epochs": 5,
    "act5_lambda_balance": 0.01,
    "act5_lambda_rank": 0.10,
    "act5_rank_margin": 0.02,
    "act5_rank_temperature": 0.50,
    "act5_rank_max_pairs": 4096,
    # Data format
    "rna_format": "Pathways",
    "gpu": 0,
}


@dataclass
class Job:
    cancer: str
    fold: int
    config: Path
    result_dir: Path
    cmd: list[str]
    env_extra: dict[str, str] | None = None

    def submit(self, dry_run: bool = False):
        if dry_run:
            print(f"[dry-run] {' '.join(shlex.quote(str(c)) for c in self.cmd)}")
            return 0
        env = dict(os.environ)
        if self.env_extra:
            env.update(self.env_extra)
        lock = None
        try:
            lock = acquire_run_lock(self.result_dir / ".run.lock", label=f"act_surv_v5_{self.cancer}_fold{self.fold}")
            print(f"Running: {self.cancer} fold {self.fold}")
            proc = subprocess.run(self.cmd, env=env)
            return proc.returncode
        except ActiveRunError as exc:
            print(f"SKIP (active run): {exc}", file=sys.stderr)
            return 0
        finally:
            release_run_lock(lock)


def build_jobs(
    cancers: list[str],
    folds: list[int],
    overrides: dict[str, object],
    dry_run: bool = False,
) -> list[Job]:
    """Build one Job per (cancer, fold) pair."""
    if not dry_run:
        verify_child_cuda(sys.executable, dict(os.environ))
    jobs: list[Job] = []

    for cancer in cancers:
        if cancer not in SUPPORTED_CANCERS:
            raise ValueError(f"Unsupported cancer: {cancer}. Choose from {SUPPORTED_CANCERS}")

        # ── Verify data ──────────────────────────────────────────────────
        csv_dir = Path(DATASET_CSV_ROOT) / "splits" / WHICH_SPLITS / cancer
        inspect_feature_directory(DATA_ROOT, cancer)
        inspect_split_directory(cancer, which_splits=WHICH_SPLITS)
        split_folds = sorted([
            int(p.name.split("_")[1].split(".")[0])
            for p in csv_dir.glob("fold_*.csv")
        ])

        for fold in folds:
            if fold not in split_folds:
                print(f"WARN: fold {fold} not in {csv_dir}, skipping")
                continue

            config = REPO_ROOT / "configs" / "act_surv_v5_blca.yaml"
            result_dir = RESULT_ROOT / cancer / f"fold{fold}"
            result_dir.mkdir(parents=True, exist_ok=True)

            fold_overrides = {
                **overrides,
                "k_start": fold,
                "k_end": fold + 1,
                "study": cancer,
                "specific_simple": f"act_surv_v5_{cancer}_fold{fold}",
            }
            if "max_epochs" in fold_overrides and fold_overrides["max_epochs"] > 2:
                # Logging pipe for long jobs
                pass
            args = _override_args(fold_overrides)
            cmd = [
                sys.executable,
                "-m",
                "survot_rank.cli",
                "train",
                "--config",
                config.as_posix(),
                *args,
            ]
            jobs.append(Job(
                cancer=cancer,
                fold=fold,
                config=config,
                result_dir=result_dir,
                cmd=cmd,
                env_extra={"PYTHONPATH": str(REPO_ROOT)},
            ))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run ACT-Surv v5")
    parser.add_argument(
        "--cancers", default=",".join(DEFAULT_CANCERS),
        help="Comma-separated cancer codes (default: blca,kirc,ucec,hnsc,lusc,skcm)",
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS),
        help="Fold indices (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT,
        help="Override data root",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--result-root", default=str(RESULT_ROOT),
        help="Override result root directory",
    )
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = args.data_root

    cancers = [c.strip() for c in args.cancers.split(",")]
    folds = args.folds

    overrides = {**FINAL_OVERRIDES, "results_dir": args.result_root}
    jobs = build_jobs(cancers, folds, overrides, dry_run=args.dry_run)

    print(f"Total jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j.cancer} fold {j.fold}  →  {j.result_dir}")

    if args.dry_run:
        return 0

    for j in jobs:
        rc = j.submit(dry_run=False)
        if rc != 0:
            print(f"FAILED: {j.cancer} fold {j.fold} (exit {rc})", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
