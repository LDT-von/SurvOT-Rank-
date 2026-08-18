#!/usr/bin/env python3
"""Run ACT-Surv v5 (archetypal_transport_composition_v5) on BLCA + cross-cancer.

This launcher is intentionally minimal — it mirrors the frozen DCT v3.8.2 fixed-full
launcher structure but launches only ACT-Surv v5 under the same protocol so results
are directly comparable.

Usage:
    # BLCA 5-fold with old community-standard split (recommended)
    python scripts/run_act_surv_v5.py --cancers blca --folds 0 1 2 3 4 --which-splits 5fold_legacy

    # BLCA 5-fold with new rebalanced split (default)
    python scripts/run_act_surv_v5.py --cancers blca --folds 0 1 2 3 4

    # v5.1 / v5.2 BLCA variants (BLCA-only tuning configs)
    python scripts/run_act_surv_v5.py --cancers blca --variant v5_1 --folds 0 1 2 3 4 --which-splits 5fold_legacy
    python scripts/run_act_surv_v5.py --cancers blca --variant v5_2 --folds 0 1 2 3 4 --which-splits 5fold_legacy

    # v5.3 / v5.4 1D ablations (isolate ranking vs KL×5 contribution)
    python scripts/run_act_surv_v5.py --cancers blca --variant v5_3 --folds 0 1 2 3 4 --which-splits 5fold_legacy
    python scripts/run_act_surv_v5.py --cancers blca --variant v5_4 --folds 0 1 2 3 4 --which-splits 5fold_legacy

    # v5_k4 K=6 vs K=4 ablation (proves K=6 is necessary)
    python scripts/run_act_surv_v5.py --cancers blca --variant v5_k4 --folds 0 1 2 3 4 --which-splits 5fold_legacy

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
# Each variant writes to its own top-level results dir, so identical recipes
# (same lr/batch/dim/seed → identical param_code) don't collide in trainer's
# per-experiment subdirectory. Without this, v5_1 results would be silently
# written under act_surv_v5/blca/.../sp_act_surv_v5_blca_fold*/ and become
# indistinguishable from the v5 baseline checkpoint series.
_VARIANT_RESULT_ROOT = {
    "v5": "results/act_surv_v5",
    "v5_1": "results/act_surv_v5_1",
    "v5_2": "results/act_surv_v5_2",
    "v5_3": "results/act_surv_v5_3",
    "v5_4": "results/act_surv_v5_4",
    "v5_k4": "results/act_surv_v5_k4",
}
_DEFAULT_WHICH_SPLITS = "5fold_uni2h"   # protocol default
DATA_ROOT = DEFAULT_DATA_ROOT  # mutable; main() may overwrite via --data-root
_WHICH_SPLITS = _DEFAULT_WHICH_SPLITS  # module-global for use in build_jobs

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
    "which_splits": _WHICH_SPLITS,
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
    variant: str = "v5",
) -> list[Job]:
    """Build one Job per (cancer, fold) pair.

    variant: config-name suffix before `_blca.yaml`. Default `v5` (baseline).
    Use `v5_1` for `act_surv_v5_1_blca.yaml` (no IPCW rank, KL balance x5),
    `v5_2` for `act_surv_v5_2_blca.yaml` (v5.1 + temperature/margin/max_pairs sweep).
    """
    if not dry_run:
        verify_child_cuda(sys.executable, dict(os.environ))
    jobs: list[Job] = []
    result_root = Path(overrides["results_dir"])

    for cancer in cancers:
        if cancer not in SUPPORTED_CANCERS:
            raise ValueError(f"Unsupported cancer: {cancer}. Choose from {SUPPORTED_CANCERS}")

        # ── Verify data ──────────────────────────────────────────────────
        csv_dir = Path(DATASET_CSV_ROOT) / "splits" / _WHICH_SPLITS / cancer
        inspect_feature_directory(DATA_ROOT, cancer)
        inspect_split_directory(cancer, which_splits=_WHICH_SPLITS)
        split_folds = sorted([
            int(p.name.split("_")[1].split(".")[0])
            for p in csv_dir.glob("fold_*.csv")
        ])

        for fold in folds:
            if fold not in split_folds:
                print(f"WARN: fold {fold} not in {csv_dir}, skipping")
                continue

            # Variant-aware config selection (only BLCA has v5.1/v5.2).
            # File naming: act_surv_v5{blca}.yaml = baseline,
            #               act_surv_v5_{1,2}_blca.yaml = v5.1/v5.2 (note underscore before digit)
            if variant == "v5":
                config = REPO_ROOT / "configs" / f"act_surv_v5_{cancer}.yaml"
            else:
                config = REPO_ROOT / "configs" / f"act_surv_v5_{variant.split('_', 1)[1]}_{cancer}.yaml"
            if not config.exists():
                # Fall back to baseline if a variant doesn't exist for this cancer
                print(f"WARN: {config.name} missing, falling back to act_surv_v5_{cancer}.yaml")
                config = REPO_ROOT / "configs" / f"act_surv_v5_{cancer}.yaml"
            result_dir = result_root / cancer / f"fold{fold}"
            result_dir.mkdir(parents=True, exist_ok=True)

            fold_overrides = {
                **overrides,
                "k_start": fold,
                "k_end": fold + 1,
                "study": cancer,
                # Include the variant tag in specific_simple so trainer's param_code
                # (= "<lr>_b<bs>_<label>_Dim_<dim>_e_<ep>_g_<rna>_sig_<sig>_seed<n>"
                #   "_rW_<w>_rG_<g>_sp_<specific_simple>") differs across variants.
                # Without this, v5 / v5_1 / v5_2 write to the SAME trainer experiment
                # directory (same param_code → same leaf folder), silently overwriting
                # each other.
                "specific_simple": (
                    f"act_surv_v5_{cancer}_fold{fold}"
                    if variant == "v5"
                    else f"act_surv_v5_{variant}_{cancer}_fold{fold}"
                ),
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


# Hyperparameters that come from the YAML config (variant-specific, e.g.
# v5.1 sets act5_lambda_rank=0.00; v5.2 sets act5_rank_temperature=1.0).
# These must NOT be force-overridden by FINAL_OVERRIDES — otherwise the
# variant configs become silent no-ops. Only the v5 baseline may supply them.
_V5_DEFAULT_HYPERS = {
    k: v for k, v in FINAL_OVERRIDES.items()
    if k.startswith("act5_") or k in {"alpha_surv"}
}


def overrides_for_variant(variant: str) -> dict[str, object]:
    """Build the FINAL_OVERRIDES dict for the requested variant.

    - `v5` baseline gets FINAL_OVERRIDES as-is (YAML matches defaults).
    - `v5_1` / `v5_2` get FINAL_OVERRIDES MINUS the v5-default hyperparameters,
      so the YAML's variant-specific values are honored.
    """
    base = {k: v for k, v in FINAL_OVERRIDES.items() if k not in _V5_DEFAULT_HYPERS}
    if variant == "v5":
        # Re-attach baseline hyperparam defaults (they may also be in YAML).
        return {**base, **_V5_DEFAULT_HYPERS, "results_dir": "<placeholder>"}
    return {**base, "results_dir": "<placeholder>"}


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
        "--which-splits",
        default=_DEFAULT_WHICH_SPLITS,
        choices=["5fold", "5fold_uni2h", "5fold_legacy"],
        help=f"Split directory under dataset_csv/splits/ (default: {_DEFAULT_WHICH_SPLITS}). "
             "Use '5fold_legacy' for the old community-standard split (pre-bee66a2, "
             "preserved at 5fold_legacy/ via 0e87fb4). Use '5fold' for the newer "
             "rebalanced split from bee66a2.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--result-root", default=None,
        help="Override result root directory (default: per-variant auto, "
             "e.g. results/act_surv_v5_1 for --variant v5_1)",
    )
    parser.add_argument(
        "--variant", default="v5",
        choices=["v5", "v5_1", "v5_2", "v5_3", "v5_4", "v5_k4"],
        help="Config variant: "
              "v5 (baseline, K=6, ranking+balance), "
              "v5_1 (main recipe: no ranking + KLx5), "
              "v5_2 (v5.1 + temperature/margin tweaks), "
              "v5_3 (1D ablation: no ranking only), "
              "v5_4 (1D ablation: KLx5 only), "
              "v5_k4 (K=4 ablation vs K=6 baseline)",
    )
    args = parser.parse_args()

    global DATA_ROOT, _WHICH_SPLITS
    DATA_ROOT = args.data_root
    _WHICH_SPLITS = args.which_splits

    cancers = [c.strip() for c in args.cancers.split(",")]
    folds = args.folds

    if args.result_root is None:
        base = _VARIANT_RESULT_ROOT[args.variant]
        # Tag non-default split so old-split runs don't clobber new-split results
        if _WHICH_SPLITS != _DEFAULT_WHICH_SPLITS:
            base = f"{base}_{_WHICH_SPLITS}"
        args.result_root = base

    overrides = overrides_for_variant(args.variant)
    overrides["results_dir"] = args.result_root
    overrides["which_splits"] = _WHICH_SPLITS
    jobs = build_jobs(cancers, folds, overrides, dry_run=args.dry_run, variant=args.variant)

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
