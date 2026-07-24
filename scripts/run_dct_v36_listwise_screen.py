#!/usr/bin/env python3
"""Run the controlled DCT v3.6 listwise screen without touching older results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CANCERS = ("blca", "brca", "luad", "lusc")
DEFAULT_CANCERS = ("blca", "brca")

COMMON_OVERRIDES = {
    "batch_size": 8,
    "max_epochs": 50,
    "grad_accum_steps": 1,
    "warmup_epochs": 5,
    "grad_clip_norm": 1.0,
    "early_stop_patience": 0,
    "fit_bins_on_train": True,
    "event_sampling_fraction": 0.0,
    "event_stratified_batches": True,
    "seed": 3,
    "lr": 0.0005,
    "opt": "adamW",
    "reg": 0.0005,
    "scheduler": "cosine",
    "eta_min": 0.000001,
    "bag_loss": "nll_surv",
    "alpha_surv": 0.15,
    "dct_lambda_ot": 0.0,
    "dct_lambda_rank": 0.0,
    "dct_lambda_anchor": 0.0,
    "dct_lambda_stage_risk": 0.0,
    "dct_lambda_coordinate": 0.0,
    "dct_slot_init_mode": "deterministic",
    "dct_slot_eval_seed": 1729,
    "dct_evidence_marginal_strength": 1.0,
    "dct_geometry_reliability_strength": 0.0,
    "dct_listwise_temperature": 0.50,
    "dct_listwise_memory_size": 64,
    "dct_listwise_tie_method": "breslow",
    "wsi_projection_dim": 256,
    "otehv2_layers": 2,
}

VARIANTS = {
    "nll": {
        "label": "NLL only",
        "survot_method": "distributional_counterfactual_transport",
        "dct_lambda_ipcw_rank": 0.0,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.0,
    },
    "ipcw": {
        "label": "NLL + IPCW pairwise",
        "survot_method": "distributional_counterfactual_transport",
        "dct_lambda_ipcw_rank": 0.10,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.0,
    },
    "etar": {
        "label": "NLL + ETAR",
        "survot_method": "distributional_counterfactual_transport",
        "dct_lambda_ipcw_rank": 0.0,
        "dct_lambda_etar": 0.10,
        "dct_lambda_listwise": 0.0,
    },
    "ipcw_etar": {
        "label": "NLL + IPCW + ETAR ablation",
        "survot_method": "distributional_counterfactual_transport",
        "dct_lambda_ipcw_rank": 0.10,
        "dct_lambda_etar": 0.10,
        "dct_lambda_listwise": 0.0,
    },
    "gpl": {
        "label": "DCT v3.6 global Plackett-Luce control",
        "survot_method": "dct_listwise_transport",
        "dct_lambda_ipcw_rank": 0.0,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.10,
        "dct_listwise_mode": "global",
    },
    "tcl": {
        "label": "DCT v3.6 transport-conditioned listwise",
        "survot_method": "dct_listwise_transport",
        "dct_lambda_ipcw_rank": 0.0,
        "dct_lambda_etar": 0.0,
        "dct_lambda_listwise": 0.10,
        "dct_listwise_mode": "stage_transport",
    },
}


def _selection(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {name}: {', '.join(unknown)}; choose from {', '.join(allowed)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    return selected


def parse_cancers(value: str) -> list[str]:
    return _selection(value, CANCERS, "cancer")


def parse_variants(value: str) -> list[str]:
    return _selection(value, tuple(VARIANTS), "variant")


def parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from error
    if not folds or any(fold < 0 or fold > 4 for fold in folds):
        raise argparse.ArgumentTypeError("folds must be selected from 0,1,2,3,4")
    return folds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("doctor", "plan", "smoke", "run"), nargs="?", default="run"
    )
    parser.add_argument(
        "--variants", type=parse_variants, default=parse_variants("all")
    )
    parser.add_argument(
        "--cancers",
        type=parse_cancers,
        default=list(DEFAULT_CANCERS),
    )
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,2"))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument(
        "--num-workers", default=os.environ.get("NUM_WORKERS", "4")
    )
    parser.add_argument(
        "--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable)
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun a fold even when its final result file already exists",
    )
    return parser


def _override_args(values: dict[str, object]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        result.extend(("--set", f"{key}={value}"))
    return result


def build_train_command(
    python_bin: str,
    cancer: str,
    variant: str,
    fold: int,
    gpu: str,
    num_workers: str,
    *,
    smoke: bool = False,
) -> tuple[list[str], Path]:
    config = Path("configs") / f"distributional_counterfactual_transport_{cancer}.yaml"
    result_root = "dct_v3.6_listwise_smoke" if smoke else "dct_v3.6_listwise"
    result_dir = Path("results") / result_root / variant / cancer
    overrides = dict(COMMON_OVERRIDES)
    overrides.update(VARIANTS[variant])
    label = overrides.pop("label")
    overrides.update(
        {
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": gpu,
            "num_workers": num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": f"dct36_{variant}_{cancer}",
        }
    )
    if smoke:
        overrides.update({"max_epochs": 1, "max_smoke_batches": 1})
    command = [
        python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config.as_posix(),
        *_override_args(overrides),
    ]
    return command, result_dir


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if args.mode == "doctor":
        return subprocess.run(
            [args.python_bin, "-m", "survot_rank.cli", "doctor"], check=False
        ).returncode

    for variant in args.variants:
        for cancer in args.cancers:
            config = Path("configs") / f"distributional_counterfactual_transport_{cancer}.yaml"
            if not config.exists():
                print(f"[ERROR] missing config: {config}")
                return 2
            for fold in args.folds:
                command, result_dir = build_train_command(
                    args.python_bin,
                    cancer,
                    variant,
                    fold,
                    args.gpu,
                    args.num_workers,
                    smoke=args.mode == "smoke",
                )
                completed = list(result_dir.rglob(f"split_{fold}_results_final.pkl"))
                if completed and not args.force and args.mode == "run":
                    print(
                        f"[skip] {variant.upper()} {cancer.upper()} fold{fold}: "
                        f"{completed[0]}"
                    )
                    continue

                print("\n" + "=" * 76)
                print(
                    f"[DCT v3.6 {variant.upper()}] {cancer.upper()} fold{fold} | "
                    f"{VARIANTS[variant]['label']}"
                )
                print("$ " + " ".join(command))
                print("=" * 76)
                if args.mode == "plan":
                    continue
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0:
                    return completed.returncode
                remaining = list(result_dir.rglob(f"split_{fold}_results_final.pkl"))
                if args.mode == "run" and not remaining:
                    print(
                        f"[ERROR] training returned without final result in: {result_dir}"
                    )
                    return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
