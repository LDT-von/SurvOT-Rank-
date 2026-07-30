#!/usr/bin/env python3
"""Run IST-Surv v4.0 on BLCA/BRCA five-fold without touching old results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = "/data1/TCGA-UNI2-h-features"
UNI2H_DIM = 1536
CANCERS = ("blca", "brca")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)

COMMON_OVERRIDES = {
    "survot_method": "intervention_stable_survival_transport",
    "data_root_dir": DEFAULT_DATA_ROOT,
    "wsi_encoder": "uni2-h",
    "encoding_dim": UNI2H_DIM,
    "num_patches": 2048,
    "batch_size": 8,
    "max_epochs": 20,
    "event_sampling_fraction": 0.0,
    "event_stratified_batches": False,
    "ist_eps": 0.05,
    "ist_sinkhorn_iters": 30,
    "ist_num_interventions": 2,
    "ist_keep_ratio": 0.75,
    "ist_stability_beta": 1.0,
    "ist_eval_seed": 20260725,
    "ist_deletion_penalty": 8.0,
}

PROTOCOLS = {
    "highscore": {
        "label": "legacy global-qcut score-comparison protocol",
        "fit_bins_on_train": False,
        "binning_mode": "global_qcut",
    },
    "clean": {
        "label": "train-fold-only label-bin protocol",
        "fit_bins_on_train": True,
        "binning_mode": "global_qcut",
    },
}

VARIANTS = {
    "factual": {
        "label": "additive factual OT without intervention stability",
        "ist_stability_strength": 0.0,
        "ist_lambda_plan": 0.0,
        "ist_lambda_attribution": 0.0,
        "ist_lambda_risk": 0.0,
    },
    "stable_plan": {
        "label": "stable cost and plan consistency only",
        "ist_stability_strength": 0.10,
        "ist_lambda_plan": 0.05,
        "ist_lambda_attribution": 0.0,
        "ist_lambda_risk": 0.0,
    },
    "full": {
        "label": "stable cost plus plan and exact attribution stability",
        "ist_stability_strength": 0.10,
        "ist_lambda_plan": 0.05,
        "ist_lambda_attribution": 0.05,
        "ist_lambda_risk": 0.0,
    },
}


def _selection(value: str, allowed, name: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {name}: {', '.join(unknown)}; "
            f"choose from {', '.join(allowed)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    return selected


def parse_cancers(value: str) -> list[str]:
    return _selection(value, CANCERS, "cancer")


def parse_protocols(value: str) -> list[str]:
    return _selection(value, PROTOCOLS, "protocol")


def parse_variants(value: str) -> list[str]:
    return _selection(value, VARIANTS, "variant")


def parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "folds must be comma-separated integers"
        ) from error
    if not folds or any(fold < 0 or fold > 4 for fold in folds):
        raise argparse.ArgumentTypeError("folds must use only 0,1,2,3,4")
    return folds


def _override_args(values: dict[str, object]) -> list[str]:
    rendered = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        rendered.extend(("--set", f"{key}={value}"))
    return rendered


def inspect_feature_directory(
    data_root: str | Path,
    cancer: str,
) -> dict[str, object]:
    directory = Path(data_root) / cancer / "uni2-h" / "pt_files"
    files = []
    if directory.is_dir():
        for pattern in ("*.h5", "*.hdf5", "*.pt"):
            files.extend(directory.rglob(pattern))
    report = {
        "cancer": cancer,
        "directory": directory,
        "count": len(files),
        "shape": None,
        "ok": False,
        "error": None,
    }
    if not files:
        report["error"] = "no supported UNI2-h feature files"
        return report
    try:
        sample = files[0]
        if sample.suffix == ".pt":
            import torch

            tensor = torch.load(sample, map_location="cpu")
            if isinstance(tensor, dict):
                tensor = tensor["features"]
            shape = tuple(tensor.shape)
        else:
            import h5py

            with h5py.File(sample, "r") as handle:
                shape = tuple(handle["features"].shape)
        report["shape"] = shape
        report["ok"] = len(shape) in (2, 3) and shape[-1] == UNI2H_DIM
        if not report["ok"]:
            report["error"] = (
                f"expected final dimension {UNI2H_DIM}, got {shape}"
            )
    except Exception as error:  # doctor reports all selected cancers
        report["error"] = str(error)
    return report


def build_train_command(
    python_bin: str,
    cancer: str,
    protocol: str,
    variant: str,
    fold: int,
    gpu: str,
    num_workers: str,
    data_root: str,
    *,
    smoke: bool = False,
) -> tuple[list[str], Path]:
    config = (
        Path("configs")
        / f"intervention_stable_survival_transport_{cancer}.yaml"
    )
    root = "ist_surv_v4.0_smoke" if smoke else "ist_surv_v4.0"
    result_dir = (
        Path("results") / root / protocol / variant / cancer
    )
    overrides = dict(COMMON_OVERRIDES)
    overrides.update(PROTOCOLS[protocol])
    overrides.update(VARIANTS[variant])
    overrides.pop("label", None)
    overrides.update(
        {
            "data_root_dir": data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": gpu,
            "num_workers": num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": (
                f"ist_v40_{protocol}_{variant}_{cancer}"
            ),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("doctor", "plan", "smoke", "run"),
        nargs="?",
        default="plan",
    )
    parser.add_argument(
        "--cancers", type=parse_cancers, default=list(CANCERS)
    )
    parser.add_argument(
        "--folds",
        type=parse_folds,
        default=list(DEFAULT_FOLDS),
        help="defaults to all five folds: 0,1,2,3,4",
    )
    parser.add_argument(
        "--protocols",
        type=parse_protocols,
        default=["highscore", "clean"],
        help="both protocols are kept separate; never pool their metrics",
    )
    parser.add_argument(
        "--variants", type=parse_variants, default=["full"]
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("UNI2H_ROOT", DEFAULT_DATA_ROOT),
    )
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument(
        "--num-workers", default=os.environ.get("NUM_WORKERS", "4")
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)

    if args.mode == "doctor":
        failed = False
        for cancer in args.cancers:
            report = inspect_feature_directory(args.data_root, cancer)
            status = "OK" if report["ok"] else "MISSING"
            print(
                f"{status:8s} {cancer.upper():8s} "
                f"files={report['count']:<5} shape={report['shape']} "
                f"path={report['directory']}"
            )
            if report["error"]:
                print(f"         {report['error']}")
            failed = failed or not report["ok"]
        return int(failed)

    for protocol in args.protocols:
        for variant in args.variants:
            for cancer in args.cancers:
                config = (
                    Path("configs")
                    / f"intervention_stable_survival_transport_{cancer}.yaml"
                )
                if not config.exists():
                    print(f"[ERROR] missing config: {config}")
                    return 2
                for fold in args.folds:
                    command, result_dir = build_train_command(
                        args.python_bin,
                        cancer,
                        protocol,
                        variant,
                        fold,
                        args.gpu,
                        args.num_workers,
                        args.data_root,
                        smoke=args.mode == "smoke",
                    )
                    completed = list(
                        result_dir.rglob(
                            f"split_{fold}_results_final.pkl"
                        )
                    )
                    if completed and not args.force and args.mode == "run":
                        print(
                            f"[skip] {protocol}/{variant}/"
                            f"{cancer.upper()} fold{fold}: {completed[0]}"
                        )
                        continue
                    print("\n" + "=" * 76)
                    print(
                        f"[IST-Surv v4.0/{protocol}/{variant}] "
                        f"{cancer.upper()} fold{fold}"
                    )
                    print(" ".join(command))
                    if args.mode in ("smoke", "run"):
                        completed_process = subprocess.run(
                            command, check=False
                        )
                        if completed_process.returncode != 0:
                            return completed_process.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
