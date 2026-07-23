#!/usr/bin/env python3
"""Screen DCT v3.7 with UNI2-h WSI features without touching older results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Ten DCT study keys backed by eleven extracted TCGA archives because COADREAD
# combines the separately distributed COAD and READ cohorts.
CANCERS = (
    "blca",
    "brca",
    "coadread",
    "hnsc",
    "kirc",
    "luad",
    "lusc",
    "skcm",
    "stad",
    "ucec",
)
DEFAULT_DATA_ROOT = "/data1/TCGA-UNI2-h-features"
UNI2H_DIM = 1536

COMMON_OVERRIDES = {
    "data_root_dir": DEFAULT_DATA_ROOT,
    "wsi_encoder": "uni2-h",
    "encoding_dim": UNI2H_DIM,
    "num_patches": 2048,
    "batch_size": 8,
    "max_epochs": 50,
    "binning_mode": "global_qcut",
    "event_sampling_fraction": 0.0,
    "event_stratified_batches": False,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_memory_size": 0,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_lambda_ot": 0.0,
    "dct_lambda_rank": 0.0,
    "dct_lambda_anchor": 0.0,
    "dct_lambda_stage_risk": 0.0,
    "dct_lambda_coordinate": 0.0,
}

VARIANTS = {
    "highscore": {
        "label": "v3.3 high-score protocol with UNI2-h as the only intended change",
        "fit_bins_on_train": False,
        "dct_slot_init_mode": "gaussian",
    },
    "clean": {
        "label": "train-fold-only binning and deterministic slots",
        "fit_bins_on_train": True,
        "dct_slot_init_mode": "deterministic",
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


def _override_args(values: dict[str, object]) -> list[str]:
    result = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        result.extend(("--set", f"{key}={value}"))
    return result


def feature_dir(data_root: str | Path, cancer: str) -> Path:
    return Path(data_root) / cancer / "uni2-h" / "pt_files"


def inspect_feature_directory(data_root: str | Path, cancer: str) -> dict[str, object]:
    directory = feature_dir(data_root, cancer)
    files = []
    if directory.is_dir():
        for pattern in ("*.h5", "*.hdf5", "*.pt"):
            files.extend(directory.rglob(pattern))
    result = {
        "cancer": cancer,
        "directory": directory,
        "count": len(files),
        "sample": files[0] if files else None,
        "shape": None,
        "ok": False,
        "error": None,
    }
    if not files:
        result["error"] = "no supported feature files"
        return result
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
        result["shape"] = shape
        result["ok"] = len(shape) in (2, 3) and shape[-1] == UNI2H_DIM
        if not result["ok"]:
            result["error"] = f"expected final dimension {UNI2H_DIM}, got {shape}"
    except Exception as error:  # doctor should report every cancer, not abort early
        result["error"] = str(error)
    return result


def build_train_command(
    python_bin: str,
    cancer: str,
    variant: str,
    fold: int,
    gpu: str,
    num_workers: str,
    data_root: str,
    *,
    smoke: bool = False,
) -> tuple[list[str], Path]:
    config = Path("configs") / f"distributional_counterfactual_transport_{cancer}.yaml"
    result_root = "dct_v3.7_uni2h_smoke" if smoke else "dct_v3.7_uni2h"
    result_dir = Path("results") / result_root / variant / cancer
    overrides = dict(COMMON_OVERRIDES)
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
                f"dct_v3_7_uni2h_{variant}_{cancer}{'_smoke' if smoke else ''}"
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
        "mode", choices=("doctor", "plan", "smoke", "run"), nargs="?", default="plan"
    )
    parser.add_argument("--cancers", type=parse_cancers, default=parse_cancers("all"))
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants("highscore"),
        help="highscore is the default v3.3-compatible protocol; clean is an audit control",
    )
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,2"))
    parser.add_argument("--data-root", default=os.environ.get("UNI2H_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if args.mode == "doctor":
        failed = False
        for cancer in args.cancers:
            report = inspect_feature_directory(args.data_root, cancer)
            status = "OK" if report["ok"] else "MISSING"
            print(
                f"{status:8s} {cancer.upper():8s} files={report['count']:<5} "
                f"shape={report['shape']} path={report['directory']}"
            )
            if report["error"]:
                print(f"         {report['error']}")
            failed = failed or not report["ok"]
        return int(failed)

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
                    args.data_root,
                    smoke=args.mode == "smoke",
                )
                completed = list(result_dir.rglob(f"split_{fold}_results_final.pkl"))
                if completed and not args.force and args.mode == "run":
                    print(f"[skip] {variant} {cancer.upper()} fold{fold}: {completed[0]}")
                    continue
                print("\n" + "=" * 76)
                print(
                    f"[DCT v3.7-UNI2H/{variant}] {cancer.upper()} fold{fold} | "
                    f"{VARIANTS[variant]['label']}"
                )
                print("$ " + " ".join(command))
                print("=" * 76)
                if args.mode == "plan":
                    continue
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
