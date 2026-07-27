#!/usr/bin/env python3
"""Run DCT v4.1 only on BLCA/BRCA/STAD/HNSC folds 0, 2, and 4 with UNI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CANCERS = ("blca", "brca", "stad", "hnsc")
FOLDS = (0, 2, 4)
DEFAULT_DATA_ROOT = "/data/CPathPatchFeature"
UNI_DIM = 1024


def _selection(value: str, allowed: tuple, label: str) -> list:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    cast = int if isinstance(allowed[0], int) else str
    try:
        selected = [cast(item) for item in selected]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {label}: {value}") from error
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"DCT v4.1 permits only {label} {', '.join(map(str, allowed))}; "
            f"received {', '.join(map(str, unknown))}"
        )
    return selected


def parse_cancers(value: str) -> list[str]:
    return _selection(value, CANCERS, "cancers")


def parse_folds(value: str) -> list[int]:
    return _selection(value, FOLDS, "folds")


def inspect_uni_directory(data_root: str | Path, cancer: str) -> dict[str, object]:
    directory = Path(data_root) / cancer / "uni" / "pt_files"
    files = list(directory.rglob("*.pt")) if directory.is_dir() else []
    report: dict[str, object] = {
        "cancer": cancer,
        "directory": directory,
        "count": len(files),
        "shape": None,
        "ok": False,
        "error": None,
    }
    if not files:
        report["error"] = "no UNI .pt feature files"
        return report
    try:
        import torch

        tensor = torch.load(files[0], map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor["features"]
        shape = tuple(tensor.shape)
        report["shape"] = shape
        report["ok"] = len(shape) in (2, 3) and shape[-1] == UNI_DIM
        if not report["ok"]:
            report["error"] = f"expected final dimension {UNI_DIM}, got {shape}"
    except Exception as error:  # report all cancers in doctor mode
        report["error"] = str(error)
    return report


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
    fold: int,
    gpu: str,
    num_workers: str,
    data_root: str,
    *,
    smoke: bool = False,
) -> tuple[list[str], Path]:
    if cancer not in CANCERS:
        raise ValueError(f"unsupported DCT v4.1 cancer: {cancer}")
    if fold not in FOLDS:
        raise ValueError(f"unsupported DCT v4.1 fold: {fold}")

    config = Path("configs") / f"dct_v41_survival_evidence_ledger_{cancer}.yaml"
    result_root = (
        "dct_v4.1_survival_evidence_ledger_smoke"
        if smoke
        else "dct_v4.1_survival_evidence_ledger"
    )
    result_dir = Path("results") / result_root / cancer
    overrides: dict[str, object] = {
        "survot_method": "dct_v41_survival_evidence_ledger",
        "data_root_dir": data_root,
        "wsi_encoder": "uni",
        "encoding_dim": UNI_DIM,
        "k_start": fold,
        "k_end": fold + 1,
        "gpu": gpu,
        "num_workers": num_workers,
        "results_dir": result_dir.as_posix(),
        "specific_simple": f"dct_v41_selc_uni_{cancer}",
    }
    if smoke:
        overrides.update({"max_epochs": 2, "max_smoke_batches": 2})
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
    parser.add_argument("--cancers", type=parse_cancers, default=list(CANCERS))
    parser.add_argument("--folds", type=parse_folds, default=list(FOLDS))
    parser.add_argument(
        "--data-root", default=os.environ.get("UNI_ROOT", DEFAULT_DATA_ROOT)
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
            report = inspect_uni_directory(args.data_root, cancer)
            status = "OK" if report["ok"] else "MISSING"
            print(
                f"{status:8s} {cancer.upper():8s} files={report['count']:<5} "
                f"shape={report['shape']} path={report['directory']}"
            )
            if report["error"]:
                print(f"         {report['error']}")
            failed = failed or not report["ok"]
        return int(failed)

    for cancer in args.cancers:
        config = Path("configs") / f"dct_v41_survival_evidence_ledger_{cancer}.yaml"
        if not config.exists():
            print(f"[ERROR] missing config: {config}")
            return 2
        for fold in args.folds:
            command, result_dir = build_train_command(
                args.python_bin,
                cancer,
                fold,
                args.gpu,
                args.num_workers,
                args.data_root,
                smoke=args.mode == "smoke",
            )
            completed = list(
                result_dir.rglob(f"split_{fold}_results_final.pkl")
            )
            if completed and not args.force and args.mode == "run":
                print(f"[skip] {cancer.upper()} fold{fold}: {completed[0]}")
                continue
            print(f"[DCT v4.1/SELC/UNI] {cancer.upper()} fold{fold}")
            print(" ".join(command))
            if args.mode in ("smoke", "run"):
                completed_process = subprocess.run(command, check=False)
                if completed_process.returncode != 0:
                    return completed_process.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
