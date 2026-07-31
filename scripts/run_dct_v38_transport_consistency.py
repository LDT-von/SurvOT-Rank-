#!/usr/bin/env python3
"""Screen DCT v3.8 transport-intervention losses without touching old results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.task_lock import (
        ActiveRunError,
        acquire_run_lock,
        release_run_lock,
    )
except ModuleNotFoundError:
    from task_lock import ActiveRunError, acquire_run_lock, release_run_lock

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV_ROOT = (
    REPO_ROOT
    / "survot_rank"
    / "research"
    / "legacy"
    / "slotspe_runtime"
    / "dataset_csv"
)
DEFAULT_DATA_ROOT = "/data1/TCGA-UNI2-h-features"
UNI2H_DIM = 1536
CANCERS = ("blca", "brca", "luad", "lusc", "stad")
DEFAULT_CANCERS = ("blca", "brca")

COMMON_OVERRIDES = {
    "survot_method": "dct_transport_intervention_consistency",
    "data_root_dir": DEFAULT_DATA_ROOT,
    "wsi_encoder": "uni2-h",
    "encoding_dim": UNI2H_DIM,
    "num_patches": 2048,
    "batch_size": 8,
    "max_epochs": 50,
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
    "dct_v38_direction_margin": 0.02,
    "dct_v38_dose_margin": 0.005,
    "dct_v38_reconfiguration_margin": 0.02,
    "dct_v38_temperature": 0.05,
    "dct_v38_alpha_mid": 0.50,
    "dct_v38_alpha_full": 1.00,
    "dct_v38_warmup_epochs": 1,
    "dct_v38_dose_every": 2,
    "dct_mix_ratio": 1.0,
}

PROTOCOLS = {
    "highscore": {
        "label": "v3.3 high-score global-binning protocol with UNI2-h",
        "fit_bins_on_train": False,
        "binning_mode": "global_qcut",
        "dct_slot_init_mode": "gaussian",
    },
    "stable": {
        "label": "global-binning protocol with deterministic evaluation slots",
        "fit_bins_on_train": False,
        "binning_mode": "global_qcut",
        "dct_slot_init_mode": "deterministic",
    },
    "clean": {
        "label": "train-fold binning and deterministic-slot audit protocol",
        "fit_bins_on_train": True,
        "binning_mode": "global_qcut",
        "dct_slot_init_mode": "deterministic",
    },
    "robust": {
        "label": (
            "cancer-agnostic sparse-event protocol with patient-complete "
            "event-spread batches and within-epoch ranking memory"
        ),
        "fit_bins_on_train": True,
        "binning_mode": "global_qcut",
        "dct_slot_init_mode": "deterministic",
        "event_sampling_fraction": 0.0,
        "event_stratified_batches": True,
        "dct_ipcw_rank_memory_size": 64,
        "dct_v38_warmup_epochs": 5,
        "dct_v38_ramp_epochs": 10,
    },
}

VARIANTS = {
    "base": {
        "label": (
            "v3.7-matched UNI2-h control through the v3.8 class "
            "(v3.3 NLL + IPCW objective)"
        ),
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
    },
    "direction": {
        "label": "risk-direction consistency only",
        "dct_v38_lambda_direction": 0.05,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.0,
    },
    "dose": {
        "label": "dose-monotonic transport response only",
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.03,
        "dct_v38_lambda_reconfiguration": 0.0,
    },
    "reconfiguration": {
        "label": "minimum Sinkhorn coupling reconfiguration only",
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.02,
    },
    "direction_dose": {
        "label": "risk-direction consistency + dose-monotonic response",
        "dct_v38_lambda_direction": 0.05,
        "dct_v38_lambda_dose": 0.03,
        "dct_v38_lambda_reconfiguration": 0.0,
    },
    "direction_reconfiguration": {
        "label": "risk-direction consistency + coupling reconfiguration",
        "dct_v38_lambda_direction": 0.05,
        "dct_v38_lambda_dose": 0.0,
        "dct_v38_lambda_reconfiguration": 0.02,
    },
    "dose_reconfiguration": {
        "label": "dose-monotonic response + coupling reconfiguration",
        "dct_v38_lambda_direction": 0.0,
        "dct_v38_lambda_dose": 0.03,
        "dct_v38_lambda_reconfiguration": 0.02,
    },
    "full": {
        "label": "direction + dose + coupling reconfiguration",
        "dct_v38_lambda_direction": 0.05,
        "dct_v38_lambda_dose": 0.03,
        "dct_v38_lambda_reconfiguration": 0.02,
    },
}


def inspect_feature_directory(data_root: str | Path, cancer: str) -> dict[str, object]:
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
        report["error"] = "no supported feature files"
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
    except Exception as error:  # doctor should report every cancer
        report["error"] = str(error)
    return report


def inspect_split_directory(cancer: str) -> dict[str, object]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.gen_splits_5fold import audit_existing_splits

    return audit_existing_splits(
        study=cancer,
        data_path=str(DATASET_CSV_ROOT),
        label_col="survival_months_dss",
        censor_col="censorship_dss",
        n_folds=5,
    )


def _selection(value: str, allowed, name: str) -> list[str]:
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
    return _selection(value, VARIANTS, "variant")


def parse_protocols(value: str) -> list[str]:
    return _selection(value, PROTOCOLS, "protocol")


def parse_positive_int(value: str) -> int:
    ival = int(value)
    if ival < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return ival


def parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "folds must be comma-separated integers"
        ) from error
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
    max_epochs: int | None = None,
    smoke: bool = False,
) -> tuple[list[str], Path]:
    config = Path("configs") / f"distributional_counterfactual_transport_{cancer}.yaml"
    effective_epochs = int(
        max_epochs if max_epochs is not None else COMMON_OVERRIDES["max_epochs"]
    )
    if smoke:
        result_root = "dct_v3.8_transport_consistency_smoke"
    elif effective_epochs == int(COMMON_OVERRIDES["max_epochs"]):
        result_root = "dct_v3.8_transport_consistency"
    else:
        result_root = f"dct_v3.8_transport_consistency_{effective_epochs}ep"
    result_dir = Path("results") / result_root / protocol / variant / cancer
    overrides = dict(COMMON_OVERRIDES)
    overrides.update(PROTOCOLS[protocol])
    overrides.update(VARIANTS[variant])
    overrides.pop("label", None)
    overrides["max_epochs"] = effective_epochs
    overrides.update(
        {
            "data_root_dir": data_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": gpu,
            "num_workers": num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": (
                f"dct_v38_{protocol}_{variant}_{cancer}_{effective_epochs}ep"
            ),
        }
    )
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


def scheduler_lock_path(smoke: bool, gpu: str) -> Path:
    result_root = (
        "dct_v3.8_transport_consistency_smoke"
        if smoke
        else "dct_v3.8_transport_consistency"
    )
    safe_gpu = "".join(
        character if character.isalnum() else "_" for character in gpu
    )
    return Path("results") / result_root / f".scheduler_gpu_{safe_gpu}.lock"


def task_lock_path(result_dir: Path, fold: int) -> Path:
    return result_dir / f".split_{fold}.run.lock"


def verify_child_cuda(python_bin: str, env: dict[str, str]) -> bool:
    """Prove that the exact training interpreter can initialize the selected GPU.

    ``CUDA_VISIBLE_DEVICES`` is process-local.  Checking CUDA in the scheduler
    process is therefore not enough: the check must run in a fresh child with
    the same interpreter and environment that will launch the fold training.
    """
    probe = """
import json
import os
import sys
import torch

report = {
    "python": sys.executable,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if report["cuda_available"]:
    device = torch.device("cuda:0")
    allocation = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    report["device_name"] = torch.cuda.get_device_name(device)
    report["allocated_bytes"] = allocation.element_size() * allocation.nelement()
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if report["cuda_available"] else 1)
"""
    completed = subprocess.run(
        [python_bin, "-c", probe],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    if output:
        print(f"[cuda-preflight] {output}")
    if completed.stderr.strip():
        print(f"[cuda-preflight stderr] {completed.stderr.strip()}", file=sys.stderr)
    if completed.returncode != 0:
        print(
            "[ERROR] training child cannot initialize CUDA; refusing CPU fallback.",
            file=sys.stderr,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("doctor", "plan", "smoke", "run"), nargs="?", default="plan"
    )
    parser.add_argument("--cancers", type=parse_cancers, default=list(DEFAULT_CANCERS))
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,2"))
    parser.add_argument("--protocols", type=parse_protocols, default=["robust"])
    parser.add_argument("--variants", type=parse_variants, default=["full"])
    parser.add_argument("--data-root", default=os.environ.get("UNI2H_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument(
        "--max-epochs",
        type=parse_positive_int,
        default=None,
        help=(
            "Override the 50-epoch formal horizon. Non-default horizons use "
            "a separate result directory, for example *_20ep."
        ),
    )
    parser.add_argument(
        "--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable)
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
                f"{status:8s} {cancer.upper():8s} files={report['count']:<5} "
                f"shape={report['shape']} path={report['directory']}"
            )
            if report["error"]:
                print(f"         {report['error']}")
            failed = failed or not report["ok"]
            split_report = inspect_split_directory(cancer)
            split_status = "OK" if split_report["ok"] else "INVALID"
            print(
                f"{split_status:8s} {cancer.upper():8s} "
                f"eligible={split_report['eligible_cases']:<5} "
                f"events={split_report['observed_events']:<4} "
                f"val_events={split_report['validation_event_counts']}"
            )
            for error in split_report["errors"]:
                print(f"         {error}")
            failed = failed or not split_report["ok"]
        return int(failed)

    if args.mode in ("smoke", "run") and "robust" in args.protocols:
        for cancer in args.cancers:
            split_report = inspect_split_directory(cancer)
            if not split_report["ok"]:
                print(
                    f"[ERROR] {cancer.upper()} split audit failed; "
                    "regenerate it with tools/gen_splits_5fold.py before "
                    "running the robust protocol."
                )
                for error in split_report["errors"]:
                    print(f"        {error}")
                return 2

    scheduler_lock = None
    if args.mode in ("smoke", "run"):
        lock_path = scheduler_lock_path(args.mode == "smoke", args.gpu)
        try:
            scheduler_lock = acquire_run_lock(
                lock_path,
                label=f"DCT v3.8 scheduler on GPU {args.gpu}",
            )
        except ActiveRunError as error:
            print(f"[already-running] {error}")
            return 3

    try:
        for protocol in args.protocols:
            for variant in args.variants:
                for cancer in args.cancers:
                    config = (
                        Path("configs")
                        / f"distributional_counterfactual_transport_{cancer}.yaml"
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
                            max_epochs=args.max_epochs,
                            smoke=args.mode == "smoke",
                        )
                        completed = list(
                            result_dir.rglob(f"split_{fold}_results_final.pkl")
                        )
                        if completed and not args.force and args.mode == "run":
                            print(
                                f"[skip] {protocol}/{variant} "
                                f"{cancer.upper()} fold{fold}: {completed[0]}"
                            )
                            continue
                        print("\n" + "=" * 76)
                        print(
                            f"[DCT v3.8/{protocol}/{variant}] "
                            f"{cancer.upper()} fold{fold}"
                        )
                        print(" ".join(command))
                        if args.mode not in ("smoke", "run"):
                            continue

                        task_lock = None
                        try:
                            task_lock = acquire_run_lock(
                                task_lock_path(result_dir, fold),
                                label=(
                                    f"DCT v3.8 {protocol}/{variant} "
                                    f"{cancer.upper()} fold{fold}"
                                ),
                            )
                        except ActiveRunError as error:
                            print(f"[skip-running] {error}")
                            continue

                        try:
                            # Recheck after locking to close the race between
                            # the initial completion scan and task acquisition.
                            completed = list(
                                result_dir.rglob(
                                    f"split_{fold}_results_final.pkl"
                                )
                            )
                            if (
                                completed
                                and not args.force
                                and args.mode == "run"
                            ):
                                print(
                                    f"[skip] {protocol}/{variant} "
                                    f"{cancer.upper()} fold{fold}: "
                                    f"{completed[0]}"
                                )
                                continue

                            env = os.environ.copy()
                            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                            env.setdefault("PYTHONUNBUFFERED", "1")
                            if not verify_child_cuda(args.python_bin, env):
                                return 1
                            completed_process = subprocess.run(
                                command,
                                check=False,
                                env=env,
                            )
                            if completed_process.returncode != 0:
                                return completed_process.returncode
                        finally:
                            release_run_lock(task_lock)
        return 0
    finally:
        release_run_lock(scheduler_lock)


if __name__ == "__main__":
    raise SystemExit(main())
