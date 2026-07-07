"""Unified command line interface for SurvOT-Rank."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from .config import apply_overrides, config_to_argv, load_config
from .project import PROJECT_ROOT, add_project_paths, resolve_project_path


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_train(args: argparse.Namespace) -> None:
    add_project_paths()
    config = load_config(args.config)
    config = apply_overrides(config, args.set or [])
    extra_args = args.extra_args or []
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    argv = config_to_argv(config) + extra_args

    from survot_rank.training.extended_args import process_args_extended
    from survot_rank.training.train_runner import run

    parsed = process_args_extended(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = parsed.gpu
    run(parsed)


def cmd_ensemble(args: argparse.Namespace) -> None:
    add_project_paths()
    module = _load_module_from_path(
        "survot_rank_v45_ensemble_eval",
        PROJECT_ROOT / "survot_rank" / "research" / "methods" / "prognostic_event_transport" / "ensemble_eval.py",
    )
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "ensemble_eval.py",
            "--dirs",
            *[resolve_project_path(path) for path in args.dirs],
            "--n_classes",
            str(args.n_classes),
        ]
        module.main()
    finally:
        sys.argv = old_argv


def cmd_doctor(args: argparse.Namespace) -> None:
    add_project_paths()
    checks = {
        "project_root": PROJECT_ROOT.exists(),
        "training": (PROJECT_ROOT / "survot_rank" / "training" / "train_runner.py").exists(),
        "method": (
            PROJECT_ROOT / "survot_rank" / "research" / "methods" / "prognostic_event_transport" / "model.py"
        ).exists(),
        "parent_model": (
            PROJECT_ROOT / "survot_rank" / "research" / "methods" / "ot_event_hazard_v2" / "model_v2.py"
        ).exists(),
        "legacy_dataset": (
            PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset" / "dataset_survival.py"
        ).exists(),
        "legacy_utils": (
            PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "utils" / "loss_func.py"
        ).exists(),
        "dataset_csv": (
            PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset_csv"
        ).exists(),
    }
    for name, ok in checks.items():
        status = "OK" if ok else "MISSING"
        print(f"{status:8s} {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="survot-rank")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run training from a YAML config")
    train.add_argument("--config", required=True, help="Path to a YAML experiment config")
    train.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override one flat parameter, for example --set seed=5",
    )
    train.add_argument("extra_args", nargs=argparse.REMAINDER)
    train.set_defaults(func=cmd_train)

    ensemble = subparsers.add_parser("ensemble", help="Evaluate multi-seed ensemble results")
    ensemble.add_argument("--dirs", nargs="+", required=True, help="Seed result directories")
    ensemble.add_argument("--n-classes", type=int, default=4)
    ensemble.set_defaults(func=cmd_ensemble)

    doctor = subparsers.add_parser("doctor", help="Check expected project files")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
