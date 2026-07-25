#!/usr/bin/env python3
"""Coordinate recent v3.6-v4.0 BLCA/BRCA five-fold experiments server-side."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ("v36", "v37", "v38", "v40")
PROTOCOLS = ("highscore", "clean")
FOLDS = "0,1,2,3,4"
CANCERS = "blca,brca"


def _selection(value: str, allowed, name: str) -> list[str]:
    value = value.strip().lower()
    if value == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {name}: {', '.join(unknown)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    return selected


def parse_versions(value: str) -> list[str]:
    return _selection(value, VERSIONS, "version")


def parse_protocols(value: str) -> list[str]:
    return _selection(value, PROTOCOLS, "protocol")


def build_commands(args) -> list[tuple[str, list[str]]]:
    common = [
        "--cancers",
        CANCERS,
        "--folds",
        FOLDS,
        "--gpu",
        str(args.gpu),
        "--num-workers",
        str(args.num_workers),
        "--python",
        str(args.python_bin),
    ]
    commands = []
    if "v36" in args.versions:
        commands.append(
            (
                "v3.6-TCL",
                [
                    args.python_bin,
                    "scripts/run_dct_v36_listwise_screen.py",
                    args.mode,
                    "--variants",
                    "tcl",
                    *common,
                ],
            )
        )
    protocol_value = ",".join(args.protocols)
    if "v37" in args.versions:
        commands.append(
            (
                "v3.7-UNI2H",
                [
                    args.python_bin,
                    "scripts/run_dct_v37_uni2h_screen.py",
                    args.mode,
                    "--variants",
                    protocol_value,
                    "--data-root",
                    args.data_root,
                    *common,
                ],
            )
        )
    if "v38" in args.versions:
        commands.append(
            (
                "v3.8-transport-consistency",
                [
                    args.python_bin,
                    "scripts/run_dct_v38_transport_consistency.py",
                    args.mode,
                    "--protocols",
                    protocol_value,
                    "--variants",
                    "full",
                    "--data-root",
                    args.data_root,
                    *common,
                ],
            )
        )
    if "v40" in args.versions:
        commands.append(
            (
                "v4.0-IST-Surv",
                [
                    args.python_bin,
                    "scripts/run_v40_intervention_stable_transport.py",
                    args.mode,
                    "--protocols",
                    protocol_value,
                    "--variants",
                    "full",
                    "--data-root",
                    args.data_root,
                    *common,
                ],
            )
        )
    if args.force:
        for _, command in commands:
            command.append("--force")
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("doctor", "plan", "smoke", "run"),
        nargs="?",
        default="plan",
    )
    parser.add_argument(
        "--versions", type=parse_versions, default=list(VERSIONS)
    )
    parser.add_argument(
        "--protocols",
        type=parse_protocols,
        default=list(PROTOCOLS),
        help="highscore and clean outputs stay strictly separated",
    )
    parser.add_argument(
        "--data-root", default="/data1/TCGA-UNI2-h-features"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-workers", default="4")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for label, command in build_commands(args):
        print("\n" + "#" * 80)
        print(f"# {label}: BLCA + BRCA, folds 0-4")
        print(" ".join(str(item) for item in command))
        if args.mode != "plan":
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
            )
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
