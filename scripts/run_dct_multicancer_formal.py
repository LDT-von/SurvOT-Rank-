#!/usr/bin/env python3
"""Launch the reproducible DCT v3.4 BRCA, LUAD, and LUSC protocols."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "brca": "configs/distributional_counterfactual_transport_brca_highscore.yaml",
    "luad": "configs/distributional_counterfactual_transport_luad_formal.yaml",
    "lusc": "configs/distributional_counterfactual_transport_lusc_formal.yaml",
}


def parse_cancers(value: str) -> list[str]:
    cancers = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(cancers) - set(CONFIGS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown cancer(s): {', '.join(unknown)}; choose from {', '.join(CONFIGS)}"
        )
    if not cancers:
        raise argparse.ArgumentTypeError("at least one cancer is required")
    return cancers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("doctor", "smoke", "run"), nargs="?", default="run")
    parser.add_argument(
        "--cancers",
        type=parse_cancers,
        default=parse_cancers(os.environ.get("CANCERS", "brca,luad,lusc")),
        help="comma-separated selection; default: brca,luad,lusc",
    )
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if args.mode == "doctor":
        return subprocess.run(
            [args.python_bin, "-m", "survot_rank.cli", "doctor"], check=False
        ).returncode

    for cancer in args.cancers:
        config = CONFIGS[cancer]
        command = [
            args.python_bin,
            "-m",
            "survot_rank.cli",
            "train",
            "--config",
            config,
            "--set",
            f"gpu={args.gpu}",
            "--set",
            f"num_workers={args.num_workers}",
        ]
        if args.mode == "smoke":
            command.extend(("--set", "max_epochs=1"))

        print("\n" + "=" * 68)
        print(f"[DCT v3.4] {cancer.upper()} | config={config} | GPU={args.gpu}")
        print("$ " + " ".join(command))
        print("=" * 68)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
