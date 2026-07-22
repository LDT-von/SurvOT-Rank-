#!/usr/bin/env python3
"""Run the opt-in DCT ETAR loss on fold 0 and fold 2 (2026-07-23)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "blca": "configs/distributional_counterfactual_transport_blca.yaml",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cancers",
        type=parse_cancers,
        default=parse_cancers(os.environ.get("CANCERS", "blca,brca,luad,lusc")),
        help="comma-separated cancers; default: blca,brca,luad,lusc",
    )
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--smoke", action="store_true", help="run one epoch per fold")
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    for cancer in args.cancers:
        for fold_start, fold_end in ((0, 1), (2, 3)):
            command = [
                args.python,
                "-m",
                "survot_rank.cli",
                "train",
                "--config",
                CONFIGS[cancer],
                "--set",
                f"gpu={args.gpu}",
                "--set",
                f"num_workers={args.num_workers}",
                "--set",
                f"results_dir=results/dct_etar_20260723_{cancer}",
                # Isolate ETAR from the v3.3 IPCW ranking baseline.
                "--set",
                "dct_lambda_ipcw_rank=0.0",
                "--set",
                "dct_lambda_etar=0.10",
                "--set",
                "dct_etar_margin=0.02",
                "--set",
                "dct_etar_uncertainty_weight=0.05",
                "--set",
                "dct_etar_temperature=0.50",
                "--set",
                "dct_etar_evidence_floor=0.10",
            ]
            if args.smoke:
                command.extend(("--set", "max_epochs=1"))
            command.extend(("--", "--k_start", str(fold_start), "--k_end", str(fold_end)))
            print("\n" + "=" * 72)
            print(f"[DCT-ETAR 2026-07-23] {cancer.upper()} fold {fold_start}")
            print("$ " + " ".join(command))
            print("=" * 72)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
