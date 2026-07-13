#!/usr/bin/env python3
"""Generate and optionally run focused Rank-Guided Event Transport ablations."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


VARIANTS = {
    "nll_only": {
        "description": "Survival NLL only; no OT, prognostic cost, rank, or stage loss.",
        "rg_prog_cost": 0.0,
        "rg_lambda_ot": 0.0,
        "rg_lambda_rank": 0.0,
        "rg_lambda_stage": 0.0,
    },
    "nll_ot": {
        "description": "Survival NLL plus three-cost OT regularization.",
        "rg_prog_cost": 0.0,
        "rg_lambda_ot": 0.06,
        "rg_lambda_rank": 0.0,
        "rg_lambda_stage": 0.0,
    },
    "nll_ot_rank005": {
        "description": "NLL plus OT and conservative continuous ranking.",
        "rg_prog_cost": 0.0,
        "rg_lambda_ot": 0.06,
        "rg_lambda_rank": 0.05,
        "rg_lambda_stage": 0.0,
    },
    "nll_ot_rank015": {
        "description": "NLL plus OT and the original stronger ranking weight.",
        "rg_prog_cost": 0.0,
        "rg_lambda_ot": 0.06,
        "rg_lambda_rank": 0.15,
        "rg_lambda_stage": 0.0,
    },
    "nll_ot_stage": {
        "description": "NLL plus OT and event-stage ordering.",
        "rg_prog_cost": 0.0,
        "rg_lambda_ot": 0.06,
        "rg_lambda_rank": 0.0,
        "rg_lambda_stage": 0.02,
    },
    "full": {
        "description": "Full RG-ET: prognostic OT cost, OT, ranking, and stage ordering.",
        "rg_prog_cost": 0.20,
        "rg_lambda_ot": 0.06,
        "rg_lambda_rank": 0.15,
        "rg_lambda_stage": 0.02,
    },
}


def build_config(template: dict, name: str, settings: dict, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(template)
    config["name"] = f"rg_et_ablation_{name}"
    config["description"] = settings["description"]

    split = config.setdefault("split", {})
    split["k_start"] = args.k_start
    split["k_end"] = args.k_end

    train = config.setdefault("train", {})
    train["specific_simple"] = f"rg_et_ablation_{name}"
    train["results_dir"] = f"results/rg_et_ablation/{name}"
    train["max_epochs"] = args.max_epochs
    train["batch_size"] = args.batch_size
    train["gpu"] = str(args.gpu)

    if args.num_patches is not None:
        config.setdefault("data", {})["num_patches"] = args.num_patches
    if args.slot_iters is not None:
        config.setdefault("slot", {})["slot_iters"] = args.slot_iters

    model = config.setdefault("model", {})
    for key in ("rg_prog_cost", "rg_lambda_ot", "rg_lambda_rank", "rg_lambda_stage"):
        model[key] = settings[key]
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="configs/rank_guided_event_transport_blca.yaml")
    parser.add_argument("--out-dir", default="configs/ablation/rank_guided_event_transport")
    parser.add_argument("--k-start", type=int, default=2)
    parser.add_argument("--k-end", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-patches", type=int, default=2048)
    parser.add_argument("--slot-iters", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run", action="store_true", help="Run each generated configuration sequentially.")
    args = parser.parse_args()

    template_path = Path(args.template)
    out_dir = Path(args.out_dir)
    with template_path.open("r", encoding="utf-8") as handle:
        template = yaml.safe_load(handle) or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, settings in VARIANTS.items():
        path = out_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(build_config(template, name, settings, args), handle, sort_keys=False)
        written.append(path)
        print(f"generated {path}")

    if not args.run:
        print("Configs generated. Add --run to execute them sequentially.")
        return 0

    log_dir = Path("results/rg_et_ablation/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    for path in written:
        log_path = log_dir / f"{path.stem}.log"
        print(f"running {path} -> {log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "survot_rank.cli", "train", "--config", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            print(f"FAILED {path} with exit code {result.returncode}; see {log_path}")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
