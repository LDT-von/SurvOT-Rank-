#!/usr/bin/env python3
"""Run the pre-registered IST feedback repair gate on BLCA folds 1/2/4.

The gate changes only the interpretation of intervention stability:

1. transport mass is divided by the support-specific independent coupling
   before cross-view stability is measured; and
2. cost feedback penalizes important *unstable* edges instead of penalizing
   every low-mass edge.

Expansion rule (fixed before seeing repair results): continue to BLCA folds
0/3 and then the five held-out cancers only if the repaired mean exceeds the
existing factual A mean (0.7072) by at least 0.005 and improves at least two of
the three gate folds.  Otherwise stop the IST line.
"""

from __future__ import annotations

import os

try:
    from scripts import run_ist_v40_final_cross_cancer as base
except (ModuleNotFoundError, ImportError):
    import run_ist_v40_final_cross_cancer as base


def build_parser():
    parser = base.build_parser()
    parser.set_defaults(cancers=["blca"], folds=[1, 2, 4], recipe="repaired")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(base.REPO_ROOT)
    if args.mode == "prepare":
        return base.prepare_splits(args)
    if args.mode == "doctor":
        return base.doctor(args)
    smoke = args.mode == "smoke"
    jobs = base.build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        base.print_plan(jobs, recipe=args.recipe)
        return 0
    base.print_plan(
        jobs,
        recipe=args.recipe,
        force=args.force,
        run_mode=args.mode == "run",
    )
    return base.run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
