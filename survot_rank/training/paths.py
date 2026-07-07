#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path helpers for the packaged legacy data runtime."""

import os
import sys


COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMMON_DIR))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "survot_rank", "research")


def _find_slotspe_dir():
    """Find the local runtime used by legacy dataset/loss imports."""
    env = os.environ.get("SURVOT_RUNTIME_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    candidates = [
        os.path.join(RESEARCH_DIR, "legacy", "slotspe_runtime"),
    ]
    for candidate in candidates:
        has_dataset = os.path.isfile(os.path.join(candidate, "dataset", "dataset_survival.py"))
        has_losses = os.path.isfile(os.path.join(candidate, "utils", "loss_func.py"))
        if has_dataset and has_losses:
            return os.path.abspath(candidate)
    return None


SLOTSPE_DIR = _find_slotspe_dir()


def ensure_slotspe_in_path():
    """Add the packaged legacy runtime to sys.path for dataset/utils imports."""
    if SLOTSPE_DIR is None:
        raise FileNotFoundError(
            "Could not find the packaged data runtime. Set SURVOT_RUNTIME_DIR or keep the "
            "runtime at survot_rank/research/legacy/slotspe_runtime."
        )
    if SLOTSPE_DIR not in sys.path:
        sys.path.insert(0, SLOTSPE_DIR)


def get_slotspe_dir():
    return SLOTSPE_DIR
