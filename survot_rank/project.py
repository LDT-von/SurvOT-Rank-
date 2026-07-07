"""Project path helpers for SurvOT-Rank."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = PROJECT_ROOT / "common"
RESEARCH_DIR = PROJECT_ROOT / "survot_rank" / "research"
SLOTSPE_DIR = RESEARCH_DIR / "legacy" / "slotspe_runtime"


def add_project_paths() -> None:
    """Make legacy modules importable from the clean package entrypoints."""
    for path in (PROJECT_ROOT, COMMON_DIR, SLOTSPE_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_project_path(value: str | os.PathLike[str]) -> str:
    """Resolve repo-relative paths, leaving absolute paths untouched."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())
