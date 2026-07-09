"""YAML experiment configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .project import resolve_project_path


PATH_KEYS = {"data_path", "results_dir"}
ABS_OR_EXTERNAL_PATH_KEYS = {"data_root_dir"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten sectioned YAML into argparse-style key/value pairs.

    Reserved metadata sections are ignored. All other nested sections are merged
    into one parameter dictionary because the legacy runner expects flat CLI
    flags.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"name", "description", "notes"}:
            continue
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        merged[key] = value
    return merged


def config_to_argv(config: dict[str, Any]) -> list[str]:
    flat = flatten_config(config)
    argv: list[str] = []
    for key, value in flat.items():
        if value is None or value is False:
            continue
        if key in PATH_KEYS:
            value = resolve_project_path(value)
        elif key in ABS_OR_EXTERNAL_PATH_KEYS and str(value).startswith("."):
            value = resolve_project_path(value)
        flag = f"--{key}"
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            # Special case: clinical_feature_cols is passed as comma-separated string
            if key == "clinical_feature_cols":
                argv.append(flag)
                argv.append(",".join(str(item) for item in value))
            else:
                argv.append(flag)
                argv.extend(str(item) for item in value)
        else:
            argv.extend([flag, str(value)])
    return argv

