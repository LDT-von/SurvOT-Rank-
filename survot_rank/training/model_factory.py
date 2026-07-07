#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model factory for the cleaned SurvOT-Rank framework."""

from __future__ import annotations

import importlib.util
import os
import sys


COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMMON_DIR))

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)


METHOD_REGISTRY = {
    "ot_event_hazard_v2": (
        os.path.join("survot_rank", "research", "methods", "ot_event_hazard_v2"),
        "OTEventHazardV2Survival",
    ),
    "otehv2_rankevent": (
        os.path.join("survot_rank", "research", "methods", "prognostic_event_transport"),
        "OTEHV2RankEvent",
    ),
}

METHOD_ALIASES = {
    "31": "ot_event_hazard_v2",
    "45": "otehv2_rankevent",
    "pet": "otehv2_rankevent",
    "prognostic_event_transport": "otehv2_rankevent",
}


def list_methods():
    return list(METHOD_REGISTRY.keys())


def _resolve_method_path(method_dir: str) -> str:
    path = os.path.join(PROJECT_ROOT, method_dir)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"method directory not found: {path}")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def _load_model_module(method_key: str, method_dir: str):
    method_path = _resolve_method_path(method_dir)
    model_file = os.path.join(method_path, "model.py")
    if method_key == "ot_event_hazard_v2":
        model_file = os.path.join(method_path, "model_v2.py")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"model file not found: {model_file}")

    unique_name = f"survot_rank_{method_key}_model"
    spec = importlib.util.spec_from_file_location(unique_name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model module: {model_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_model(method, args, omic_input_dim=None, omic_names=None, pathway_names=None):
    """Build a model from a public method name or alias."""
    key = METHOD_ALIASES.get(method, method)
    if key not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method: {method}. Available: {list(METHOD_REGISTRY)}")

    method_dir, cls_name = METHOD_REGISTRY[key]
    mod = _load_model_module(key, method_dir)
    cls = getattr(mod, cls_name)
    try:
        return cls(args, omic_input_dim=omic_input_dim, omic_names=omic_names, pathway_names=pathway_names)
    except TypeError:
        return cls(args, omic_input_dim=omic_input_dim, omic_names=omic_names)
