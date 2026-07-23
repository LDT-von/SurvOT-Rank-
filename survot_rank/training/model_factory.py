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
    "otehv2_rankevent_v2": (
        os.path.join("survot_rank", "research", "methods", "prognostic_event_transport"),
        "OTEHV2RankEventV2",
    ),
    "otehv2_timelocal_competing": (
        os.path.join("survot_rank", "research", "methods", "prognostic_event_transport"),
        "OTEHTimeLocalCompeting",
    ),
    "rank_guided_event_transport": (
        os.path.join("survot_rank", "research", "methods", "rank_guided_event_transport"),
        "RankGuidedEventTransport",
    ),
    "stagewise_prognostic_transport": (
        os.path.join("survot_rank", "research", "methods", "stagewise_prognostic_transport"),
        "StagewisePrognosticTransport",
    ),
    "faithful_evidence_transport": (
        os.path.join("survot_rank", "research", "methods", "faithful_evidence_transport"),
        "FaithfulEvidenceTransport",
    ),
    "distributional_counterfactual_transport": (
        os.path.join("survot_rank", "research", "methods", "distributional_counterfactual_transport"),
        "DistributionalCounterfactualTransport",
    ),
    "dct_listwise_transport": (
        os.path.join(
            "survot_rank",
            "research",
            "methods",
            "dct_listwise_transport",
        ),
        "DCTListwiseTransport",
    ),
    "censoring_aware_temporal_evidence_transport": (
        os.path.join("survot_rank", "research", "methods", "censoring_aware_temporal_evidence_transport"),
        "CensoringAwareTemporalEvidenceTransport",
    ),
    "v60_ot_event_rank": (
        os.path.join("survot_rank", "research", "methods", "v60_ot_event_rank"),
        "V60OTEventRank",
    ),
    "cohort_anchored_adaptive_prognostic_slot_attention": (
        os.path.join(
            "survot_rank",
            "research",
            "methods",
            "cohort_anchored_adaptive_prognostic_slot_attention",
        ),
        "CohortAnchoredAdaptivePrognosticSlotAttention",
    ),
    "v70_patient_specific_prognostic_circuits": (
        os.path.join(
            "survot_rank",
            "research",
            "methods",
            "v70_patient_specific_prognostic_circuits",
        ),
        "V70PatientSpecificPrognosticCircuits",
    ),
}

METHOD_ALIASES = {
    "31": "ot_event_hazard_v2",
    "45": "otehv2_rankevent",
    "pet": "otehv2_rankevent",
    "prognostic_event_transport": "otehv2_rankevent",
    "45v2": "otehv2_rankevent_v2",
    "50": "otehv2_timelocal_competing",
    "60": "v60_ot_event_rank",
    "ca_psa": "cohort_anchored_adaptive_prognostic_slot_attention",
    "capsa": "cohort_anchored_adaptive_prognostic_slot_attention",
    "70": "v70_patient_specific_prognostic_circuits",
    "pspc_surv": "v70_patient_specific_prognostic_circuits",
    "pspc": "v70_patient_specific_prognostic_circuits",
}


# OTEHV2RankEventV2 新增能力字段的声明类型（用于 _validate_config 的类型校验）。
# 说明：argparse CLI 路径下，未知的 `--flag` 名称与无法解析为声明类型的取值已经由
# argparse 自身在解析阶段拦截；这里主要覆盖“绕过 argparse 类型强制转换”的路径，例如
# 通过 YAML 配置 + `--set key=value`（`survot_rank/config.py` 中 `--set` 的值经
# `yaml.safe_load` 解析，可能得到字符串而非期望的 bool/int/float）或测试代码里手工
# 构造的 Namespace 对象直接赋值。
_OTEHV2V2_BOOL_FIELDS = (
    "otehv2v2_use_clinical",
    "otehv2v2_use_unified_objective",
    "otehv2v2_slot_disentangled",
    "otehv2v2_slot_cross_modal_cond",
    "otehv2v2_slot_adaptive_iters",
    "otehv2v2_learnable_loss_weights",
)
_OTEHV2V2_INT_FIELDS = (
    "otehv2v2_clinical_feature_dim",
    "otehv2v2_num_slots_clinical",
    "otehv2v2_sinkhorn_max_iters",
)
_OTEHV2V2_FLOAT_FIELDS = (
    "lambda_unified_rank",
    "otehv2v2_convergence_threshold",
)
_OTEHV2V2_CHOICE_FIELDS = {
    "otehv2v2_slot_router": ("softmax", "sinkhorn"),
}

# 互斥能力组合预留位（需求 6 AC8）：当前没有任何新增能力被声明为互斥，因此该列表
# 保持为空，循环不会产生任何拒绝行为。未来若引入互斥组合，向此列表追加
# `(field_a, field_b, ...)` 形式的元组即可自动获得 ValueError 拒绝逻辑。
_MUTUALLY_EXCLUSIVE_GROUPS: list[tuple[str, ...]] = []


def _validate_config(args) -> None:
    """在构建模型前对新增配置字段做轻量 schema 校验。

    当前版本职责（对应需求 7 AC2/AC3、需求 6 AC8）：
    1. 字段类型校验：若字段存在于 `args` 上，但其取值无法解析为声明的数据类型
       （例如 YAML `--set` 覆盖产生的字符串 "true" 而非真正的 bool），抛出
       `ValueError`，消息中包含具体字段名与问题原因。
    2. 字段缺失不视为错误：新增字段均通过 `getattr(args, name, default)` 的方式
       在模型内部读取，缺失时使用默认值，因此本函数对缺失字段不做任何处理。
    3. 互斥组合冲突检测预留接口：当前没有已声明的互斥能力组合，`_MUTUALLY_EXCLUSIVE_GROUPS`
       为空列表，循环体不会执行任何拒绝逻辑；未来新增互斥组合后，此处将抛出
       `ValueError` 并在消息中指明冲突的能力名称。

    本函数是纯校验函数：不实例化任何模型、不创建 `results_dir`、不写入任何文件。
    """
    # --- 1. 字段类型校验 ---
    for field in _OTEHV2V2_BOOL_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            if not isinstance(value, bool):
                raise ValueError(
                    f"配置字段 '{field}' 的取值无法解析为声明的数据类型 bool："
                    f"实际取值为 {value!r}（类型 {type(value).__name__}）"
                )

    for field in _OTEHV2V2_INT_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            # bool 是 int 的子类，但语义上不属于本处期望的整数字段，需排除。
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"配置字段 '{field}' 的取值无法解析为声明的数据类型 int："
                    f"实际取值为 {value!r}（类型 {type(value).__name__}）"
                )

    for field in _OTEHV2V2_FLOAT_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"配置字段 '{field}' 的取值无法解析为声明的数据类型 float："
                    f"实际取值为 {value!r}（类型 {type(value).__name__}）"
                )

    for field, choices in _OTEHV2V2_CHOICE_FIELDS.items():
        if hasattr(args, field):
            value = getattr(args, field)
            if value not in choices:
                raise ValueError(
                    f"配置字段 '{field}' 的取值不是合法的可选项："
                    f"实际取值为 {value!r}，允许的取值为 {choices}"
                )

    # --- 2. 互斥能力组合冲突检测（当前为空，预留后续扩展） ---
    for group in _MUTUALLY_EXCLUSIVE_GROUPS:
        enabled = [field for field in group if getattr(args, field, False)]
        if len(enabled) > 1:
            raise ValueError(
                f"配置中同时启用的能力存在互斥冲突: {enabled}"
            )


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
    _validate_config(args)

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
