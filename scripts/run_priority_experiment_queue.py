#!/usr/bin/env python3
"""按论文优先级串行运行精简实验队列，并支持断点续跑。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.run_dct_v38_transport_consistency import (
        VARIANTS as V38_VARIANTS,
        _override_args,
        build_train_command as build_v38_command,
        inspect_feature_directory,
        inspect_split_directory,
        verify_child_cuda,
    )
    from scripts.run_dct_v382_mgptr import build_train_command as build_v382_command
    from scripts.run_dct_v41_survival_evidence_ledger import inspect_uni_directory
    from scripts.run_v40_intervention_stable_transport import (
        COMMON_OVERRIDES as V40_COMMON,
        PROTOCOLS as V40_PROTOCOLS,
        VARIANTS as V40_VARIANTS,
    )
    from scripts.task_lock import ActiveRunError, acquire_run_lock, release_run_lock
except ModuleNotFoundError:
    from run_dct_v38_transport_consistency import (
        VARIANTS as V38_VARIANTS,
        _override_args,
        build_train_command as build_v38_command,
        inspect_feature_directory,
        inspect_split_directory,
        verify_child_cuda,
    )
    from run_dct_v382_mgptr import build_train_command as build_v382_command
    from run_dct_v41_survival_evidence_ledger import inspect_uni_directory
    from run_v40_intervention_stable_transport import (
        COMMON_OVERRIDES as V40_COMMON,
        PROTOCOLS as V40_PROTOCOLS,
        VARIANTS as V40_VARIANTS,
    )
    from task_lock import ActiveRunError, acquire_run_lock, release_run_lock

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGES = (
    # 当前主线：先用 6 次训练判定 MGPTR 是否成立。
    "b0_mgptr_control",
    "b1_mgptr",
    # v3.3 在 2026-07-30 重划之前的旧划分上复现历史分数 (0.7311)。
    "v33_blca_legacy_repro",
    # A 组 clean 基线：显式写出 fit_bins_on_train，使 0.7400 可溯源。
    "v33_clean_baseline",
    # A 组底座候选：去掉稀疏事件下高方差的 IPCW pairwise rank。
    "v33_clean_no_ipcw_rank",
    # 修复分阶段激活后重跑（统一 50ep）。
    "v40_staged_rerun",
    "v41_staged_rerun",
    "arcsurv_staged_rerun",
    # 单变量消融：稳定性分数是否应回写 factual transport cost。
    "v40_no_cost_feedback",
    # 单变量消融：完整模态任务上人为删模态是否有害。
    "v41_no_modality_dropout",
    # 单变量消融：v4.1 的退化是否来自继承的 IPCW rank。
    "v41_no_ipcw_rank",
    # v4.2 已实现但刻意不入默认队列：需先由 arcsurv_staged_rerun 确认
    # archetype 真的分化开（act_archetype_cosine / act_hazard_spread 诊断），
    # 否则「凸组合 + 精确可加归因」的第二卖点没有立足点。
    "v42_act_surv",
    # 以下为历史阶段，保留以便复跑/审计。
    "v33_blca_uni5",
    "v38_lusc_screen",
    "v382_blca_fold124",
    "v383_blca_fold124",
    "v39_blca_fold124",
    "v40_blca_fold124",
    "v41_blca_fold124",
    "arcsurv_blca_fold124",
)
DEFAULT_STAGES = (
    # 当前主线只保留三件事：把 A 组底座钉死、修复后的 v4.0、以及三个单变量消融。
    # v4.1 与 ArcSurv 的完整重跑不入默认队列（见 FINAL_SUMMARY 的判定）。
    "v33_clean_baseline",
    "v33_clean_no_ipcw_rank",
    "v40_staged_rerun",
    "v40_no_cost_feedback",
    "v41_no_modality_dropout",
    "v41_no_ipcw_rank",
)
FOLDS_124 = (1, 2, 4)
# B0/B1 唯一变量：MGPTR 权重。其余全部相同。
MGPTR_SHARED = {
    "survot_method": "dct_v382_prognostic_transport_reconstruction",
    "max_epochs": 50,
    # 前 5 轮只训 NLL + IPCW，第 6-15 轮线性拉起 MGPTR。
    "dct_v382_warmup_epochs": 5,
    "dct_v382_ramp_epochs": 10,
    "dct_v382_adaptive_aux_weights": False,
    "dct_v382_distill_weight": 0.50,
    # v3.8 的三个干预损失全部关闭（direction 无任一癌种正向证据）。
    "dct_v38_lambda_direction": 0.0,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    # robust 协议。
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_slot_init_mode": "deterministic",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_ipcw_rank_memory_size": 64,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_mix_ratio": 1.0,
    "num_patches": 2048,
    "batch_size": 8,
}


@dataclass(frozen=True)
class Job:
    stage: str
    label: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path
    encoder: str
    cancer: str
    which_splits: str


def _generic_command(
    args: argparse.Namespace,
    *,
    stage: str,
    label: str,
    config: str,
    fold: int,
    result_dir: Path,
    encoder: str,
    cancer: str = "blca",
    which_splits: str = "5fold",
    overrides: dict[str, object] | None = None,
    smoke: bool = False,
) -> Job:
    values: dict[str, object] = {
        "k_start": fold,
        "k_end": fold + 1,
        "gpu": args.gpu,
        "num_workers": args.num_workers,
        "data_root_dir": args.uni_root if encoder == "uni" else args.uni2h_root,
        "wsi_encoder": encoder,
        "encoding_dim": 1024 if encoder == "uni" else 1536,
        "which_splits": which_splits,
        "on_missing_wsi": "error",
        "results_dir": result_dir.as_posix(),
    }
    if overrides:
        values.update(overrides)
    if smoke:
        values.update({"max_epochs": 1, "max_smoke_batches": 2})
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config,
        *_override_args(values),
    )
    return Job(
        stage=stage,
        label=label,
        fold=fold,
        command=tuple(command),
        result_dir=result_dir,
        config=Path(config),
        encoder=encoder,
        cancer=cancer,
        which_splits=which_splits,
    )


def _replace_override(command: list[str], key: str, value: object) -> None:
    prefix = f"{key}="
    for index, item in enumerate(command):
        if item == "--set" and index + 1 < len(command):
            if command[index + 1].startswith(prefix):
                command[index + 1] = f"{key}={value}"
                return
    command.extend(("--set", f"{key}={value}"))


def _selected_stages(args: argparse.Namespace) -> list[str]:
    selected = list(args.stages or DEFAULT_STAGES)
    if args.from_stage:
        selected = [name for name in selected if STAGES.index(name) >= STAGES.index(args.from_stage)]
    return [name for name in STAGES if name in selected]


def _smoke_dir(stage: str, suffix: str = "") -> Path:
    path = Path("results") / "priority_experiment_queue_smoke" / stage
    return path / suffix if suffix else path


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[Job]:
    """构造固定顺序队列；正式队列共 31 个 fold/variant 任务。"""
    selected = set(_selected_stages(args))
    jobs: list[Job] = []

    # B0/B1：MGPTR 单变量对照。B0 同时充当 UNI2-h clean 基线（台账 #8）。
    mgptr_specs = [
        ("b0_mgptr_control", "B0 v3.8.2 base (MGPTR=0) BLCA UNI2-h", 0.0, "base"),
        ("b1_mgptr", "B1 v3.8.2 mgptr (MGPTR=0.05) BLCA UNI2-h", 0.05, "mgptr"),
    ]
    for stage, label, mgptr_weight, variant in mgptr_specs:
        if stage not in selected:
            continue
        result_dir = _smoke_dir(stage) if smoke else Path(
            "results/dct_v3.8.2/robust"
        ) / variant / "blca"
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            overrides = dict(MGPTR_SHARED)
            overrides["dct_v382_lambda_mgptr"] = mgptr_weight
            overrides["specific_simple"] = f"dct_v382_robust_{variant}_blca_50ep"
            jobs.append(
                _generic_command(
                    args,
                    stage=stage,
                    label=label,
                    config="configs/distributional_counterfactual_transport_blca.yaml",
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni2-h",
                    which_splits="5fold_uni2h",
                    overrides=overrides,
                    smoke=smoke,
                )
            )

    # v3.3 在旧划分上复现历史分数（0.7311）。沿用当时的 leaky 分箱（默认 False）。
    if "v33_blca_legacy_repro" in selected:
        result_dir = _smoke_dir("v33_blca_legacy_repro") if smoke else Path(
            "results/dct_v3.3_score_first_blca_legacy_repro"
        )
        for fold in (range(1) if smoke else range(5)):
            jobs.append(
                _generic_command(
                    args,
                    stage="v33_blca_legacy_repro",
                    label="v3.3 Score-First BLCA UNI legacy-split repro",
                    config="configs/diagnostics/dct_v3_score_blca.yaml",
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni",
                    which_splits="5fold_legacy",
                    overrides={
                        "survot_method": "distributional_counterfactual_transport",
                        "max_epochs": 50,
                        "specific_simple": "dct_v3_score_first_legacy_repro",
                    },
                    smoke=smoke,
                )
            )

    # A 组 clean 基线与「去掉 IPCW rank」的底座候选。
    # 冻结 YAML configs/diagnostics/dct_v3_score_blca.yaml 未设 fit_bins_on_train，
    # 而该参数在 extended_args.py 中是 action="store_true"（默认 False）。
    # 因此 clean 协议必须在这里显式覆盖，否则跑出来的是 leaky 分箱。
    v33_clean_specs = [
        (
            "v33_clean_baseline",
            "v3.3 Score-First BLCA UNI clean baseline",
            "results/dct_v3.3_score_first_blca_clean_50ep/blca",
            "dct_v3_score_first_clean_50ep",
            {},
        ),
        (
            "v33_clean_no_ipcw_rank",
            "v3.3 Score-First BLCA UNI clean, IPCW rank off",
            "results/dct_v3.3_score_first_blca_clean_no_ipcw_50ep/blca",
            "dct_v3_score_first_clean_no_ipcw_50ep",
            {"dct_lambda_ipcw_rank": 0.0},
        ),
    ]
    for stage, label, root, identity, extra in v33_clean_specs:
        if stage not in selected:
            continue
        result_dir = _smoke_dir(stage) if smoke else Path(root)
        for fold in (FOLDS_124[:1] if smoke else FOLDS_124):
            overrides = {
                "survot_method": "distributional_counterfactual_transport",
                "max_epochs": 50,
                "fit_bins_on_train": True,
                "binning_mode": "global_qcut",
                "specific_simple": identity,
            }
            overrides.update(extra)
            jobs.append(
                _generic_command(
                    args,
                    stage=stage,
                    label=label,
                    config="configs/diagnostics/dct_v3_score_blca.yaml",
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni",
                    which_splits="5fold",
                    overrides=overrides,
                    smoke=smoke,
                )
            )

    # 修复分阶段激活后重跑。三者的唯一共同改动是「辅助约束不再从第一轮生效」。
    staged_specs = [
        (
            "v40_staged_rerun",
            "v4.0 IST-Surv staged stability BLCA UNI2-h",
            "configs/intervention_stable_survival_transport_blca.yaml",
            "results/ist_surv_v4.0_staged_50ep/clean/full/blca",
            "intervention_stable_survival_transport",
            "ist_v40_staged_full_blca_50ep",
            "uni2-h",
            "5fold_uni2h",
            {
                "ist_warmup_epochs": 5,
                "ist_ramp_epochs": 10,
                "ist_stability_strength": 0.10,
                "ist_lambda_plan": 0.05,
                "ist_lambda_attribution": 0.05,
                "ist_lambda_risk": 0.0,
                "fit_bins_on_train": True,
                "binning_mode": "global_qcut",
            },
        ),
        (
            "v41_staged_rerun",
            "v4.1 Evidence Ledger staged aux BLCA UNI",
            "configs/dct_v41_survival_evidence_ledger_blca.yaml",
            "results/dct_v4.1_survival_evidence_ledger_staged_50ep/blca",
            "dct_v41_survival_evidence_ledger",
            "dct_v41_staged_blca_50ep",
            "uni",
            "5fold",
            {
                "v41_warmup_epochs": 5,
                "v41_ramp_epochs": 10,
                "v41_modality_dropout": 0.20,
            },
        ),
        (
            "arcsurv_staged_rerun",
            "ArcSurv staged structure losses BLCA UNI",
            "configs/archetypal_risk_composition_blca.yaml",
            "results/archetypal_risk_composition_staged_50ep/blca",
            "archetypal_risk_composition",
            "arcsurv_staged_blca_50ep",
            "uni",
            "5fold",
            {
                "arc_warmup_epochs": 5,
                "arc_ramp_epochs": 10,
                "arc_bank_update_epochs": -1,
                "batch_size": 8,
            },
        ),
    ]
    for (
        stage,
        label,
        config,
        root,
        method,
        identity,
        encoder,
        split,
        extra,
    ) in staged_specs:
        if stage not in selected:
            continue
        result_dir = _smoke_dir(stage) if smoke else Path(root)
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            overrides = {
                "survot_method": method,
                # 统一 50ep：ArcSurv fold1 在 30ep 下峰值贴在 e29 且仍在上升。
                "max_epochs": 50,
                "specific_simple": identity,
            }
            overrides.update(extra)
            jobs.append(
                _generic_command(
                    args,
                    stage=stage,
                    label=label,
                    config=config,
                    fold=fold,
                    result_dir=result_dir,
                    encoder=encoder,
                    which_splits=split,
                    overrides=overrides,
                    smoke=smoke,
                )
            )

    # 单变量消融。每个 stage 相对其 staged 基线只改一个键，其余全部继承，
    # 便于做逐折配对比较。
    ablation_specs = [
        (
            "v40_no_cost_feedback",
            "v4.0 IST-Surv staged, stability cost feedback off",
            "configs/intervention_stable_survival_transport_blca.yaml",
            "results/ist_surv_v4.0_staged_50ep/clean/no_cost_feedback/blca",
            "intervention_stable_survival_transport",
            "ist_v40_staged_no_cost_feedback_blca_50ep",
            "uni2-h",
            "5fold_uni2h",
            {
                "ist_warmup_epochs": 5,
                "ist_ramp_epochs": 10,
                # 唯一变量：稳定性分数不再回写 factual cost。
                # 辅助损失仍然保留，用于区分「回写运输计划」与「稳定性正则」。
                "ist_stability_strength": 0.0,
                "ist_lambda_plan": 0.05,
                "ist_lambda_attribution": 0.05,
                "ist_lambda_risk": 0.0,
                "fit_bins_on_train": True,
                "binning_mode": "global_qcut",
            },
        ),
        (
            "v41_no_modality_dropout",
            "v4.1 Evidence Ledger staged, modality dropout off",
            "configs/dct_v41_survival_evidence_ledger_blca.yaml",
            "results/dct_v4.1_survival_evidence_ledger_staged_50ep/no_dropout/blca",
            "dct_v41_survival_evidence_ledger",
            "dct_v41_staged_no_dropout_blca_50ep",
            "uni",
            "5fold",
            {
                "v41_warmup_epochs": 5,
                "v41_ramp_epochs": 10,
                # 唯一变量：不再在完整双模态病例上人为删模态。
                "v41_modality_dropout": 0.0,
            },
        ),
        (
            "v41_no_ipcw_rank",
            "v4.1 Evidence Ledger staged, inherited IPCW rank off",
            "configs/dct_v41_survival_evidence_ledger_blca.yaml",
            "results/dct_v4.1_survival_evidence_ledger_staged_50ep/no_ipcw/blca",
            "dct_v41_survival_evidence_ledger",
            "dct_v41_staged_no_ipcw_blca_50ep",
            "uni",
            "5fold",
            {
                "v41_warmup_epochs": 5,
                "v41_ramp_epochs": 10,
                "v41_modality_dropout": 0.20,
                # 唯一变量：去掉从 v3.3 继承的 pairwise IPCW rank。
                "dct_lambda_ipcw_rank": 0.0,
            },
        ),
    ]
    for (
        stage,
        label,
        config,
        root,
        method,
        identity,
        encoder,
        split,
        extra,
    ) in ablation_specs:
        if stage not in selected:
            continue
        result_dir = _smoke_dir(stage) if smoke else Path(root)
        for fold in (FOLDS_124[:1] if smoke else FOLDS_124):
            overrides = {
                "survot_method": method,
                "max_epochs": 50,
                "specific_simple": identity,
            }
            overrides.update(extra)
            jobs.append(
                _generic_command(
                    args,
                    stage=stage,
                    label=label,
                    config=config,
                    fold=fold,
                    result_dir=result_dir,
                    encoder=encoder,
                    which_splits=split,
                    overrides=overrides,
                    smoke=smoke,
                )
            )

    if "v42_act_surv" in selected:
        result_dir = _smoke_dir("v42_act_surv") if smoke else Path(
            "results/act_surv_v4.2/blca"
        )
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            jobs.append(
                _generic_command(
                    args,
                    stage="v42_act_surv",
                    label="v4.2 ACT-Surv archetypal transport BLCA UNI",
                    config="configs/archetypal_transport_composition_blca.yaml",
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni",
                    which_splits="5fold",
                    overrides={
                        "survot_method": "archetypal_transport_composition",
                        "max_epochs": 50,
                        "specific_simple": "act_surv_v42_blca_50ep",
                    },
                    smoke=smoke,
                )
            )

    if "v33_blca_uni5" in selected:
        result_dir = _smoke_dir("v33_blca_uni5") if smoke else Path(
            "results/dct_v3.3_score_first_blca_uni_rep"
        )
        for fold in (range(1) if smoke else range(5)):
            jobs.append(
                _generic_command(
                    args,
                    stage="v33_blca_uni5",
                    label="DCT v3.3 Score-First/full BLCA UNI",
                    config="configs/diagnostics/dct_v3_score_blca.yaml",
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni",
                    overrides={
                        "survot_method": "distributional_counterfactual_transport",
                        "max_epochs": 50,
                        "specific_simple": "dct_v3_score_first_full",
                    },
                    smoke=smoke,
                )
            )

    if "v38_lusc_screen" in selected:
        for variant in V38_VARIANTS:
            command, result_dir = build_v38_command(
                args.python_bin,
                "lusc",
                "robust",
                variant,
                0,
                args.gpu,
                args.num_workers,
                args.uni2h_root,
                max_epochs=20,
                smoke=smoke,
            )
            _replace_override(command, "on_missing_wsi", "error")
            if smoke:
                _replace_override(command, "max_epochs", 1)
            jobs.append(
                Job(
                    stage="v38_lusc_screen",
                    label=f"DCT v3.8 robust/{variant} LUSC UNI2-h",
                    fold=0,
                    command=tuple(command),
                    result_dir=result_dir,
                    config=Path("configs/distributional_counterfactual_transport_lusc.yaml"),
                    encoder="uni2-h",
                    cancer="lusc",
                    which_splits="5fold_uni2h",
                )
            )

    if "v382_blca_fold124" in selected:
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            command, result_dir = build_v382_command(
                args.python_bin,
                "blca",
                "robust",
                "adaptive_full",
                fold,
                args.gpu,
                args.num_workers,
                args.uni2h_root,
                max_epochs=30,
                smoke=smoke,
            )
            _replace_override(command, "on_missing_wsi", "error")
            if smoke:
                _replace_override(command, "max_epochs", 1)
            jobs.append(
                Job(
                    stage="v382_blca_fold124",
                    label="DCT v3.8.2 robust/adaptive_full BLCA UNI2-h",
                    fold=fold,
                    command=tuple(command),
                    result_dir=result_dir,
                    config=Path("configs/distributional_counterfactual_transport_blca.yaml"),
                    encoder="uni2-h",
                    cancer="blca",
                    which_splits="5fold_uni2h",
                )
            )

    generic_specs = [
        (
            "v383_blca_fold124",
            "DCT v3.8.3 centered/full BLCA UNI2-h",
            "configs/dct_v383_intervention_consistency_centered_blca.yaml",
            "dct_v3.8.3_intervention_consistency_centered_30ep",
            "dct_v383_intervention_consistency_centered",
            "dct_v383_centered_full_blca_30ep",
            "uni2-h",
            "5fold_uni2h",
        ),
        (
            "v39_blca_fold124",
            "DCT v3.9 Risk-Simplex BLCA UNI2-h",
            "configs/dct_v39_risk_simplex_transport_blca.yaml",
            "dct_v3.9_risk_simplex_transport_30ep",
            "dct_v39_risk_simplex_transport",
            "dct_v39_risk_simplex_blca_30ep",
            "uni2-h",
            "5fold_uni2h",
        ),
    ]
    for stage, label, config, root, method, identity, encoder, split in generic_specs:
        if stage not in selected:
            continue
        result_dir = _smoke_dir(stage) if smoke else Path("results") / root / "blca"
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            jobs.append(
                _generic_command(
                    args,
                    stage=stage,
                    label=label,
                    config=config,
                    fold=fold,
                    result_dir=result_dir,
                    encoder=encoder,
                    which_splits=split,
                    overrides={
                        "survot_method": method,
                        "max_epochs": 30,
                        "specific_simple": identity,
                    },
                    smoke=smoke,
                )
            )

    if "v40_blca_fold124" in selected:
        v40 = dict(V40_COMMON)
        v40.update(V40_PROTOCOLS["clean"])
        v40.update(V40_VARIANTS["full"])
        v40.pop("label", None)
        v40.pop("data_root_dir", None)
        v40.update(
            {
                "survot_method": "intervention_stable_survival_transport",
                "max_epochs": 30,
                "specific_simple": "ist_v40_clean_full_blca_30ep",
            }
        )
        result_dir = _smoke_dir("v40_blca_fold124") if smoke else Path(
            "results/ist_surv_v4.0_30ep/clean/full/blca"
        )
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            jobs.append(
                _generic_command(
                    args,
                    stage="v40_blca_fold124",
                    label="IST-Surv v4.0 clean/full BLCA UNI2-h",
                    config="configs/intervention_stable_survival_transport_blca.yaml",
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni2-h",
                    which_splits="5fold_uni2h",
                    overrides=v40,
                    smoke=smoke,
                )
            )

    final_specs = [
        (
            "v41_blca_fold124",
            "DCT v4.1 Evidence Ledger BLCA UNI",
            "configs/dct_v41_survival_evidence_ledger_blca.yaml",
            "dct_v4.1_survival_evidence_ledger_30ep",
            "dct_v41_survival_evidence_ledger",
            "dct_v41_selc_uni_blca_30ep",
        ),
        (
            "arcsurv_blca_fold124",
            "ArcSurv BLCA UNI",
            "configs/archetypal_risk_composition_blca.yaml",
            "archetypal_risk_composition_30ep",
            "archetypal_risk_composition",
            "arcsurv_blca_uni_30ep",
        ),
    ]
    for stage, label, config, root, method, identity in final_specs:
        if stage not in selected:
            continue
        result_dir = _smoke_dir(stage) if smoke else Path("results") / root / "blca"
        folds = FOLDS_124[:1] if smoke else FOLDS_124
        for fold in folds:
            jobs.append(
                _generic_command(
                    args,
                    stage=stage,
                    label=label,
                    config=config,
                    fold=fold,
                    result_dir=result_dir,
                    encoder="uni",
                    overrides={
                        "survot_method": method,
                        "max_epochs": 30,
                        "specific_simple": identity,
                    },
                    smoke=smoke,
                )
            )
    return jobs


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def _safe_gpu_name(gpu: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in gpu)


def _task_lock_path(job: Job) -> Path:
    return job.result_dir / f".split_{job.fold}.priority_queue.lock"


def _scheduler_lock_path(gpu: str, smoke: bool) -> Path:
    kind = "smoke" if smoke else "run"
    return Path("results/priority_experiment_queue") / f".{kind}_gpu_{_safe_gpu_name(gpu)}.lock"


def _override_value(job: Job, key: str) -> str | None:
    prefix = f"{key}="
    for index, item in enumerate(job.command[:-1]):
        if item == "--set" and job.command[index + 1].startswith(prefix):
            return job.command[index + 1][len(prefix) :]
    return None


def doctor(args: argparse.Namespace, jobs: list[Job]) -> int:
    """在启动 GPU 任务前检查配置、注册、特征维度和 split。"""
    failed = False
    configs = sorted({job.config for job in jobs})
    for config in configs:
        exists = (REPO_ROOT / config).is_file()
        print(f"{'OK' if exists else 'MISSING':8s} config {config}")
        failed = failed or not exists

    factory_path = REPO_ROOT / "survot_rank/training/model_factory.py"
    factory_text = factory_path.read_text(encoding="utf-8") if factory_path.is_file() else ""
    methods = sorted(filter(None, {_override_value(job, "survot_method") for job in jobs}))
    for method in methods:
        registered = f'"{method}"' in factory_text
        print(f"{'OK' if registered else 'MISSING':8s} method {method}")
        failed = failed or not registered

    feature_specs = sorted({(job.encoder, job.cancer) for job in jobs})
    for encoder, cancer in feature_specs:
        if encoder == "uni":
            report = inspect_uni_directory(args.uni_root, cancer)
        else:
            report = inspect_feature_directory(args.uni2h_root, cancer)
        status = "OK" if report["ok"] else "MISSING"
        print(
            f"{status:8s} feature {cancer.upper()} {encoder} "
            f"files={report['count']} shape={report['shape']} path={report['directory']}"
        )
        if report["error"]:
            print(f"         {report['error']}")
        failed = failed or not bool(report["ok"])

    split_specs = sorted({(job.cancer, job.which_splits, job.encoder) for job in jobs})
    for cancer, which_splits, encoder in split_specs:
        root = args.uni_root if encoder == "uni" else args.uni2h_root
        # 旧划分是刻意恢复的历史产物，已知含缺 DSS 标签患者且事件分层更差。
        # 它只用于复现历史分数，因此审计失败只告警，不阻塞。
        legacy = which_splits == "5fold_legacy"
        try:
            report = inspect_split_directory(
                cancer,
                data_root=root,
                which_splits=which_splits,
            )
            ok = bool(report["ok"])
            status = "OK" if ok else ("LEGACY" if legacy else "INVALID")
            print(
                f"{status:8s} split {cancer.upper()} "
                f"{which_splits} eligible={report['eligible_cases']} "
                f"val_events={report['validation_event_counts']}"
            )
            for error in report["errors"]:
                print(f"         {error}")
            if legacy and not ok:
                print("         (已知缺陷划分，仅用于历史复现，不阻塞运行)")
            failed = failed or not (ok or legacy)
        except Exception as error:  # doctor 要一次报告全部问题
            status = "LEGACY" if legacy else "INVALID"
            print(f"{status:8s} split {cancer.upper()} {which_splits}: {error}")
            if legacy:
                print("         (已知缺陷划分，仅用于历史复现，不阻塞运行)")
            else:
                failed = True
    return int(failed)


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print(f"队列共 {len(jobs)} 个任务，严格串行执行：")
    current_stage = None
    for index, job in enumerate(jobs, start=1):
        if job.stage != current_stage:
            current_stage = job.stage
            print(f"\n[{STAGES.index(job.stage) + 1}. {job.stage}]")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} fold{job.fold} | {job.label}")
        print("    " + shlex.join(job.command))
        if completion:
            print(f"    已完成：{completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if doctor(args, jobs):
        print("[ERROR] doctor 检查未通过，拒绝启动训练。")
        return 2

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not verify_child_cuda(args.python_bin, environment):
        return 1

    scheduler_lock = None
    try:
        scheduler_lock = acquire_run_lock(
            _scheduler_lock_path(args.gpu, smoke),
            label=f"priority experiment queue on GPU {args.gpu}",
        )
    except ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3

    try:
        for index, job in enumerate(jobs, start=1):
            completion = _completion(job)
            if completion and not args.force and not smoke:
                print(f"[{index:02d}/{len(jobs):02d}] [skip] {job.label} fold{job.fold}: {completion}")
                continue
            print(f"\n[{index:02d}/{len(jobs):02d}] {job.label} fold{job.fold}")
            print(shlex.join(job.command))
            task_lock = None
            try:
                task_lock = acquire_run_lock(
                    _task_lock_path(job),
                    label=f"{job.stage} fold{job.fold}",
                )
            except ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                completion = _completion(job)
                if completion and not args.force and not smoke:
                    print(f"[skip] 锁内复检已完成：{completion}")
                    continue
                completed = subprocess.run(job.command, check=False, env=environment)
                if completed.returncode != 0:
                    print(f"[ERROR] 任务失败，返回码 {completed.returncode}；队列停止。")
                    return completed.returncode
            finally:
                release_run_lock(task_lock)
        return 0
    finally:
        release_run_lock(scheduler_lock)


def parse_stages(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(STAGES)
    if value.strip().lower() == "default":
        return list(DEFAULT_STAGES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"未知阶段：{', '.join(unknown)}；可选：{', '.join(STAGES)}"
        )
    if not selected:
        raise argparse.ArgumentTypeError("至少选择一个阶段")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("plan", "doctor", "smoke", "run"),
        nargs="?",
        default="plan",
    )
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=None,
        help=(
            "默认只跑当前主线：A 组 clean 底座 (v3.3 clean / 去 IPCW rank)、"
            "修复后的 v4.0，以及三个单变量消融；"
            "用 all 跑全部历史阶段，或逗号分隔指定。"
        ),
    )
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    parser.add_argument(
        "--uni-root",
        default=os.environ.get("UNI_ROOT", "/data/CPathPatchFeature"),
    )
    parser.add_argument(
        "--uni2h-root",
        default=os.environ.get("UNI2H_ROOT", "/data1/TCGA-UNI2-h-features"),
    )
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    if args.mode == "doctor":
        return doctor(args, jobs)
    print_plan(jobs, force=args.force, run_mode=args.mode == "run")
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
