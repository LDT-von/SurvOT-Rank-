#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument parser for the cleaned SurvOT-Rank training path."""

from __future__ import annotations

import argparse


METHOD_CHOICES = [
    "otehv2_rankevent",
    "otehv2_rankevent_v2",
    "otehv2_timelocal_competing",
    "pet",
    "prognostic_event_transport",
    "ot_event_hazard_v2",
    "rank_guided_event_transport",
    "stagewise_prognostic_transport",
    "faithful_evidence_transport",
    "distributional_counterfactual_transport",
    "censoring_aware_temporal_evidence_transport",
    "v60_ot_event_rank",
    "cohort_anchored_adaptive_prognostic_slot_attention",
    "ca_psa",
    "capsa",
    "v70_patient_specific_prognostic_circuits",
    "pspc_surv",
    "pspc",
]


def build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SurvOT-Rank training")

    # Experiment and data.
    parser.add_argument("--study", type=str, default="blca")
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--specific_simple", default="")
    parser.add_argument("--data_root_dir", type=str, default="")
    parser.add_argument("--data_path", type=str, default="./dataset_csv")
    parser.add_argument("--num_patches", type=int, default=4096)
    parser.add_argument("--num_genes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_col", type=str, default="survival_months_dss")
    parser.add_argument("--wsi_encoder", type=str, default="uni", choices=["uni", "gigap", "r50", "chief"])
    parser.add_argument("--rna_format", type=str, default="Pathways", choices=["RNASeq", "Pathways", "GeneEmbedding"])
    parser.add_argument("--signature", type=str, default="combine", choices=["all", "six", "hallmarks", "combine", "xena"])
    parser.add_argument("--clinical_feature_cols", type=str, default=None, help="Comma-separated clinical feature column names, e.g. 'age,gender'")

    # Splits.
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k_start", type=int, default=-1)
    parser.add_argument("--k_end", type=int, default=-1)
    parser.add_argument("--which_splits", type=str, default="5fold")

    # Training.
    parser.add_argument("--survot_method", type=str, default=None, choices=METHOD_CHOICES)
    parser.add_argument("--newslot_method", type=str, default=None, choices=METHOD_CHOICES)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd", "adamW"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bag_loss", type=str, default="nll_surv", choices=["nll_surv", "rank_surv", "cox_surv", "sinkhorn_surv"])
    parser.add_argument("--alpha_surv", type=float, default=0.5)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--max_cindex", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--step_size", type=int, default=10)
    # 梯度累积 / LR warmup / 梯度裁剪（默认值 = 原有行为，不影响历史实验）。
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="累积多少个 micro-batch 再更新一次；有效 batch = batch_size * grad_accum_steps")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="cosine 前的线性 warmup epoch 数；0 = 无 warmup（原行为）")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0,
                        help="梯度裁剪范数上限；0 = 不裁剪（原行为）")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--only_test", action="store_true", default=False)
    parser.add_argument("--omic_missing", action="store_true", default=False)
    parser.add_argument("--max_smoke_batches", type=int, default=0)
    parser.add_argument("--min_free_space_gb", type=float, default=2.0)
    parser.add_argument(
        "--fit_bins_on_train",
        action="store_true",
        default=False,
        help="Fit discrete survival bins from the current fold's uncensored training cases only.",
    )
    parser.add_argument(
        "--event_sampling_fraction",
        type=float,
        default=0.0,
        help=(
            "Target observed-event fraction for weighted training sampling. "
            "0 disables event-aware sampling and preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--event_stratified_batches",
        action="store_true",
        default=False,
        help=(
            "Spread observed events across batches without replacement. Every "
            "training patient is used exactly once per epoch."
        ),
    )

    # Early stopping.
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_metric", type=str, default="val_cindex", choices=["val_cindex", "val_cindex_ipcw", "val_iauc"])
    parser.add_argument("--early_stop_warmup", type=int, default=0)

    # Slot and projection settings.
    parser.add_argument("--method", type=str, default="SurvOTRank")
    parser.add_argument("--encoding_dim", type=int, default=1024)
    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--slot_num_wsi", type=int, default=8)
    parser.add_argument("--slot_num_omics", type=int, default=8)
    parser.add_argument("--slot_iters", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--topk_ratio", type=float, default=0.25)
    parser.add_argument("--top_k_method", type=str, default="parallel_topk_st", choices=["gumbel_topk_st", "parallel_topk_st"])

    # OT event hazard v2 / PET model settings.
    parser.add_argument("--otehv2_eps", type=float, default=0.05)
    parser.add_argument("--otehv2_iter", type=int, default=50)
    parser.add_argument("--otehv2_warmup", type=int, default=5)
    parser.add_argument("--otehv2_num_events", type=int, default=24)
    parser.add_argument("--otehv2_heads", type=int, default=4)
    parser.add_argument("--otehv2_layers", type=int, default=4)
    parser.add_argument("--otehv2_dropout", type=float, default=0.1)
    parser.add_argument("--lambda_otehv2_ot", type=float, default=0.06)
    parser.add_argument("--lambda_otehv2_div", type=float, default=0.01)
    parser.add_argument("--lambda_otehv2_event_surv", type=float, default=0.25)
    parser.add_argument("--lambda_otehv2_recon", type=float, default=0.2)

    parser.add_argument("--lambda_rankevent_per_event", type=float, default=0.15)
    parser.add_argument("--lambda_rankevent_rank", type=float, default=0.15)
    parser.add_argument("--lambda_rankevent_global_cons", type=float, default=0.02)
    parser.add_argument("--lambda_rankevent_gate_ent", type=float, default=0.005)
    parser.add_argument("--rankevent_eps_start", type=float, default=0.10)
    parser.add_argument("--rankevent_eps_end", type=float, default=0.05)
    parser.add_argument("--rankevent_eps_anneal_epochs", type=int, default=12)
    parser.add_argument("--rankevent_global_init", type=float, default=-2.0)
    parser.add_argument("--rankevent_dropout", type=float, default=0.1)
    parser.add_argument("--rankevent_rank_margin", type=float, default=0.0)
    parser.add_argument("--rankevent_rank_max_pairs", type=int, default=4096)

    # OTEHV2RankEventV2 新增能力配置（默认全部关闭/退化为 V45 行为）。
    # 三模态融合（临床模态）。
    parser.add_argument("--otehv2v2_use_clinical", action="store_true", default=False)
    parser.add_argument("--otehv2v2_clinical_feature_dim", type=int, default=0)
    parser.add_argument("--otehv2v2_num_slots_clinical", type=int, default=8)
    # 统一生存目标（Unified Objective）。
    parser.add_argument("--otehv2v2_use_unified_objective", action="store_true", default=False)
    parser.add_argument("--lambda_unified_rank", type=float, default=0.15)
    # 可学习自适应损失加权（Kendall 2018 同方差不确定性加权）。开启后用可学习对数
    # 方差替代人工固定 lambda 配平多项损失；默认关闭，等价于 V45 的固定权重路径。
    parser.add_argument("--otehv2v2_learnable_loss_weights", action="store_true", default=False)

    # OTEHTimeLocalCompeting (V50) 专属超参数。骨架超参数 (otehv2_* / rankevent_*)
    # 继承自 V45，含义不变，无需重复声明。
    parser.add_argument("--lambda_timelocal_spec", type=float, default=0.01)
    parser.add_argument("--lambda_timelocal_cover", type=float, default=0.01)
    parser.add_argument("--lambda_compete_reg", type=float, default=0.001)
    parser.add_argument("--compete_beta_init", type=float, default=-2.0)
    # Slot 身份/状态解耦与路由机制重设计。
    parser.add_argument("--otehv2v2_slot_disentangled", action="store_true", default=False)
    parser.add_argument("--otehv2v2_slot_router", type=str, default="softmax", choices=["softmax", "sinkhorn"])
    parser.add_argument("--otehv2v2_slot_cross_modal_cond", action="store_true", default=False)
    parser.add_argument("--otehv2v2_slot_adaptive_iters", action="store_true", default=False)
    parser.add_argument("--otehv2v2_sinkhorn_max_iters", type=int, default=20)
    parser.add_argument("--otehv2v2_convergence_threshold", type=float, default=0.0)

    # Rank-guided event transport method.
    parser.add_argument("--rg_num_events", type=int, default=4)
    parser.add_argument("--rg_prog_cost", type=float, default=0.20)
    parser.add_argument("--rg_lambda_ot", type=float, default=0.06)
    parser.add_argument("--rg_lambda_rank", type=float, default=0.15)
    parser.add_argument("--rg_lambda_stage", type=float, default=0.02)
    parser.add_argument("--rg_rank_margin", type=float, default=0.0)
    parser.add_argument("--rg_rank_max_pairs", type=int, default=4096)
    parser.add_argument("--rg_stage_margin", type=float, default=0.25)
    parser.add_argument("--rg_eps_start", type=float, default=0.10)
    parser.add_argument("--rg_eps_anneal", type=int, default=12)

    # Stagewise prognostic transport method.
    parser.add_argument("--spt_num_stages", type=int, default=4)
    parser.add_argument("--spt_prog_cost", type=float, default=0.20)
    parser.add_argument("--spt_lambda_ot", type=float, default=0.06)
    parser.add_argument("--spt_lambda_rank", type=float, default=0.05)
    parser.add_argument("--spt_lambda_stage", type=float, default=0.02)
    parser.add_argument("--spt_stage_margin", type=float, default=0.25)

    # Faithful evidence transport method.
    parser.add_argument("--fet_num_stages", type=int, default=4)
    parser.add_argument("--fet_lambda_sparse", type=float, default=0.01)
    parser.add_argument("--fet_lambda_faith", type=float, default=0.05)
    parser.add_argument("--fet_keep_ratio", type=float, default=0.25)
    parser.add_argument("--fet_faith_margin", type=float, default=0.05)

    # Distributional counterfactual transport method.
    parser.add_argument("--dct_num_stages", type=int, default=4)
    parser.add_argument("--dct_lambda_ipcw_rank", type=float, default=0.10)
    parser.add_argument("--dct_ipcw_rank_margin", type=float, default=0.02)
    parser.add_argument("--dct_ipcw_rank_temperature", type=float, default=0.50)
    parser.add_argument("--dct_ipcw_max_weight", type=float, default=10.0)
    parser.add_argument(
        "--dct_ipcw_rank_memory_size",
        type=int,
        default=0,
        help="Within-epoch detached risk memory used to provide IPCW pairs beyond one micro-batch; 0 disables it.",
    )
    # Legacy auxiliary objectives are opt-in ablations. The score-first recipe
    # intentionally does not optimise transport energy or duplicate rank losses.
    parser.add_argument("--dct_lambda_ot", type=float, default=0.0)
    parser.add_argument("--dct_lambda_rank", type=float, default=0.0)
    parser.add_argument("--dct_lambda_anchor", type=float, default=0.0)
    parser.add_argument("--dct_lambda_stage_risk", type=float, default=0.0)
    parser.add_argument("--dct_stage_risk_margin", type=float, default=0.02)
    parser.add_argument("--dct_anchor_margin", type=float, default=0.02)
    parser.add_argument("--dct_anchor_momentum", type=float, default=0.90)
    parser.add_argument("--dct_evidence_cost_weight", type=float, default=0.0)
    parser.add_argument("--dct_evidence_mass_floor", type=float, default=0.05)
    parser.add_argument("--dct_coupling_projection_iters", type=int, default=1000)
    parser.add_argument("--dct_coupling_projection_tol", type=float, default=1e-4)
    parser.add_argument("--dct_lambda_coordinate", type=float, default=0.0)
    parser.add_argument("--dct_coordinate_temperature", type=float, default=0.30)
    parser.add_argument("--dct_mix_ratio", type=float, default=0.50)
    parser.add_argument(
        "--dct_slot_init_mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "deterministic", "learned"],
        help=(
            "gaussian preserves the legacy stochastic evaluation; deterministic "
            "uses fixed distinct evaluation slots; learned uses per-slot queries."
        ),
    )
    parser.add_argument("--dct_slot_eval_seed", type=int, default=1729)
    parser.add_argument(
        "--dct_evidence_marginal_strength",
        type=float,
        default=1.0,
        help="Mix evidence-conditioned OT marginals with uniform mass; 1 is legacy, 0 is uniform.",
    )
    parser.add_argument(
        "--dct_geometry_reliability_strength",
        type=float,
        default=0.0,
        help=(
            "RTEM diagnostic: temper evidence-conditioned marginals using agreement "
            "among cosine/euclidean/dot OT geometries; 0 exactly preserves legacy DCT."
        ),
    )
    parser.add_argument(
        "--dct_geometry_reliability_temperature",
        type=float,
        default=0.25,
        help="Softmax temperature used to estimate cross-geometry edge agreement.",
    )

    # Censoring-aware temporal evidence transport mainline.
    parser.add_argument("--catet_num_stages", type=int, default=4)
    parser.add_argument("--catet_prog_cost", type=float, default=0.20)
    parser.add_argument("--catet_lambda_ot", type=float, default=0.04)
    parser.add_argument("--catet_lambda_rank", type=float, default=0.08)
    parser.add_argument("--catet_lambda_intervention", type=float, default=0.05)
    parser.add_argument("--catet_keep_ratio", type=float, default=0.25)
    parser.add_argument("--catet_intervention_margin", type=float, default=0.05)
    parser.add_argument("--catet_rank_margin", type=float, default=0.0)
    parser.add_argument("--catet_rank_max_pairs", type=int, default=4096)

    # V60 OT Event Rank method.
    parser.add_argument("--v60_num_events", type=int, default=24)
    parser.add_argument("--v60_lambda_per_event", type=float, default=0.15)
    parser.add_argument("--v60_lambda_rank", type=float, default=0.15)
    parser.add_argument("--v60_rank_margin", type=float, default=0.0)
    parser.add_argument("--v60_rank_max_pairs", type=int, default=4096)

    # Cohort-Anchored Adaptive Prognostic Slot Attention (CA-PSA).
    parser.add_argument("--capsa_max_slots", type=int, default=16)
    parser.add_argument("--capsa_slot_iters", type=int, default=3)
    parser.add_argument("--capsa_heads", type=int, default=4)
    parser.add_argument("--capsa_dropout", type=float, default=0.15)
    parser.add_argument("--capsa_gate_temperature", type=float, default=2.0 / 3.0)
    parser.add_argument("--capsa_gate_gamma", type=float, default=-0.1)
    parser.add_argument("--capsa_gate_zeta", type=float, default=1.1)
    parser.add_argument("--capsa_gate_threshold", type=float, default=0.5)
    parser.add_argument("--capsa_gate_prior_start", type=float, default=-1.0)
    parser.add_argument("--capsa_gate_prior_end", type=float, default=-2.2)
    parser.add_argument("--capsa_lambda_sparse", type=float, default=0.01)
    parser.add_argument("--capsa_lambda_align", type=float, default=0.02)

    # V70 Patient-Specific Prognostic Circuits (PSPC-Surv).
    parser.add_argument("--pspc_max_modules", type=int, default=16)
    parser.add_argument("--pspc_heads", type=int, default=4)
    parser.add_argument("--pspc_layers", type=int, default=3)
    parser.add_argument("--pspc_dropout", type=float, default=0.15)
    parser.add_argument("--pspc_gate_temperature", type=float, default=2.0 / 3.0)
    parser.add_argument("--pspc_gate_gamma", type=float, default=-0.1)
    parser.add_argument("--pspc_gate_zeta", type=float, default=1.1)
    parser.add_argument("--pspc_gate_threshold", type=float, default=0.5)
    parser.add_argument("--pspc_edge_temperature", type=float, default=0.75)
    parser.add_argument("--pspc_edge_threshold", type=float, default=0.5)
    parser.add_argument("--pspc_edge_rank", type=int, default=4)
    parser.add_argument("--pspc_lambda_node_sparse", type=float, default=0.01)
    parser.add_argument("--pspc_lambda_edge_sparse", type=float, default=0.005)

    return parser


def process_args_extended(argv=None):
    parser = build_base_parser()
    args = parser.parse_args(argv)
    args.survot_method = args.survot_method or args.newslot_method or "otehv2_rankevent"
    args.newslot_method = args.survot_method
    return args
