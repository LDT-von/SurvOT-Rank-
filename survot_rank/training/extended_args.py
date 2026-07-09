#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument parser for the cleaned SurvOT-Rank training path."""

from __future__ import annotations

import argparse


METHOD_CHOICES = [
    "otehv2_rankevent",
    "otehv2_rankevent_v2",
    "pet",
    "prognostic_event_transport",
    "ot_event_hazard_v2",
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
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--only_test", action="store_true", default=False)
    parser.add_argument("--omic_missing", action="store_true", default=False)
    parser.add_argument("--max_smoke_batches", type=int, default=0)
    parser.add_argument("--min_free_space_gb", type=float, default=2.0)

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
    # Slot 身份/状态解耦与路由机制重设计。
    parser.add_argument("--otehv2v2_slot_disentangled", action="store_true", default=False)
    parser.add_argument("--otehv2v2_slot_router", type=str, default="softmax", choices=["softmax", "sinkhorn"])
    parser.add_argument("--otehv2v2_slot_cross_modal_cond", action="store_true", default=False)
    parser.add_argument("--otehv2v2_slot_adaptive_iters", action="store_true", default=False)
    parser.add_argument("--otehv2v2_sinkhorn_max_iters", type=int, default=20)
    parser.add_argument("--otehv2v2_convergence_threshold", type=float, default=0.0)

    return parser


def process_args_extended(argv=None):
    parser = build_base_parser()
    args = parser.parse_args(argv)
    args.survot_method = args.survot_method or args.newslot_method or "otehv2_rankevent"
    args.newslot_method = args.survot_method
    return args
