#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument parser for the cleaned SurvOT-Rank training path."""

from __future__ import annotations

import argparse


from survot_rank.research.methods.catalog import METHOD_CHOICES


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
    parser.add_argument(
        "--on_missing_wsi",
        type=str,
        default="error",
        choices=["error", "zero"],
        help=(
            "缺失 WSI 特征时的处理：error（默认，拒绝运行，防止零填充污染，"
            "如 BRCA 在 UNI2-h 下仅 74% 覆盖）；zero（旧行为，静默零填充，不推荐）。"
        ),
    )
    parser.add_argument(
        "--wsi_encoder",
        type=str,
        default="uni",
        choices=["uni", "uni2-h", "gigap", "r50", "chief"],
    )
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
    parser.add_argument("--wsi_missing", action="store_true", default=False)
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
        "--binning_mode",
        type=str,
        default="global_qcut",
        choices=["global_qcut", "legacy_equal_width"],
        help=(
            "global_qcut (default): equal-frequency qcut on all uncensored data. "
            "legacy_equal_width: original SlotSPE equal-width pd.cut(bins=4)."
        ),
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
    parser.add_argument("--dct_lambda_etar", type=float, default=0.0)
    parser.add_argument("--dct_etar_margin", type=float, default=0.02)
    parser.add_argument("--dct_etar_uncertainty_weight", type=float, default=0.05)
    parser.add_argument("--dct_etar_temperature", type=float, default=0.50)
    parser.add_argument("--dct_etar_evidence_floor", type=float, default=0.10)
    parser.add_argument("--dct_anchor_momentum", type=float, default=0.90)
    parser.add_argument("--dct_evidence_cost_weight", type=float, default=0.0)
    parser.add_argument("--dct_evidence_mass_floor", type=float, default=0.05)
    parser.add_argument("--dct_coupling_projection_iters", type=int, default=1000)
    parser.add_argument("--dct_coupling_projection_tol", type=float, default=1e-4)
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
    parser.add_argument(
        "--dct_listwise_mode",
        type=str,
        default="stage_transport",
        choices=["global", "stage_transport"],
        help=(
            "DCT v3.6 listwise channel: final factual risk (global/GPL) or "
            "the event-time stage's factual transport representation (TCL)."
        ),
    )
    parser.add_argument("--dct_lambda_listwise", type=float, default=0.10)
    parser.add_argument("--dct_listwise_temperature", type=float, default=0.50)
    parser.add_argument("--dct_listwise_memory_size", type=int, default=64)
    parser.add_argument(
        "--dct_listwise_tie_method",
        type=str,
        default="breslow",
        choices=["breslow"],
    )
    # DCT v3.8.3: v3.8 干预一致性损失 + 去塌缩的中心化运输几何。与 v3.8 的
    # 唯一区别是 slot 编码方式，三个损失与超参全部继承 v3.8。
    parser.add_argument(
        "--dct_v383_center_slots",
        type=lambda value: str(value).lower() not in {"0", "false", "no"},
        default=True,
        help=(
            "跨 slot 中心化，移除共模分量。关闭它即退回 v3.8 的塌缩行为，"
            "作为单变量对照。"
        ),
    )
    parser.add_argument(
        "--dct_v383_keep_legacy_slot_init",
        action="store_true",
        help="保留 v3.8 的高斯 slot 初始化，仅用于消融对照。",
    )

    # DCT v3.9 risk-simplex transport. 预测被定义为低危/高危锚定运输几何之间
    # 的坐标，方向一致性与剂量单调性由参数化保证，因此不需要 v3.8 的 margin 项。
    parser.add_argument(
        "--dct_v39_center_slots",
        type=lambda value: str(value).lower() not in {"0", "false", "no"},
        default=True,
        help=(
            "跨 slot 中心化，移除共模分量。关闭它可复现 v3.3 的塌缩行为，"
            "作为消融对照使用。"
        ),
    )
    parser.add_argument(
        "--dct_v39_residual_scale",
        type=float,
        default=0.0,
        help=(
            "可选残差旁路权重。0 表示预测严格落在锚定 hazard 的凸包内；"
            "调大用于'结构约束 vs 模型容量'的消融。"
        ),
    )
    parser.add_argument(
        "--dct_v39_tau_init",
        type=float,
        default=0.02,
        help=(
            "锚点代价差的初始温度（可学习）。实测该代价差尺度约 0.01，温度取 "
            "0.25 会让 lambda 挤在 0.5 附近导致学不动。"
        ),
    )
    parser.add_argument(
        "--dct_v39_tau_autoscale",
        type=lambda value: str(value).lower() not in {"0", "false", "no"},
        default=True,
        help=(
            "用首个可用 batch 的代价差标准差自动标定温度，消除尺度错配这一类"
            "失效模式；关闭后使用 dct_v39_tau_init。"
        ),
    )
    parser.add_argument(
        "--dct_v39_anchor_freeze_epoch",
        type=int,
        default=0,
        help=(
            "到该 epoch 后冻结队列锚点，避免坐标追逐移动目标；0 表示始终用 EMA 更新。"
        ),
    )
    parser.add_argument(
        "--dct_v39_lambda_spread_target",
        type=float,
        default=0.0,
        help=(
            "坐标铺开项的目标方差，只惩罚所有患者挤在同一 lambda 的退化解，"
            "不规定任何方向或次序。0 表示目标严格等于 v3.3 的两项。"
        ),
    )
    parser.add_argument(
        "--dct_v39_projection_iters",
        type=int,
        default=3,
        help="边缘投影迭代次数；log-domain Sinkhorn 已收敛，这里只做数值兜底。",
    )
    parser.add_argument(
        "--dct_v39_keep_legacy_slot_init",
        action="store_true",
        help="保留 v3.3 的高斯 slot 初始化，仅用于消融对照。",
    )

    # DCT v3.8 transport-intervention consistency. Each term is independently
    # switchable so direction, dose response, and coupling reconfiguration can
    # be ablated without changing the v3.3 factual path.
    parser.add_argument("--dct_v38_lambda_direction", type=float, default=0.05)
    parser.add_argument("--dct_v38_lambda_dose", type=float, default=0.03)
    parser.add_argument(
        "--dct_v38_lambda_reconfiguration", type=float, default=0.02
    )
    parser.add_argument("--dct_v38_direction_margin", type=float, default=0.02)
    parser.add_argument("--dct_v38_dose_margin", type=float, default=0.005)
    parser.add_argument(
        "--dct_v38_reconfiguration_margin", type=float, default=0.02
    )
    parser.add_argument("--dct_v38_temperature", type=float, default=0.05)
    parser.add_argument("--dct_v38_alpha_mid", type=float, default=0.50)
    parser.add_argument("--dct_v38_alpha_full", type=float, default=1.00)
    parser.add_argument("--dct_v38_warmup_epochs", type=int, default=1)
    parser.add_argument(
        "--dct_v38_ramp_epochs",
        type=int,
        default=0,
        help=(
            "Linearly ramp all v3.8 structural-loss weights after warmup. "
            "Zero preserves the historical immediate full-weight behavior."
        ),
    )
    parser.add_argument(
        "--dct_v38_dose_every",
        type=int,
        default=1,
        help="Evaluate the extra midpoint Sinkhorn branches every N post-warmup epochs.",
    )
    # DCT v3.8.2 Multi-Geometry Prognostic Transport Reconstruction. The loss
    # reuses factual couplings and the shared decoder, so it adds no parameters
    # and no additional Sinkhorn solves.
    parser.add_argument("--dct_v382_lambda_mgptr", type=float, default=0.05)
    parser.add_argument("--dct_v382_distill_weight", type=float, default=0.50)
    parser.add_argument("--dct_v382_warmup_epochs", type=int, default=1)
    parser.add_argument("--dct_v382_ramp_epochs", type=int, default=4)
    parser.add_argument(
        "--dct_v382_adaptive_aux_weights", action="store_true", default=False
    )
    parser.add_argument(
        "--dct_v382_adaptive_prior_fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--dct_v382_adaptive_temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--dct_v382_adaptive_kl_strength", type=float, default=0.01
    )

    # Ablation switches shared across DCT variants.
    parser.add_argument(
        "--dct_perm_labels_seed", type=int, default=0,
        help=(
            "Permute event times before stage fitting (seed > 0 triggers null calibration). "
            "Used in the Targeted Null experiment."
        ),
    )
    parser.add_argument(
        "--dct_random_anchors", action="store_true", default=False,
        help="Use random anchor costs instead of learned ones (ablation).",
    )
    parser.add_argument(
        "--dct_fixed_coupling", action="store_true", default=False,
        help="Fix coupling to uniform distribution (ablation).",
    )
    parser.add_argument(
        "--dct_stage_jitter_fraction", type=float, default=0.0,
        help="Jitter stage edges by this fraction of the total span (ablation).",
    )

    # DCT v4.1 Survival-Evidence Ledger (SELC). This method replaces the
    # inherited slot mechanism while retaining the verified v3.3 DCT path.
    parser.add_argument("--v41_modality_dropout", type=float, default=0.35)
    parser.add_argument("--v41_ledger_temperature", type=float, default=0.25)
    parser.add_argument(
        "--v41_missing_confidence_cap", type=float, default=0.65
    )
    parser.add_argument("--v41_confidence_floor", type=float, default=0.05)
    parser.add_argument("--v41_lambda_completion", type=float, default=0.05)
    parser.add_argument("--v41_lambda_ledger", type=float, default=0.02)
    parser.add_argument("--v41_lambda_survival", type=float, default=0.05)
    parser.add_argument("--v41_lambda_private", type=float, default=0.02)
    parser.add_argument("--v41_shared_rank", type=int, default=64)
    parser.add_argument(
        "--v41_min_log_variance",
        type=float,
        default=-4.0,
        help=(
            "补全损失的 log-variance 下界。高斯 NLL 在方差无约束时下界为负无穷，"
            "而补全目标是模型自身的 detached 账本表示，误差易被压到 0，"
            "方差项因此成为免费的下降通道（实测补全项降到 -1.3 ~ -1.9，"
            "使总目标与训练损失变负）。取 0 表示固定方差。"
        ),
    )
    # v4.1 账本辅助损失与模态删除的分阶段激活。四项损失 + modality dropout
    # 从第一轮就与生存似然竞争：BLCA fold2 最佳 C-index 出现在 epoch 3 后持续走低。
    parser.add_argument(
        "--v41_warmup_epochs",
        type=int,
        default=5,
        help="前 N 轮关闭 completion/ledger/survival/private 及模态删除；0 = 旧行为",
    )
    parser.add_argument(
        "--v41_ramp_epochs",
        type=int,
        default=10,
        help="warmup 之后用多少轮线性拉到满权重；0 = 立即满权重",
    )

    # V4.0 intervention-stable survival transport.  Unlike the DCT evolution,
    # this is an independent raw patch-pathway architecture whose hazard logits
    # exactly decompose into signed transport-edge contributions.
    parser.add_argument("--ist_eps", type=float, default=0.05)
    parser.add_argument("--ist_sinkhorn_iters", type=int, default=30)
    parser.add_argument("--ist_num_interventions", type=int, default=2)
    parser.add_argument("--ist_keep_ratio", type=float, default=0.75)
    parser.add_argument("--ist_stability_beta", type=float, default=1.0)
    parser.add_argument("--ist_stability_strength", type=float, default=0.10)
    parser.add_argument(
        "--ist_stability_normalization",
        choices=("raw_mass", "independence_lift"),
        default="raw_mass",
        help=(
            "raw_mass reproduces v4.0; independence_lift removes deterministic "
            "support-marginal rescaling before measuring edge stability"
        ),
    )
    parser.add_argument(
        "--ist_feedback_mode",
        choices=("legacy_product", "importance_weighted_instability"),
        default="legacy_product",
        help=(
            "legacy_product reproduces v4.0; importance_weighted_instability "
            "penalizes only important edges that are intervention-unstable"
        ),
    )
    parser.add_argument("--ist_lambda_plan", type=float, default=0.05)
    parser.add_argument("--ist_lambda_attribution", type=float, default=0.05)
    parser.add_argument("--ist_lambda_risk", type=float, default=0.0)
    parser.add_argument("--ist_edge_value_scale", type=float, default=4.0)
    parser.add_argument("--ist_eval_seed", type=int, default=20260725)
    # v4.0 干预稳定性的分阶段激活。稳定性项会改写 stable_cost（进而改写用于
    # 预测的运输计划），从第一轮生效会在生存头学到东西之前压制它：
    # BLCA fold1 最佳 C-index 出现在 epoch 0，随后 29 轮持续下滑。
    parser.add_argument(
        "--ist_warmup_epochs",
        type=int,
        default=5,
        help="前 N 轮完全关闭干预稳定性（成本项与三个辅助损失）；0 = 旧行为",
    )
    parser.add_argument(
        "--ist_ramp_epochs",
        type=int,
        default=10,
        help="warmup 之后用多少轮线性拉到满权重；0 = 立即满权重",
    )
    parser.add_argument("--ist_deletion_penalty", type=float, default=8.0)

    # Censoring-aware temporal evidence transport mainline.
    parser.add_argument("--catet_num_stages", type=int, default=4)
    parser.add_argument("--catet_prog_cost", type=float, default=0.20)
    parser.add_argument("--catet_lambda_ot", type=float, default=0.04)
    parser.add_argument("--catet_lambda_rank", type=float, default=0.08)
    parser.add_argument("--catet_lambda_stage", type=float, default=0.04)
    parser.add_argument("--catet_lambda_intervention", type=float, default=0.05)
    parser.add_argument("--catet_keep_ratio", type=float, default=0.25)
    parser.add_argument("--catet_intervention_margin", type=float, default=0.05)
    parser.add_argument("--catet_intervention_cost", type=float, default=1.0)
    parser.add_argument("--catet_plan_diversity_margin", type=float, default=0.01)
    parser.add_argument("--catet_rank_margin", type=float, default=0.0)
    parser.add_argument("--catet_rank_temperature", type=float, default=0.50)
    parser.add_argument("--catet_ipcw_max_weight", type=float, default=10.0)
    parser.add_argument("--catet_rank_max_pairs", type=int, default=4096)
    # v2 (three_method_final_2026_08_13): cohort-anchored pre-routing.
    parser.add_argument(
        "--catet_cohort_routes",
        type=int,
        default=4,
        help="v2 cohort routing slots — OT now operates on routes×routes.",
    )
    parser.add_argument(
        "--catet_cohort_topk",
        type=int,
        default=2,
        help="v2 top-K active routes per slot in cohort router.",
    )
    parser.add_argument(
        "--catet_lambda_route",
        type=float,
        default=0.02,
        help="v2 KL weight between wsi/omic cohort-routing soft assignments.",
    )
    parser.add_argument(
        "--catet_use_archetype_prior",
        type=int,
        default=0,
        choices=(0, 1),
        help="v2 enable lazy archetype-derived per-stage OT bias.",
    )

    # V60 OT Event Rank method.
    parser.add_argument("--v60_num_events", type=int, default=24)
    parser.add_argument("--v60_lambda_per_event", type=float, default=0.15)
    parser.add_argument("--v60_lambda_rank", type=float, default=0.15)
    parser.add_argument("--v60_rank_margin", type=float, default=0.0)
    parser.add_argument("--v60_rank_max_pairs", type=int, default=4096)

    # ArcSurv cohort-level archetypal risk composition.
    parser.add_argument("--arc_num_archetypes", type=int, default=6)
    parser.add_argument("--arc_bank_size", type=int, default=256)
    parser.add_argument("--arc_temperature", type=float, default=0.25)
    parser.add_argument("--arc_beta_init_scale", type=float, default=1.5)
    parser.add_argument("--arc_lambda_recon", type=float, default=0.05)
    parser.add_argument("--arc_lambda_align", type=float, default=0.05)
    # v2 (three_method_final_2026_08_13): balanced OT re-transport + hard gate.
    parser.add_argument(
        "--arc_lambda_ot",
        type=float,
        default=0.04,
        help="v2 balanced Sinkhorn between wsi/omic archetype compositions.",
    )
    parser.add_argument(
        "--arc_lambda_gate",
        type=float,
        default=0.01,
        help="v2 gate budget loss keeps per-patient active archetype count near topk_active.",
    )
    parser.add_argument(
        "--arc_topk_active",
        type=int,
        default=3,
        help="v2 hard top-K active archetypes per patient composition.",
    )
    parser.add_argument(
        "--arc_ot_eps",
        type=float,
        default=0.05,
        help="v2 OT epsilon for the cross-modal re-transport Sinkhorn.",
    )
    parser.add_argument(
        "--arc_ot_iters",
        type=int,
        default=25,
        help="v2 Sinkhorn max iterations for cross-modal re-transport.",
    )
    parser.add_argument("--arc_lambda_balance", type=float, default=0.01)
    parser.add_argument("--arc_lambda_volume", type=float, default=0.01)
    parser.add_argument("--arc_lambda_rank", type=float, default=0.10)
    parser.add_argument("--arc_rank_margin", type=float, default=0.0)
    parser.add_argument("--arc_rank_max_pairs", type=int, default=4096)
    parser.add_argument("--arc_seed_anchors", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--arc_freeze_state_encoder", type=int, choices=(0, 1), default=1
    )
    # ArcSurv 分阶段激活四项结构损失（rank 与 NLL 同期生效，不走 ramp）。
    # BLCA fold1 最佳 C-index 出现在 epoch 29 且最后 5 轮仍在上升 = 欠训练。
    parser.add_argument(
        "--arc_warmup_epochs",
        type=int,
        default=5,
        help="前 N 轮只训 NLL + rank，关闭 recon/align/balance/volume；0 = 旧行为",
    )
    parser.add_argument(
        "--arc_ramp_epochs",
        type=int,
        default=10,
        help="warmup 之后用多少轮线性拉到满权重；0 = 立即满权重",
    )
    # v4.2 ACT-Surv：hazard = archetype hazard 曲线在运输质量坐标下的凸组合。
    # 可加归因与删除反事实都是凸组合的推论，因此没有对应的辅助损失。
    parser.add_argument("--act_num_archetypes", type=int, default=6)
    parser.add_argument(
        "--act_epsilon",
        type=float,
        default=0.10,
        help="行约束熵正则运输的温度；越小分配越尖锐",
    )
    parser.add_argument(
        "--act_lambda_balance",
        type=float,
        default=0.01,
        help="唯一的辅助损失：批内平均 β 对均匀分布的 KL，抑制 archetype 塌缩",
    )
    parser.add_argument("--act_lambda_rank", type=float, default=0.10)
    parser.add_argument("--act_rank_margin", type=float, default=0.0)
    parser.add_argument("--act_rank_max_pairs", type=int, default=4096)
    parser.add_argument(
        "--act_hazard_scale",
        type=float,
        default=1.0,
        help="archetype hazard 曲线的整体尺度；凸包半径由它控制",
    )
    parser.add_argument(
        "--arc_bank_update_epochs",
        type=int,
        default=-1,
        help=(
            "原型库更新轮数，-1 表示跟随 arc_warmup_epochs。"
            "原实现只在 epoch 0 建库并冻结，此时编码器尚未被生存目标塑形。"
        ),
    )
    # ArcSurv 原型使用塌缩修复。实测组合熵 ≈ ln(6) = 1.7918、患者间组合方差
    # ≈ 1e-4，即几乎所有患者都均匀使用全部原型，凸组合退化为常向量。
    parser.add_argument(
        "--arc_distance_reduction",
        choices=("mean", "scaled"),
        default="scaled",
        help=(
            "patient-archetype 距离的归一方式。mean = 旧行为（对 dim 取均值，"
            "把距离量级压掉 dim 倍，softmax 必然接近均匀）；"
            "scaled = 按 sqrt(dim) 归一，使尺度不随投影维度塌缩。"
        ),
    )
    parser.add_argument(
        "--arc_anchor_logit",
        type=float,
        default=6.0,
        help=(
            "memory 冻结后对每个原型做 furthest-point 锚定时加在该锚点上的 "
            "logit。0 = 关闭锚定，退回纯随机 beta 初始化（会重现塌缩）。"
        ),
    )
    parser.add_argument(
        "--arc_lambda_sharpness",
        type=float,
        default=0.0,
        help=(
            "个体 composition 熵惩罚。balance 只把批次平均推向均匀，"
            "此前没有任何一项奖励单个患者的组合变尖；0 = 旧行为。"
        ),
    )

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
    parser.add_argument("--capsa_lambda_budget", type=float, default=0.01)
    parser.add_argument("--capsa_lambda_identity", type=float, default=0.02)
    parser.add_argument("--capsa_target_active_ratio", type=float, default=0.25)
    parser.add_argument("--capsa_identity_temperature", type=float, default=0.10)
    parser.add_argument("--capsa_anchor_cosine_margin", type=float, default=0.20)
    parser.add_argument("--capsa_anchor_scale", type=float, default=0.50)
    # v2 (three_method_final_2026_08_13): cohort-archetype anchors.
    parser.add_argument(
        "--capsa_archetype_bank_size",
        type=int,
        default=256,
        help="v2 cohort memory bank size — slot anchors are convex combinations of this bank.",
    )
    parser.add_argument(
        "--capsa_archetype_beta_init_scale",
        type=float,
        default=1.5,
        help="v2 init scale for the Beta row-stochastic convex-combination logits.",
    )
    parser.add_argument(
        "--capsa_lambda_archetypal_recon",
        type=float,
        default=0.02,
        help="v2 reconstruction loss: archetypes should reconstruct patient state.",
    )

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

    # ACT-Surv v5 (archetypal_transport_composition_v5) — minimal-rewrite flags.
    parser.add_argument("--act5_num_archetypes", type=int, default=6)
    parser.add_argument("--act5_epsilon", type=float, default=0.10)
    parser.add_argument("--act5_hazard_scale", type=float, default=1.0)
    parser.add_argument("--act5_warmup_epochs", type=int, default=5)
    parser.add_argument("--act5_lambda_balance", type=float, default=0.01)
    parser.add_argument("--act5_lambda_rank", type=float, default=0.10)
    parser.add_argument("--act5_rank_margin", type=float, default=0.02)
    parser.add_argument("--act5_rank_temperature", type=float, default=0.50)
    parser.add_argument("--act5_rank_max_pairs", type=int, default=4096)

    return parser


def process_args_extended(argv=None):
    parser = build_base_parser()
    args = parser.parse_args(argv)
    args.survot_method = args.survot_method or args.newslot_method or "otehv2_rankevent"
    args.newslot_method = args.survot_method
    return args
