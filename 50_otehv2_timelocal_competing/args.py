#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@file: args.py（50_otehv2_timelocal_competing 独立参数解析器）
@desc: 只保留：
       1. 训练必需的基础参数（study/data/split/training/slot 等）
       2. 灵活停止训练参数（早停 + 手动中断安全落盘）
       3. V45 骨架超参数（otehv2_* + rankevent_*，本方法继承复用）
       4. V50 本方法专属超参数（timelocal_* + compete_*）

       本文件只服务于 timelocal_competing 这一个模型。

用法:
    from args import process_args
    args = process_args()
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="45_otehv2_rankevent (V45) standalone training")

    # ---- study / data ----
    parser.add_argument("--study", type=str, default="blca")
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--specific_simple", default="")
    parser.add_argument("--data_root_dir", type=str, default="")
    parser.add_argument("--data_path", type=str, default="./dataset_csv")
    parser.add_argument("--num_patches", type=int, default=4096)
    parser.add_argument("--num_genes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4,
                         help="Number of workers for data loading (0=single thread)")
    parser.add_argument("--label_col", type=str, default="survival_months_dss")
    parser.add_argument("--rna_format", type=str, default="Pathways",
                         choices=["RNASeq", "Pathways", "GeneEmbedding"])
    parser.add_argument("--signature", type=str, default="combine",
                         choices=["all", "six", "hallmarks", "combine", "xena"])

    # ---- split ----
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k_start", type=int, default=-1)
    parser.add_argument("--k_end", type=int, default=-1)
    parser.add_argument("--which_splits", type=str, default="5fold")

    # ---- training ----
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bag_loss", type=str, default="nll_surv",
                         choices=["nll_surv", "rank_surv", "cox_surv", "sinkhorn_surv"])
    parser.add_argument("--alpha_surv", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--max_cindex", type=float, default=0.0)

    # ---- model (SlotSPE 通用字段, model_factory 里靠 --method 拼实验目录) ----
    parser.add_argument("--method", type=str, default="SlotSPE_otehv2_timelocal_competing",
                         help="保留兼容字段，仅用于结果目录命名")
    parser.add_argument("--newslot_method", type=str, default="otehv2_timelocal_competing",
                         help="固定为本文件夹的方法名，train.py 只认这一个")
    parser.add_argument("--encoding_dim", type=int, default=1024)
    parser.add_argument("--wsi_projection_dim", type=int, default=256)

    # ---- loss ----
    parser.add_argument("--lambda_recon_loss", type=float, default=0.01)
    parser.add_argument("--lambda_aux_surv", type=float, default=1.0)

    # ---- scheduler ----
    parser.add_argument("--scheduler", type=str, default="cosine",
                         choices=["cosine", "step"])
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--step_size", type=int, default=10)

    # ---- gpu ----
    parser.add_argument("--gpu", type=str, default="0")

    # ---- test / misc ----
    parser.add_argument("--only_test", action="store_true", default=False)
    parser.add_argument("--omic_missing", action="store_true", default=False)
    parser.add_argument("--max_smoke_batches", type=int, default=0,
                         help="debug only: stop each train epoch after N batches when >0")
    parser.add_argument("--min_free_space_gb", type=float, default=2.0,
                         help="minimum free disk space required for results directory before writes")

    # ---- slot attention ----
    parser.add_argument("--slot_num_wsi", type=int, default=8)
    parser.add_argument("--slot_num_omics", type=int, default=8)
    parser.add_argument("--slot_iters", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--topk_ratio", type=float, default=0.25)
    parser.add_argument("--top_k_method", type=str, default="parallel_topk_st",
                         choices=["gumbel_topk_st", "parallel_topk_st"])

    # ---- 灵活停止训练（早停 + 手动中断安全落盘） ----
    parser.add_argument("--early_stop_patience", type=int, default=0,
                         help="验证指标连续 N 个 epoch 不提升则早停；0=关闭")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_metric", type=str, default="val_cindex",
                         choices=["val_cindex", "val_cindex_ipcw", "val_iauc"])
    parser.add_argument("--early_stop_warmup", type=int, default=0)

    # ================================================================
    # V45 (otehv2_rankevent) 专属超参数
    # ================================================================
    # 骨架参数（继承自 v9 强 OT 配置，OTEventHazardV2Survival 用）
    g_backbone = parser.add_argument_group("v45_backbone")
    g_backbone.add_argument("--otehv2_eps", type=float, default=0.05)
    g_backbone.add_argument("--otehv2_iter", type=int, default=50)
    g_backbone.add_argument("--otehv2_warmup", type=int, default=5)
    g_backbone.add_argument("--otehv2_num_events", type=int, default=24)
    g_backbone.add_argument("--otehv2_heads", type=int, default=4)
    g_backbone.add_argument("--otehv2_layers", type=int, default=4)
    g_backbone.add_argument("--otehv2_dropout", type=float, default=0.1)
    g_backbone.add_argument("--lambda_otehv2_ot", type=float, default=0.06)
    g_backbone.add_argument("--lambda_otehv2_div", type=float, default=0.01)
    g_backbone.add_argument("--lambda_otehv2_event_surv", type=float, default=0.25)
    g_backbone.add_argument("--lambda_otehv2_recon", type=float, default=0.2)

    # V45 自己新增的 4 个协同改进
    g_v45 = parser.add_argument_group("v45_rankevent")
    g_v45.add_argument("--lambda_rankevent_per_event", type=float, default=0.15,
                        help="per-event NLL 监督权重")
    g_v45.add_argument("--lambda_rankevent_rank", type=float, default=0.15,
                        help="Cox 风格成对排序损失权重")
    g_v45.add_argument("--lambda_rankevent_global_cons", type=float, default=0.02,
                        help="全局残差头与事件均值 logits 的一致性损失权重")
    g_v45.add_argument("--lambda_rankevent_gate_ent", type=float, default=0.005,
                        help="门控熵正则化权重（防止门控坍缩）")
    g_v45.add_argument("--rankevent_eps_start", type=float, default=0.10,
                        help="OT epsilon 退火起始值")
    g_v45.add_argument("--rankevent_eps_end", type=float, default=0.05,
                        help="OT epsilon 退火终止值")
    g_v45.add_argument("--rankevent_eps_anneal_epochs", type=int, default=12,
                        help="epsilon 退火所需 epoch 数")
    g_v45.add_argument("--rankevent_global_init", type=float, default=-2.0,
                        help="sigmoid 全局残差 scale 的初始 logit")
    g_v45.add_argument("--rankevent_dropout", type=float, default=0.1)
    g_v45.add_argument("--rankevent_rank_margin", type=float, default=0.0)
    g_v45.add_argument("--rankevent_rank_max_pairs", type=int, default=4096)

    # ================================================================
    # V50 (timelocal_competing) 专属超参数
    # ================================================================
    g_v50 = parser.add_argument_group("v50_timelocal_competing")
    g_v50.add_argument("--lambda_timelocal_spec", type=float, default=0.01,
                        help="时间特化正则：每个事件的时间责任分布应尖锐(低熵)")
    g_v50.add_argument("--lambda_timelocal_cover", type=float, default=0.01,
                        help="时间覆盖正则：所有事件的总责任应铺满时间轴(高熵)")
    g_v50.add_argument("--lambda_compete_reg", type=float, default=0.001,
                        help="竞争稳定正则：约束保护通路幅度，防止负贡献发散")
    g_v50.add_argument("--compete_beta_init", type=float, default=-2.0,
                        help="保护通路竞争强度 beta 的初始 logit (softplus 前)")

    return parser


def process_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # newslot_method 固定，防止被命令行意外改成别的模型名
    args.newslot_method = "otehv2_timelocal_competing"
    return args
