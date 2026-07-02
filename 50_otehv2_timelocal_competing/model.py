#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V50: Time-local Competing Event Hazard Decomposition（时间局部竞争事件危险率分解）。

本文件夹 = 一个独立方法，单独训练、单独出结果，不与 V45 混。

在 V45 (OTEHV2RankEvent) 的 OT 诱导事件骨架之上做两个结构性创新
（不是已知 trick 的堆叠，而是改变了 事件 -> hazard 的建模机制）：

  创新 A —— 时间局部事件头 (Time-local event heads)
    不再让每个事件预测一整条完整 hazard 曲线，而是给每个事件在"每个离散
    生存时间 bin"上一个可学习责任权重；对每个时间 bin，事件之间相互竞争
    (softmax over events)，从而自发分化出"早期风险事件 / 中期事件 / 晚期事件"。
    配套两个正则：
      - 时间特化 (specialization)：每个事件在时间轴上的责任分布应尖锐(低熵)。
      - 时间覆盖 (coverage)：所有事件的总责任应铺满时间轴(高熵)，避免挤在同一 bin。

  创新 B —— 竞争性风险/保护门控 (Competing risk/protective gate)
    事件拆成"风险增强"和"风险保护"两条通路，各自有独立 hazard 头和时间局部门控。
    最终 hazard = 风险贡献 - beta * 保护贡献 (beta 可学习、softplus 非负)。
    这让模型能表达"某些事件降低风险"，医学上对应保护性因素，可解释性显著增强；
    配一个小 L2 稳定项防止保护通路发散。

预测主路仍不使用 SlotDecoder / CrossAttention trunk / SelfAttention trunk /
D*3 concat 分类器，继承 V45 "脱离 SlotSPE 主预测链"的性质。

论文卖点组合名："Time-Localized Optimal Transport Event Decomposition"。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from paths import ensure_slotspe_in_path  # noqa
import backbone as _parent  # noqa: 本文件夹自己的 OT 事件危险率骨架
from model_v45 import OTEHV2RankEvent  # noqa: 复用 V45 的排序/eps退火/全局残差机制

ensure_slotspe_in_path()
from utils.loss_func import NLLSurvLoss  # noqa


class OTEHTimeLocalCompeting(OTEHV2RankEvent):
    """V45 骨架 + 时间局部竞争事件 hazard 分解（本文件夹主模型）。"""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        # 新增超参数默认值（不覆盖命令行/args.py 已给的值）
        self._set_default(args, "lambda_timelocal_spec", 0.01)
        self._set_default(args, "lambda_timelocal_cover", 0.01)
        self._set_default(args, "lambda_compete_reg", 0.001)
        self._set_default(args, "compete_beta_init", -2.0)

        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        num_classes = self.num_classes

        # 创新 B：风险 / 保护 两条 hazard 头
        self.risk_hazard = nn.Linear(dim, num_classes)
        self.protect_hazard = nn.Linear(dim, num_classes)
        # 创新 A：每个事件在每个时间 bin 上的责任（风险 / 保护各一套）
        self.risk_time_gate = nn.Linear(dim, num_classes)
        self.protect_time_gate = nn.Linear(dim, num_classes)
        # 竞争强度 beta（softplus 保证非负）
        self.beta_protect = nn.Parameter(torch.tensor(float(args.compete_beta_init)))

        self.lambda_time_spec = float(args.lambda_timelocal_spec)
        self.lambda_time_cover = float(args.lambda_timelocal_cover)
        self.lambda_compete = float(args.lambda_compete_reg)

    # ------------------------------------------------------------------
    # 时间局部竞争事件 hazard 分解
    # ------------------------------------------------------------------
    def _make_timelocal_logits(self, event_tokens):
        """event_tokens: [B, E, D] -> 聚合成 [B, C] 的 hazard logits。"""
        risk_h = self.risk_hazard(event_tokens)             # [B, E, C]
        prot_h = self.protect_hazard(event_tokens)          # [B, E, C]
        risk_g_raw = self.risk_time_gate(event_tokens)      # [B, E, C]
        prot_g_raw = self.protect_time_gate(event_tokens)   # [B, E, C]

        # 时间局部竞争：对每个时间 bin，事件之间竞争责任 (softmax over events, dim=1)
        risk_g = torch.softmax(risk_g_raw, dim=1)           # [B, E, C]
        prot_g = torch.softmax(prot_g_raw, dim=1)           # [B, E, C]

        risk_logits = (risk_g * risk_h).sum(dim=1)          # [B, C]
        prot_logits = (prot_g * prot_h).sum(dim=1)          # [B, C]
        beta = F.softplus(self.beta_protect)
        gated_logits = risk_logits - beta * prot_logits     # [B, C]

        # 事件重要度（跨时间 bin 平均风险责任），用于全局残差池化 + 门控熵正则
        importance = risk_g.mean(dim=2)                     # [B, E], sum_e = 1
        global_feat = torch.einsum("be,bed->bd", importance, event_tokens)
        global_logits = self.global_head(global_feat)
        scale = torch.sigmoid(self.global_scale)
        logits = gated_logits + scale * global_logits

        # --- 时间局部相关正则 ---
        # 每个事件跨时间 bin 的责任分布 (softmax over bins, dim=2)
        time_prof = torch.softmax(risk_g_raw, dim=2)        # [B, E, C], sum_c = 1
        # 特化：责任分布应尖锐 -> 最小化其熵
        spec = -(time_prof.clamp_min(1e-8).log() * time_prof).sum(dim=2).mean()
        # 覆盖：所有事件的总责任应铺满时间轴 -> 最大化 per-bin 覆盖分布的熵
        cover_dist = time_prof.mean(dim=1)                  # [B, C]
        cover_ent = -(cover_dist.clamp_min(1e-8).log() * cover_dist).sum(dim=1).mean()
        cover = cover_dist.new_tensor(float(np.log(time_prof.shape[2]))) - cover_ent
        # 竞争稳定：约束保护通路幅度，防止负贡献发散
        compete = prot_logits.pow(2).mean()

        extra = {"spec": spec, "cover": cover, "compete": compete}
        return logits, risk_h, importance, global_logits, extra

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"]
        x_wsi_proj = self.wsi_mlp(x_wsi)
        x_omics = self._encode_omics(kwargs)

        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        cost_cos = _parent.cosine_cost(slots_wsi, slots_omic)
        cost_euc = _parent.euclidean_cost(slots_wsi, slots_omic)
        cost_dot = _parent.dot_cost(slots_wsi, slots_omic)

        eps = self._current_eps(kwargs)
        plan_cos, ot_dist_cos = _parent.log_sinkhorn_plan(cost_cos, eps=eps, max_iter=self.ot_iter)
        plan_euc, ot_dist_euc = _parent.log_sinkhorn_plan(cost_euc, eps=eps, max_iter=self.ot_iter)
        plan_dot, ot_dist_dot = _parent.log_sinkhorn_plan(cost_dot, eps=eps, max_iter=self.ot_iter)

        event_tokens, _ = self.fusion(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot)
        event_tokens = self.event_norm(self.event_encoder(event_tokens))

        logits, risk_h, importance, global_logits, extra = self._make_timelocal_logits(event_tokens)

        if not self.training:
            return logits, 0.0

        epoch = self._current_epoch(kwargs)
        ramp = 0.0 if epoch < self.ot_warmup else min(1.0, (epoch - self.ot_warmup) / max(1, self.ot_warmup))
        ot_mean = (ot_dist_cos + ot_dist_euc + ot_dist_dot).mean() / 3.0

        recon_wsi = self.recon_wsi(slots_wsi)
        recon_omic = self.recon_omic(slots_omic)
        recon_loss = F.mse_loss(recon_wsi, slots_omic) + F.mse_loss(recon_omic, slots_wsi)

        aux_loss = (
            ramp * self.lambda_ot * ot_mean
            + self.lambda_div * self._diversity_loss(event_tokens)
            + self.lambda_recon * recon_loss
            + self.lambda_gate_ent * self._gate_entropy_penalty(importance)
            + self.lambda_time_spec * extra["spec"]
            + self.lambda_time_cover * extra["cover"]
            + self.lambda_compete * extra["compete"]
        )

        if "y" in kwargs and "c" in kwargs:
            y, c = kwargs["y"], kwargs["c"]
            event_mean_logits = risk_h.mean(dim=1)
            loss_fn = NLLSurvLoss(alpha=getattr(self.args, "alpha_surv", 0.0))
            aux_loss = aux_loss + self.lambda_event_surv * loss_fn(event_mean_logits, y=y, c=c, t=None)
            aux_loss = aux_loss + self.lambda_per_event * self._per_event_surv_loss(risk_h, y, c)
            aux_loss = aux_loss + self.lambda_rank * self._ranking_loss(logits, y, c)
            aux_loss = aux_loss + self.lambda_global_cons * F.mse_loss(
                global_logits, event_mean_logits.detach()
            )

        return logits, aux_loss


__all__ = ["OTEHTimeLocalCompeting"]
