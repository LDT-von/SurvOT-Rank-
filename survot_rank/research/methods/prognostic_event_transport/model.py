#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V45: Rank-guided OT event hazard decomposition.

This variant keeps the v9/31 OT-induced event backbone, but targets the
observed C-index gap more directly:

1. Per-event survival supervision instead of only supervising mean event logits.
2. Cox-style pairwise ranking loss on the final risk score.
3. Global event residual head outside the event-gated hazard path.
4. OT epsilon annealing from a softer early plan to the v9 sharp plan.

The prediction path still avoids SlotDecoder/CrossAttention/SelfAttention/D*3.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.training.paths import ensure_slotspe_in_path  # noqa

ensure_slotspe_in_path()
from utils.loss_func import NLLSurvLoss, UnifiedSurvivalObjective  # noqa
from survot_rank.research.components.slot_attention import (  # noqa
    build_slot_attention as _build_slot_attention,
)

_PARENT_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ot_event_hazard_v2", "model_v2.py")
)
_PARENT_DIR = os.path.dirname(_PARENT_MODEL)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_parent_otehv2_rankevent_model", _PARENT_MODEL)
_parent = _ilu.module_from_spec(_spec)
sys.modules[_spec.name] = _parent
_spec.loader.exec_module(_parent)


class OTEHV2RankEvent(_parent.OTEventHazardV2Survival):
    """OT-event backbone with ranking/event-level supervision and global residual."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        self._set_default(args, "otehv2_eps", 0.05)
        self._set_default(args, "otehv2_iter", 50)
        self._set_default(args, "otehv2_warmup", 5)
        self._set_default(args, "otehv2_num_events", 24)
        self._set_default(args, "otehv2_heads", 4)
        self._set_default(args, "otehv2_layers", 4)
        self._set_default(args, "otehv2_dropout", 0.1)
        self._set_default(args, "lambda_otehv2_ot", 0.06)
        self._set_default(args, "lambda_otehv2_div", 0.01)
        self._set_default(args, "lambda_otehv2_event_surv", 0.25)
        self._set_default(args, "lambda_otehv2_recon", 0.2)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.global_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(getattr(args, "rankevent_dropout", 0.1)),
            nn.Linear(dim, self.num_classes),
        )
        init = float(getattr(args, "rankevent_global_init", -2.0))
        self.global_scale = nn.Parameter(torch.tensor(init))

        self.lambda_per_event = getattr(args, "lambda_rankevent_per_event", 0.15)
        self.lambda_rank = getattr(args, "lambda_rankevent_rank", 0.15)
        self.lambda_global_cons = getattr(args, "lambda_rankevent_global_cons", 0.02)
        self.lambda_gate_ent = getattr(args, "lambda_rankevent_gate_ent", 0.005)
        self.eps_start = getattr(args, "rankevent_eps_start", 0.10)
        self.eps_end = getattr(args, "rankevent_eps_end", 0.05)
        self.eps_anneal_epochs = getattr(args, "rankevent_eps_anneal_epochs", 12)
        self.rank_margin = getattr(args, "rankevent_rank_margin", 0.0)
        self.rank_max_pairs = getattr(args, "rankevent_rank_max_pairs", 4096)

    @staticmethod
    def _set_default(args, name, value):
        if not hasattr(args, name):
            setattr(args, name, value)

    def _current_epoch(self, kwargs):
        return int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))

    def _current_eps(self, kwargs):
        epoch = self._current_epoch(kwargs)
        if epoch <= 0:
            return self.eps_start
        frac = min(1.0, epoch / max(1, self.eps_anneal_epochs))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @staticmethod
    def _nll_per_sample(logits, y, c, eps=1e-7):
        y = y.view(-1, 1).long()
        c = c.view(-1, 1).float()
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        survival_pad = torch.cat([torch.ones_like(c), survival], dim=1)
        s_prev = torch.gather(survival_pad, 1, y).clamp_min(eps)
        h_this = torch.gather(hazards, 1, y).clamp_min(eps)
        s_this = torch.gather(survival_pad, 1, y + 1).clamp_min(eps)
        uncensored = -(1.0 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored = -c * torch.log(s_this)
        return (uncensored + censored).view(-1)

    def _per_event_surv_loss(self, event_logits, y, c):
        bsz, num_events, num_classes = event_logits.shape
        flat = event_logits.reshape(bsz * num_events, num_classes)
        y_rep = y.view(-1, 1).expand(-1, num_events).reshape(-1)
        c_rep = c.view(-1, 1).expand(-1, num_events).reshape(-1)
        return self._nll_per_sample(flat, y_rep, c_rep).mean()

    def _ranking_loss(self, logits, y, c):
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risk = -survival.sum(dim=1)
        t = y.float().view(-1)
        e = (1.0 - c.float()).view(-1)
        if risk.numel() < 2 or e.sum() <= 0:
            return risk.sum() * 0.0
        ti = t.view(-1, 1)
        tj = t.view(1, -1)
        comparable = (e.view(-1, 1) > 0.5) & (ti < tj)
        if comparable.sum() == 0:
            return risk.sum() * 0.0
        diff = risk.view(-1, 1) - risk.view(1, -1)
        values = F.softplus(-(diff - self.rank_margin))[comparable]
        if values.numel() > self.rank_max_pairs:
            idx = torch.randperm(values.numel(), device=values.device)[: self.rank_max_pairs]
            values = values[idx]
        return values.mean()

    @staticmethod
    def _gate_entropy_penalty(gate):
        entropy = -(gate.clamp_min(1e-8).log() * gate).sum(dim=1).mean()
        max_entropy = np.log(gate.shape[1])
        return gate.new_tensor(max_entropy) - entropy

    def _make_logits(self, event_tokens):
        event_logits = self.event_hazard(event_tokens)
        gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
        gated_logits = torch.einsum("be,bec->bc", gate, event_logits)
        global_feat = torch.einsum("be,bed->bd", gate, event_tokens)
        global_logits = self.global_head(global_feat)
        scale = torch.sigmoid(self.global_scale)
        logits = gated_logits + scale * global_logits
        return logits, event_logits, gate, global_logits

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
        logits, event_logits, gate, global_logits = self._make_logits(event_tokens)

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
            + self.lambda_gate_ent * self._gate_entropy_penalty(gate)
        )

        if "y" in kwargs and "c" in kwargs:
            y, c = kwargs["y"], kwargs["c"]
            event_mean_logits = event_logits.mean(dim=1)
            loss_fn = NLLSurvLoss(alpha=getattr(self.args, "alpha_surv", 0.0))
            aux_loss = aux_loss + self.lambda_event_surv * loss_fn(event_mean_logits, y=y, c=c, t=None)
            aux_loss = aux_loss + self.lambda_per_event * self._per_event_surv_loss(event_logits, y, c)
            aux_loss = aux_loss + self.lambda_rank * self._ranking_loss(logits, y, c)
            aux_loss = aux_loss + self.lambda_global_cons * F.mse_loss(global_logits, event_mean_logits.detach())

        return logits, aux_loss


class ThreeWayOTFusion(nn.Module):
    """三模态（WSI/Omic/Clinical）OT 融合，输出契约与 `MultiScaleOTFusion` 一致：
    输入任意三组 slot 张量 [B, K_i, dim]（K_i 可不同），输出 event_tokens
    形状为 [B, num_events, dim]，供下游 event_encoder/event_hazard 等复用。

    实现方式：对三对模态组合（wsi-omic, wsi-clinical, omic-clinical）分别构造
    pair token（拼接/乘积/差的绝对值）与 OT 代价/plan（用 cosine cost +
    log_sinkhorn_plan），加权投影后拼在一起，再用与 MultiScaleOTFusion 同构的
    event-query 注意力聚合到 num_events 个事件 token 上，最后接一个
    TransformerEncoderLayer 精炼。
    """

    def __init__(self, dim, num_events, nhead=4, dropout=0.1):
        super().__init__()
        self.num_events = num_events
        self.dim = dim
        # 每对模态贡献 2 个通道（pair-context 投影 + cost 投影），3 对模态共 6 个通道，
        # 对应任务描述“输入通道数从 3 变为 6”。
        self.pair_proj = nn.Linear(dim * 4, dim)
        self.cost_proj = nn.Linear(1, dim)
        self.combine_proj = nn.Linear(dim * 2 * 3, dim)  # 3 对模态 x (pair_proj+cost_proj) = 6 路 -> dim
        self.event_queries = nn.Parameter(torch.randn(num_events, dim) * 0.02)
        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.refine = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    @staticmethod
    def _build_pair_tokens(a, b):
        bsz, ka, kb, dim = a.shape[0], a.shape[1], b.shape[1], a.shape[-1]
        a_exp = a.unsqueeze(2).expand(bsz, ka, kb, dim)
        b_exp = b.unsqueeze(1).expand(bsz, ka, kb, dim)
        return torch.cat([a_exp, b_exp, a_exp * b_exp, (a_exp - b_exp).abs()], dim=-1)

    def _pair_stream(self, a, b):
        # a: [B, Ka, dim], b: [B, Kb, dim] -> event-query-ready tokens [B, Ka*Kb, dim*2]
        bsz, ka, dim = a.shape
        kb = b.shape[1]
        pair_tokens_raw = self._build_pair_tokens(a, b)  # [B, Ka, Kb, dim*4]
        pair_ctx = self.pair_proj(pair_tokens_raw)  # [B, Ka, Kb, dim]
        cost = 1.0 - torch.bmm(F.normalize(a, dim=-1), F.normalize(b, dim=-1).transpose(1, 2))  # [B, Ka, Kb]
        plan, _ = _parent.log_sinkhorn_plan(cost, eps=0.05, max_iter=20)
        cost_feat = self.cost_proj(plan.unsqueeze(-1))  # [B, Ka, Kb, dim]
        combined = torch.cat([pair_ctx, cost_feat], dim=-1)  # [B, Ka, Kb, dim*2]
        return combined.reshape(bsz, ka * kb, dim * 2)

    def forward(self, slots_wsi, slots_omic, slots_clinical):
        bsz, _, dim = slots_wsi.shape
        stream_wo = self._pair_stream(slots_wsi, slots_omic)
        stream_wc = self._pair_stream(slots_wsi, slots_clinical)
        stream_oc = self._pair_stream(slots_omic, slots_clinical)

        # 对齐 token 数量：三条流的 token 数（Ka*Kb）可能不同，分别聚合到 num_events
        # 再拼接通道，比强行拼接不同长度的序列更稳健。
        streams = [stream_wo, stream_wc, stream_oc]  # each [B, P_i, dim*2]
        # 对每条流各自做 event-query 聚合到 [B, num_events, dim*2]，再在最后一维拼接 3 条流
        # 得到 [B, num_events, dim*6]，用 combine_proj 投回 dim。
        agg_list = []
        q = F.normalize(self.event_queries, dim=-1)  # [num_events, dim]
        for stream in streams:
            # stream: [B, P, dim*2]；用其前 dim 维（pair_ctx 部分）做相似度打分，
            # 避免维度不匹配（q 是 dim 维）。
            stream_key = stream[..., :dim]
            t = F.normalize(stream_key, dim=-1)
            scores = torch.einsum("kd,bpd->bpk", q, t)  # [B, P, num_events]
            assign = torch.softmax(scores.transpose(1, 2), dim=-1)  # [B, num_events, P]
            agg = torch.bmm(assign, stream)  # [B, num_events, dim*2]
            agg_list.append(agg)

        combined = torch.cat(agg_list, dim=-1)  # [B, num_events, dim*6]
        events = self.combine_proj(combined)  # [B, num_events, dim]
        events = self.norm(self.cross_attn(events))
        return events + self.refine(events), None


class OTEHV2RankEventV2(OTEHV2RankEvent):
    """在 `OTEHV2RankEvent`（V45）基础上新增可选能力的骨架子类。

    新增能力包括：三模态融合（临床模态）、统一生存目标（Unified Objective）、
    slot 身份/状态解耦、Slot Attention 路由机制重设计（Sinkhorn/跨模态条件化/
    自适应迭代次数）。这些能力均由 `getattr(args, name, default)` 读取的配置开关
    控制，默认值使所有新开关保持关闭/退化为旧行为。

    本任务（8.1）只搭建骨架：`__init__` 仅读取并保存新增配置字段，不实例化任何新
    的 `nn.Module` 子模块（新模块的实例化留给后续任务，如 9.3 的临床编码器接入、
    11.1 的路由重设计接入）；`forward` 直接调用父类 `OTEHV2RankEvent.forward`，
    保证当前默认行为与父类完全一致。
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        # 三模态融合相关配置（需求1：临床模态编码器与三模态融合），默认关闭。
        self.otehv2v2_use_clinical = getattr(args, "otehv2v2_use_clinical", False)
        self.otehv2v2_clinical_feature_dim = getattr(args, "otehv2v2_clinical_feature_dim", 0)
        self.otehv2v2_num_slots_clinical = getattr(args, "otehv2v2_num_slots_clinical", 8)

        # 统一生存目标相关配置（需求3：统一的 NLL 与排序目标），默认关闭。
        self.otehv2v2_use_unified_objective = getattr(args, "otehv2v2_use_unified_objective", False)
        self.lambda_unified_rank = getattr(args, "lambda_unified_rank", 0.15)
        self._unified_objective = UnifiedSurvivalObjective(
            margin=self.rank_margin, rank_weight=self.lambda_unified_rank
        )

        # 可学习自适应损失加权（Kendall 2018 同方差不确定性加权），默认关闭。
        # 开启后用可学习对数方差替代人工固定 lambda 来配平多项损失，缓解
        # “损失函数太多、手工权重配不平”导致的协同崩塌（对照历史 V44 教训）。
        self.otehv2v2_learnable_loss_weights = getattr(
            args, "otehv2v2_learnable_loss_weights", False
        )
        if self.otehv2v2_learnable_loss_weights:
            from survot_rank.research.components.adaptive_loss_weighter import (
                AdaptiveLossWeighter,
            )
            # 注册所有可能出现的损失项名称超集（不同能力开关下实际用到的子集不同，
            # forward 时只传入当前激活的子集）。
            self._loss_weighter = AdaptiveLossWeighter(
                [
                    "nll",
                    "per_event",
                    "rank",
                    "global_cons",
                    "unified",
                    "ot",
                    "div",
                    "recon",
                    "gate_ent",
                ]
            )

        # Slot 身份/状态解耦与路由机制重设计相关配置（需求4+5），默认关闭/使用旧行为。
        self.otehv2v2_slot_disentangled = getattr(args, "otehv2v2_slot_disentangled", False)
        self.otehv2v2_slot_router = getattr(args, "otehv2v2_slot_router", "softmax")
        self.otehv2v2_slot_cross_modal_cond = getattr(args, "otehv2v2_slot_cross_modal_cond", False)
        self.otehv2v2_slot_adaptive_iters = getattr(args, "otehv2v2_slot_adaptive_iters", False)
        self.otehv2v2_sinkhorn_max_iters = getattr(args, "otehv2v2_sinkhorn_max_iters", 20)
        self.otehv2v2_convergence_threshold = getattr(args, "otehv2v2_convergence_threshold", 0.0)

        # 需求5：只要 router/disentangled/cross_modal_cond/adaptive_iters 任一开关
        # 偏离默认值，就用工厂函数重建 WSI/Omic 的 slot attention 模块（用
        # MultiHeadSlotAttentionV2 替换父类 __init__ 中实例化的普通
        # MultiHeadSlotAttention）。所有开关均为默认值时，保留父类实例化的原始
        # 模块不做任何改动，避免不必要的重新实例化/参数重置，保证 V45 默认行为
        # 的数值完全一致（需求6.7 的向后兼容性检查）。
        _uses_routing_redesign = (
            self.otehv2v2_slot_router != "softmax"
            or self.otehv2v2_slot_disentangled
            or self.otehv2v2_slot_cross_modal_cond
            or self.otehv2v2_slot_adaptive_iters
        )
        if _uses_routing_redesign:
            dim = self.wsi_projection_dim
            self.slot_attention_wsi = _build_slot_attention(
                dim=dim, num_slots=args.slot_num_wsi, heads=8, iters=args.slot_iters, config=args
            )
            self.slot_attention_omic = _build_slot_attention(
                dim=dim, num_slots=args.slot_num_omics, heads=8, iters=args.slot_iters, config=args
            )

        if self.otehv2v2_use_clinical:
            from survot_rank.research.components.clinical_encoder import ClinicalEncoder
            dim = self.wsi_projection_dim
            self.clinical_encoder = ClinicalEncoder(
                clinical_feature_dim=self.otehv2v2_clinical_feature_dim, dim=dim
            )
            self.slot_attention_clinical = _build_slot_attention(
                dim=dim, num_slots=self.otehv2v2_num_slots_clinical, heads=8, iters=args.slot_iters, config=args
            )
            self.three_way_fusion = ThreeWayOTFusion(
                dim=dim, num_events=self.fusion.num_events,
                nhead=getattr(args, "otehv2_heads", 4),
                dropout=getattr(args, "otehv2_dropout", 0.1),
            )

        # 注意：`otehv2v2_slot_disentangled` / `otehv2v2_slot_router` 等路由重设计相关
        # 新模块的实例化已在上面通过 `_build_slot_attention` 工厂函数完成；此处只在
        # `otehv2v2_use_clinical=True` 时额外实例化临床编码器、临床 slot attention 与
        # 三模态 OT 融合模块（需求1，任务9.3）。

    def forward(self, **kwargs):
        # 默认关闭时（临床模态 + 跨模态条件化均未启用）：直接复用父类前向逻辑，
        # 保证与 OTEHV2RankEvent 完全一致的默认行为（需求6.7 的向后兼容性检查）。
        if not self.otehv2v2_use_clinical and not self.otehv2v2_slot_cross_modal_cond:
            return super().forward(**kwargs)

        if self.otehv2v2_use_clinical:
            x_clinical = kwargs.get("x_clinical")
            if x_clinical is None:
                raise ValueError(
                    "Clinical 输入缺失或维度不匹配：三模态开关已启用但未提供 x_clinical"
                )
            if x_clinical.shape[-1] != self.otehv2v2_clinical_feature_dim:
                raise ValueError(
                    f"Clinical 输入缺失或维度不匹配：期望最后一维为 "
                    f"{self.otehv2v2_clinical_feature_dim}，实际为 {x_clinical.shape[-1]}"
                )

        x_wsi = kwargs["x_wsi"]
        x_wsi_proj = self.wsi_mlp(x_wsi)
        x_omics = self._encode_omics(kwargs)

        if self.otehv2v2_slot_cross_modal_cond:
            # 需求5b：双向跨模态条件化。先各自跑一次不带条件化的前向拿到对方
            # 上一轮的 slot_state 近似值，再各自用对方的结果作为
            # cross_modal_state 重新计算一次，实现“用对方模态信息条件化自身
            # slot 更新”。
            slots_wsi_pre = self.slot_attention_wsi(x_wsi_proj)
            slots_omic_pre = self.slot_attention_omic(x_omics)
            slots_wsi = self.slot_attention_wsi(x_wsi_proj, cross_modal_state=slots_omic_pre)
            slots_omic = self.slot_attention_omic(x_omics, cross_modal_state=slots_wsi_pre)
        else:
            slots_wsi = self.slot_attention_wsi(x_wsi_proj)
            slots_omic = self.slot_attention_omic(x_omics)

        if self.otehv2v2_use_clinical:
            x_clinical_proj = self.clinical_encoder(x_clinical).unsqueeze(1)  # [B, 1, dim]
            slots_clinical = self.slot_attention_clinical(x_clinical_proj)  # [B, num_slots_clinical, dim]
            event_tokens, _ = self.three_way_fusion(slots_wsi, slots_omic, slots_clinical)
        else:
            # 非临床但启用跨模态条件化：复用父类相同的双模态 OT 融合逻辑
            # （无法再委托给 super().forward()，因为这里需要传入
            # cross_modal_state 的 slot attention 调用）。
            cost_cos = _parent.cosine_cost(slots_wsi, slots_omic)
            cost_euc = _parent.euclidean_cost(slots_wsi, slots_omic)
            cost_dot = _parent.dot_cost(slots_wsi, slots_omic)
            eps = self._current_eps(kwargs)
            plan_cos, ot_dist_cos = _parent.log_sinkhorn_plan(cost_cos, eps=eps, max_iter=self.ot_iter)
            plan_euc, ot_dist_euc = _parent.log_sinkhorn_plan(cost_euc, eps=eps, max_iter=self.ot_iter)
            plan_dot, ot_dist_dot = _parent.log_sinkhorn_plan(cost_dot, eps=eps, max_iter=self.ot_iter)
            event_tokens, _ = self.fusion(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot)

        event_tokens = self.event_norm(self.event_encoder(event_tokens))
        logits, event_logits, gate, global_logits = self._make_logits(event_tokens)

        if not self.training:
            return logits, 0.0

        # ------------------------------------------------------------------
        # 收集各损失项的“原始（未乘 lambda）值”与对应的固定 lambda 权重。
        # - 关闭可学习加权时（默认）：aux_loss = sum(lambda_k * raw_k)，与原实现
        #   逐项等价（含 OT warmup ramp）。
        # - 开启可学习加权时：把原始损失项交给 AdaptiveLossWeighter，用可学习对数
        #   方差自动配平，不再使用人工固定 lambda。
        # ------------------------------------------------------------------
        raw_losses: dict = {}
        lambdas: dict = {}

        if self.otehv2v2_use_clinical:
            # 三模态融合路径：沿用原实现，只有 diversity + gate entropy 两个正则项。
            raw_losses["div"] = self._diversity_loss(event_tokens)
            lambdas["div"] = self.lambda_div
            raw_losses["gate_ent"] = self._gate_entropy_penalty(gate)
            lambdas["gate_ent"] = self.lambda_gate_ent
        else:
            epoch = self._current_epoch(kwargs)
            ramp = 0.0 if epoch < self.ot_warmup else min(1.0, (epoch - self.ot_warmup) / max(1, self.ot_warmup))
            ot_mean = (ot_dist_cos + ot_dist_euc + ot_dist_dot).mean() / 3.0
            recon_wsi = self.recon_wsi(slots_wsi)
            recon_omic = self.recon_omic(slots_omic)
            recon_loss = F.mse_loss(recon_wsi, slots_omic) + F.mse_loss(recon_omic, slots_wsi)
            # OT warmup ramp 是训练进度调度（非损失权重），无论是否可学习加权都保留，
            # 直接折进 OT 项的原始值里。
            raw_losses["ot"] = ramp * ot_mean
            lambdas["ot"] = self.lambda_ot
            raw_losses["div"] = self._diversity_loss(event_tokens)
            lambdas["div"] = self.lambda_div
            raw_losses["recon"] = recon_loss
            lambdas["recon"] = self.lambda_recon
            raw_losses["gate_ent"] = self._gate_entropy_penalty(gate)
            lambdas["gate_ent"] = self.lambda_gate_ent

        if "y" in kwargs and "c" in kwargs:
            y, c = kwargs["y"], kwargs["c"]
            if self.otehv2v2_use_unified_objective:
                # 统一目标本身已含 per-event NLL + 排序惩罚，作为单个监督项。
                raw_losses["unified"] = self._unified_objective(
                    event_logits=event_logits, risk_logits=logits, y=y, c=c
                )
                lambdas["unified"] = 1.0
            else:
                event_mean_logits = event_logits.mean(dim=1)
                loss_fn = NLLSurvLoss(alpha=getattr(self.args, "alpha_surv", 0.0))
                raw_losses["nll"] = loss_fn(event_mean_logits, y=y, c=c, t=None)
                lambdas["nll"] = self.lambda_event_surv
                raw_losses["per_event"] = self._per_event_surv_loss(event_logits, y, c)
                lambdas["per_event"] = self.lambda_per_event
                raw_losses["rank"] = self._ranking_loss(logits, y, c)
                lambdas["rank"] = self.lambda_rank
                raw_losses["global_cons"] = F.mse_loss(global_logits, event_mean_logits.detach())
                lambdas["global_cons"] = self.lambda_global_cons

        if self.otehv2v2_learnable_loss_weights:
            aux_loss = self._loss_weighter(raw_losses)
        else:
            aux_loss = None
            for name, value in raw_losses.items():
                term = lambdas[name] * value
                aux_loss = term if aux_loss is None else aux_loss + term
            if aux_loss is None:
                aux_loss = logits.sum() * 0.0

        return logits, aux_loss


__all__ = ["OTEHV2RankEvent", "OTEHV2RankEventV2"]
