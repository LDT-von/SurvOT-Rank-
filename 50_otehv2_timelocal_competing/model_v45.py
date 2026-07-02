#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V45: Rank-guided OT event hazard decomposition.

This variant keeps the v9/31 OT-induced event backbone, but targets the
observed C-index gap more directly:

1. Per-event survival supervision instead of only supervising mean event logits.
2. Cox-style pairwise ranking loss on the final risk score.
3. Global event residual head, which is not the SlotSPE baseline trunk.
4. OT epsilon annealing from a softer early plan to the v9 sharp plan.

The prediction path still avoids SlotDecoder/CrossAttention/SelfAttention/D*3.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 本文件夹现在完全自包含：骨架代码在同目录的 backbone.py 里（不再跨文件夹
# 动态 import 31_ot_event_hazard_v2/model_v2.py），路径解析用同目录的 paths.py
# （不再依赖 common/paths.py）。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from paths import ensure_slotspe_in_path  # noqa
import backbone as _parent  # noqa: 内嵌的 OT 事件危险率骨架

ensure_slotspe_in_path()
from utils.loss_func import NLLSurvLoss  # noqa


class OTEHV2RankEvent(_parent.OTEventHazardV2Survival):
    """v9 backbone + ranking/event-level supervision and non-baseline residual."""

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


__all__ = ["OTEHV2RankEvent"]
