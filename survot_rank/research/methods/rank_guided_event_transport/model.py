"""Rank-guided event transport with a compact survival objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    OTEventHazardV2Survival,
    cosine_cost,
    euclidean_cost,
    log_sinkhorn_plan,
)


class RankGuidedEventTransport(OTEventHazardV2Survival):
    """OT event model whose transport is trained by continuous survival order."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.otehv2_num_events = int(getattr(args, "rg_num_events", 4))
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.num_events = args.otehv2_num_events
        self.rg_prog_cost_weight = float(getattr(args, "rg_prog_cost", 0.20))
        self.rg_lambda_ot = float(getattr(args, "rg_lambda_ot", 0.06))
        self.rg_lambda_rank = float(getattr(args, "rg_lambda_rank", 0.15))
        self.rg_lambda_stage = float(getattr(args, "rg_lambda_stage", 0.02))
        self.rg_rank_margin = float(getattr(args, "rg_rank_margin", 0.0))
        self.rg_rank_max_pairs = int(getattr(args, "rg_rank_max_pairs", 4096))
        self.rg_stage_margin = float(getattr(args, "rg_stage_margin", 0.25))

        self.prognostic_pair_cost = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.stage_head = nn.Linear(dim, self.num_classes)
        stage_embedding = torch.linspace(-1.0, 1.0, self.num_events).unsqueeze(1)
        self.register_buffer("stage_embedding", stage_embedding.repeat(1, dim))

        del self.recon_wsi
        del self.recon_omic

    @staticmethod
    def _positive_dot_cost(x, y):
        similarity = torch.bmm(x, y.transpose(1, 2))
        return F.softplus(-similarity)

    @staticmethod
    def _normalize_cost(cost):
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _pair_tokens(slots_wsi, slots_omic):
        bsz, sw, dim = slots_wsi.shape
        so = slots_omic.shape[1]
        w = slots_wsi.unsqueeze(2).expand(bsz, sw, so, dim)
        o = slots_omic.unsqueeze(1).expand(bsz, sw, so, dim)
        return torch.cat([w, o, w * o, (w - o).abs()], dim=-1)

    def _transport_plans(self, slots_wsi, slots_omic, epoch):
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        prognostic_cost = F.softplus(
            self.prognostic_pair_cost(pair_tokens).squeeze(-1)
        )
        prognostic_cost = self._normalize_cost(prognostic_cost)
        costs = [
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        ]
        costs = [cost + self.rg_prog_cost_weight * prognostic_cost for cost in costs]

        # 单调退火：epoch 0 用软起点 start(=0.10)，随 epoch 单调收紧到 otehv2_eps(=0.05)。
        # 之前的 `if epoch > 0` guard 会让 epoch 0 直接落到终点值 0.05（最尖锐），
        # epoch 1 又跳回 0.096（软），造成 plan 在首个 epoch 间断变化、假峰出现在 epoch 0。
        end = getattr(self.args, "otehv2_eps", 0.05)
        start = float(getattr(self.args, "rg_eps_start", end * 2.0))
        frac = min(1.0, epoch / max(1, int(getattr(self.args, "rg_eps_anneal", 12))))
        eps = start + frac * (end - start)
        plans = [
            log_sinkhorn_plan(cost, eps=eps, max_iter=self.ot_iter)
            for cost in costs
        ]
        return plans

    def _stage_order_loss(self, event_tokens):
        stage_prob = torch.softmax(self.stage_head(event_tokens), dim=-1)
        time_index = torch.arange(
            self.num_classes, device=event_tokens.device, dtype=event_tokens.dtype
        )
        centers = (stage_prob * time_index).sum(dim=-1)
        if self.num_events < 2:
            return centers.sum() * 0.0
        adjacent_gap = centers[:, 1:] - centers[:, :-1]
        return F.relu(self.rg_stage_margin - adjacent_gap).mean()

    def _continuous_ranking_loss(self, logits, event_time, censorship):
        hazards = torch.sigmoid(logits)
        risk = -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)
        times = event_time.float().view(-1)
        observed = (1.0 - censorship.float()).view(-1) > 0.5
        if risk.numel() < 2 or not observed.any():
            return risk.sum() * 0.0

        comparable = observed[:, None] & (times[:, None] < times[None, :])
        if not comparable.any():
            return risk.sum() * 0.0
        differences = risk[:, None] - risk[None, :]
        values = F.softplus(-(differences - self.rg_rank_margin))[comparable]
        if values.numel() > self.rg_rank_max_pairs:
            keep = torch.randperm(values.numel(), device=values.device)[: self.rg_rank_max_pairs]
            values = values[keep]
        return values.mean()

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        (plan_cos, ot_cos), (plan_euc, ot_euc), (plan_dot, ot_dot) = [
            (plan, dist) for plan, dist in self._transport_plans(slots_wsi, slots_omic, epoch)
        ]
        event_tokens, _ = self.fusion(
            slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot
        )
        event_tokens = event_tokens + self.stage_embedding.unsqueeze(0)
        event_tokens = self.event_norm(self.event_encoder(event_tokens))

        event_logits = self.event_hazard(event_tokens)
        gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)

        if not self.training:
            return logits, 0.0

        aux_loss = self.rg_lambda_ot * (ot_cos + ot_euc + ot_dot).mean() / 3.0
        if "event_time" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.rg_lambda_rank * self._continuous_ranking_loss(
                logits, kwargs["event_time"], kwargs["c"]
            )
        aux_loss = aux_loss + self.rg_lambda_stage * self._stage_order_loss(event_tokens)
        return logits, aux_loss
