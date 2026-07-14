"""Stage-specific prognostic transport for multimodal survival prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    cosine_cost,
    euclidean_cost,
    log_sinkhorn_plan,
)
from survot_rank.research.methods.rank_guided_event_transport.model import (
    RankGuidedEventTransport,
)


class StagewisePrognosticTransport(RankGuidedEventTransport):
    """Use a separate prognosis-conditioned OT plan for each event stage.

    The parent RG-ET model learns one prognostic pair cost shared by all events.
    This method predicts a stage-specific pair cost and feeds the corresponding
    Sinkhorn plans into the matching event token. The old methods remain intact.
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.rg_num_events = int(getattr(args, "spt_num_stages", 4))
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.spt_num_stages = self.num_events
        self.spt_prog_cost_weight = float(getattr(args, "spt_prog_cost", 0.20))
        self.spt_lambda_ot = float(getattr(args, "spt_lambda_ot", 0.06))
        self.spt_lambda_rank = float(getattr(args, "spt_lambda_rank", 0.05))
        self.spt_lambda_stage = float(getattr(args, "spt_lambda_stage", 0.02))
        self.spt_stage_margin = float(getattr(args, "spt_stage_margin", 0.25))

        self.stage_pair_cost = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, self.spt_num_stages),
        )
        self.stage_score = nn.Linear(dim, 1)

    @staticmethod
    def _normalize_stage_cost(cost):
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _stage_transport_plans(self, slots_wsi, slots_omic, epoch):
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        stage_cost = F.softplus(self.stage_pair_cost(pair_tokens))
        stage_cost = stage_cost.permute(0, 3, 1, 2)

        base_costs = [
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        ]
        # 单调退火：epoch 0 用软起点 start(=0.10)，随 epoch 单调收紧到 otehv2_eps(=0.05)。
        # 去掉原 `if epoch > 0` guard，避免 epoch 0 直接落到最尖锐值造成的假峰与 epoch 间断。
        end = getattr(self.args, "otehv2_eps", 0.05)
        start = float(getattr(self.args, "rg_eps_start", end * 2.0))
        frac = min(1.0, epoch / max(1, int(getattr(self.args, "rg_eps_anneal", 12))))
        eps = start + frac * (end - start)

        plans = []
        distances = []
        for stage_idx in range(self.spt_num_stages):
            prognostic = self._normalize_stage_cost(stage_cost[:, stage_idx])
            stage_plans = []
            stage_distances = []
            for base_cost in base_costs:
                plan, distance = log_sinkhorn_plan(
                    base_cost + self.spt_prog_cost_weight * prognostic,
                    eps=eps,
                    max_iter=self.ot_iter,
                )
                stage_plans.append(plan)
                stage_distances.append(distance)
            plans.append(stage_plans)
            distances.append(torch.stack(stage_distances).mean())
        return plans, torch.stack(distances).mean()

    def _stagewise_events(self, slots_wsi, slots_omic, plans):
        selected_events = []
        for stage_idx, (plan_cos, plan_euc, plan_dot) in enumerate(plans):
            all_events, _ = self.fusion(
                slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot
            )
            selected_events.append(all_events[:, stage_idx:stage_idx + 1])
        return torch.cat(selected_events, dim=1)

    def _stage_order_loss(self, event_tokens):
        scores = self.stage_score(event_tokens).squeeze(-1)
        if scores.size(1) < 2:
            return scores.sum() * 0.0
        gaps = scores[:, 1:] - scores[:, :-1]
        return F.relu(self.spt_stage_margin - gaps).mean()

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        plans, ot_distance = self._stage_transport_plans(slots_wsi, slots_omic, epoch)
        event_tokens = self._stagewise_events(slots_wsi, slots_omic, plans)
        event_tokens = event_tokens + self.stage_embedding.unsqueeze(0)
        event_tokens = self.event_norm(self.event_encoder(event_tokens))

        event_logits = self.event_hazard(event_tokens)
        gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)

        if not self.training:
            return logits, 0.0

        aux_loss = self.spt_lambda_ot * ot_distance
        if "event_time" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.spt_lambda_rank * self._continuous_ranking_loss(
                logits, kwargs["event_time"], kwargs["c"]
            )
        aux_loss = aux_loss + self.spt_lambda_stage * self._stage_order_loss(event_tokens)
        return logits, aux_loss
