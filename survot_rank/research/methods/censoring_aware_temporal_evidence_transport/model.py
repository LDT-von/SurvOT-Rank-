"""Censoring-aware temporal evidence transport for multimodal survival.

The method has one coupled mechanism: a stage-specific OT plan is modulated by
an evidence gate, while a risk-set objective supervises the plan-weighted edge
evidence.  The model also performs keep/remove interventions on the same plan
used for prediction.  It never constructs artificial low/high-risk labels or
claims a causal treatment effect.
"""

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


class CensoringAwareTemporalEvidenceTransport(OTEventHazardV2Survival):
    """Stage-specific, risk-set supervised OT evidence model.

    ``c == 0`` is an observed event and ``c == 1`` is right-censored.  The
    ranking term only anchors comparisons at observed events and weights each
    anchor by its empirical at-risk set, so censored observations are used as
    risk-set context rather than treated as observed outcomes.
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.catet_num_stages = int(getattr(args, "catet_num_stages", 4))
        args.otehv2_num_events = args.catet_num_stages
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.catet_num_stages = args.catet_num_stages
        self.catet_prog_cost_weight = float(getattr(args, "catet_prog_cost", 0.20))
        self.catet_lambda_ot = float(getattr(args, "catet_lambda_ot", 0.04))
        self.catet_lambda_rank = float(getattr(args, "catet_lambda_rank", 0.08))
        self.catet_lambda_intervention = float(
            getattr(args, "catet_lambda_intervention", 0.05)
        )
        self.catet_keep_ratio = float(getattr(args, "catet_keep_ratio", 0.25))
        self.catet_intervention_margin = float(
            getattr(args, "catet_intervention_margin", 0.05)
        )
        self.catet_rank_margin = float(getattr(args, "catet_rank_margin", 0.0))
        self.catet_rank_max_pairs = int(getattr(args, "catet_rank_max_pairs", 4096))
        stage_embedding = torch.linspace(-1.0, 1.0, self.catet_num_stages).unsqueeze(1)
        self.register_buffer("stage_embedding", stage_embedding.repeat(1, dim))

        self.stage_edge_risk = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.temporal_evidence_gate = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.last_explanations = None

        # These heads belong to the legacy reconstruction objective and are
        # intentionally absent from this compact mainline.
        del self.recon_wsi
        del self.recon_omic

    @staticmethod
    def _normalize_cost(cost):
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _positive_dot_cost(x, y):
        return F.softplus(-torch.bmm(x, y.transpose(1, 2)))

    @staticmethod
    def _pair_tokens(slots_wsi, slots_omic):
        bsz, sw, dim = slots_wsi.shape
        so = slots_omic.shape[1]
        w = slots_wsi.unsqueeze(2).expand(bsz, sw, so, dim)
        o = slots_omic.unsqueeze(1).expand(bsz, sw, so, dim)
        return torch.cat([w, o, w * o, (w - o).abs()], dim=-1)

    def _stage_transport_plans(self, slots_wsi, slots_omic, epoch):
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        edge_risk = torch.sigmoid(self.stage_edge_risk(pair_tokens).squeeze(-1))
        edge_risk = edge_risk.unsqueeze(1).expand(-1, self.catet_num_stages, -1, -1)

        # The learned prognostic edge signal changes the OT geometry itself.
        # High-risk evidence has lower transport cost, so it can be selected by
        # the plan and is not merely plotted after prediction.
        stage_bias = 1.0 - edge_risk
        stage_bias = self._normalize_cost(stage_bias.flatten(0, 1)).view_as(stage_bias)
        base_costs = [
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        ]
        eps = float(getattr(self.args, "otehv2_eps", 0.05))
        if epoch > 0:
            start = float(getattr(self.args, "rg_eps_start", eps * 2.0))
            frac = min(1.0, epoch / max(1, int(getattr(self.args, "rg_eps_anneal", 12))))
            eps = start + frac * (eps - start)

        plans, distances = [], []
        for stage_idx in range(self.catet_num_stages):
            stage_plans, stage_distances = [], []
            for base_cost in base_costs:
                plan, distance = log_sinkhorn_plan(
                    base_cost + self.catet_prog_cost_weight * stage_bias[:, stage_idx],
                    eps=eps,
                    max_iter=self.ot_iter,
                )
                stage_plans.append(plan)
                stage_distances.append(distance)
            plans.append(tuple(stage_plans))
            distances.append(torch.stack(stage_distances).mean())
        return plans, torch.stack(distances).mean(), edge_risk

    @staticmethod
    def _renormalize_plan(plan, weights):
        weighted = plan * weights
        return weighted / weighted.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _gated_plans(self, slots_wsi, slots_omic, plans):
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        bsz, sw, so, _ = pair_tokens.shape
        gated, removed, gates = [], [], []
        for stage_idx, stage_plans in enumerate(plans):
            stage_code = self.stage_embedding[stage_idx].view(1, 1, 1, -1)
            stage_code = stage_code.expand(bsz, sw, so, -1)
            gate = torch.sigmoid(
                self.temporal_evidence_gate(
                    torch.cat([pair_tokens, stage_code], dim=-1)
                ).squeeze(-1)
            )
            gated.append(tuple(self._renormalize_plan(p, gate) for p in stage_plans))
            removed.append(tuple(self._renormalize_plan(p, 1.0 - gate) for p in stage_plans))
            gates.append(gate)
        return gated, removed, torch.stack(gates, dim=1)

    def _stage_events(self, slots_wsi, slots_omic, plans):
        events = []
        for stage_idx, stage_plans in enumerate(plans):
            all_events, _ = self.fusion(slots_wsi, slots_omic, *stage_plans)
            events.append(all_events[:, stage_idx:stage_idx + 1])
        return torch.cat(events, dim=1)

    def _decode(self, slots_wsi, slots_omic, plans):
        tokens = self._stage_events(slots_wsi, slots_omic, plans)
        tokens = self.event_norm(
            self.event_encoder(tokens + self.stage_embedding.unsqueeze(0))
        )
        event_logits = self.event_hazard(tokens)
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)
        return logits, tokens, gate

    @staticmethod
    def _risk_score(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    def _risk_set_transport_loss(self, transport_evidence, event_time, censorship):
        risk = transport_evidence
        times = event_time.float().view(-1)
        observed = (1.0 - censorship.float().view(-1)) > 0.5
        comparable = observed[:, None] & (times[:, None] < times[None, :])
        if comparable.sum() == 0:
            return risk.sum() * 0.0
        at_risk = (times[None, :] >= times[:, None]).float().sum(dim=1)
        anchor_weight = (times.numel() / at_risk.clamp_min(1.0)).detach()
        values = F.softplus(
            -(risk[:, None] - risk[None, :] - self.catet_rank_margin)
        )
        weights = anchor_weight[:, None].expand_as(values)
        values = values[comparable] * weights[comparable]
        if values.numel() > self.catet_rank_max_pairs:
            keep = torch.randperm(values.numel(), device=values.device)[: self.catet_rank_max_pairs]
            values = values[keep]
        return values.mean()

    def _intervention_loss(self, full_logits, evidence_logits, removed_logits, evidence):
        full_risk = self._risk_score(full_logits).detach()
        evidence_risk = self._risk_score(evidence_logits)
        removed_risk = self._risk_score(removed_logits)
        # Sufficiency: selected evidence should reproduce the factual risk.
        sufficiency = F.smooth_l1_loss(evidence_risk, full_risk)
        flat = evidence.flatten(2)
        keep_n = max(1, int(flat.size(-1) * self.catet_keep_ratio))
        threshold = flat.topk(keep_n, dim=-1).values[..., -1:].detach()
        selected = (flat >= threshold).to(flat.dtype).view_as(evidence)
        selected_mass = (evidence * selected).sum(dim=(2, 3))
        full_mass = evidence.sum(dim=(2, 3)).clamp_min(1e-6)
        sparsity = (selected_mass / full_mass).mean()
        # Comprehensiveness is direction-free: removal must matter, but its
        # direction is determined by the factual model rather than prescribed.
        change = (full_risk - removed_risk).abs()
        comprehensiveness = F.relu(self.catet_intervention_margin - change).mean()
        return sufficiency + comprehensiveness + 0.1 * sparsity

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        plans, ot_distance, edge_risk = self._stage_transport_plans(
            slots_wsi, slots_omic, epoch
        )
        evidence_plans, removed_plans, evidence_gate = self._gated_plans(
            slots_wsi, slots_omic, plans
        )
        full_logits, _, _ = self._decode(slots_wsi, slots_omic, plans)
        logits, tokens, event_gate = self._decode(slots_wsi, slots_omic, evidence_plans)
        removed_logits, _, _ = self._decode(slots_wsi, slots_omic, removed_plans)

        transport_evidence = (edge_risk * evidence_gate).flatten(2).mean(dim=-1)
        transport_evidence = (transport_evidence * event_gate).sum(dim=1)
        self.last_explanations = {
            "stage_slot_pair_evidence": evidence_gate.detach(),
            "stage_slot_pair_risk": edge_risk.detach(),
            "transport_evidence_risk": transport_evidence.detach(),
            "event_gate": event_gate.detach(),
            "factual_risk": self._risk_score(logits).detach(),
            "removed_risk": self._risk_score(removed_logits).detach(),
        }

        if not self.training:
            return logits, 0.0

        aux_loss = self.catet_lambda_ot * ot_distance
        if "event_time" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.catet_lambda_rank * self._risk_set_transport_loss(
                transport_evidence, kwargs["event_time"], kwargs["c"]
            )
        aux_loss = aux_loss + self.catet_lambda_intervention * self._intervention_loss(
            full_logits, logits, removed_logits, evidence_gate
        )
        return logits, aux_loss

    def explain_last_batch(self):
        if self.last_explanations is None:
            raise RuntimeError("Run a forward pass before requesting explanations")
        return self.last_explanations
