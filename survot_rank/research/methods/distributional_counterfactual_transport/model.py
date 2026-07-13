"""Distributional counterfactual prognostic transport."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.faithful_evidence_transport.model import (
    FaithfulEvidenceTransport,
)


class DistributionalCounterfactualTransport(FaithfulEvidenceTransport):
    """Build risk-directed latent counterfactuals by transporting evidence plans.

    The method learns two stage-wise evidence distributions (low-risk and
    high-risk). For each patient it interpolates the factual evidence plan
    toward both distributions and reports the predicted risk change and the
    transport distance. This is a model-faithful latent counterfactual, not a
    causal treatment recommendation.
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.fet_num_stages = int(getattr(args, "dct_num_stages", 4))
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_lambda_ot = float(getattr(args, "dct_lambda_ot", 0.06))
        self.dct_lambda_rank = float(getattr(args, "dct_lambda_rank", 0.05))
        self.dct_lambda_cf = float(getattr(args, "dct_lambda_cf", 0.10))
        self.dct_lambda_proto = float(getattr(args, "dct_lambda_proto", 0.01))
        self.dct_cf_margin = float(getattr(args, "dct_cf_margin", 0.05))
        self.dct_mix_ratio = float(getattr(args, "dct_mix_ratio", 0.50))

        sw = int(getattr(args, "slot_num_wsi"))
        so = int(getattr(args, "slot_num_omics"))
        self.risk_prototypes = nn.Parameter(
            torch.zeros(2, self.spt_num_stages, sw, so)
        )
        nn.init.normal_(self.risk_prototypes, mean=0.0, std=0.02)
        self.cf_strength = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.wsi_projection_dim // 2),
            nn.GELU(),
            nn.Linear(self.wsi_projection_dim // 2, self.spt_num_stages),
        )

    def _prototype_plans(self, batch_size, device, dtype):
        prototypes = self.risk_prototypes.flatten(-2).softmax(dim=-1)
        prototypes = prototypes.view_as(self.risk_prototypes)
        plans = []
        for risk_idx in range(2):
            stage_plans = []
            for stage_idx in range(self.spt_num_stages):
                proto = prototypes[risk_idx, stage_idx].to(device=device, dtype=dtype)
                proto = proto.unsqueeze(0).expand(batch_size, -1, -1)
                stage_plans.append((proto, proto, proto))
            plans.append(stage_plans)
        return plans, prototypes

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
        tokens = tokens + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        event_logits = self.event_hazard(tokens)
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)
        return logits, tokens, gate

    @staticmethod
    def _risk(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    def _counterfactual_loss(self, factual_risk, low_risk, high_risk):
        low_gap = factual_risk - low_risk
        high_gap = high_risk - factual_risk
        return (
            F.relu(self.dct_cf_margin - low_gap).mean()
            + F.relu(self.dct_cf_margin - high_gap).mean()
        )

    def _prototype_sparsity_loss(self, prototypes):
        entropy = -(prototypes.clamp_min(1e-8) * prototypes.clamp_min(1e-8).log()).sum(
            dim=(-2, -1)
        )
        normalizer = max(1.0, float(torch.log(torch.tensor(prototypes.size(-1) * prototypes.size(-2))).item()))
        return (entropy / normalizer).mean()

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        factual_plans, ot_distance = self._stage_transport_plans(
            slots_wsi, slots_omic, epoch
        )
        factual_plans, _, evidence = self._evidence_plan_pairs(
            slots_wsi, slots_omic, factual_plans
        )

        factual_logits, factual_tokens, factual_gate = self._encode_logits_from_plans(
            slots_wsi, slots_omic, factual_plans
        )
        factual_risk = self._risk(factual_logits)

        prototype_plans, prototypes = self._prototype_plans(
            factual_logits.size(0), factual_logits.device, factual_logits.dtype
        )
        low_proto, high_proto = prototype_plans
        strength = self.dct_mix_ratio * torch.sigmoid(
            self.cf_strength(factual_tokens.mean(dim=1))
        )
        low_plans = []
        high_plans = []
        for stage_idx in range(self.spt_num_stages):
            eta = strength[:, stage_idx].reshape(-1, 1, 1)
            low_stage = []
            high_stage = []
            for cost_idx in range(3):
                factual = factual_plans[stage_idx][cost_idx]
                low_stage.append((1.0 - eta) * factual + eta * low_proto[stage_idx][cost_idx])
                high_stage.append((1.0 - eta) * factual + eta * high_proto[stage_idx][cost_idx])
            low_plans.append(tuple(low_stage))
            high_plans.append(tuple(high_stage))

        low_logits, _, _ = self._encode_logits_from_plans(
            slots_wsi, slots_omic, low_plans
        )
        high_logits, _, _ = self._encode_logits_from_plans(
            slots_wsi, slots_omic, high_plans
        )
        low_risk = self._risk(low_logits)
        high_risk = self._risk(high_logits)

        low_distance = torch.stack([
            (factual_plans[k][0] - low_plans[k][0]).abs().mean(dim=(1, 2))
            for k in range(self.spt_num_stages)
        ], dim=1).mean(dim=1)
        high_distance = torch.stack([
            (factual_plans[k][0] - high_plans[k][0]).abs().mean(dim=(1, 2))
            for k in range(self.spt_num_stages)
        ], dim=1).mean(dim=1)
        wsi_assignment, omic_assignment = self._slot_assignments(
            x_wsi_proj, x_omics, slots_wsi, slots_omic
        )

        self.last_explanations = {
            "stage_slot_pair_evidence": evidence,
            "wsi_slot_assignment": wsi_assignment,
            "omic_slot_assignment": omic_assignment,
            "factual_risk": factual_risk.detach(),
            "low_risk_counterfactual": low_risk.detach(),
            "high_risk_counterfactual": high_risk.detach(),
            "low_risk_prototype": prototypes[0].detach(),
            "high_risk_prototype": prototypes[1].detach(),
            "counterfactual_risk_delta_low": (low_risk - factual_risk).detach(),
            "counterfactual_risk_delta_high": (high_risk - factual_risk).detach(),
            "counterfactual_transport_distance_low": low_distance.detach(),
            "counterfactual_transport_distance_high": high_distance.detach(),
            "event_gate": factual_gate.detach(),
        }

        if not self.training:
            return factual_logits, 0.0

        aux_loss = self.dct_lambda_ot * ot_distance
        if "event_time" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.dct_lambda_rank * self._continuous_ranking_loss(
                factual_logits, kwargs["event_time"], kwargs["c"]
            )
        aux_loss = aux_loss + self.dct_lambda_cf * self._counterfactual_loss(
            factual_risk, low_risk, high_risk
        )
        aux_loss = aux_loss + self.dct_lambda_proto * self._prototype_sparsity_loss(
            prototypes
        )
        return factual_logits, aux_loss
