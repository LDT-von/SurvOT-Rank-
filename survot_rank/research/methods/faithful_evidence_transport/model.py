"""Faithful, stage-aware evidence transport for multimodal survival."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.stagewise_prognostic_transport.model import (
    StagewisePrognosticTransport,
)


class FaithfulEvidenceTransport(StagewisePrognosticTransport):
    """Expose and train a causal-to-the-model evidence path.

    Each stage has a learned slot-pair evidence gate. The gate changes the OT
    plan used by the corresponding event token, so the explanation is coupled
    to the prediction path rather than produced by a detached attention map.
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.spt_num_stages = int(getattr(args, "fet_num_stages", 4))
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.fet_lambda_sparse = float(getattr(args, "fet_lambda_sparse", 0.01))
        self.fet_lambda_faith = float(getattr(args, "fet_lambda_faith", 0.05))
        self.fet_keep_ratio = float(getattr(args, "fet_keep_ratio", 0.25))
        self.fet_faith_margin = float(getattr(args, "fet_faith_margin", 0.05))
        self.evidence_gate = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.last_explanations = None

    @staticmethod
    def _renormalize_plan(plan, weights):
        weighted = plan * weights
        return weighted / weighted.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _evidence_plan_pairs(self, slots_wsi, slots_omic, plans):
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        bsz, sw, so, dim4 = pair_tokens.shape
        dim = dim4 // 4
        evidence_plans = []
        removed_plans = []
        evidence_scores = []

        for stage_idx, stage_plans in enumerate(plans):
            stage_code = self.stage_embedding[stage_idx].view(1, 1, 1, dim)
            stage_code = stage_code.expand(bsz, sw, so, dim)
            gate = torch.sigmoid(
                self.evidence_gate(torch.cat([pair_tokens, stage_code], dim=-1)).squeeze(-1)
            )
            keep_plans = tuple(self._renormalize_plan(plan, gate) for plan in stage_plans)
            remove_plans = tuple(
                self._renormalize_plan(plan, 1.0 - gate) for plan in stage_plans
            )
            evidence_plans.append(keep_plans)
            removed_plans.append(remove_plans)
            evidence_scores.append(keep_plans[0].detach())

        return evidence_plans, removed_plans, torch.stack(evidence_scores, dim=1)

    def _selected_stage_events(self, slots_wsi, slots_omic, plans):
        selected = []
        for stage_idx, (plan_cos, plan_euc, plan_dot) in enumerate(plans):
            all_events, _ = self.fusion(
                slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot
            )
            selected.append(all_events[:, stage_idx:stage_idx + 1])
        return torch.cat(selected, dim=1)

    @staticmethod
    def _risk_score(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    def _evidence_losses(self, logits, removed_logits, evidence):
        probs = evidence.flatten(2)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(1.0, float(torch.log(torch.tensor(probs.size(-1))).item()))
        sparse_loss = entropy.mean()

        risk_full = self._risk_score(logits)
        risk_removed = self._risk_score(removed_logits)
        faithfulness = F.relu(
            self.fet_faith_margin - (risk_full - risk_removed).abs()
        ).mean()
        return sparse_loss, faithfulness

    def _slot_assignments(self, x_wsi_proj, x_omics, slots_wsi, slots_omic):
        wsi_assign = torch.softmax(
            F.normalize(x_wsi_proj, dim=-1) @ F.normalize(slots_wsi, dim=-1).transpose(1, 2),
            dim=-1,
        )
        omic_assign = torch.softmax(
            F.normalize(x_omics, dim=-1) @ F.normalize(slots_omic, dim=-1).transpose(1, 2),
            dim=-1,
        )
        return wsi_assign.detach(), omic_assign.detach()

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        plans, ot_distance = self._stage_transport_plans(slots_wsi, slots_omic, epoch)
        evidence_plans, removed_plans, evidence = self._evidence_plan_pairs(
            slots_wsi, slots_omic, plans
        )

        event_tokens = self._selected_stage_events(slots_wsi, slots_omic, evidence_plans)
        event_tokens = event_tokens + self.stage_embedding.unsqueeze(0)
        event_tokens = self.event_norm(self.event_encoder(event_tokens))
        event_logits = self.event_hazard(event_tokens)
        gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)

        removed_tokens = self._selected_stage_events(slots_wsi, slots_omic, removed_plans)
        removed_tokens = removed_tokens + self.stage_embedding.unsqueeze(0)
        removed_tokens = self.event_norm(self.event_encoder(removed_tokens))
        removed_event_logits = self.event_hazard(removed_tokens)
        removed_gate = torch.softmax(self.event_gate(removed_tokens).squeeze(-1), dim=1)
        removed_logits = torch.einsum("be,bec->bc", removed_gate, removed_event_logits)

        wsi_assignment, omic_assignment = self._slot_assignments(
            x_wsi_proj, x_omics, slots_wsi, slots_omic
        )
        self.last_explanations = {
            "stage_slot_pair_evidence": evidence,
            "wsi_slot_assignment": wsi_assignment,
            "omic_slot_assignment": omic_assignment,
            "event_gate": gate.detach(),
        }

        if not self.training:
            return logits, 0.0

        sparse_loss, faithfulness_loss = self._evidence_losses(
            logits, removed_logits, evidence
        )
        aux_loss = self.spt_lambda_ot * ot_distance
        if "event_time" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.spt_lambda_rank * self._continuous_ranking_loss(
                logits, kwargs["event_time"], kwargs["c"]
            )
        aux_loss = aux_loss + self.spt_lambda_stage * self._stage_order_loss(event_tokens)
        aux_loss = aux_loss + self.fet_lambda_sparse * sparse_loss
        aux_loss = aux_loss + self.fet_lambda_faith * faithfulness_loss
        return logits, aux_loss

    def explain_last_batch(self):
        """Return detached stage/slot evidence from the most recent forward pass."""
        if self.last_explanations is None:
            raise RuntimeError("Run a forward pass before requesting explanations")
        return self.last_explanations
