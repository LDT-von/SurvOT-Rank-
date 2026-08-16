"""CATET v2 — original base + cohort-anchored pre-routing.

# Base
Original from commit 6f6295c (catet_old BLCA f0=0.6458, f2=0.6837, mean
0.6648). The mechanism: stage-specific OT plan (4 stages) is biased by a
learned prognostic edge signal; a temporal evidence gate selectively keeps a
fraction of each plan; ranking risk-set supervises the gated evidence; and
keep/remove interventions audit factual risk faithfulness.

# Inherited from 15aa637 (catet_fix)
Monotonic OT eps annealing (start->end), no `epoch > 0` guard. Avoids the
discontinuity bug where the e0 lambda dropped straight to the sharpest value.

# Additions (cross-method fusion)
1. CA-PSA-style *cohort-anchored pre-routing* on the wsi and omic slot
   spaces before the stage-conditioned OT.  This routes each slot to a
   small set of cohort routes and reduces the OT cost-matrix dimension
   from (slots × slots) to (routes × routes).  Empirically stabilises
   training on small cohorts and aligns the two modalities on a shared
   cohort identity.
2. ArcSurv-style *archetype priors* on the stage edge bias: stage 0 uses
   the cohort archetype average as a constant per-stage prior; this gives
   the OT plan an initial hierarchy without relying entirely on the
   learned edge risk.

# Defaults match the original: batch=4, max_epochs=30, lr=5e-4.  Use
configs/catet_v2_blca.yaml to override to batch=16 with warmup/clip.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    OTEventHazardV2Survival,
    cosine_cost,
    log_sinkhorn_plan,
)


def _safe_euclidean_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Numerically stable euclidean cost.

    The parent OTEventHazardV2 implementation uses ``sqrt((x-y)^2)`` which
    produces NaN gradients via ``PowBackward0`` whenever ``x == y`` at any
    (i, j) pair — common after the cohort router aligns slots tightly.  We
    clamp the inner sum before sqrt so the backward path is well-defined.
    """
    x_u = x.unsqueeze(2)        # [B, K_w, 1, D]
    y_u = y.unsqueeze(1)        # [B, 1, K_o, D]
    squared = (x_u - y_u).pow(2).sum(-1).clamp_min(1e-12)
    return squared.sqrt()


# Local alias so downstream code reads the same as the original import.
euclidean_cost = _safe_euclidean_cost


class CohortAnchoredRouter(nn.Module):
    """Coarse-grained cohort router — a CA-PSA-style module adapted to OT.

    Each modality encodes a soft assignment to `num_routes` cohort identities,
    and we return a slot embedding computed by reading from the bank weighted
    by that assignment. The OT then operates on the reduced `routes × routes`
    cost matrix rather than the noisy `slots × slots` matrix.
    """

    def __init__(self, dim: int, num_routes: int = 4, topk: int = 2):
        super().__init__()
        if num_routes < 2:
            raise ValueError("catet_cohort_routes must be >= 2")
        if topk < 1 or topk > num_routes:
            raise ValueError("catet_cohort_topk must be in [1, num_routes]")
        self.dim = dim
        self.num_routes = num_routes
        self.topk = topk
        self.queries = nn.Parameter(torch.randn(num_routes, dim) * 0.02)
        self.slot_to_query = nn.Linear(dim, dim, bias=False)
        self.route_norm = nn.LayerNorm(dim)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map slot tokens to a slimmed-down route summary.

        Returns (routed [B, R, D], assignment [B, N, R]).

        Top-K gating uses a straight-through estimator: forward keeps only
        the K highest-weighted routes per token, backward flows through the
        full softmax so the gradient does not vanish on the masked-out
        routes.  Without this, the autograd through
        ``(assignment * gate).sum(...)`` produces NaN gradients whenever a
        masked-out route carries weight.
        """
        keys = self.slot_to_query(slots)
        scores = torch.einsum("bnd,rd->bnr", keys, self.queries)
        assignment = scores.softmax(dim=-1)
        # Straight-through hard top-K gate.
        topk_idx = assignment.topk(self.topk, dim=-1).indices
        gate = torch.zeros_like(assignment)
        gate.scatter_(-1, topk_idx, 1.0)
        # Forward: hard mask.  Backward: pass-through of `assignment`.
        masked = assignment * gate + (assignment - assignment.detach())
        # Renormalise so the convex weight sums to 1.
        masked_sum = masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        soft_assignment = masked / masked_sum
        routed = torch.einsum("bnr,bnd->brd", soft_assignment, slots)
        routed = self.route_norm(routed)
        return routed, assignment


class CensoringAwareTemporalEvidenceTransport(OTEventHazardV2Survival):
    """Stage-specific, risk-set supervised OT evidence model — v2."""

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

        # v2 additions
        self.catet_cohort_routes = int(getattr(args, "catet_cohort_routes", 4))
        self.catet_cohort_topk = int(getattr(args, "catet_cohort_topk", 2))
        self.catet_lambda_route = float(getattr(args, "catet_lambda_route", 0.02))
        self.catet_use_archetype_prior = bool(
            int(getattr(args, "catet_use_archetype_prior", 0))
        )

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

        # v2: cohort router instances (one per modality, shared cohort space).
        self.cohort_router_wsi = CohortAnchoredRouter(
            dim,
            num_routes=self.catet_cohort_routes,
            topk=self.catet_cohort_topk,
        )
        self.cohort_router_omic = CohortAnchoredRouter(
            dim,
            num_routes=self.catet_cohort_routes,
            topk=self.catet_cohort_topk,
        )

        # v2: archetype-derived stage priors if enabled.  The first call to
        # `forward` will populate this lazily from batch output; subsequent
        # calls reuse the registered parameter.
        self.register_buffer(
            "archetype_prior_per_stage",
            torch.zeros(self.catet_num_stages, dim),
        )
        self.register_buffer("archetype_prior_filled", torch.zeros((), dtype=torch.bool))

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
        # v2: route both modalities through a shared cohort space first.
        routed_wsi, _ = self.cohort_router_wsi(slots_wsi)
        routed_omic, _ = self.cohort_router_omic(slots_omic)
        pair_tokens = self._pair_tokens(routed_wsi, routed_omic)
        edge_risk = torch.sigmoid(self.stage_edge_risk(pair_tokens).squeeze(-1))
        edge_risk = edge_risk.unsqueeze(1).expand(-1, self.catet_num_stages, -1, -1)

        stage_bias = 1.0 - edge_risk
        stage_bias = self._normalize_cost(stage_bias.flatten(0, 1)).view_as(stage_bias)

        # v2: add an optional archetype-style stage prior.  Even if not yet
        # filled, we use a learnable zero buffer (which the einsum below
        # absorbs gracefully).
        if self.catet_use_archetype_prior and bool(self.archetype_prior_filled):
            prior = self.archetype_prior_per_stage.view(
                1, self.catet_num_stages, 1, 1, -1
            )
            # routed_diff[B, 1, R, R, D]: wsi slot i minus omic slot j, for
            # every (i, j) pair.  Broadcasting against prior[S, 1, 1, 1, D]
            # yields [B, S, R, R, D]; summing the D dim gives the per-stage
            # bias [B, S, R, R].  L2-normalise the prior so its scale is
            # independent of the slot dimension and the gradient stays stable.
            prior = prior / prior.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            routed_diff = (
                routed_wsi.unsqueeze(2) - routed_omic.unsqueeze(1)
            ).unsqueeze(1)
            archetype_term = -(
                prior * routed_diff
            ).sum(dim=-1)
            archetype_term = torch.tanh(
                self._normalize_cost(
                    archetype_term.flatten(0, 1)
                ).view_as(stage_bias)
            )
            stage_bias = 0.5 * stage_bias + 0.5 * archetype_term

        base_costs = [
            self._normalize_cost(cosine_cost(routed_wsi, routed_omic)),
            self._normalize_cost(euclidean_cost(routed_wsi, routed_omic)),
            self._normalize_cost(self._positive_dot_cost(routed_wsi, routed_omic)),
        ]
        # Inherited monotonic OT-eps anneal (15aa637 fix).
        end = float(getattr(self.args, "otehv2_eps", 0.05))
        start = float(getattr(self.args, "rg_eps_start", end * 2.0))
        frac = min(1.0, epoch / max(1, int(getattr(self.args, "rg_eps_anneal", 12))))
        eps = start + frac * (end - start)

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

    def _gated_plans_on(
        self,
        routed_wsi: torch.Tensor,
        routed_omic: torch.Tensor,
        plans,
    ):
        # v2: gating uses the routed pair tokens so that the gate tensor
        # agrees with the (R, R) plan shape used everywhere downstream.
        pair_tokens = self._pair_tokens(routed_wsi, routed_omic)
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
            gated.append(
                tuple(self._renormalize_plan(p, gate) for p in stage_plans)
            )
            removed.append(
                tuple(self._renormalize_plan(p, 1.0 - gate) for p in stage_plans)
            )
            gates.append(gate)
        return gated, removed, torch.stack(gates, dim=1)

    # Backwards-compatible alias — unchanged behaviour routed through the
    # routed pair tokens, but kept so older external callers do not break.
    def _gated_plans(self, slots_wsi, slots_omic, plans):
        routed_wsi, _ = self.cohort_router_wsi(slots_wsi)
        routed_omic, _ = self.cohort_router_omic(slots_omic)
        return self._gated_plans_on(routed_wsi, routed_omic, plans)

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
        sufficiency = F.smooth_l1_loss(evidence_risk, full_risk)
        flat = evidence.flatten(2)
        keep_n = max(1, int(flat.size(-1) * self.catet_keep_ratio))
        threshold = flat.topk(keep_n, dim=-1).values[..., -1:].detach()
        selected = (flat >= threshold).to(flat.dtype).view_as(evidence)
        selected_mass = (evidence * selected).sum(dim=(2, 3))
        full_mass = evidence.sum(dim=(2, 3)).clamp_min(1e-6)
        sparsity = (selected_mass / full_mass).mean()
        change = (full_risk - removed_risk).abs()
        comprehensiveness = F.relu(self.catet_intervention_margin - change).mean()
        return sufficiency + comprehensiveness + 0.1 * sparsity

    def _route_consistency_loss(
        self,
        slots_wsi: torch.Tensor,
        slots_omic: torch.Tensor,
    ) -> torch.Tensor:
        """v2: encourage both modalities to share cohort routing."""
        _, assign_wsi = self.cohort_router_wsi(slots_wsi)
        _, assign_omic = self.cohort_router_omic(slots_omic)
        # Soft assignment average per patient; symmetric KL keeps the JS-style
        # alignment bounded.
        wsi_avg = assign_wsi.mean(dim=1)
        omic_avg = assign_omic.mean(dim=1)
        eps = 1e-8
        kl_w_o = F.kl_div(
            (wsi_avg + eps).log(),
            omic_avg,
            reduction="batchmean",
        )
        kl_o_w = F.kl_div(
            (omic_avg + eps).log(),
            wsi_avg,
            reduction="batchmean",
        )
        return 0.5 * (kl_w_o + kl_o_w)

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        # v2: route both modalities through a shared cohort space first.
        routed_wsi, _ = self.cohort_router_wsi(slots_wsi)
        routed_omic, _ = self.cohort_router_omic(slots_omic)

        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        plans, ot_distance, edge_risk = self._stage_transport_plans(
            slots_wsi, slots_omic, epoch
        )
        # v2: gating now operates on the routed pair tokens (`routed_wsi`,
        # `routed_omic`) so the cost-matrix dimension stays consistent with
        # the plans returned by `_stage_transport_plans`.
        evidence_plans, removed_plans, evidence_gate = self._gated_plans_on(
            routed_wsi, routed_omic, plans
        )
        # v2: decode using routed slots so fusion's internal projection does
        # not see a mismatch between the slot tensor and plan tensor shape.
        full_logits, _, _ = self._decode(routed_wsi, routed_omic, plans)
        logits, tokens, event_gate = self._decode(
            routed_wsi, routed_omic, evidence_plans
        )
        removed_logits, _, _ = self._decode(
            routed_wsi, routed_omic, removed_plans
        )

        # v2: lazily fill archetype prior from the first-batch mean of routed
        # WSI/omic states.  Once populated, the prior acts as a stage-zero
        # anchor.
        if self.catet_use_archetype_prior and not bool(self.archetype_prior_filled):
            mean_state = 0.5 * (
                routed_wsi.mean(dim=1) + routed_omic.mean(dim=1)
            ).mean(dim=0)
            self.archetype_prior_per_stage.copy_(
                mean_state.unsqueeze(0).expand(self.catet_num_stages, -1)
            )
            self.archetype_prior_filled.fill_(True)

        transport_evidence = (edge_risk * evidence_gate).flatten(2).mean(dim=-1)
        transport_evidence = (transport_evidence * event_gate).sum(dim=1)
        self.last_explanations = {
            "stage_slot_pair_evidence": evidence_gate.detach(),
            "stage_slot_pair_risk": edge_risk.detach(),
            "transport_evidence_risk": transport_evidence.detach(),
            "event_gate": event_gate.detach(),
            "factual_risk": self._risk_score(logits).detach(),
            "removed_risk": self._risk_score(removed_logits).detach(),
            "cohort_assignment_wsi": self.cohort_router_wsi(slots_wsi)[1].detach(),
            "cohort_assignment_omic": self.cohort_router_omic(slots_omic)[1].detach(),
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
        # v2: cohort-routing KL encourages cross-modal slot fusion.
        aux_loss = aux_loss + self.catet_lambda_route * self._route_consistency_loss(
            slots_wsi, slots_omic
        )
        return logits, aux_loss

    def explain_last_batch(self):
        if self.last_explanations is None:
            raise RuntimeError("Run a forward pass before requesting explanations")
        return self.last_explanations
