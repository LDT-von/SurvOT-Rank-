"""Censor-aware, risk-anchored counterfactual transport for survival."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.faithful_evidence_transport.model import (
    FaithfulEvidenceTransport,
)
from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    cosine_cost,
    euclidean_cost,
)


class DistributionalCounterfactualTransport(FaithfulEvidenceTransport):
    """Risk-set anchored interventions on re-optimised multimodal transport.

    Patient tokens are pooled by global WSI/pathway prototype dictionaries, so a
    slot index has a shared coordinate system across patients.  Train-fold event
    times define survival stages and a censoring Kaplan--Meier estimate supplies
    IPCW weights.  Each stage therefore has empirical high-event and low-risk-set
    cost anchors.  Interventions happen in cost space and always re-solve OT.

    This is model-based counterfactual sensitivity analysis, not a causal
    treatment recommendation.  No loss imposes an ordering on CF risk outputs.
    """

    _LOW_RISK = 0
    _HIGH_RISK = 1

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.fet_num_stages = int(getattr(args, "dct_num_stages", 4))
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_lambda_ot = float(getattr(args, "dct_lambda_ot", 0.06))
        self.dct_lambda_rank = float(getattr(args, "dct_lambda_rank", 0.05))
        self.dct_lambda_anchor = float(getattr(args, "dct_lambda_anchor", 0.03))
        self.dct_lambda_coordinate = float(getattr(args, "dct_lambda_coordinate", 0.01))
        self.dct_anchor_margin = float(getattr(args, "dct_anchor_margin", 0.02))
        self.dct_anchor_momentum = float(getattr(args, "dct_anchor_momentum", 0.90))
        self.dct_evidence_cost_weight = float(
            getattr(args, "dct_evidence_cost_weight", 0.10)
        )
        self.dct_coupling_projection_iters = int(
            getattr(args, "dct_coupling_projection_iters", 1000)
        )
        self.dct_coupling_projection_tol = float(
            getattr(args, "dct_coupling_projection_tol", 1e-4)
        )
        # A fixed query-time fraction; it is not a learned risk-ordering head.
        self.dct_mix_ratio = float(getattr(args, "dct_mix_ratio", 0.50))
        self.dct_coordinate_temperature = float(
            getattr(args, "dct_coordinate_temperature", 0.20)
        )

        sw = int(getattr(args, "slot_num_wsi"))
        so = int(getattr(args, "slot_num_omics"))
        dim = self.wsi_projection_dim
        # These dictionaries provide global, index-stable coordinates.  Unlike
        # independent Slot Attention slots, prototype j has one identity for all
        # patients before any population cost anchor is averaged.
        self.shared_wsi_prototypes = nn.Parameter(torch.empty(sw, dim))
        self.shared_omic_prototypes = nn.Parameter(torch.empty(so, dim))
        nn.init.normal_(self.shared_wsi_prototypes, std=0.02)
        nn.init.normal_(self.shared_omic_prototypes, std=0.02)

        self.register_buffer(
            "risk_anchor_costs",
            torch.zeros(self.spt_num_stages, 2, 3, sw, so),
        )
        self.register_buffer(
            "risk_anchor_seen", torch.zeros(self.spt_num_stages, 2, dtype=torch.bool)
        )
        # Set only by configure_train_reference(), which receives this fold's
        # training labels.  Validation labels never enter these buffers.
        self.register_buffer("dct_stage_edges", torch.empty(0))
        self.register_buffer("dct_censor_times", torch.empty(0))
        self.register_buffer("dct_censor_survival", torch.empty(0))

    @staticmethod
    def _risk(logits):
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    @property
    def has_train_reference(self):
        return self.dct_stage_edges.numel() == self.spt_num_stages + 1

    @torch.no_grad()
    def configure_train_reference(self, event_times, censorship):
        """Fit time stages and IPCW censoring survival from one fold's train set.

        ``c == 0`` denotes an observed event and ``c == 1`` denotes censoring.
        Stage upper edges are event-time quantiles.  The final edge is finite so
        patients followed beyond it can contribute to the final low-risk anchor.
        """
        device = self.risk_anchor_costs.device
        times = torch.as_tensor(event_times, dtype=torch.float32, device=device).flatten()
        cens = torch.as_tensor(censorship, dtype=torch.float32, device=device).flatten()
        observed = times[cens < 0.5]
        if observed.numel() < self.spt_num_stages:
            raise ValueError(
                "DCT needs at least dct_num_stages observed training events to fit stage anchors."
            )
        quantiles = torch.linspace(
            1.0 / self.spt_num_stages, 1.0, self.spt_num_stages, device=device
        )
        upper = torch.quantile(observed, quantiles)
        # Strictly increasing edges make stage membership deterministic even with ties.
        upper = torch.maximum(upper, torch.cummax(upper, dim=0).values)
        eps = torch.finfo(upper.dtype).eps
        for idx in range(1, upper.numel()):
            upper[idx] = torch.maximum(upper[idx], upper[idx - 1] + eps)
        self.dct_stage_edges = torch.cat([upper.new_tensor([-float("inf")]), upper])

        unique_times = torch.unique(times, sorted=True)
        censor_survival = torch.ones_like(unique_times)
        value = torch.ones((), dtype=times.dtype, device=device)
        for idx, time in enumerate(unique_times):
            at_risk = (times >= time).sum().to(times.dtype).clamp_min(1.0)
            censor_events = ((times == time) & (cens >= 0.5)).sum().to(times.dtype)
            value = value * (1.0 - censor_events / at_risk)
            censor_survival[idx] = value
        self.dct_censor_times = unique_times
        self.dct_censor_survival = censor_survival.clamp_min(0.05)
        self.risk_anchor_costs.zero_()
        self.risk_anchor_seen.zero_()

    def _ipcw(self, query_times):
        if self.dct_censor_times.numel() == 0:
            return torch.ones_like(query_times)
        indices = torch.searchsorted(self.dct_censor_times, query_times, right=True) - 1
        indices = indices.clamp_min(0)
        values = self.dct_censor_survival[indices]
        return values.clamp_min(0.05).reciprocal()

    def _semantic_slots(self, tokens, prototypes):
        """Pool variable patient tokens into globally indexed prototype slots."""
        keys = F.normalize(prototypes, dim=-1)
        normalized_tokens = F.normalize(tokens, dim=-1)
        scores = torch.einsum("bnd,kd->bkn", normalized_tokens, keys)
        weights = torch.softmax(scores / self.dct_coordinate_temperature, dim=-1)
        slots = torch.einsum("bkn,bnd->bkd", weights, tokens)
        return slots, weights

    def _coordinate_loss(self):
        def _orthogonality(prototypes):
            gram = F.normalize(prototypes, dim=-1) @ F.normalize(prototypes, dim=-1).transpose(0, 1)
            return (gram - torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)).square().mean()

        return _orthogonality(self.shared_wsi_prototypes) + _orthogonality(self.shared_omic_prototypes)

    def _sinkhorn_eps(self, epoch):
        end = float(getattr(self.args, "otehv2_eps", 0.05))
        start = float(getattr(self.args, "rg_eps_start", end * 2.0))
        anneal = max(1, int(getattr(self.args, "rg_eps_anneal", 12)))
        return start + min(1.0, epoch / anneal) * (end - start)

    def _cost_tensor(self, slots_wsi, slots_omic):
        """Return stage costs and evidence-conditioned OT marginals."""
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        bsz, sw, so, dim4 = pair_tokens.shape
        dim = dim4 // 4
        stage_cost = F.softplus(self.stage_pair_cost(pair_tokens)).permute(0, 3, 1, 2)
        base_costs = (
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        )

        all_stage_costs, row_marginals, col_marginals, gates = [], [], [], []
        for stage_idx in range(self.spt_num_stages):
            stage_code = self.stage_embedding[stage_idx].view(1, 1, 1, dim)
            stage_code = stage_code.expand(bsz, sw, so, dim)
            gate = torch.sigmoid(
                self.evidence_gate(torch.cat([pair_tokens, stage_code], dim=-1)).squeeze(-1)
            )
            evidence_cost = self._normalize_stage_cost(-torch.log(gate.clamp_min(1e-6)))
            prognostic_cost = self._normalize_stage_cost(stage_cost[:, stage_idx])
            all_stage_costs.append(torch.stack([
                base_cost
                + self.spt_prog_cost_weight * prognostic_cost
                + self.dct_evidence_cost_weight * evidence_cost
                for base_cost in base_costs
            ], dim=1))
            # Gate affects both the energy and how much each semantic slot is
            # allowed to transport.  This avoids forcing weak evidence to carry
            # uniform mass merely because standard balanced OT requires it.
            row_marginals.append(gate.mean(dim=-1).clamp_min(1e-6))
            col_marginals.append(gate.mean(dim=-2).clamp_min(1e-6))
            gates.append(gate)
        rows = torch.stack(row_marginals, dim=1)
        cols = torch.stack(col_marginals, dim=1)
        rows = rows / rows.sum(dim=-1, keepdim=True)
        cols = cols / cols.sum(dim=-1, keepdim=True)
        return torch.stack(all_stage_costs, dim=1), rows, cols, torch.stack(gates, dim=1)

    @staticmethod
    def _log_sinkhorn(cost, rows, cols, eps, max_iter):
        kernel = -cost / eps
        log_rows = rows.clamp_min(1e-8).log()
        log_cols = cols.clamp_min(1e-8).log()
        log_u = torch.zeros_like(log_rows)
        log_v = torch.zeros_like(log_cols)
        for _ in range(max_iter):
            log_u = log_rows - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
            log_v = log_cols - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)
        return (kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)).exp()

    def _project_coupling(self, plan, rows, cols):
        """Numerically project a positive plan to its evidence-conditioned marginals."""
        for _ in range(self.dct_coupling_projection_iters):
            plan = plan * (rows.unsqueeze(-1) / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8))
            plan = plan * (cols.unsqueeze(1) / plan.sum(dim=-2, keepdim=True).clamp_min(1e-8))
            row_error = (plan.sum(dim=-1) - rows).abs().amax()
            col_error = (plan.sum(dim=-2) - cols).abs().amax()
            if bool(torch.maximum(row_error, col_error).detach() <= self.dct_coupling_projection_tol):
                break
        return plan

    def _plans_from_cost_tensor(self, costs, rows, cols, epoch):
        eps = self._sinkhorn_eps(epoch)
        plans, distances = [], []
        for stage_idx in range(self.spt_num_stages):
            stage_plans, stage_distances = [], []
            for cost_idx in range(costs.size(2)):
                plan = self._log_sinkhorn(
                    costs[:, stage_idx, cost_idx], rows[:, stage_idx], cols[:, stage_idx],
                    eps=eps, max_iter=self.ot_iter,
                )
                plan = self._project_coupling(plan, rows[:, stage_idx], cols[:, stage_idx])
                stage_plans.append(plan)
                stage_distances.append((plan * costs[:, stage_idx, cost_idx]).sum(dim=(1, 2)))
            plans.append(tuple(stage_plans))
            distances.append(torch.stack(stage_distances).mean())
        return plans, torch.stack(distances).mean()

    def _stage_membership_weights(self, event_time, censorship):
        """IPCW weights for event-in-stage (high) and survived-past-stage (low)."""
        bsz = event_time.size(0)
        high = event_time.new_zeros(bsz, self.spt_num_stages)
        low = event_time.new_zeros(bsz, self.spt_num_stages)
        if not self.has_train_reference:
            return low, high
        observed = censorship < 0.5
        for stage_idx in range(self.spt_num_stages):
            lower = self.dct_stage_edges[stage_idx]
            upper = self.dct_stage_edges[stage_idx + 1]
            in_stage = observed & (event_time > lower) & (event_time <= upper)
            # Event after the upper boundary or censoring after it proves survival
            # through this stage.  Censoring before the boundary contributes zero.
            survived_stage = (event_time > upper) | ((censorship >= 0.5) & (event_time >= upper))
            high[:, stage_idx] = in_stage.to(event_time.dtype) * self._ipcw(event_time)
            low[:, stage_idx] = survived_stage.to(event_time.dtype) * self._ipcw(
                torch.full_like(event_time, upper)
            )
        return low, high

    def _anchor_contrastive_loss(self, costs, low_weights, high_weights):
        losses = []
        for stage_idx in range(self.spt_num_stages):
            for risk_idx, weights in (
                (self._LOW_RISK, low_weights[:, stage_idx]),
                (self._HIGH_RISK, high_weights[:, stage_idx]),
            ):
                if not bool(self.risk_anchor_seen[stage_idx].all()):
                    continue
                valid = weights > 0
                if not bool(valid.any()):
                    continue
                values = costs[valid, stage_idx]
                own = self.risk_anchor_costs[stage_idx, risk_idx].to(dtype=costs.dtype)
                other = self.risk_anchor_costs[stage_idx, 1 - risk_idx].to(dtype=costs.dtype)
                own_distance = (values - own).square().mean(dim=(1, 2, 3))
                other_distance = (values - other).square().mean(dim=(1, 2, 3))
                hinge = F.relu(self.dct_anchor_margin + own_distance - other_distance)
                losses.append((hinge * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6))
        return torch.stack(losses).mean() if losses else costs.new_zeros(())

    @torch.no_grad()
    def _update_risk_anchors(self, costs, low_weights, high_weights):
        for stage_idx in range(self.spt_num_stages):
            for risk_idx, weights in (
                (self._LOW_RISK, low_weights[:, stage_idx]),
                (self._HIGH_RISK, high_weights[:, stage_idx]),
            ):
                if not bool((weights > 0).any()):
                    continue
                weighted = costs[:, stage_idx] * weights.view(-1, 1, 1, 1)
                current = weighted.sum(dim=0) / weights.sum().clamp_min(1e-6)
                if bool(self.risk_anchor_seen[stage_idx, risk_idx]):
                    self.risk_anchor_costs[stage_idx, risk_idx].lerp_(
                        current.to(dtype=self.risk_anchor_costs.dtype),
                        1.0 - self.dct_anchor_momentum,
                    )
                else:
                    self.risk_anchor_costs[stage_idx, risk_idx].copy_(
                        current.to(dtype=self.risk_anchor_costs.dtype)
                    )
                    self.risk_anchor_seen[stage_idx, risk_idx] = True

    def _counterfactual_costs(self, factual_costs):
        alpha = min(1.0, max(0.0, self.dct_mix_ratio))
        bsz = factual_costs.size(0)
        anchors = self.risk_anchor_costs.to(device=factual_costs.device, dtype=factual_costs.dtype)
        low_anchor = anchors[:, self._LOW_RISK].unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        high_anchor = anchors[:, self._HIGH_RISK].unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        low_seen = self.risk_anchor_seen[:, self._LOW_RISK].view(1, -1, 1, 1, 1)
        high_seen = self.risk_anchor_seen[:, self._HIGH_RISK].view(1, -1, 1, 1, 1)
        low_anchor = torch.where(low_seen, low_anchor, factual_costs)
        high_anchor = torch.where(high_seen, high_anchor, factual_costs)
        return (
            (1.0 - alpha) * factual_costs + alpha * low_anchor,
            (1.0 - alpha) * factual_costs + alpha * high_anchor,
        )

    @staticmethod
    def _marginal_error(plans, rows, cols):
        errors = []
        for stage_idx, stage_plans in enumerate(plans):
            for plan in stage_plans:
                row_error = (plan.sum(dim=-1) - rows[:, stage_idx]).abs().amax(dim=-1)
                col_error = (plan.sum(dim=-2) - cols[:, stage_idx]).abs().amax(dim=-1)
                errors.append(torch.maximum(row_error, col_error))
        return torch.stack(errors, dim=1).amax(dim=1)

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
        tokens = tokens + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        event_logits = self.event_hazard(tokens)
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)
        return logits, gate

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi, wsi_coordinate_assignment = self._semantic_slots(
            x_wsi_proj, self.shared_wsi_prototypes
        )
        slots_omic, omic_coordinate_assignment = self._semantic_slots(
            x_omics, self.shared_omic_prototypes
        )
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))

        factual_costs, rows, cols, evidence_gate = self._cost_tensor(slots_wsi, slots_omic)
        factual_plans, ot_distance = self._plans_from_cost_tensor(factual_costs, rows, cols, epoch)
        factual_logits, factual_gate = self._encode_logits_from_plans(slots_wsi, slots_omic, factual_plans)
        factual_risk = self._risk(factual_logits)

        low_weights = factual_costs.new_zeros(factual_costs.size(0), self.spt_num_stages)
        high_weights = torch.zeros_like(low_weights)
        anchor_loss = factual_costs.new_zeros(())
        if kwargs.get("event_time") is not None and kwargs.get("c") is not None:
            low_weights, high_weights = self._stage_membership_weights(kwargs["event_time"], kwargs["c"])
            anchor_loss = self._anchor_contrastive_loss(factual_costs, low_weights, high_weights)
            if self.training:
                self._update_risk_anchors(factual_costs.detach(), low_weights, high_weights)

        low_costs, high_costs = self._counterfactual_costs(factual_costs)
        low_plans, _ = self._plans_from_cost_tensor(low_costs, rows, cols, epoch)
        high_plans, _ = self._plans_from_cost_tensor(high_costs, rows, cols, epoch)
        low_logits, _ = self._encode_logits_from_plans(slots_wsi, slots_omic, low_plans)
        high_logits, _ = self._encode_logits_from_plans(slots_wsi, slots_omic, high_plans)
        low_risk, high_risk = self._risk(low_logits), self._risk(high_logits)

        low_distance = (factual_costs - low_costs).abs().mean(dim=(1, 2, 3, 4))
        high_distance = (factual_costs - high_costs).abs().mean(dim=(1, 2, 3, 4))
        self.last_explanations = {
            "stage_slot_pair_evidence": torch.stack([item[0] for item in factual_plans], dim=1).detach(),
            "evidence_gate": evidence_gate.detach(),
            "wsi_coordinate_assignment": wsi_coordinate_assignment.detach(),
            "omic_coordinate_assignment": omic_coordinate_assignment.detach(),
            "factual_risk": factual_risk.detach(),
            "low_risk_counterfactual": low_risk.detach(),
            "high_risk_counterfactual": high_risk.detach(),
            "counterfactual_risk_delta_low": (low_risk - factual_risk).detach(),
            "counterfactual_risk_delta_high": (high_risk - factual_risk).detach(),
            "counterfactual_transport_distance_low": low_distance.detach(),
            "counterfactual_transport_distance_high": high_distance.detach(),
            "risk_anchor_costs": self.risk_anchor_costs.detach(),
            "risk_anchor_seen": self.risk_anchor_seen.detach(),
            "stage_edges": self.dct_stage_edges.detach(),
            "low_risk_set_ipcw": low_weights.detach(),
            "high_event_ipcw": high_weights.detach(),
            "factual_coupling_marginal_error": self._marginal_error(factual_plans, rows, cols).detach(),
            "low_coupling_marginal_error": self._marginal_error(low_plans, rows, cols).detach(),
            "high_coupling_marginal_error": self._marginal_error(high_plans, rows, cols).detach(),
            "event_gate": factual_gate.detach(),
        }

        if not self.training:
            return factual_logits, 0.0

        aux_loss = self.dct_lambda_ot * ot_distance
        if "event_time" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.dct_lambda_rank * self._continuous_ranking_loss(
                factual_logits, kwargs["event_time"], kwargs["c"]
            )
        aux_loss = aux_loss + self.dct_lambda_anchor * anchor_loss
        aux_loss = aux_loss + self.dct_lambda_coordinate * self._coordinate_loss()
        return factual_logits, aux_loss
