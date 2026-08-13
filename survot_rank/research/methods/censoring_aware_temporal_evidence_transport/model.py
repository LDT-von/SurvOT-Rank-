"""Censoring-aware temporal evidence re-transport for multimodal survival.

CATET Final couples four quantities that refer to the same prediction path:
stage-conditioned balanced OT, factual hazard prediction, training-fold IPCW
supervision, and balanced keep/remove re-transport interventions.  The method
does not claim a treatment effect; its counterfactuals are model-faithfulness
tests over cross-modal evidence.
"""

from __future__ import annotations

import math

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
    """Stage-conditioned balanced OT with censor-aware faithfulness audits."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.catet_num_stages = int(getattr(args, "catet_num_stages", 4))
        args.otehv2_num_events = args.catet_num_stages
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.catet_num_stages = int(args.catet_num_stages)
        self.catet_prog_cost_weight = float(getattr(args, "catet_prog_cost", 0.20))
        self.catet_lambda_ot = float(getattr(args, "catet_lambda_ot", 0.04))
        self.catet_lambda_rank = float(getattr(args, "catet_lambda_rank", 0.08))
        self.catet_lambda_stage = float(getattr(args, "catet_lambda_stage", 0.04))
        self.catet_lambda_intervention = float(
            getattr(args, "catet_lambda_intervention", 0.05)
        )
        self.catet_keep_ratio = float(getattr(args, "catet_keep_ratio", 0.25))
        self.catet_intervention_margin = float(
            getattr(args, "catet_intervention_margin", 0.05)
        )
        self.catet_intervention_cost = float(
            getattr(args, "catet_intervention_cost", 1.0)
        )
        self.catet_plan_diversity_margin = float(
            getattr(args, "catet_plan_diversity_margin", 0.01)
        )
        self.catet_rank_margin = float(getattr(args, "catet_rank_margin", 0.0))
        self.catet_rank_temperature = float(
            getattr(args, "catet_rank_temperature", 0.50)
        )
        self.catet_ipcw_max_weight = float(
            getattr(args, "catet_ipcw_max_weight", 10.0)
        )
        self.catet_rank_max_pairs = int(getattr(args, "catet_rank_max_pairs", 4096))
        self._validate_hyperparameters()

        stage_embedding = torch.linspace(-1.0, 1.0, self.catet_num_stages).unsqueeze(1)
        self.register_buffer("stage_embedding", stage_embedding.repeat(1, dim))
        self.register_buffer("catet_stage_edges", torch.empty(0))
        self.register_buffer("catet_censor_times", torch.empty(0))
        self.register_buffer("catet_censor_survival", torch.empty(0))

        # Stage identity enters the risk cost before Sinkhorn.  This is the
        # decisive difference from the old implementation, which expanded one
        # edge-risk matrix over all stages.
        self.stage_edge_risk = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.temporal_evidence_gate = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.last_explanations: dict[str, torch.Tensor] | None = None
        self.last_training_losses: dict[str, torch.Tensor] = {}

        # Reconstruction belongs to the legacy OTEH objective and is not part
        # of the final CATET claim.
        del self.recon_wsi
        del self.recon_omic

    def _validate_hyperparameters(self) -> None:
        if self.catet_num_stages < 2:
            raise ValueError("catet_num_stages must be at least 2")
        non_negative = {
            "catet_prog_cost": self.catet_prog_cost_weight,
            "catet_lambda_ot": self.catet_lambda_ot,
            "catet_lambda_rank": self.catet_lambda_rank,
            "catet_lambda_stage": self.catet_lambda_stage,
            "catet_lambda_intervention": self.catet_lambda_intervention,
            "catet_intervention_margin": self.catet_intervention_margin,
            "catet_intervention_cost": self.catet_intervention_cost,
            "catet_plan_diversity_margin": self.catet_plan_diversity_margin,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.catet_keep_ratio < 1.0:
            raise ValueError("catet_keep_ratio must be in (0, 1)")
        if self.catet_rank_temperature <= 0.0:
            raise ValueError("catet_rank_temperature must be positive")
        if self.catet_ipcw_max_weight < 1.0:
            raise ValueError("catet_ipcw_max_weight must be at least 1")
        if self.catet_rank_max_pairs < 1:
            raise ValueError("catet_rank_max_pairs must be at least 1")

    @property
    def has_train_reference(self) -> bool:
        return self.catet_stage_edges.numel() == self.catet_num_stages + 1

    @torch.no_grad()
    def configure_train_reference(self, event_times, censorship) -> None:
        """Fit event stages and censoring KM using this fold's train patients."""
        device = self.stage_embedding.device
        times = torch.as_tensor(event_times, dtype=torch.float32, device=device).flatten()
        cens = torch.as_tensor(censorship, dtype=torch.float32, device=device).flatten()
        if times.numel() != cens.numel() or times.numel() == 0:
            raise ValueError("event_times and censorship must be non-empty and aligned")
        observed_times = times[cens < 0.5]
        if observed_times.numel() < self.catet_num_stages:
            raise ValueError(
                "CATET needs at least catet_num_stages observed training events"
            )

        quantiles = torch.linspace(
            1.0 / self.catet_num_stages,
            1.0,
            self.catet_num_stages,
            device=device,
        )
        upper = torch.quantile(observed_times, quantiles)
        eps = torch.finfo(upper.dtype).eps
        for index in range(1, upper.numel()):
            upper[index] = torch.maximum(upper[index], upper[index - 1] + eps)
        self.catet_stage_edges = torch.cat(
            [upper.new_tensor([-float("inf")]), upper]
        )

        unique_times = torch.unique(times, sorted=True)
        censor_survival = torch.ones_like(unique_times)
        value = torch.ones((), dtype=times.dtype, device=device)
        for index, time in enumerate(unique_times):
            at_risk = (times >= time).sum().to(times.dtype).clamp_min(1.0)
            censor_events = ((times == time) & (cens >= 0.5)).sum().to(times.dtype)
            value = value * (1.0 - censor_events / at_risk)
            censor_survival[index] = value
        self.catet_censor_times = unique_times
        self.catet_censor_survival = censor_survival.clamp_min(0.05)

    def _ipcw(self, query_times: torch.Tensor) -> torch.Tensor:
        if self.catet_censor_times.numel() == 0:
            return torch.ones_like(query_times)
        indices = torch.searchsorted(self.catet_censor_times, query_times, right=True) - 1
        values = torch.ones_like(query_times)
        valid = indices >= 0
        values[valid] = self.catet_censor_survival[indices[valid]]
        return values.clamp_min(0.05).reciprocal()

    @staticmethod
    def _normalize_cost(cost: torch.Tensor) -> torch.Tensor:
        cost = cost - cost.amin(dim=(1, 2), keepdim=True)
        return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _positive_dot_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.softplus(-torch.bmm(x, y.transpose(1, 2)))

    @staticmethod
    def _pair_tokens(slots_wsi: torch.Tensor, slots_omic: torch.Tensor) -> torch.Tensor:
        batch, wsi_count, dim = slots_wsi.shape
        omic_count = slots_omic.size(1)
        wsi = slots_wsi.unsqueeze(2).expand(batch, wsi_count, omic_count, dim)
        omic = slots_omic.unsqueeze(1).expand(batch, wsi_count, omic_count, dim)
        return torch.cat([wsi, omic, wsi * omic, (wsi - omic).abs()], dim=-1)

    def _epsilon(self, epoch: int) -> float:
        end = float(getattr(self.args, "otehv2_eps", 0.05))
        start = float(getattr(self.args, "rg_eps_start", end * 2.0))
        duration = max(1, int(getattr(self.args, "rg_eps_anneal", 12)))
        fraction = min(1.0, max(0, epoch) / duration)
        return start + fraction * (end - start)

    def _stage_costs(
        self, slots_wsi: torch.Tensor, slots_omic: torch.Tensor
    ) -> tuple[list[tuple[torch.Tensor, ...]], torch.Tensor, torch.Tensor]:
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        batch, wsi_count, omic_count, _ = pair_tokens.shape
        pair_by_stage = pair_tokens.unsqueeze(1).expand(
            -1, self.catet_num_stages, -1, -1, -1
        )
        stage_code = self.stage_embedding.view(
            1, self.catet_num_stages, 1, 1, -1
        ).expand(batch, -1, wsi_count, omic_count, -1)
        stage_features = torch.cat([pair_by_stage, stage_code], dim=-1)
        edge_risk = torch.sigmoid(
            self.stage_edge_risk(stage_features).squeeze(-1)
        )
        evidence_gate = torch.sigmoid(
            self.temporal_evidence_gate(stage_features).squeeze(-1)
        )
        stage_bias = self._normalize_cost(
            (1.0 - edge_risk).flatten(0, 1)
        ).view_as(edge_risk)
        base_costs = (
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        )
        costs: list[tuple[torch.Tensor, ...]] = []
        for stage_index in range(self.catet_num_stages):
            costs.append(
                tuple(
                    base + self.catet_prog_cost_weight * stage_bias[:, stage_index]
                    for base in base_costs
                )
            )
        return costs, edge_risk, evidence_gate

    def _solve_costs(
        self, costs: list[tuple[torch.Tensor, ...]], eps: float
    ) -> tuple[list[tuple[torch.Tensor, ...]], torch.Tensor]:
        plans: list[tuple[torch.Tensor, ...]] = []
        distances = []
        for stage_costs in costs:
            stage_plans = []
            stage_distances = []
            for cost in stage_costs:
                plan, _ = log_sinkhorn_plan(
                    cost,
                    eps=eps,
                    max_iter=self.ot_iter,
                )
                # Low-epsilon costs can converge slowly in a short smoke run.
                # A few differentiable IPFP projections make the balanced-plan
                # invariant explicit for factual and counterfactual families.
                row_target = 1.0 / plan.size(1)
                col_target = 1.0 / plan.size(2)
                with torch.no_grad():
                    balanced = plan.detach().clamp_min(1e-12)
                    for _ in range(256):
                        balanced = balanced * (
                            row_target
                            / balanced.sum(dim=2, keepdim=True).clamp_min(1e-12)
                        )
                        balanced = balanced * (
                            col_target
                            / balanced.sum(dim=1, keepdim=True).clamp_min(1e-12)
                        )
                # Straight-through projection: forward explanations satisfy
                # the balanced-OT invariant while backward follows the stable
                # log-domain Sinkhorn solution.
                plan = plan + (balanced - plan).detach()
                distance = (plan * cost).sum(dim=(1, 2)).clamp(0.0, 10.0)
                stage_plans.append(plan)
                stage_distances.append(distance.mean())
            plans.append(tuple(stage_plans))
            distances.append(torch.stack(stage_distances).mean())
        return plans, torch.stack(distances).mean()

    def _counterfactual_plans(
        self,
        costs: list[tuple[torch.Tensor, ...]],
        evidence_gate: torch.Tensor,
        eps: float,
    ) -> tuple[list[tuple[torch.Tensor, ...]], list[tuple[torch.Tensor, ...]]]:
        keep_costs: list[tuple[torch.Tensor, ...]] = []
        remove_costs: list[tuple[torch.Tensor, ...]] = []
        for stage_index, stage_costs in enumerate(costs):
            gate = evidence_gate[:, stage_index]
            keep_costs.append(
                tuple(
                    cost + self.catet_intervention_cost * (1.0 - gate)
                    for cost in stage_costs
                )
            )
            remove_costs.append(
                tuple(
                    cost + self.catet_intervention_cost * gate
                    for cost in stage_costs
                )
            )
        keep_plans, _ = self._solve_costs(keep_costs, eps)
        remove_plans, _ = self._solve_costs(remove_costs, eps)
        return keep_plans, remove_plans

    def _stage_events(self, slots_wsi, slots_omic, plans):
        events = []
        for stage_index, stage_plans in enumerate(plans):
            all_events, _ = self.fusion(slots_wsi, slots_omic, *stage_plans)
            events.append(all_events[:, stage_index : stage_index + 1])
        return torch.cat(events, dim=1)

    def _decode(self, slots_wsi, slots_omic, plans):
        tokens = self._stage_events(slots_wsi, slots_omic, plans)
        tokens = self.event_norm(
            self.event_encoder(tokens + self.stage_embedding.unsqueeze(0))
        )
        event_logits = self.event_hazard(tokens)
        event_gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", event_gate, event_logits)
        return logits, tokens, event_gate

    @staticmethod
    def _risk_score(logits: torch.Tensor) -> torch.Tensor:
        hazards = torch.sigmoid(logits)
        return -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)

    @staticmethod
    def _mean_plan(plans: list[tuple[torch.Tensor, ...]]) -> torch.Tensor:
        return torch.stack(
            [torch.stack(stage_plans, dim=0).mean(dim=0) for stage_plans in plans],
            dim=1,
        )

    @staticmethod
    def _marginal_error(plan_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = plan_stack.size(-2)
        cols = plan_stack.size(-1)
        row_target = 1.0 / rows
        col_target = 1.0 / cols
        row_error = (plan_stack.sum(dim=-1) - row_target).abs().amax(dim=(-1, -2))
        col_error = (plan_stack.sum(dim=-2) - col_target).abs().amax(dim=(-1, -2))
        return row_error, col_error

    def _ipcw_ranking_loss(self, logits, event_time, censorship):
        risk = self._risk_score(logits)
        times = event_time.float().view(-1)
        observed = censorship.float().view(-1) < 0.5
        comparable = observed[:, None] & (times[:, None] < times[None, :])
        if not bool(comparable.any()):
            return risk.sum() * 0.0, comparable.sum()
        differences = risk[:, None] - risk[None, :]
        losses = self.catet_rank_temperature * F.softplus(
            (self.catet_rank_margin - differences) / self.catet_rank_temperature
        )
        weights = self._ipcw(times).square().clamp_max(
            self.catet_ipcw_max_weight
        )[:, None].expand_as(losses)
        pair_losses = losses[comparable]
        pair_weights = weights[comparable]
        if pair_losses.numel() > self.catet_rank_max_pairs:
            keep = torch.randperm(pair_losses.numel(), device=pair_losses.device)[
                : self.catet_rank_max_pairs
            ]
            pair_losses = pair_losses[keep]
            pair_weights = pair_weights[keep]
        loss = (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
        return loss, comparable.sum()

    def _censored_stage_loss(self, probability, event_time, censorship):
        if not self.has_train_reference:
            return probability.sum() * 0.0
        times = event_time.float().view(-1)
        cens = censorship.float().view(-1)
        boundaries = self.catet_stage_edges[1:-1]
        stage_index = torch.bucketize(times, boundaries).clamp_max(
            self.catet_num_stages - 1
        )
        probability = probability.clamp_min(1e-8)
        observed = cens < 0.5
        losses = probability.new_zeros(times.shape)
        if bool(observed.any()):
            event_probability = probability[
                torch.arange(times.numel(), device=times.device), stage_index
            ]
            event_weight = self._ipcw(times).square().clamp_max(
                self.catet_ipcw_max_weight
            )
            losses[observed] = -event_probability[observed].log() * event_weight[observed]
        censored = ~observed
        for row in torch.where(censored)[0]:
            index = int(stage_index[row].item())
            if index + 1 < self.catet_num_stages:
                losses[row] = -probability[row, index + 1 :].sum().clamp_min(1e-8).log()
        informative = observed | (censored & (stage_index + 1 < self.catet_num_stages))
        return losses[informative].mean() if bool(informative.any()) else losses.sum() * 0.0

    def _plan_diversity_loss(self, plan_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adjacent = (plan_stack[:, 1:] - plan_stack[:, :-1]).abs().mean(dim=(-1, -2))
        return F.relu(self.catet_plan_diversity_margin - adjacent).mean(), adjacent.mean()

    def _intervention_loss(self, factual_logits, keep_logits, remove_logits, gate):
        factual_risk = self._risk_score(factual_logits).detach()
        keep_risk = self._risk_score(keep_logits)
        remove_risk = self._risk_score(remove_logits)
        sufficiency = F.smooth_l1_loss(keep_risk, factual_risk)
        remove_change = (factual_risk - remove_risk).abs()
        comprehensiveness = F.relu(
            self.catet_intervention_margin - remove_change
        ).mean()
        gate_budget = (gate.mean(dim=(-1, -2, -3)) - self.catet_keep_ratio).square().mean()
        return (
            sufficiency + comprehensiveness + 0.1 * gate_budget,
            sufficiency,
            comprehensiveness,
            gate_budget,
        )

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        epoch = int(
            getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0))
        )
        eps = self._epsilon(epoch)
        costs, edge_risk, evidence_gate = self._stage_costs(slots_wsi, slots_omic)
        factual_plans, ot_distance = self._solve_costs(costs, eps)
        keep_plans, remove_plans = self._counterfactual_plans(
            costs, evidence_gate, eps
        )

        factual_logits, _, event_gate = self._decode(
            slots_wsi, slots_omic, factual_plans
        )
        keep_logits, _, _ = self._decode(slots_wsi, slots_omic, keep_plans)
        remove_logits, _, _ = self._decode(slots_wsi, slots_omic, remove_plans)

        factual_plan_stack = self._mean_plan(factual_plans)
        keep_plan_stack = self._mean_plan(keep_plans)
        remove_plan_stack = self._mean_plan(remove_plans)
        stage_evidence = (
            factual_plan_stack * edge_risk * evidence_gate
        ).sum(dim=(-1, -2))
        stage_probability = stage_evidence * event_gate
        stage_probability = stage_probability / stage_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

        factual_row_error, factual_col_error = self._marginal_error(
            factual_plan_stack
        )
        keep_row_error, keep_col_error = self._marginal_error(keep_plan_stack)
        remove_row_error, remove_col_error = self._marginal_error(remove_plan_stack)
        adjacent_plan_l1 = (
            factual_plan_stack[:, 1:] - factual_plan_stack[:, :-1]
        ).abs().mean(dim=(-1, -2))

        factual_risk = self._risk_score(factual_logits)
        keep_risk = self._risk_score(keep_logits)
        remove_risk = self._risk_score(remove_logits)
        self.last_explanations = {
            "stage_slot_pair_evidence": evidence_gate.detach(),
            "stage_slot_pair_risk": edge_risk.detach(),
            "factual_plan": factual_plan_stack.detach(),
            "keep_plan": keep_plan_stack.detach(),
            "remove_plan": remove_plan_stack.detach(),
            "stage_probability": stage_probability.detach(),
            "event_gate": event_gate.detach(),
            "factual_risk": factual_risk.detach(),
            "kept_risk": keep_risk.detach(),
            "removed_risk": remove_risk.detach(),
            "sufficiency_gap": (keep_risk - factual_risk).abs().detach(),
            "comprehensiveness_gap": (remove_risk - factual_risk).abs().detach(),
            "adjacent_stage_plan_l1": adjacent_plan_l1.detach(),
            "factual_row_marginal_error": factual_row_error.detach(),
            "factual_col_marginal_error": factual_col_error.detach(),
            "keep_row_marginal_error": keep_row_error.detach(),
            "keep_col_marginal_error": keep_col_error.detach(),
            "remove_row_marginal_error": remove_row_error.detach(),
            "remove_col_marginal_error": remove_col_error.detach(),
        }

        if not self.training:
            self.last_training_losses = {}
            return factual_logits, factual_logits.new_zeros(())

        rank_loss = factual_logits.sum() * 0.0
        rank_pairs = factual_logits.new_zeros(())
        stage_loss = factual_logits.sum() * 0.0
        if "event_time" in kwargs and "c" in kwargs:
            rank_loss, rank_pairs = self._ipcw_ranking_loss(
                factual_logits, kwargs["event_time"], kwargs["c"]
            )
            stage_loss = self._censored_stage_loss(
                stage_probability, kwargs["event_time"], kwargs["c"]
            )
        diversity_loss, plan_l1 = self._plan_diversity_loss(factual_plan_stack)
        (
            intervention_loss,
            sufficiency,
            comprehensiveness,
            gate_budget,
        ) = self._intervention_loss(
            factual_logits, keep_logits, remove_logits, evidence_gate
        )
        transport_loss = ot_distance + diversity_loss
        aux_loss = (
            self.catet_lambda_ot * transport_loss
            + self.catet_lambda_rank * rank_loss
            + self.catet_lambda_stage * stage_loss
            + self.catet_lambda_intervention * intervention_loss
        )
        finite = torch.stack(
            [
                torch.isfinite(factual_logits).all(),
                torch.isfinite(aux_loss),
                torch.isfinite(factual_plan_stack).all(),
            ]
        ).all()
        self.last_training_losses = {
            "catet_ot": ot_distance.detach(),
            "catet_plan_diversity": diversity_loss.detach(),
            "catet_adjacent_plan_l1": plan_l1.detach(),
            "catet_ipcw_rank": rank_loss.detach(),
            "catet_ipcw_pairs": rank_pairs.detach().to(factual_logits.dtype),
            "catet_censored_stage": stage_loss.detach(),
            "catet_sufficiency": sufficiency.detach(),
            "catet_comprehensiveness": comprehensiveness.detach(),
            "catet_gate_budget": gate_budget.detach(),
            "catet_factual_marginal_error": torch.stack(
                [factual_row_error.amax(), factual_col_error.amax()]
            ).amax().detach(),
            "catet_keep_marginal_error": torch.stack(
                [keep_row_error.amax(), keep_col_error.amax()]
            ).amax().detach(),
            "catet_remove_marginal_error": torch.stack(
                [remove_row_error.amax(), remove_col_error.amax()]
            ).amax().detach(),
            "catet_mean_sufficiency_gap": (keep_risk - factual_risk).abs().mean().detach(),
            "catet_mean_comprehensiveness_gap": (
                remove_risk - factual_risk
            ).abs().mean().detach(),
            "catet_stage_probability_entropy": (
                -(
                    stage_probability.clamp_min(1e-8)
                    * stage_probability.clamp_min(1e-8).log()
                ).sum(dim=1).mean()
            ).detach(),
            "catet_auxiliary": aux_loss.detach(),
            "catet_finite": finite.detach().to(factual_logits.dtype),
        }
        return factual_logits, aux_loss

    def explain_last_batch(self) -> dict[str, torch.Tensor]:
        if self.last_explanations is None:
            raise RuntimeError("Run a forward pass before requesting explanations")
        return dict(self.last_explanations)


__all__ = ["CensoringAwareTemporalEvidenceTransport"]
