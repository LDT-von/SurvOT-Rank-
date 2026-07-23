"""DCT v3.6: censor-aware listwise learning on factual transport events."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


def censor_aware_plackett_luce_loss(
    scores: torch.Tensor,
    event_time: torch.Tensor,
    censorship: torch.Tensor,
    *,
    temperature: float = 0.5,
    event_stage: torch.Tensor | None = None,
    current_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Breslow risk-set listwise loss with current-batch event numerators.

    ``scores`` is ``[patients, score_channels]``.  Global GPL uses one channel;
    TCL supplies one transport score per train-fold event stage.  Detached
    patients from the within-epoch memory may enter denominators, but only the
    first ``current_count`` patients can contribute event numerators.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [patients, score_channels]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    times = event_time.float().view(-1)
    cens = censorship.float().view(-1)
    if scores.size(0) != times.numel() or times.numel() != cens.numel():
        raise ValueError("scores, event_time, and censorship must share patient count")

    current = scores.size(0) if current_count is None else int(current_count)
    if current < 0 or current > scores.size(0):
        raise ValueError("current_count is outside the score batch")
    if event_stage is None:
        stages = torch.zeros(current, dtype=torch.long, device=scores.device)
    else:
        stages = event_stage.to(device=scores.device, dtype=torch.long).view(-1)
        if stages.numel() != current:
            raise ValueError("event_stage must have current_count entries")
        if bool(((stages < 0) | (stages >= scores.size(1))).any()):
            raise ValueError("event_stage contains an invalid score channel")

    losses = []
    risk_set_sizes = []
    used_stages = []
    for patient_idx in range(current):
        if bool(cens[patient_idx] >= 0.5):
            continue
        risk_set = times >= times[patient_idx]
        risk_set_size = int(risk_set.sum().item())
        # A one-patient denominator produces an exact zero and supplies no
        # ordering information, so it is not counted as a training list.
        if risk_set_size < 2:
            continue
        stage_idx = int(stages[patient_idx].item())
        scaled = scores[:, stage_idx] / float(temperature)
        losses.append(
            torch.logsumexp(scaled[risk_set], dim=0) - scaled[patient_idx]
        )
        risk_set_sizes.append(risk_set_size)
        used_stages.append(stage_idx)

    zero = scores.sum() * 0.0
    if not losses:
        diagnostics = {
            "list_count": scores.new_zeros(()),
            "avg_risk_set_size": scores.new_zeros(()),
            "stage_coverage": scores.new_zeros(()),
        }
        return zero, diagnostics

    loss = torch.stack(losses).mean()
    unique_stages = len(set(used_stages))
    diagnostics = {
        "list_count": scores.new_tensor(float(len(losses))),
        "avg_risk_set_size": scores.new_tensor(
            float(sum(risk_set_sizes)) / float(len(risk_set_sizes))
        ),
        "stage_coverage": scores.new_tensor(
            float(unique_stages) / float(scores.size(1))
        ),
    }
    return loss, diagnostics


class DCTListwiseTransport(DistributionalCounterfactualTransport):
    """Compare global score listwise learning with stage-transport listwise learning.

    GPL applies the risk-set softmax to DCT's final factual risk and is a
    conventional control.  TCL instead scores every factual stage event token
    and uses the event patient's train-fold stage to select the corresponding
    score channel for the full censor-aware risk set.
    """

    _LISTWISE_MODES = {"global", "stage_transport"}

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_listwise_mode = str(
            getattr(args, "dct_listwise_mode", "stage_transport")
        ).lower()
        self.dct_lambda_listwise = float(
            getattr(args, "dct_lambda_listwise", 0.10)
        )
        self.dct_listwise_temperature = float(
            getattr(args, "dct_listwise_temperature", 0.50)
        )
        self.dct_listwise_memory_size = int(
            getattr(args, "dct_listwise_memory_size", 64)
        )
        self.dct_listwise_tie_method = str(
            getattr(args, "dct_listwise_tie_method", "breslow")
        ).lower()
        if self.dct_listwise_mode not in self._LISTWISE_MODES:
            raise ValueError(
                "dct_listwise_mode must be one of: global, stage_transport"
            )
        if self.dct_lambda_listwise < 0:
            raise ValueError("dct_lambda_listwise must be non-negative")
        if self.dct_listwise_temperature <= 0:
            raise ValueError("dct_listwise_temperature must be positive")
        if self.dct_listwise_memory_size < 0:
            raise ValueError("dct_listwise_memory_size must be non-negative")
        if self.dct_listwise_tie_method != "breslow":
            raise ValueError("DCT v3.6 currently supports only Breslow ties")

        # The v3.6 candidate intentionally has one structural auxiliary loss.
        # Controls using IPCW and/or ETAR continue to use the original DCT class.
        self.dct_lambda_ipcw_rank = 0.0
        self.dct_lambda_etar = 0.0
        self.dct_lambda_ot = 0.0
        self.dct_lambda_rank = 0.0
        self.dct_lambda_anchor = 0.0
        self.dct_lambda_stage_risk = 0.0
        self.dct_lambda_coordinate = 0.0

        dim = self.wsi_projection_dim
        self.stage_listwise_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

        self._listwise_memory_epoch = None
        self._listwise_memory_scores = None
        self._listwise_memory_times = None
        self._listwise_memory_censorship = None

        self._capture_eval = False
        self._captured_semantic_components: list[dict[str, torch.Tensor]] = []
        self._captured_plan_tensors: list[torch.Tensor] = []
        self._captured_cost_tensors: list[torch.Tensor] = []
        self._captured_stage_tokens: list[torch.Tensor] = []
        self._captured_stage_scores: list[torch.Tensor] = []
        self._captured_rows = None
        self._captured_cols = None
        self._captured_evidence_gate = None
        self._last_eval_cache: dict[str, Any] | None = None

    def _semantic_slots(self, tokens, prototypes):
        """Preserve DCT pooling while retaining both assignment normalizations."""

        keys = F.normalize(prototypes, dim=-1)
        normalized_tokens = F.normalize(tokens, dim=-1)
        scores = torch.einsum("bnd,kd->bkn", normalized_tokens, keys)
        assignment = torch.softmax(
            scores / self.dct_coordinate_temperature, dim=1
        )
        pooling = assignment / assignment.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        slots = torch.einsum("bkn,bnd->bkd", pooling, tokens)
        if self._capture_eval:
            self._captured_semantic_components.append(
                {
                    "token_to_global_assignment": assignment.detach(),
                    "global_pooling_attention": pooling.detach(),
                    "global_slots": slots.detach(),
                }
            )
        return slots, pooling

    def _cost_tensor(self, slots_wsi, slots_omic):
        result = super()._cost_tensor(slots_wsi, slots_omic)
        if self._capture_eval:
            costs, rows, cols, evidence_gate = result
            self._captured_rows = rows.detach()
            self._captured_cols = cols.detach()
            self._captured_evidence_gate = evidence_gate.detach()
        return result

    @staticmethod
    def _stack_plans(plans) -> torch.Tensor:
        return torch.stack(
            [torch.stack(list(stage_plans), dim=1) for stage_plans in plans],
            dim=1,
        )

    @staticmethod
    def _unstack_plans(plan_tensor: torch.Tensor):
        return [
            tuple(plan_tensor[:, stage_idx, geometry_idx] for geometry_idx in range(plan_tensor.size(2)))
            for stage_idx in range(plan_tensor.size(1))
        ]

    def _plans_from_cost_tensor(self, costs, rows, cols, epoch):
        plans, distance = super()._plans_from_cost_tensor(
            costs, rows, cols, epoch
        )
        if self._capture_eval:
            self._captured_cost_tensors.append(costs.detach())
            self._captured_plan_tensors.append(self._stack_plans(plans).detach())
        return plans, distance

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
        tokens = tokens + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        event_logits = self.event_hazard(tokens)
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)
        self._captured_stage_tokens.append(tokens)
        self._captured_stage_scores.append(
            self.stage_listwise_head(tokens).squeeze(-1)
        )
        return logits, gate

    def _reset_listwise_memory_for_epoch(self, epoch: int) -> None:
        if self._listwise_memory_epoch != int(epoch):
            self._listwise_memory_epoch = int(epoch)
            self._listwise_memory_scores = None
            self._listwise_memory_times = None
            self._listwise_memory_censorship = None

    def _remember_listwise_batch(
        self,
        scores: torch.Tensor,
        event_time: torch.Tensor,
        censorship: torch.Tensor,
    ) -> None:
        if self.dct_listwise_memory_size <= 0:
            return
        score_parts = [scores.detach()]
        time_parts = [event_time.detach().float().view(-1)]
        censor_parts = [censorship.detach().float().view(-1)]
        if self._listwise_memory_scores is not None:
            score_parts.insert(0, self._listwise_memory_scores.to(scores.device))
            time_parts.insert(0, self._listwise_memory_times.to(scores.device))
            censor_parts.insert(
                0, self._listwise_memory_censorship.to(scores.device)
            )
        keep = self.dct_listwise_memory_size
        self._listwise_memory_scores = torch.cat(score_parts, dim=0)[-keep:].clone()
        self._listwise_memory_times = torch.cat(time_parts, dim=0)[-keep:].clone()
        self._listwise_memory_censorship = torch.cat(
            censor_parts, dim=0
        )[-keep:].clone()

    def _event_stages(self, event_time: torch.Tensor) -> torch.Tensor:
        if not self.has_train_reference:
            raise RuntimeError(
                "configure_train_reference must be called before TCL training"
            )
        upper = self.dct_stage_edges[1:].to(event_time.device)
        return torch.bucketize(
            event_time.float().view(-1), upper, right=False
        ).clamp(max=self.spt_num_stages - 1)

    def _listwise_loss(
        self,
        logits: torch.Tensor,
        event_time: torch.Tensor,
        censorship: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        current_count = logits.size(0)
        if self.dct_listwise_mode == "global":
            current_scores = self._risk(logits).unsqueeze(-1)
            current_stages = torch.zeros(
                current_count, dtype=torch.long, device=logits.device
            )
        else:
            current_scores = self._captured_stage_scores[0]
            current_stages = self._event_stages(event_time)

        scores = current_scores
        times = event_time.float().view(-1)
        cens = censorship.float().view(-1)
        if self._listwise_memory_scores is not None:
            scores = torch.cat(
                [scores, self._listwise_memory_scores.to(scores.device)], dim=0
            )
            times = torch.cat(
                [times, self._listwise_memory_times.to(times.device)], dim=0
            )
            cens = torch.cat(
                [cens, self._listwise_memory_censorship.to(cens.device)], dim=0
            )

        loss, diagnostics = censor_aware_plackett_luce_loss(
            scores,
            times,
            cens,
            temperature=self.dct_listwise_temperature,
            event_stage=current_stages,
            current_count=current_count,
        )
        return loss, diagnostics, current_scores

    @staticmethod
    def _compose_global_token_maps(
        local_token_assignment: torch.Tensor,
        local_pooling_attention: torch.Tensor,
        semantic_assignment: torch.Tensor,
        semantic_pooling: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        membership = torch.einsum(
            "bks,bsn->bkn", semantic_assignment, local_token_assignment
        )
        membership = membership / membership.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        pooling = torch.einsum(
            "bks,bsn->bkn", semantic_pooling, local_pooling_attention
        )
        pooling = pooling / pooling.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return membership, pooling

    def _finalize_explanations(self, logits: torch.Tensor, epoch: int) -> None:
        if len(self._captured_semantic_components) != 2:
            raise RuntimeError("DCT v3.6 did not capture both semantic modalities")
        if len(self._captured_plan_tensors) != 3:
            raise RuntimeError("DCT v3.6 expected factual, low, and high plans")

        wsi_semantic, omic_semantic = self._captured_semantic_components
        wsi_local_assignment = self.slot_attention_wsi.last_token_assignment
        wsi_local_pooling = self.slot_attention_wsi.last_pooling_attention
        omic_local_assignment = self.slot_attention_omic.last_token_assignment
        omic_local_pooling = self.slot_attention_omic.last_pooling_attention
        if any(
            value is None
            for value in (
                wsi_local_assignment,
                wsi_local_pooling,
                omic_local_assignment,
                omic_local_pooling,
            )
        ):
            raise RuntimeError("Slot attention capture was not available")

        wsi_membership, wsi_pooling = self._compose_global_token_maps(
            wsi_local_assignment,
            wsi_local_pooling,
            wsi_semantic["token_to_global_assignment"],
            wsi_semantic["global_pooling_attention"],
        )
        omic_membership, omic_pooling = self._compose_global_token_maps(
            omic_local_assignment,
            omic_local_pooling,
            omic_semantic["token_to_global_assignment"],
            omic_semantic["global_pooling_attention"],
        )

        factual_plans, low_plans, high_plans = self._captured_plan_tensors
        factual_costs, low_costs, high_costs = self._captured_cost_tensors
        self.last_explanations.update(
            {
                "wsi_patch_to_local_slot_assignment": wsi_local_assignment,
                "wsi_patch_pooling_attention": wsi_local_pooling,
                "wsi_local_to_global_assignment": wsi_semantic[
                    "token_to_global_assignment"
                ],
                "wsi_global_pooling_attention": wsi_semantic[
                    "global_pooling_attention"
                ],
                "wsi_patch_to_global_prototype": wsi_membership.detach(),
                "wsi_patch_to_global_pooling": wsi_pooling.detach(),
                "omic_pathway_to_local_slot_assignment": omic_local_assignment,
                "omic_pathway_pooling_attention": omic_local_pooling,
                "omic_local_to_global_assignment": omic_semantic[
                    "token_to_global_assignment"
                ],
                "omic_global_pooling_attention": omic_semantic[
                    "global_pooling_attention"
                ],
                "omic_pathway_to_global_prototype": omic_membership.detach(),
                "omic_pathway_to_global_pooling": omic_pooling.detach(),
                "factual_stage_costs": factual_costs,
                "low_stage_costs": low_costs,
                "high_stage_costs": high_costs,
                "factual_stage_couplings": factual_plans,
                "low_stage_couplings": low_plans,
                "high_stage_couplings": high_plans,
                "factual_row_marginals": self._captured_rows,
                "factual_col_marginals": self._captured_cols,
                "factual_stage_transport_scores": self._captured_stage_scores[0].detach(),
                "low_stage_transport_scores": self._captured_stage_scores[1].detach(),
                "high_stage_transport_scores": self._captured_stage_scores[2].detach(),
            }
        )
        self._last_eval_cache = {
            "slots_wsi": wsi_semantic["global_slots"],
            "slots_omic": omic_semantic["global_slots"],
            "factual_costs": factual_costs,
            "rows": self._captured_rows,
            "cols": self._captured_cols,
            "epoch": int(epoch),
            "factual_risk": self._risk(logits).detach(),
        }

    def forward(self, **kwargs):
        epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
        self._captured_semantic_components = []
        self._captured_plan_tensors = []
        self._captured_cost_tensors = []
        self._captured_stage_tokens = []
        self._captured_stage_scores = []
        self._captured_rows = None
        self._captured_cols = None
        self._captured_evidence_gate = None
        self._capture_eval = not self.training
        if self._capture_eval:
            self.slot_attention_wsi.capture_attention = True
            self.slot_attention_omic.capture_attention = True

        try:
            logits, aux_loss = super().forward(**kwargs)
        finally:
            self.slot_attention_wsi.capture_attention = False
            self.slot_attention_omic.capture_attention = False
            self._capture_eval = False

        if self.training:
            listwise_loss = logits.sum() * 0.0
            diagnostics = {
                "list_count": logits.new_zeros(()),
                "avg_risk_set_size": logits.new_zeros(()),
                "stage_coverage": logits.new_zeros(()),
            }
            event_time = kwargs.get("event_time")
            censorship = kwargs.get("c")
            if (
                self.dct_lambda_listwise != 0.0
                and event_time is not None
                and censorship is not None
            ):
                self._reset_listwise_memory_for_epoch(epoch)
                listwise_loss, diagnostics, current_scores = self._listwise_loss(
                    logits, event_time, censorship
                )
                self._remember_listwise_batch(
                    current_scores, event_time, censorship
                )
            self.last_training_losses.update(
                {
                    "listwise": listwise_loss.detach(),
                    "listwise_lists": diagnostics["list_count"].detach(),
                    "listwise_avg_risk_set": diagnostics[
                        "avg_risk_set_size"
                    ].detach(),
                    "listwise_stage_coverage": diagnostics[
                        "stage_coverage"
                    ].detach(),
                    "listwise_finite_scores": torch.isfinite(
                        self._captured_stage_scores[0]
                    ).to(logits.dtype).mean().detach(),
                }
            )
            return logits, aux_loss + self.dct_lambda_listwise * listwise_loss

        self._finalize_explanations(logits, epoch)
        return logits, aux_loss

    @torch.no_grad()
    def counterfactual_sweep(
        self,
        alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        *,
        random_seed: int = 1729,
    ) -> dict[str, torch.Tensor]:
        """Re-solve low/high and matched-value random anchors over a dose grid."""

        if self.training or self._last_eval_cache is None:
            raise RuntimeError(
                "Run an evaluation forward pass before counterfactual_sweep"
            )
        alpha_values = tuple(float(value) for value in alphas)
        if not alpha_values or any(value < 0.0 or value > 1.0 for value in alpha_values):
            raise ValueError("alphas must be a non-empty sequence in [0, 1]")

        cache = self._last_eval_cache
        factual_costs = cache["factual_costs"]
        rows = cache["rows"]
        cols = cache["cols"]
        slots_wsi = cache["slots_wsi"]
        slots_omic = cache["slots_omic"]
        epoch = cache["epoch"]
        factual_risk = cache["factual_risk"]
        anchors = self.risk_anchor_costs.to(
            device=factual_costs.device, dtype=factual_costs.dtype
        )
        batch_size = factual_costs.size(0)
        low_anchor = anchors[:, self._LOW_RISK].unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )
        high_anchor = anchors[:, self._HIGH_RISK].unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )
        low_seen = self.risk_anchor_seen[:, self._LOW_RISK].view(
            1, -1, 1, 1, 1
        )
        high_seen = self.risk_anchor_seen[:, self._HIGH_RISK].view(
            1, -1, 1, 1, 1
        )
        low_anchor = torch.where(
            low_seen, low_anchor, factual_costs
        )
        high_anchor = torch.where(
            high_seen, high_anchor, factual_costs
        )

        low_risks = []
        high_risks = []
        for alpha in alpha_values:
            low_costs = (1.0 - alpha) * factual_costs + alpha * low_anchor
            high_costs = (1.0 - alpha) * factual_costs + alpha * high_anchor
            low_plans, _ = super()._plans_from_cost_tensor(
                low_costs, rows, cols, epoch
            )
            high_plans, _ = super()._plans_from_cost_tensor(
                high_costs, rows, cols, epoch
            )
            low_logits, _ = self._encode_logits_from_plans(
                slots_wsi, slots_omic, low_plans
            )
            high_logits, _ = self._encode_logits_from_plans(
                slots_wsi, slots_omic, high_plans
            )
            low_risks.append(self._risk(low_logits))
            high_risks.append(self._risk(high_logits))

        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        flat_high = high_anchor.flatten(3)
        permutation = torch.randperm(
            flat_high.size(-1), generator=generator
        ).to(flat_high.device)
        random_anchor = flat_high[..., permutation].view_as(high_anchor)
        random_costs = random_anchor
        random_plans, _ = super()._plans_from_cost_tensor(
            random_costs, rows, cols, epoch
        )
        random_logits, _ = self._encode_logits_from_plans(
            slots_wsi, slots_omic, random_plans
        )
        random_risk = self._risk(random_logits)

        return {
            "alpha": factual_costs.new_tensor(alpha_values),
            "factual_risk": factual_risk,
            "low_risk": torch.stack(low_risks, dim=1),
            "high_risk": torch.stack(high_risks, dim=1),
            "random_anchor_risk": random_risk,
            # The decoder consumes the coupling, not cost directly.  Holding the
            # factual coupling fixed is therefore an architectural zero-response
            # control rather than another learned branch.
            "frozen_coupling_risk": factual_risk.clone(),
        }
