"""DCT v3.8: losses tied to re-optimised transport interventions.

This method keeps the v3.3 factual path and its discrete-time survival NLL plus
IPCW ranking objective.  Its only addition is a coherent structural objective:

1. high/low transport interventions must move risk in opposite directions;
2. a stronger cost intervention must produce a monotone risk response;
3. re-solving Sinkhorn must materially reconfigure the coupling.

The anchors are train-fold statistics and are detached when updated.  These are
model-based structural interventions, not identified causal treatment effects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


class DCTTransportInterventionConsistency(DistributionalCounterfactualTransport):
    """Train DCT's cost-intervention -> Sinkhorn -> risk-response chain."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_v38_lambda_direction = float(
            getattr(args, "dct_v38_lambda_direction", 0.05)
        )
        self.dct_v38_lambda_dose = float(
            getattr(args, "dct_v38_lambda_dose", 0.03)
        )
        self.dct_v38_lambda_reconfiguration = float(
            getattr(args, "dct_v38_lambda_reconfiguration", 0.02)
        )
        self.dct_v38_direction_margin = float(
            getattr(args, "dct_v38_direction_margin", 0.02)
        )
        self.dct_v38_dose_margin = float(
            getattr(args, "dct_v38_dose_margin", 0.005)
        )
        self.dct_v38_reconfiguration_margin = float(
            getattr(args, "dct_v38_reconfiguration_margin", 0.02)
        )
        self.dct_v38_temperature = float(
            getattr(args, "dct_v38_temperature", 0.05)
        )
        self.dct_v38_alpha_mid = float(getattr(args, "dct_v38_alpha_mid", 0.50))
        self.dct_v38_alpha_full = float(getattr(args, "dct_v38_alpha_full", 1.00))
        self.dct_v38_warmup_epochs = int(
            getattr(args, "dct_v38_warmup_epochs", 1)
        )
        self.dct_v38_dose_every = int(getattr(args, "dct_v38_dose_every", 1))

        nonnegative = {
            "dct_v38_lambda_direction": self.dct_v38_lambda_direction,
            "dct_v38_lambda_dose": self.dct_v38_lambda_dose,
            "dct_v38_lambda_reconfiguration": self.dct_v38_lambda_reconfiguration,
            "dct_v38_direction_margin": self.dct_v38_direction_margin,
            "dct_v38_dose_margin": self.dct_v38_dose_margin,
            "dct_v38_reconfiguration_margin": self.dct_v38_reconfiguration_margin,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.dct_v38_temperature <= 0.0:
            raise ValueError("dct_v38_temperature must be positive")
        if not 0.0 < self.dct_v38_alpha_mid < self.dct_v38_alpha_full <= 1.0:
            raise ValueError(
                "DCT v3.8 requires 0 < dct_v38_alpha_mid "
                "< dct_v38_alpha_full <= 1"
            )
        if self.dct_v38_warmup_epochs < 0:
            raise ValueError("dct_v38_warmup_epochs must be non-negative")
        if self.dct_v38_dose_every <= 0:
            raise ValueError("dct_v38_dose_every must be positive")

        # v3.8 is a controlled v3.3 extension.  Do not silently combine it with
        # ETAR or the older five-objective ablations.
        self.dct_lambda_etar = 0.0
        self.dct_lambda_ot = 0.0
        self.dct_lambda_rank = 0.0
        self.dct_lambda_anchor = 0.0
        self.dct_lambda_stage_risk = 0.0
        self.dct_lambda_coordinate = 0.0

    def _costs_at_alpha(self, factual_costs, alpha):
        alpha = min(1.0, max(0.0, float(alpha)))
        bsz = factual_costs.size(0)
        anchors = self.risk_anchor_costs.to(
            device=factual_costs.device, dtype=factual_costs.dtype
        )
        low_anchor = (
            anchors[:, self._LOW_RISK]
            .unsqueeze(0)
            .expand(bsz, -1, -1, -1, -1)
        )
        high_anchor = (
            anchors[:, self._HIGH_RISK]
            .unsqueeze(0)
            .expand(bsz, -1, -1, -1, -1)
        )
        low_seen = self.risk_anchor_seen[:, self._LOW_RISK].view(1, -1, 1, 1, 1)
        high_seen = self.risk_anchor_seen[:, self._HIGH_RISK].view(
            1, -1, 1, 1, 1
        )
        low_anchor = torch.where(low_seen, low_anchor, factual_costs)
        high_anchor = torch.where(high_seen, high_anchor, factual_costs)
        return (
            torch.lerp(factual_costs, low_anchor, alpha),
            torch.lerp(factual_costs, high_anchor, alpha),
        )

    @staticmethod
    def _stack_plans(plans):
        """Return couplings as [batch, stage, geometry, WSI slot, omics slot]."""

        return torch.stack(
            [torch.stack(tuple(stage_plans), dim=1) for stage_plans in plans],
            dim=1,
        )

    @staticmethod
    def _split_plans(plans, chunks):
        split_by_stage = [
            [plan.chunk(chunks, dim=0) for plan in stage_plans]
            for stage_plans in plans
        ]
        return [
            [
                tuple(split_by_stage[stage_idx][geometry_idx][chunk_idx]
                      for geometry_idx in range(len(split_by_stage[stage_idx])))
                for stage_idx in range(len(split_by_stage))
            ]
            for chunk_idx in range(chunks)
        ]

    def _solve_interventions(
        self,
        cost_tensors,
        *,
        slots_wsi,
        slots_omic,
        rows,
        cols,
        epoch,
    ):
        """Solve several intervention strengths in one batched Sinkhorn call."""

        count = len(cost_tensors)
        joint_costs = torch.cat(tuple(cost_tensors), dim=0)
        joint_rows = rows.repeat((count, 1, 1))
        joint_cols = cols.repeat((count, 1, 1))
        joint_wsi = slots_wsi.repeat((count, 1, 1))
        joint_omic = slots_omic.repeat((count, 1, 1))
        joint_plans, _ = self._plans_from_cost_tensor(
            joint_costs, joint_rows, joint_cols, epoch
        )
        joint_logits, _ = self._encode_logits_from_plans(
            joint_wsi, joint_omic, joint_plans
        )
        return (
            self._split_plans(joint_plans, count),
            list(joint_logits.chunk(count, dim=0)),
        )

    def _smooth_lower_bound(self, values, margin):
        return self.dct_v38_temperature * F.softplus(
            (float(margin) - values) / self.dct_v38_temperature
        )

    def _direction_loss(self, factual_risk, low_risk, high_risk):
        high_gain = high_risk - factual_risk
        low_gain = factual_risk - low_risk
        loss = 0.5 * (
            self._smooth_lower_bound(high_gain, self.dct_v38_direction_margin)
            + self._smooth_lower_bound(low_gain, self.dct_v38_direction_margin)
        ).mean()
        return loss, high_gain, low_gain

    def _dose_loss(
        self,
        factual_risk,
        low_mid_risk,
        low_full_risk,
        high_mid_risk,
        high_full_risk,
    ):
        increments = torch.stack(
            (
                high_mid_risk - factual_risk,
                high_full_risk - high_mid_risk,
                factual_risk - low_mid_risk,
                low_mid_risk - low_full_risk,
            ),
            dim=1,
        )
        return self._smooth_lower_bound(
            increments, self.dct_v38_dose_margin
        ).mean()

    def _reconfiguration_loss(
        self,
        factual_plans,
        low_plans,
        high_plans,
        active_stages,
    ):
        factual = self._stack_plans(factual_plans)
        low = self._stack_plans(low_plans)
        high = self._stack_plans(high_plans)
        # Total variation is in [0, 1] for each unit-mass coupling.
        low_shift = 0.5 * (low - factual).abs().sum(dim=(-1, -2))
        high_shift = 0.5 * (high - factual).abs().sum(dim=(-1, -2))
        mask = active_stages.view(1, -1, 1).expand_as(low_shift)
        if not bool(mask.any()):
            zero = factual.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        low_active = low_shift[mask]
        high_active = high_shift[mask]
        loss = 0.5 * (
            self._smooth_lower_bound(
                low_active, self.dct_v38_reconfiguration_margin
            ).mean()
            + self._smooth_lower_bound(
                high_active, self.dct_v38_reconfiguration_margin
            ).mean()
        )
        return loss, low_active.mean(), high_active.mean()

    def _zero_metrics(self, factual_costs, active_stage_fraction):
        zero = factual_costs.new_zeros(())
        return zero, {
            "v38_direction": zero,
            "v38_dose": zero,
            "v38_reconfiguration": zero,
            "v38_total": zero,
            "v38_active_stage_fraction": active_stage_fraction,
            "v38_high_risk_gain": zero,
            "v38_low_risk_gain": zero,
            "v38_high_plan_shift": zero,
            "v38_low_plan_shift": zero,
            "v38_finite": factual_costs.new_ones(()),
        }

    def _training_transport_objective(
        self,
        *,
        factual_costs,
        factual_plans,
        factual_logits,
        slots_wsi,
        slots_omic,
        rows,
        cols,
        epoch,
    ):
        active_stages = self.risk_anchor_seen.all(dim=1)
        active_stage_fraction = active_stages.to(factual_costs.dtype).mean()
        enabled = (
            self.dct_v38_lambda_direction > 0.0
            or self.dct_v38_lambda_dose > 0.0
            or self.dct_v38_lambda_reconfiguration > 0.0
        )
        if (
            not enabled
            or epoch < self.dct_v38_warmup_epochs
            or not bool(active_stages.any())
        ):
            return self._zero_metrics(factual_costs, active_stage_fraction)

        full_low_costs, full_high_costs = self._costs_at_alpha(
            factual_costs, self.dct_v38_alpha_full
        )
        use_dose = (
            self.dct_v38_lambda_dose > 0.0
            and (epoch - self.dct_v38_warmup_epochs) % self.dct_v38_dose_every == 0
        )
        intervention_costs = [full_low_costs, full_high_costs]
        if use_dose:
            mid_low_costs, mid_high_costs = self._costs_at_alpha(
                factual_costs, self.dct_v38_alpha_mid
            )
            intervention_costs.extend((mid_low_costs, mid_high_costs))

        plans, logits = self._solve_interventions(
            intervention_costs,
            slots_wsi=slots_wsi,
            slots_omic=slots_omic,
            rows=rows,
            cols=cols,
            epoch=epoch,
        )
        full_low_plans, full_high_plans = plans[:2]
        full_low_risk, full_high_risk = (self._risk(item) for item in logits[:2])
        factual_risk = self._risk(factual_logits)

        direction, high_gain, low_gain = self._direction_loss(
            factual_risk, full_low_risk, full_high_risk
        )
        reconfiguration, low_shift, high_shift = self._reconfiguration_loss(
            factual_plans,
            full_low_plans,
            full_high_plans,
            active_stages,
        )
        dose = factual_costs.new_zeros(())
        if use_dose:
            mid_low_risk, mid_high_risk = (self._risk(item) for item in logits[2:4])
            dose = self._dose_loss(
                factual_risk,
                mid_low_risk,
                full_low_risk,
                mid_high_risk,
                full_high_risk,
            )

        total = (
            self.dct_v38_lambda_direction * direction
            + self.dct_v38_lambda_dose * dose
            + self.dct_v38_lambda_reconfiguration * reconfiguration
        )
        finite = torch.stack(
            (direction, dose, reconfiguration, total)
        ).isfinite().all()
        metrics = {
            "v38_direction": direction,
            "v38_dose": dose,
            "v38_reconfiguration": reconfiguration,
            "v38_total": total,
            "v38_active_stage_fraction": active_stage_fraction,
            "v38_high_risk_gain": high_gain.mean(),
            "v38_low_risk_gain": low_gain.mean(),
            "v38_high_plan_shift": high_shift,
            "v38_low_plan_shift": low_shift,
            "v38_finite": finite.to(factual_costs.dtype),
        }
        return total, metrics
