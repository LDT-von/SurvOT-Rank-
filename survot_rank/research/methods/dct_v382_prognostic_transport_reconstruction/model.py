"""DCT v3.8.2: deep supervision for factual transport geometries.

The v3.8 objective asks re-optimised counterfactual transport to move risk in
the requested direction.  This extension addresses a different failure mode:
the fused decoder can hide a factual geometry that carries no prognostic
signal.  MGPTR therefore asks each already-computed factual coupling to support
the patient's survival prediction on its own.

No additional Sinkhorn problem or trainable decoder is introduced.  A
geometry-isolated branch reuses the same fusion and survival heads, while the
fused prediction is a detached teacher only over the patient's observed time
horizon.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)


class BudgetedAdaptiveAuxiliaryWeighter(nn.Module):
    """Learn a bounded allocation of a fixed auxiliary-loss budget.

    Trainable logits redistribute the sum of the active base weights.  A
    prior fraction guarantees that every active objective keeps part of its
    original coefficient, so the optimiser cannot reduce the objective by
    switching every auxiliary loss off.
    """

    def __init__(
        self,
        base_weights,
        *,
        prior_fraction=0.25,
        temperature=1.0,
        kl_strength=0.01,
    ):
        super().__init__()
        names = tuple(base_weights)
        values = torch.tensor(
            [float(base_weights[name]) for name in names], dtype=torch.float32
        )
        if not names or bool((values <= 0.0).any()):
            raise ValueError("Adaptive auxiliary base weights must be positive")
        if not 0.0 <= float(prior_fraction) <= 1.0:
            raise ValueError("prior_fraction must be in [0, 1]")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if float(kl_strength) < 0.0:
            raise ValueError("kl_strength must be non-negative")

        self.loss_names = names
        self._name_to_index = {name: index for index, name in enumerate(names)}
        self.prior_fraction = float(prior_fraction)
        self.temperature = float(temperature)
        self.kl_strength = float(kl_strength)
        self.register_buffer("base_weights", values)
        prior = values / values.sum()
        self.allocation_logits = nn.Parameter(prior.log())

    def forward(self, losses):
        if not losses:
            raise ValueError("Adaptive auxiliary weighting received no active losses")
        unknown = sorted(set(losses) - set(self._name_to_index))
        if unknown:
            raise KeyError(f"Unknown adaptive auxiliary losses: {unknown}")

        names = tuple(name for name in self.loss_names if name in losses)
        indices = torch.tensor(
            [self._name_to_index[name] for name in names],
            device=self.allocation_logits.device,
            dtype=torch.long,
        )
        raw = torch.stack(tuple(losses[name] for name in names))
        base = self.base_weights.index_select(0, indices).to(raw)
        active_budget = base.sum()
        prior = base / active_budget
        logits = self.allocation_logits.index_select(0, indices).to(raw)
        learned = torch.softmax(logits / self.temperature, dim=0)
        allocation = (
            self.prior_fraction * prior
            + (1.0 - self.prior_fraction) * learned
        )
        weights = active_budget * allocation
        weighted = (weights * raw).sum()
        kl = (
            allocation
            * (allocation.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())
        ).sum()
        total = weighted + self.kl_strength * kl
        return total, dict(zip(names, weights)), kl


class DCTV382PrognosticTransportReconstruction(
    DCTTransportInterventionConsistency
):
    """Add Multi-Geometry Prognostic Transport Reconstruction (MGPTR)."""

    _NUM_GEOMETRIES = 3

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        self.dct_v382_lambda_mgptr = float(
            getattr(args, "dct_v382_lambda_mgptr", 0.05)
        )
        self.dct_v382_distill_weight = float(
            getattr(args, "dct_v382_distill_weight", 0.50)
        )
        self.dct_v382_warmup_epochs = int(
            getattr(args, "dct_v382_warmup_epochs", 1)
        )
        self.dct_v382_ramp_epochs = int(
            getattr(args, "dct_v382_ramp_epochs", 4)
        )
        self.dct_v382_adaptive_aux_weights = bool(
            getattr(args, "dct_v382_adaptive_aux_weights", False)
        )
        self.dct_v382_adaptive_prior_fraction = float(
            getattr(args, "dct_v382_adaptive_prior_fraction", 0.25)
        )
        self.dct_v382_adaptive_temperature = float(
            getattr(args, "dct_v382_adaptive_temperature", 1.0)
        )
        self.dct_v382_adaptive_kl_strength = float(
            getattr(args, "dct_v382_adaptive_kl_strength", 0.01)
        )

        if self.dct_v382_lambda_mgptr < 0.0:
            raise ValueError("dct_v382_lambda_mgptr must be non-negative")
        if self.dct_v382_distill_weight < 0.0:
            raise ValueError("dct_v382_distill_weight must be non-negative")
        if self.dct_v382_warmup_epochs < 0:
            raise ValueError("dct_v382_warmup_epochs must be non-negative")
        if self.dct_v382_ramp_epochs < 0:
            raise ValueError("dct_v382_ramp_epochs must be non-negative")
        if not 0.0 <= self.dct_v382_adaptive_prior_fraction <= 1.0:
            raise ValueError("dct_v382_adaptive_prior_fraction must be in [0, 1]")
        if self.dct_v382_adaptive_temperature <= 0.0:
            raise ValueError("dct_v382_adaptive_temperature must be positive")
        if self.dct_v382_adaptive_kl_strength < 0.0:
            raise ValueError("dct_v382_adaptive_kl_strength must be non-negative")

        self.adaptive_auxiliary_weighter = None
        if self.dct_v382_adaptive_aux_weights:
            base_weights = {
                "ipcw_rank": self.dct_lambda_ipcw_rank,
                "direction": self.dct_v38_lambda_direction,
                "dose": self.dct_v38_lambda_dose,
                "reconfiguration": self.dct_v38_lambda_reconfiguration,
                "mgptr": self.dct_v382_lambda_mgptr,
            }
            base_weights = {
                name: value for name, value in base_weights.items() if value > 0.0
            }
            if len(base_weights) < 2:
                raise ValueError(
                    "Adaptive v3.8.2 requires at least two enabled auxiliary losses"
                )
            self.adaptive_auxiliary_weighter = (
                BudgetedAdaptiveAuxiliaryWeighter(
                    base_weights,
                    prior_fraction=self.dct_v382_adaptive_prior_fraction,
                    temperature=self.dct_v382_adaptive_temperature,
                    kl_strength=self.dct_v382_adaptive_kl_strength,
                )
            )

        # Labels are made available only during the synchronous parent forward
        # call.  Keeping them out of persistent buffers prevents patient labels
        # from entering checkpoints.
        self._v382_y = None
        self._v382_c = None

    def forward(self, **kwargs):
        self._v382_y = kwargs.get("y")
        self._v382_c = kwargs.get("c")
        try:
            return super().forward(**kwargs)
        finally:
            self._v382_y = None
            self._v382_c = None

    def _mgptr_scale(self, epoch):
        epoch = int(epoch)
        if epoch < self.dct_v382_warmup_epochs:
            return 0.0
        if self.dct_v382_ramp_epochs <= 0:
            return 1.0
        post_warmup_epoch = epoch - self.dct_v382_warmup_epochs + 1
        return min(1.0, post_warmup_epoch / self.dct_v382_ramp_epochs)

    def _geometry_isolated_logits(self, slots_wsi, slots_omic, factual_plans):
        """Decode one prediction per factual geometry without another OT solve.

        ``MultiScaleOTFusion`` uses three geometry-specific projections and the
        cosine input as its attention-mass prior.  Repeating the selected
        coupling through all three inputs keeps that decoder numerically intact
        while ensuring that no other coupling can contribute to the branch.
        The three branches are concatenated into one shared decoder call.
        """

        geometry_count = len(factual_plans[0])
        if geometry_count != self._NUM_GEOMETRIES:
            raise ValueError(
                "MGPTR expects exactly three factual transport geometries, "
                f"got {geometry_count}"
            )
        if any(len(stage_plans) != geometry_count for stage_plans in factual_plans):
            raise ValueError("Every DCT stage must expose the same geometry count")

        joint_plans = []
        for stage_plans in factual_plans:
            selected = torch.cat(tuple(stage_plans), dim=0)
            joint_plans.append((selected, selected, selected))

        repeats = (geometry_count, 1, 1)
        joint_logits, _ = self._encode_logits_from_plans(
            slots_wsi.repeat(repeats),
            slots_omic.repeat(repeats),
            joint_plans,
        )
        return torch.stack(tuple(joint_logits.chunk(geometry_count, dim=0)), dim=0)

    @staticmethod
    def _per_patient_survival_nll(logits, y, c, alpha):
        """Discrete-time censored NLL with no dependency on the trainer loss."""

        eps = torch.finfo(logits.dtype).eps
        y = y.long().view(-1, 1)
        c = c.to(dtype=logits.dtype).view(-1, 1)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        survival_pad = torch.cat([torch.ones_like(c), survival], dim=1)
        previous_survival = torch.gather(survival_pad, 1, y).clamp_min(eps)
        current_hazard = torch.gather(hazards, 1, y).clamp_min(eps)
        current_survival = torch.gather(survival_pad, 1, y + 1).clamp_min(eps)
        uncensored = -(1.0 - c) * (
            previous_survival.log() + current_hazard.log()
        )
        censored = -c * current_survival.log()
        negative_likelihood = uncensored + censored
        if alpha is not None:
            negative_likelihood = (
                (1.0 - float(alpha)) * negative_likelihood
                + float(alpha) * uncensored
            )
        return negative_likelihood.view(-1)

    @staticmethod
    def _observed_horizon_kl(branch_logits, teacher_logits, y):
        """Bernoulli hazard KL only through each patient's observed time bin."""

        teacher_probability = torch.sigmoid(teacher_logits.detach()).clamp(
            min=1e-6, max=1.0 - 1e-6
        )
        teacher_probability = teacher_probability.unsqueeze(0)
        log_teacher = teacher_probability.log()
        log_teacher_complement = torch.log1p(-teacher_probability)
        log_branch = F.logsigmoid(branch_logits)
        log_branch_complement = F.logsigmoid(-branch_logits)
        kl = teacher_probability * (log_teacher - log_branch)
        kl = kl + (1.0 - teacher_probability) * (
            log_teacher_complement - log_branch_complement
        )

        time_index = torch.arange(
            branch_logits.size(-1), device=branch_logits.device
        ).view(1, 1, -1)
        observed = time_index <= y.long().view(1, -1, 1)
        observed = observed.to(dtype=branch_logits.dtype)
        per_branch_patient = (kl * observed).sum(dim=-1) / observed.sum(
            dim=-1
        ).clamp_min(1.0)
        return per_branch_patient.mean()

    def _mgptr_loss(self, factual_logits, geometry_logits, y, c):
        alpha = getattr(self.args, "alpha_surv", 0.0)
        geometry_nll = torch.stack(
            tuple(
                self._per_patient_survival_nll(branch, y, c, alpha).mean()
                for branch in geometry_logits
            )
        ).mean()
        reconstruction = self._observed_horizon_kl(
            geometry_logits, factual_logits, y
        )
        raw = geometry_nll + self.dct_v382_distill_weight * reconstruction
        geometry_risk = torch.stack(
            tuple(self._risk(branch) for branch in geometry_logits), dim=0
        )
        risk_spread = geometry_risk.std(dim=0, unbiased=False).mean()
        return raw, geometry_nll, reconstruction, risk_spread

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
        v38_total, metrics = super()._training_transport_objective(
            factual_costs=factual_costs,
            factual_plans=factual_plans,
            factual_logits=factual_logits,
            slots_wsi=slots_wsi,
            slots_omic=slots_omic,
            rows=rows,
            cols=cols,
            epoch=epoch,
        )

        zero = factual_costs.new_zeros(())
        scale = self._mgptr_scale(epoch)
        enabled = (
            self.dct_v382_lambda_mgptr > 0.0
            and scale > 0.0
            and self._v382_y is not None
            and self._v382_c is not None
        )
        if enabled:
            geometry_logits = self._geometry_isolated_logits(
                slots_wsi, slots_omic, factual_plans
            )
            raw, geometry_nll, reconstruction, risk_spread = self._mgptr_loss(
                factual_logits,
                geometry_logits,
                self._v382_y,
                self._v382_c,
            )
            weighted = self.dct_v382_lambda_mgptr * scale * raw
        else:
            raw = geometry_nll = reconstruction = risk_spread = weighted = zero

        total = v38_total + weighted
        finite = torch.stack(
            (raw, geometry_nll, reconstruction, risk_spread, weighted, total)
        ).isfinite().all()
        metrics.update(
            {
                "v382_mgptr_nll": geometry_nll,
                "v382_mgptr_reconstruction": reconstruction,
                "v382_mgptr_raw": raw,
                "v382_mgptr_weighted": weighted,
                "v382_mgptr_scale": factual_costs.new_tensor(scale),
                "v382_geometry_risk_spread": risk_spread,
                "v382_total": total,
                "v382_finite": finite.to(factual_costs.dtype),
            }
        )
        return total, metrics

    def _combine_auxiliary_objectives(
        self,
        *,
        ipcw_rank_loss,
        etar_loss,
        transport_objective,
        transport_metrics,
        epoch,
    ):
        if not self.dct_v382_adaptive_aux_weights:
            return super()._combine_auxiliary_objectives(
                ipcw_rank_loss=ipcw_rank_loss,
                etar_loss=etar_loss,
                transport_objective=transport_objective,
                transport_metrics=transport_metrics,
                epoch=epoch,
            )

        del etar_loss, epoch
        zero = ipcw_rank_loss.new_zeros(())
        active_losses = {}
        pair_count = getattr(self, "last_ipcw_pair_count", zero)
        if self.dct_lambda_ipcw_rank > 0.0 and bool(
            (pair_count.detach() > 0).item()
        ):
            active_losses["ipcw_rank"] = ipcw_rank_loss

        structural_scale = transport_metrics.get("v38_loss_scale", zero)
        if bool((structural_scale.detach() > 0).item()):
            for name in ("direction", "dose", "reconfiguration"):
                if name in self.adaptive_auxiliary_weighter.loss_names:
                    active_losses[name] = (
                        structural_scale * transport_metrics[f"v38_{name}"]
                    )

        mgptr_scale = transport_metrics.get("v382_mgptr_scale", zero)
        if (
            "mgptr" in self.adaptive_auxiliary_weighter.loss_names
            and bool((mgptr_scale.detach() > 0).item())
        ):
            active_losses["mgptr"] = (
                mgptr_scale * transport_metrics["v382_mgptr_raw"]
            )

        if active_losses:
            total, weights, kl = self.adaptive_auxiliary_weighter(active_losses)
        else:
            total, weights, kl = zero, {}, zero

        transport_metrics["v382_fixed_total"] = transport_metrics.get(
            "v382_total", transport_objective
        )
        transport_metrics["v382_total"] = total
        transport_metrics["v382_adaptive_total"] = total
        transport_metrics["v382_adaptive_kl"] = kl
        transport_metrics["v382_adaptive_active_terms"] = zero.new_tensor(
            len(active_losses)
        )
        transport_metrics["v382_adaptive_enabled"] = zero.new_ones(())
        for name in (
            "ipcw_rank",
            "direction",
            "dose",
            "reconfiguration",
            "mgptr",
        ):
            weight = weights.get(name, zero)
            raw = active_losses.get(name, zero)
            transport_metrics[f"v382_adaptive_weight_{name}"] = weight
            transport_metrics[f"v382_adaptive_contribution_{name}"] = weight * raw
        return total
