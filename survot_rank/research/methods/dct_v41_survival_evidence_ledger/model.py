#!/usr/bin/env python3
"""DCT v4.1 with missingness-calibrated survival-evidence ledger slots.

The slot mechanism in this file is independent of Slot Attention. Tokens do
not iteratively update randomly or query-initialised slots. Instead, every
token writes a non-negative precision mass into a patient-conditioned ledger;
the softmax over ledger addresses makes that mass exactly conserved. When a
modality is absent, the counterpart ledger predicts a distribution over the
missing slots and exposes its uncertainty to DCT's OT marginals.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


def _harmonic_ledger(num_slots: int, dim: int) -> torch.Tensor:
    """Create deterministic, non-learned ledger addresses."""
    positions = (torch.arange(num_slots, dtype=torch.float32) + 0.5) / num_slots
    frequencies = torch.arange(1, (dim + 1) // 2 + 1, dtype=torch.float32)
    angles = 2.0 * math.pi * positions[:, None] * frequencies[None, :]
    codes = torch.cat([angles.sin(), angles.cos()], dim=1)[:, :dim]
    return F.normalize(codes, dim=-1)


class SurvivalEvidenceLedger(nn.Module):
    """Conserve token evidence while forming patient-conditioned slots.

    The responsibility matrix is normalized over ledger addresses for every
    token. Consequently, the precision mass written by all tokens equals the
    mass read from all slots, which provides an auditable alternative to
    unconstrained attention pooling.
    """

    def __init__(self, dim: int, num_slots: int, temperature: float):
        super().__init__()
        if num_slots < 2:
            raise ValueError("v4.1 requires at least two ledger slots")
        if temperature <= 0.0:
            raise ValueError("v41_ledger_temperature must be positive")

        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.temperature = float(temperature)
        self.register_buffer(
            "ledger_codes",
            _harmonic_ledger(self.num_slots, self.dim),
            persistent=True,
        )

        self.token_norm = nn.LayerNorm(self.dim)
        self.token_key = nn.Linear(self.dim, self.dim, bias=False)
        self.token_value = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.token_precision = nn.Linear(self.dim, 1)
        self.context_field = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_slots * self.dim),
        )
        self.slot_norm = nn.LayerNorm(self.dim)
        self._last_audit: dict[str, torch.Tensor] = {}

    def audit_last_forward(self) -> dict[str, torch.Tensor]:
        """Return detached per-patient invariants from the latest ledger write.

        These values are diagnostics, not additional optimization targets.  In
        particular, ``mass_error`` and ``responsibility_error`` should remain
        at floating-point round-off: a material increase points to a broken
        ledger write before it can silently contaminate OT.
        """
        if not self._last_audit:
            raise RuntimeError("ledger audit requested before a forward pass")
        return {name: value.detach() for name, value in self._last_audit.items()}

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.size(-1) != self.dim:
            raise ValueError(
                f"ledger expects [B,N,{self.dim}] tokens, got {tuple(tokens.shape)}"
            )

        normalized = self.token_norm(tokens)
        keys = F.normalize(self.token_key(normalized), dim=-1)
        values = self.token_value(normalized)
        token_precision = F.softplus(self.token_precision(normalized).squeeze(-1)) + 1e-4

        context = values.mean(dim=1)
        patient_field = self.context_field(context).view(
            tokens.size(0), self.num_slots, self.dim
        )
        addresses = F.normalize(
            self.ledger_codes.unsqueeze(0) + patient_field,
            dim=-1,
        )

        logits = torch.einsum("bnd,bkd->bnk", keys, addresses)
        responsibilities = torch.softmax(logits / self.temperature, dim=-1)
        weighted_mass = responsibilities * token_precision.unsqueeze(-1)
        slot_mass = weighted_mass.sum(dim=1)
        slots = torch.einsum("bnk,bnd->bkd", weighted_mass, values)
        slots = slots / slot_mass.unsqueeze(-1).clamp_min(1e-6)
        slots = self.slot_norm(slots + addresses)

        expected_mass = (
            token_precision.sum(dim=1, keepdim=True) / float(self.num_slots)
        ).clamp_min(1e-6)
        confidence = 1.0 - torch.exp(-slot_mass / expected_mass)
        assignment = responsibilities.transpose(1, 2)

        # Keep the two structural invariants and collapse indicators beside the
        # ledger output.  They make a bad run diagnosable without changing the
        # model objective or injecting a hand-tuned anti-collapse penalty.
        mass_ratio = slot_mass / expected_mass
        assignment_entropy = -(
            responsibilities.clamp_min(1e-8)
            * responsibilities.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(2, self.num_slots))
        self._last_audit = {
            "written_mass": token_precision.sum(dim=1),
            "read_mass": slot_mass.sum(dim=1),
            "mass_error": (
                slot_mass.sum(dim=1) - token_precision.sum(dim=1)
            ).abs(),
            "responsibility_error": (
                responsibilities.sum(dim=-1) - 1.0
            ).abs().amax(dim=1),
            "active_slot_fraction": (mass_ratio >= 0.01).to(tokens.dtype).mean(dim=1),
            "minimum_slot_mass_ratio": mass_ratio.amin(dim=1),
            "assignment_entropy": assignment_entropy.mean(dim=1),
        }
        return slots, assignment, slot_mass, confidence.clamp(0.0, 1.0)


class CrossLedgerCompletion(nn.Module):
    """Recover shared evidence while retaining modality-private uncertainty."""

    def __init__(self, dim: int, confidence_cap: float, shared_rank: int):
        super().__init__()
        if not 0.0 < confidence_cap <= 1.0:
            raise ValueError("v41_missing_confidence_cap must be in (0, 1]")
        if shared_rank < 1:
            raise ValueError("v41_shared_rank must be positive")
        self.confidence_cap = float(confidence_cap)
        self.shared_rank = min(int(shared_rank), max(1, dim // 2))
        self.source_shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.shared_rank),
            nn.SiLU(),
        )
        self.target_shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.shared_rank),
            nn.SiLU(),
        )
        self.shared_decoder = nn.Linear(self.shared_rank, dim)
        self.shared_log_variance = nn.Linear(self.shared_rank, dim)
        self.recoverability = nn.Linear(self.shared_rank, 1)
        self.private_uncertainty = nn.Linear(self.shared_rank, 1)

    def decompose_target(
        self, target_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a low-rank shared target and its exact private residual."""
        shared = self.shared_decoder(self.target_shared(target_slots))
        private = target_slots - shared
        return shared, private

    def forward(
        self, source_slots: torch.Tensor, source_confidence: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        shared_latent = self.source_shared(source_slots)
        shared_evidence = self.shared_decoder(shared_latent)
        log_variance = self.shared_log_variance(shared_latent).clamp(-5.0, 3.0)
        recoverability = torch.sigmoid(
            self.recoverability(shared_latent).squeeze(-1)
        )
        private_uncertainty = F.softplus(
            self.private_uncertainty(shared_latent).squeeze(-1)
        )
        confidence = (
            source_confidence
            * recoverability
            * torch.exp(-private_uncertainty)
        ).clamp(0.0, self.confidence_cap)
        return (
            shared_evidence,
            log_variance,
            confidence,
            private_uncertainty,
            recoverability,
        )


class DCTV41SurvivalEvidenceLedger(DistributionalCounterfactualTransport):
    """Missing-aware ledger slots followed by the verified DCT v3.3 backbone.

    Reused DCT components:
    stage-specific three-geometry costs, evidence-conditioned Sinkhorn,
    train-fold censoring/IPCW references, empirical low/high cost anchors,
    cost-space interventions, re-solved counterfactual transport, and the
    shared discrete-time survival decoder.

    New v4.1 components:
    evidence-conserving ledger slots, uncertainty-aware cross-ledger
    completion, confidence-tempered OT marginals, and the SELC objective.
    """

    def __init__(
        self,
        args,
        omic_input_dim=None,
        omic_names=None,
        pathway_names=None,
    ):
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.v41_modality_dropout = float(
            getattr(args, "v41_modality_dropout", 0.35)
        )
        self.v41_lambda_completion = float(
            getattr(args, "v41_lambda_completion", 0.05)
        )
        self.v41_lambda_ledger = float(getattr(args, "v41_lambda_ledger", 0.02))
        self.v41_lambda_survival = float(
            getattr(args, "v41_lambda_survival", 0.05)
        )
        self.v41_lambda_private = float(
            getattr(args, "v41_lambda_private", 0.02)
        )
        self.v41_confidence_floor = float(
            getattr(args, "v41_confidence_floor", 0.05)
        )
        self.v41_missing_confidence_cap = float(
            getattr(args, "v41_missing_confidence_cap", 0.65)
        )
        temperature = float(getattr(args, "v41_ledger_temperature", 0.25))

        if not 0.0 <= self.v41_modality_dropout < 1.0:
            raise ValueError("v41_modality_dropout must be in [0, 1)")
        for name, value in (
            ("v41_lambda_completion", self.v41_lambda_completion),
            ("v41_lambda_ledger", self.v41_lambda_ledger),
            ("v41_lambda_survival", self.v41_lambda_survival),
            ("v41_lambda_private", self.v41_lambda_private),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.v41_confidence_floor < 1.0:
            raise ValueError("v41_confidence_floor must be in (0, 1)")

        # v4.1 deliberately contains neither v3.3 Slot Attention nor its
        # shared-prototype coordinate dictionary.
        del self.slot_attention_wsi
        del self.slot_attention_omic
        del self.shared_wsi_prototypes
        del self.shared_omic_prototypes

        self.wsi_ledger = SurvivalEvidenceLedger(
            dim, int(args.slot_num_wsi), temperature
        )
        self.omic_ledger = SurvivalEvidenceLedger(
            dim, int(args.slot_num_omics), temperature
        )
        if int(args.slot_num_wsi) != int(args.slot_num_omics):
            raise ValueError(
                "v4.1 requires equal WSI and omics ledger sizes for counterpart completion"
            )
        self.wsi_from_omic = CrossLedgerCompletion(
            dim,
            self.v41_missing_confidence_cap,
            int(getattr(args, "v41_shared_rank", 64)),
        )
        self.omic_from_wsi = CrossLedgerCompletion(
            dim,
            self.v41_missing_confidence_cap,
            int(getattr(args, "v41_shared_rank", 64)),
        )
        self.null_wsi_ledger = nn.Parameter(
            torch.zeros(1, int(args.slot_num_wsi), dim)
        )
        self.null_omic_ledger = nn.Parameter(
            torch.zeros(1, int(args.slot_num_omics), dim)
        )

        # Score-first remains the factual supervision. ETAR is not combined
        # with the new missingness objective.
        self.dct_lambda_etar = 0.0
        self._v41_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _availability(
        kwargs: dict[str, Any],
        name: str,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = kwargs.get(name)
        if value is None:
            return reference.new_ones(batch)
        value = torch.as_tensor(value, device=reference.device).flatten()
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(f"{name} must be scalar or have one value per patient")
        return (value > 0).to(reference.dtype)

    @staticmethod
    def _event_distribution(logits: torch.Tensor) -> torch.Tensor:
        hazards = torch.sigmoid(logits)
        survival_before = torch.cat(
            [torch.ones_like(hazards[:, :1]), torch.cumprod(1.0 - hazards, dim=1)[:, :-1]],
            dim=1,
        )
        event_mass = survival_before * hazards
        tail = torch.cumprod(1.0 - hazards, dim=1)[:, -1:]
        distribution = torch.cat([event_mass, tail], dim=1)
        return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _gaussian_completion_loss(
        predicted: torch.Tensor,
        log_variance: torch.Tensor,
        target: torch.Tensor,
        target_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        error = (predicted - target.detach()).square()
        nll = 0.5 * (error * torch.exp(-log_variance) + log_variance)
        nll = nll.mean(dim=-1)
        weights = target_confidence.detach() * valid[:, None].to(nll.dtype)
        return (nll * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _ledger_distribution_loss(
        predicted_confidence: torch.Tensor,
        target_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        p = predicted_confidence / predicted_confidence.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        q = target_confidence.detach() / target_confidence.detach().sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        midpoint = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=1)
            + (q * (q.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=1)
        )
        return (js * valid.to(js.dtype)).sum() / valid.to(js.dtype).sum().clamp_min(1.0)

    @staticmethod
    def _private_uncertainty_loss(
        predicted_uncertainty: torch.Tensor,
        private_residual: torch.Tensor,
        target_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate irrecoverable uncertainty to private residual energy."""
        target = private_residual.detach().square().mean(dim=-1).sqrt()
        loss = F.smooth_l1_loss(
            predicted_uncertainty,
            target,
            reduction="none",
        )
        weights = target_confidence.detach() * valid[:, None].to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _global_missing_flag(value: Any, name: str) -> bool:
        if value is None:
            return False
        flag = torch.as_tensor(value).flatten()
        if flag.numel() != 1:
            raise ValueError(
                f"{name} is a global switch; use the per-patient availability flag instead"
            )
        return bool(flag.item())

    def _sample_availability(
        self,
        wsi_available: torch.Tensor,
        omic_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.v41_modality_dropout <= 0.0:
            dropped = torch.zeros_like(wsi_available, dtype=torch.bool)
            return wsi_available, omic_available, dropped

        both = (wsi_available > 0) & (omic_available > 0)
        draw = torch.rand_like(wsi_available)
        half = self.v41_modality_dropout * 0.5
        drop_wsi = both & (draw < half)
        drop_omic = both & (draw >= half) & (
            draw < self.v41_modality_dropout
        )
        used_wsi = wsi_available * (~drop_wsi).to(wsi_available.dtype)
        used_omic = omic_available * (~drop_omic).to(omic_available.dtype)
        return used_wsi, used_omic, drop_wsi | drop_omic

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        (
            full_wsi,
            wsi_assignment,
            wsi_mass,
            full_wsi_confidence,
        ) = self.wsi_ledger(x_wsi_proj)
        (
            full_omic,
            omic_assignment,
            omic_mass,
            full_omic_confidence,
        ) = self.omic_ledger(x_omics)

        (
            shared_wsi,
            wsi_log_variance,
            completed_wsi_confidence,
            wsi_private_uncertainty,
            wsi_recoverability,
        ) = self.wsi_from_omic(full_omic, full_omic_confidence)
        (
            shared_omic,
            omic_log_variance,
            completed_omic_confidence,
            omic_private_uncertainty,
            omic_recoverability,
        ) = self.omic_from_wsi(full_wsi, full_wsi_confidence)
        target_wsi_shared, target_wsi_private = self.wsi_from_omic.decompose_target(
            full_wsi
        )
        target_omic_shared, target_omic_private = self.omic_from_wsi.decompose_target(
            full_omic
        )

        batch = x_wsi_proj.size(0)
        externally_available_wsi = self._availability(
            kwargs, "wsi_available", batch, x_wsi_proj
        )
        externally_available_omic = self._availability(
            kwargs, "omics_available", batch, x_wsi_proj
        )
        if self._global_missing_flag(kwargs.get("wsi_missing"), "wsi_missing"):
            externally_available_wsi = torch.zeros_like(externally_available_wsi)
        if self._global_missing_flag(kwargs.get("omic_missing"), "omic_missing"):
            externally_available_omic = torch.zeros_like(externally_available_omic)

        used_wsi, used_omic, artificially_dropped = self._sample_availability(
            externally_available_wsi,
            externally_available_omic,
        )
        wsi_observed = used_wsi[:, None, None] > 0
        omic_observed = used_omic[:, None, None] > 0
        wsi_can_complete = (~wsi_observed) & (used_omic[:, None, None] > 0)
        omic_can_complete = (~omic_observed) & (used_wsi[:, None, None] > 0)

        null_wsi = self.null_wsi_ledger.expand(batch, -1, -1)
        null_omic = self.null_omic_ledger.expand(batch, -1, -1)
        slots_wsi = torch.where(
            wsi_observed,
            full_wsi,
            torch.where(wsi_can_complete, shared_wsi, null_wsi),
        )
        slots_omic = torch.where(
            omic_observed,
            full_omic,
            torch.where(omic_can_complete, shared_omic, null_omic),
        )

        floor_wsi = torch.full_like(full_wsi_confidence, self.v41_confidence_floor)
        floor_omic = torch.full_like(full_omic_confidence, self.v41_confidence_floor)
        wsi_confidence = torch.where(
            used_wsi[:, None] > 0,
            full_wsi_confidence,
            torch.where(
                used_omic[:, None] > 0,
                completed_wsi_confidence,
                floor_wsi,
            ),
        )
        omic_confidence = torch.where(
            used_omic[:, None] > 0,
            full_omic_confidence,
            torch.where(
                used_wsi[:, None] > 0,
                completed_omic_confidence,
                floor_omic,
            ),
        )

        complete_pair = (
            (externally_available_wsi > 0) & (externally_available_omic > 0)
        )
        completion_loss = self._gaussian_completion_loss(
            shared_wsi,
            wsi_log_variance,
            target_wsi_shared,
            full_wsi_confidence,
            complete_pair,
        ) + self._gaussian_completion_loss(
            shared_omic,
            omic_log_variance,
            target_omic_shared,
            full_omic_confidence,
            complete_pair,
        )
        shared_autoencoding_loss = 0.5 * (
            self._gaussian_completion_loss(
                target_wsi_shared,
                torch.zeros_like(target_wsi_shared),
                full_wsi,
                full_wsi_confidence,
                complete_pair,
            )
            + self._gaussian_completion_loss(
                target_omic_shared,
                torch.zeros_like(target_omic_shared),
                full_omic,
                full_omic_confidence,
                complete_pair,
            )
        )
        completion_loss = completion_loss + 0.25 * shared_autoencoding_loss
        private_loss = self._private_uncertainty_loss(
            wsi_private_uncertainty,
            target_wsi_private,
            full_wsi_confidence,
            complete_pair,
        ) + self._private_uncertainty_loss(
            omic_private_uncertainty,
            target_omic_private,
            full_omic_confidence,
            complete_pair,
        )
        ledger_loss = self._ledger_distribution_loss(
            completed_wsi_confidence,
            full_wsi_confidence,
            complete_pair,
        ) + self._ledger_distribution_loss(
            completed_omic_confidence,
            full_omic_confidence,
            complete_pair,
        )

        self._v41_cache = {
            "full_wsi": full_wsi,
            "full_omic": full_omic,
            "full_wsi_confidence": full_wsi_confidence,
            "full_omic_confidence": full_omic_confidence,
            "wsi_confidence": wsi_confidence,
            "omic_confidence": omic_confidence,
            "wsi_mass": wsi_mass,
            "omic_mass": omic_mass,
            "wsi_ledger_audit": self.wsi_ledger.audit_last_forward(),
            "omic_ledger_audit": self.omic_ledger.audit_last_forward(),
            "completion_loss": completion_loss,
            "private_loss": private_loss,
            "ledger_loss": ledger_loss,
            "shared_wsi": shared_wsi,
            "shared_omic": shared_omic,
            "wsi_private_uncertainty": wsi_private_uncertainty,
            "omic_private_uncertainty": omic_private_uncertainty,
            "wsi_recoverability": wsi_recoverability,
            "omic_recoverability": omic_recoverability,
            "artificially_dropped": artificially_dropped,
            "used_wsi": used_wsi,
            "used_omic": used_omic,
        }

        wsi_assignment = wsi_assignment * used_wsi[:, None, None]
        omic_assignment = omic_assignment * used_omic[:, None, None]
        return slots_wsi, slots_omic, wsi_assignment, omic_assignment

    def _temper_marginals(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        wsi_confidence: torch.Tensor,
        omic_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_scale = wsi_confidence.clamp_min(self.v41_confidence_floor).unsqueeze(1)
        col_scale = omic_confidence.clamp_min(self.v41_confidence_floor).unsqueeze(1)
        rows = rows * row_scale
        cols = cols * col_scale
        rows = rows / rows.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cols = cols / cols.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return rows, cols

    def _cost_tensor(self, slots_wsi, slots_omic):
        costs, rows, cols, evidence_gate = super()._cost_tensor(
            slots_wsi, slots_omic
        )
        if self._v41_cache:
            rows, cols = self._temper_marginals(
                rows,
                cols,
                self._v41_cache["wsi_confidence"],
                self._v41_cache["omic_confidence"],
            )
        return costs, rows, cols, evidence_gate

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
        cache = self._v41_cache
        completion_loss = cache["completion_loss"]
        private_loss = cache["private_loss"]
        ledger_loss = cache["ledger_loss"]
        dropped = cache["artificially_dropped"]

        survival_consistency = factual_logits.new_zeros(())
        if bool(dropped.any()):
            with torch.no_grad():
                full_costs, full_rows, full_cols, _ = (
                    DistributionalCounterfactualTransport._cost_tensor(
                        self,
                        cache["full_wsi"],
                        cache["full_omic"],
                    )
                )
                full_rows, full_cols = self._temper_marginals(
                    full_rows,
                    full_cols,
                    cache["full_wsi_confidence"],
                    cache["full_omic_confidence"],
                )
                full_plans, _ = self._plans_from_cost_tensor(
                    full_costs, full_rows, full_cols, epoch
                )
                full_logits, _ = self._encode_logits_from_plans(
                    cache["full_wsi"], cache["full_omic"], full_plans
                )
                target_distribution = self._event_distribution(full_logits)

            predicted_distribution = self._event_distribution(factual_logits)
            per_patient = (
                target_distribution
                * (
                    target_distribution.clamp_min(1e-8).log()
                    - predicted_distribution.clamp_min(1e-8).log()
                )
            ).sum(dim=1)
            survival_consistency = (
                per_patient * dropped.to(per_patient.dtype)
            ).sum() / dropped.sum().clamp_min(1)

        objective = (
            self.v41_lambda_completion * completion_loss
            + self.v41_lambda_ledger * ledger_loss
            + self.v41_lambda_survival * survival_consistency
            + self.v41_lambda_private * private_loss
        )
        metrics = {
            "v41_completion": completion_loss,
            "v41_private_uncertainty": private_loss,
            "v41_ledger": ledger_loss,
            "v41_survival_consistency": survival_consistency,
            "v41_missing_fraction": dropped.to(factual_logits.dtype).mean(),
            "v41_wsi_confidence": cache["wsi_confidence"].mean(),
            "v41_omic_confidence": cache["omic_confidence"].mean(),
            "v41_wsi_ledger_mass_error": cache["wsi_ledger_audit"][
                "mass_error"
            ].mean(),
            "v41_omic_ledger_mass_error": cache["omic_ledger_audit"][
                "mass_error"
            ].mean(),
            "v41_wsi_ledger_assignment_error": cache["wsi_ledger_audit"][
                "responsibility_error"
            ].mean(),
            "v41_omic_ledger_assignment_error": cache["omic_ledger_audit"][
                "responsibility_error"
            ].mean(),
            "v41_wsi_active_slot_fraction": cache["wsi_ledger_audit"][
                "active_slot_fraction"
            ].mean(),
            "v41_omic_active_slot_fraction": cache["omic_ledger_audit"][
                "active_slot_fraction"
            ].mean(),
            "v41_wsi_assignment_entropy": cache["wsi_ledger_audit"][
                "assignment_entropy"
            ].mean(),
            "v41_omic_assignment_entropy": cache["omic_ledger_audit"][
                "assignment_entropy"
            ].mean(),
            "v41_objective": objective,
        }
        return objective, metrics

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        if not self.training and self.last_explanations is not None:
            cache = self._v41_cache
            self.last_explanations.update(
                {
                    "wsi_ledger_assignment": self.last_explanations[
                        "wsi_coordinate_assignment"
                    ],
                    "omic_ledger_assignment": self.last_explanations[
                        "omic_coordinate_assignment"
                    ],
                    "wsi_ledger_confidence": cache["wsi_confidence"].detach(),
                    "omic_ledger_confidence": cache["omic_confidence"].detach(),
                    "wsi_available": cache["used_wsi"].detach(),
                    "omic_available": cache["used_omic"].detach(),
                    "wsi_recoverable_shared": cache["shared_wsi"].detach(),
                    "omic_recoverable_shared": cache["shared_omic"].detach(),
                    "wsi_private_uncertainty": cache[
                        "wsi_private_uncertainty"
                    ].detach(),
                    "omic_private_uncertainty": cache[
                        "omic_private_uncertainty"
                    ].detach(),
                    "wsi_recoverability": cache["wsi_recoverability"].detach(),
                    "omic_recoverability": cache["omic_recoverability"].detach(),
                    "wsi_ledger_written_mass": cache["wsi_ledger_audit"][
                        "written_mass"
                    ],
                    "omic_ledger_written_mass": cache["omic_ledger_audit"][
                        "written_mass"
                    ],
                    "wsi_ledger_read_mass": cache["wsi_ledger_audit"]["read_mass"],
                    "omic_ledger_read_mass": cache["omic_ledger_audit"]["read_mass"],
                    "wsi_ledger_mass_error": cache["wsi_ledger_audit"]["mass_error"],
                    "omic_ledger_mass_error": cache["omic_ledger_audit"]["mass_error"],
                    "wsi_ledger_assignment_error": cache["wsi_ledger_audit"][
                        "responsibility_error"
                    ],
                    "omic_ledger_assignment_error": cache["omic_ledger_audit"][
                        "responsibility_error"
                    ],
                    "wsi_ledger_active_slot_fraction": cache["wsi_ledger_audit"][
                        "active_slot_fraction"
                    ],
                    "omic_ledger_active_slot_fraction": cache["omic_ledger_audit"][
                        "active_slot_fraction"
                    ],
                    "wsi_ledger_assignment_entropy": cache["wsi_ledger_audit"][
                        "assignment_entropy"
                    ],
                    "omic_ledger_assignment_entropy": cache["omic_ledger_audit"][
                        "assignment_entropy"
                    ],
                }
            )
        return output
