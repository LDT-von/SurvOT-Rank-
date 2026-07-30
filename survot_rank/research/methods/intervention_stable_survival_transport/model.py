"""Intervention-stable transport with exact patch-pathway risk attribution.

The method is deliberately independent from DCT's score-first objectives.  It
uses raw projected WSI patches and pathway tokens, tests their transport under
controlled masking interventions, writes the resulting stability back into
the transport cost, and re-solves Sinkhorn.  The survival head is additive by
construction, so every stage logit is exactly the bias plus the sum of signed
patch-pathway contributions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp


def _normalise_mask(mask: torch.Tensor) -> torch.Tensor:
    """Turn a non-empty boolean token mask into a probability marginal."""

    weights = mask.to(dtype=torch.float32)
    empty = weights.sum(dim=1) == 0
    if empty.any():
        weights = weights.clone()
        weights[empty, 0] = 1.0
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


def masked_log_sinkhorn_plan(
    cost: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    *,
    eps: float = 0.05,
    max_iter: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced log-domain Sinkhorn on the visible rows and columns only.

    Returns the plan, row marginal and column marginal.  Invalid edges are
    exactly zero in the returned plan, which makes intervention support
    comparisons auditable.
    """

    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, patches, pathways]")
    if row_mask.shape != cost.shape[:2]:
        raise ValueError("row_mask shape does not match cost")
    if col_mask.shape != (cost.size(0), cost.size(2)):
        raise ValueError("col_mask shape does not match cost")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    row_mask = row_mask.bool()
    col_mask = col_mask.bool()
    rows = _normalise_mask(row_mask).to(device=cost.device, dtype=cost.dtype)
    cols = _normalise_mask(col_mask).to(device=cost.device, dtype=cost.dtype)
    row_mask = rows > 0
    col_mask = cols > 0
    edge_mask = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)

    negative = cost.new_tensor(-1.0e4)
    kernel = (-cost / float(eps)).masked_fill(~edge_mask, negative)
    log_rows = torch.where(row_mask, rows.clamp_min(1e-12).log(), negative)
    log_cols = torch.where(col_mask, cols.clamp_min(1e-12).log(), negative)
    log_u = torch.zeros_like(log_rows)
    log_v = torch.zeros_like(log_cols)
    for _ in range(int(max_iter)):
        log_u = log_rows - torch.logsumexp(
            kernel + log_v.unsqueeze(1), dim=2
        )
        log_u = torch.where(row_mask, log_u, negative)
        log_v = log_cols - torch.logsumexp(
            kernel + log_u.unsqueeze(2), dim=1
        )
        log_v = torch.where(col_mask, log_v, negative)

    log_plan = kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    plan = log_plan.exp().masked_fill(~edge_mask, 0.0)
    # Project the forward value tightly onto the requested marginals.  The
    # straight-through correction keeps gradients from the stable log-domain
    # solver while avoiding a very large autograd graph through 100+ scaling
    # iterations on 2k-patch slides.
    with torch.no_grad():
        projected = plan.detach().clone()
        for _ in range(max(100, int(max_iter) * 4)):
            projected = projected * (
                rows / projected.sum(dim=2).clamp_min(1e-12)
            ).unsqueeze(2)
            projected = projected.masked_fill(~edge_mask, 0.0)
            projected = projected * (
                cols / projected.sum(dim=1).clamp_min(1e-12)
            ).unsqueeze(1)
            projected = projected.masked_fill(~edge_mask, 0.0)
        projected = projected / projected.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-12)
    plan = plan + (projected - plan).detach()
    return plan, rows, cols


def _normalised_js(
    factual: torch.Tensor,
    intervened: torch.Tensor,
    visible_edges: torch.Tensor,
) -> torch.Tensor:
    """Jensen-Shannon divergence on an intervention's common support."""

    mask = visible_edges.to(dtype=factual.dtype)
    p = factual * mask
    q = intervened * mask
    p = p / p.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    q = q / q.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    midpoint = 0.5 * (p + q)
    p_term = torch.where(
        p > 0,
        p * (p.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()),
        torch.zeros_like(p),
    )
    q_term = torch.where(
        q > 0,
        q * (q.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()),
        torch.zeros_like(q),
    )
    return 0.5 * (p_term + q_term).sum(dim=(1, 2)).mean()


def _survival_risk(logits: torch.Tensor) -> torch.Tensor:
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)


class InterventionStableSurvivalTransport(nn.Module):
    """V4.0 intervention-stable transport with intrinsic explanations.

    The final stage logits obey the exact identity

    ``logit[b, s] = stage_bias[s] + edge_contribution[b, s].sum()``.

    No opaque attention or Transformer is placed after the stable transport
    plan.  This makes WSI-patch, pathway and patch-pathway explanations exact
    decompositions of the model's hazard logits rather than post-hoc maps.
    """

    def __init__(
        self,
        args,
        omic_input_dim=None,
        omic_names=None,
        pathway_names=None,
    ):
        super().__init__()
        self.args = args
        self.num_classes = int(args.n_classes)
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.dim = int(args.wsi_projection_dim)
        self.omics_input_dim = omic_input_dim
        self.omic_sizes = args.omic_sizes
        self.pathway_names = list(pathway_names or [])

        self.ist_eps = float(getattr(args, "ist_eps", 0.05))
        self.ist_sinkhorn_iters = int(getattr(args, "ist_sinkhorn_iters", 30))
        self.ist_num_interventions = int(
            getattr(args, "ist_num_interventions", 2)
        )
        self.ist_keep_ratio = float(getattr(args, "ist_keep_ratio", 0.75))
        self.ist_stability_beta = float(
            getattr(args, "ist_stability_beta", 1.0)
        )
        self.ist_stability_strength = float(
            getattr(args, "ist_stability_strength", 0.10)
        )
        self.ist_lambda_plan = float(getattr(args, "ist_lambda_plan", 0.05))
        self.ist_lambda_attribution = float(
            getattr(args, "ist_lambda_attribution", 0.05)
        )
        self.ist_lambda_risk = float(getattr(args, "ist_lambda_risk", 0.0))
        self.ist_edge_value_scale = float(
            getattr(args, "ist_edge_value_scale", 4.0)
        )
        self.ist_eval_seed = int(getattr(args, "ist_eval_seed", 20260725))
        self.ist_deletion_penalty = float(
            getattr(args, "ist_deletion_penalty", 8.0)
        )
        self._validate_hyperparameters()

        self.wsi_mlp = WSI_Mlp(
            dim_in=self.wsi_embedding_dim,
            feat_dim=self.dim,
        )
        self._init_omics_encoder()

        self.wsi_stage_value = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes),
        )
        self.omic_stage_value = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes),
        )
        self.edge_pair_scale = nn.Parameter(
            torch.zeros(self.num_classes, 2)
        )
        self.stage_bias = nn.Parameter(torch.zeros(self.num_classes))

        self.last_explanations: dict[str, torch.Tensor] | None = None
        self.last_training_losses: dict[str, torch.Tensor] = {}
        self._last_eval_cache: dict[str, torch.Tensor] | None = None

        if omic_names:
            try:
                self.all_gene_names = list(np.unique(np.concatenate(omic_names)))
            except (TypeError, ValueError):
                pass

    def _validate_hyperparameters(self) -> None:
        if self.ist_eps <= 0:
            raise ValueError("ist_eps must be positive")
        if self.ist_sinkhorn_iters <= 0:
            raise ValueError("ist_sinkhorn_iters must be positive")
        if self.ist_num_interventions < 1:
            raise ValueError("ist_num_interventions must be at least one")
        if not 0 < self.ist_keep_ratio <= 1:
            raise ValueError("ist_keep_ratio must be in (0, 1]")
        for name in (
            "ist_stability_beta",
            "ist_stability_strength",
            "ist_lambda_plan",
            "ist_lambda_attribution",
            "ist_lambda_risk",
            "ist_edge_value_scale",
            "ist_deletion_penalty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def _init_omics_encoder(self) -> None:
        rna_format = self.args.rna_format
        if rna_format in {"Pathways", "RankedGenes"}:
            if self.omic_sizes is None:
                raise ValueError("omic_sizes are required for pathway inputs")
            self.num_pathways = len(self.omic_sizes)
            self.sig_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        SNN_Block(dim1=int(size), dim2=self.dim),
                        SNN_Block(
                            dim1=self.dim,
                            dim2=self.dim,
                            dropout=0.25,
                        ),
                    )
                    for size in self.omic_sizes
                ]
            )
        elif rna_format == "GeneEmbedding":
            self.sig_networks = SNN_Block(dim1=768, dim2=self.dim)
        elif rna_format == "RNASeq":
            if self.omics_input_dim is None:
                raise ValueError("omic_input_dim is required for RNASeq")
            self.sig_networks = SNN_Block(
                dim1=int(self.omics_input_dim),
                dim2=self.dim,
            )
        else:
            raise ValueError(f"Invalid omics_format: {rna_format}")

    def _encode_omics(self, kwargs) -> torch.Tensor:
        if self.args.rna_format in {"Pathways", "RankedGenes"}:
            values = [
                kwargs[f"x_omic{index}"]
                for index in range(1, self.num_pathways + 1)
            ]
            tokens = [
                self.sig_networks[index](value)
                for index, value in enumerate(values)
            ]
            return torch.stack(tokens, dim=1)
        encoded = self.sig_networks(kwargs["x_omics"])
        return encoded.unsqueeze(1) if encoded.ndim == 2 else encoded

    @staticmethod
    def _valid_wsi_mask(x_wsi: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x_wsi).all(dim=-1) & (x_wsi.abs().sum(dim=-1) > 0)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        return valid

    @staticmethod
    def _valid_omic_mask(x_omics: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x_omics).all(dim=-1)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        return valid

    @staticmethod
    def _cosine_cost(x_wsi: torch.Tensor, x_omics: torch.Tensor) -> torch.Tensor:
        wsi = F.normalize(x_wsi, dim=-1)
        omics = F.normalize(x_omics, dim=-1)
        return 1.0 - torch.bmm(wsi, omics.transpose(1, 2))

    def _subset_mask(
        self,
        valid: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        selected = torch.zeros_like(valid)
        for batch_index in range(valid.size(0)):
            indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            count = max(1, int(math.ceil(indices.numel() * self.ist_keep_ratio)))
            if generator is None:
                order = torch.randperm(indices.numel(), device=indices.device)
            else:
                order = torch.randperm(
                    indices.numel(),
                    generator=generator,
                    device="cpu",
                ).to(indices.device)
            selected[batch_index, indices[order[:count]]] = True
        return selected

    def _intervention_masks(
        self,
        row_valid: torch.Tensor,
        col_valid: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        generator = None
        if not self.training:
            generator = torch.Generator(device="cpu").manual_seed(
                self.ist_eval_seed
            )
        masks = []
        for view_index in range(self.ist_num_interventions):
            mode = view_index % 3
            row_mask = row_valid
            col_mask = col_valid
            if mode in (0, 2):
                row_mask = self._subset_mask(row_valid, generator=generator)
            if mode in (1, 2):
                col_mask = self._subset_mask(col_valid, generator=generator)
            masks.append((row_mask, col_mask))
        return masks

    def _solve(
        self,
        cost: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return masked_log_sinkhorn_plan(
            cost,
            row_mask,
            col_mask,
            eps=self.ist_eps,
            max_iter=self.ist_sinkhorn_iters,
        )

    def _solve_mask_views(
        self,
        cost: torch.Tensor,
        masks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve factual/intervention masks in one vectorized Sinkhorn batch."""
        if not masks:
            raise ValueError("at least one transport mask view is required")
        batch, rows, cols = cost.shape
        view_count = len(masks)
        row_masks = torch.stack([item[0] for item in masks], dim=1)
        col_masks = torch.stack([item[1] for item in masks], dim=1)
        expanded_cost = (
            cost[:, None]
            .expand(batch, view_count, rows, cols)
            .reshape(batch * view_count, rows, cols)
        )
        flat_plan, flat_rows, flat_cols = self._solve(
            expanded_cost,
            row_masks.reshape(batch * view_count, rows),
            col_masks.reshape(batch * view_count, cols),
        )
        return (
            flat_plan.reshape(batch, view_count, rows, cols),
            flat_rows.reshape(batch, view_count, rows),
            flat_cols.reshape(batch, view_count, cols),
        )

    def _stable_edge_score(
        self,
        factual_plan: torch.Tensor,
        intervention_plans: list[torch.Tensor],
        intervention_masks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plan_values = [factual_plan]
        availability = [factual_plan > 0]
        for plan, (row_mask, col_mask) in zip(
            intervention_plans, intervention_masks
        ):
            plan_values.append(plan)
            availability.append(row_mask.unsqueeze(2) & col_mask.unsqueeze(1))
        plans = torch.stack(plan_values, dim=0)
        visible = torch.stack(availability, dim=0)
        weights = visible.to(dtype=plans.dtype)
        count = weights.sum(dim=0).clamp_min(1.0)
        mean = (plans * weights).sum(dim=0) / count
        variance = ((plans - mean.unsqueeze(0)).square() * weights).sum(dim=0)
        variance = variance / count
        coefficient_variation = variance / mean.square().clamp_min(1e-10)
        reliability = torch.exp(
            -self.ist_stability_beta * coefficient_variation
        ).clamp(min=1e-4, max=1.0)
        relative_mass = mean / mean.amax(dim=(1, 2), keepdim=True).clamp_min(
            1e-12
        )
        score = (relative_mass * reliability).clamp(min=1e-4, max=1.0)
        return score, reliability, variance

    def _edge_values(
        self,
        x_wsi: torch.Tensor,
        x_omics: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        wsi_value = self.wsi_stage_value(x_wsi).unsqueeze(2)
        omic_value = self.omic_stage_value(x_omics).unsqueeze(1)
        cosine_similarity = 1.0 - cost
        euclidean = torch.cdist(
            F.normalize(x_wsi, dim=-1),
            F.normalize(x_omics, dim=-1),
        ) / 2.0
        pair_terms = torch.stack((cosine_similarity, -euclidean), dim=-1)
        pair_value = torch.einsum(
            "bngq,sq->bngs",
            pair_terms,
            self.edge_pair_scale,
        )
        raw = wsi_value + omic_value + pair_value
        return self.ist_edge_value_scale * torch.tanh(raw)

    def _logits_from_plan(
        self,
        plan: torch.Tensor,
        edge_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contribution = plan.unsqueeze(-1) * edge_values
        logits = self.stage_bias + contribution.sum(dim=(1, 2))
        return logits, contribution.permute(0, 3, 1, 2)

    @staticmethod
    def _marginal_error(
        plan: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        row_error = (plan.sum(dim=2) - rows).abs().amax(dim=1)
        col_error = (plan.sum(dim=1) - cols).abs().amax(dim=1)
        return torch.maximum(row_error, col_error)

    def _intervention_losses(
        self,
        factual_plan: torch.Tensor,
        stable_logits: torch.Tensor,
        edge_values: torch.Tensor,
        intervention_plans: list[torch.Tensor],
        intervention_masks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        plan_losses = []
        attribution_losses = []
        risk_losses = []
        view_logits = []
        for plan, (row_mask, col_mask) in zip(
            intervention_plans, intervention_masks
        ):
            edge_mask = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)
            plan_losses.append(
                _normalised_js(factual_plan, plan, edge_mask)
            )

            visible = edge_mask.to(dtype=plan.dtype)
            factual_visible = factual_plan * visible
            factual_visible = factual_visible / factual_visible.sum(
                dim=(1, 2), keepdim=True
            ).clamp_min(1e-12)
            intervention_visible = plan * visible
            intervention_visible = intervention_visible / intervention_visible.sum(
                dim=(1, 2), keepdim=True
            ).clamp_min(1e-12)
            factual_attr = factual_visible.unsqueeze(-1) * edge_values
            intervention_attr = intervention_visible.unsqueeze(-1) * edge_values
            factual_attr = factual_attr / factual_attr.abs().sum(
                dim=(1, 2), keepdim=True
            ).clamp_min(1e-12)
            intervention_attr = intervention_attr / intervention_attr.abs().sum(
                dim=(1, 2), keepdim=True
            ).clamp_min(1e-12)
            attribution_losses.append(
                F.smooth_l1_loss(intervention_attr, factual_attr)
            )

            current_logits, _ = self._logits_from_plan(plan, edge_values)
            view_logits.append(current_logits)
            risk_losses.append(
                F.smooth_l1_loss(current_logits, stable_logits.detach())
            )

        return (
            torch.stack(plan_losses).mean(),
            torch.stack(attribution_losses).mean(),
            torch.stack(risk_losses).mean(),
            torch.stack(view_logits, dim=1),
        )

    def forward(self, **kwargs):
        raw_wsi = kwargs["x_wsi"].float()
        # Determine transport support before sanitising numerical values.  If a
        # patch embedding contains even one NaN/Inf, replacing only that value
        # with zero must not let the otherwise non-zero patch receive mass.
        row_valid = self._valid_wsi_mask(raw_wsi)
        raw_wsi = torch.nan_to_num(raw_wsi)
        x_wsi = self.wsi_mlp(raw_wsi)
        raw_omics = self._encode_omics(kwargs).float()
        # The pathway encoder propagates non-finite source values.  Preserve
        # that signal for the mask, then use a finite tensor for arithmetic.
        col_valid = self._valid_omic_mask(raw_omics)
        x_omics = torch.nan_to_num(raw_omics)

        factual_cost = self._cosine_cost(x_wsi, x_omics)
        intervention_masks = self._intervention_masks(row_valid, col_valid)
        all_plans, all_rows, all_cols = self._solve_mask_views(
            factual_cost,
            [(row_valid, col_valid), *intervention_masks],
        )
        factual_plan = all_plans[:, 0]
        rows = all_rows[:, 0]
        cols = all_cols[:, 0]
        intervention_plan_tensor = all_plans[:, 1:]
        intervention_plans = list(intervention_plan_tensor.unbind(dim=1))
        stability_score, reliability, stability_variance = (
            self._stable_edge_score(
                factual_plan,
                intervention_plans,
                intervention_masks,
            )
        )
        stable_cost = factual_cost + self.ist_stability_strength * (
            -stability_score.clamp_min(1e-8).log()
        )
        stable_plan, stable_rows, stable_cols = self._solve(
            stable_cost, row_valid, col_valid
        )

        edge_values = self._edge_values(x_wsi, x_omics, factual_cost)
        logits, stage_contribution = self._logits_from_plan(
            stable_plan, edge_values
        )
        completeness_error = (
            logits
            - (
                self.stage_bias
                + stage_contribution.sum(dim=(2, 3))
            )
        ).abs().amax(dim=1)
        marginal_error = self._marginal_error(
            stable_plan, stable_rows, stable_cols
        )

        plan_loss, attribution_loss, risk_loss, intervention_logits = (
            self._intervention_losses(
                factual_plan,
                logits,
                edge_values,
                intervention_plans,
                intervention_masks,
            )
        )
        aux_loss = (
            self.ist_lambda_plan * plan_loss
            + self.ist_lambda_attribution * attribution_loss
            + self.ist_lambda_risk * risk_loss
        )

        self.last_training_losses = {
            "ist_plan_stability": plan_loss.detach(),
            "ist_attribution_stability": attribution_loss.detach(),
            "ist_risk_stability": risk_loss.detach(),
            "ist_total": aux_loss.detach(),
            "ist_completeness_error": completeness_error.mean().detach(),
            "ist_marginal_error": marginal_error.mean().detach(),
            "ist_reliability": reliability.mean().detach(),
            "ist_sinkhorn_batches": logits.new_tensor(2.0).detach(),
            "ist_finite": torch.stack(
                [
                    torch.isfinite(logits).all(),
                    torch.isfinite(aux_loss),
                    torch.isfinite(stable_plan).all(),
                ]
            ).to(dtype=logits.dtype).mean().detach(),
        }

        row_mask_tensor = torch.stack(
            [item[0] for item in intervention_masks], dim=1
        )
        col_mask_tensor = torch.stack(
            [item[1] for item in intervention_masks], dim=1
        )
        self.last_explanations = {
            "factual_cost": factual_cost.detach(),
            "factual_plan": factual_plan.detach(),
            "intervention_plans": intervention_plan_tensor.detach(),
            "intervention_row_masks": row_mask_tensor.detach(),
            "intervention_col_masks": col_mask_tensor.detach(),
            "transport_stability_score": stability_score.detach(),
            "transport_reliability": reliability.detach(),
            "transport_stability_variance": stability_variance.detach(),
            "stable_cost": stable_cost.detach(),
            "stable_plan": stable_plan.detach(),
            "row_marginals": stable_rows.detach(),
            "col_marginals": stable_cols.detach(),
            "edge_values": edge_values.detach().permute(0, 3, 1, 2),
            "stage_edge_contribution": stage_contribution.detach(),
            "stage_patch_contribution": stage_contribution.sum(dim=3).detach(),
            "stage_pathway_contribution": stage_contribution.sum(dim=2).detach(),
            "stage_logits": logits.detach(),
            "stage_bias": self.stage_bias.detach(),
            "hazards": torch.sigmoid(logits).detach(),
            "survival": torch.cumprod(1.0 - torch.sigmoid(logits), dim=1).detach(),
            "risk": _survival_risk(logits).detach(),
            "intervention_logits": intervention_logits.detach(),
            "intervention_risk": _survival_risk(
                intervention_logits.flatten(0, 1)
            ).view(logits.size(0), -1).detach(),
            "completeness_error": completeness_error.detach(),
            "marginal_error": marginal_error.detach(),
            "wsi_valid_mask": row_valid.detach(),
            "omic_valid_mask": col_valid.detach(),
        }
        if not self.training:
            self._last_eval_cache = {
                "stable_cost": stable_cost.detach(),
                "stable_plan": stable_plan.detach(),
                "rows": stable_rows.detach(),
                "cols": stable_cols.detach(),
                "row_valid": row_valid.detach(),
                "col_valid": col_valid.detach(),
                "edge_values": edge_values.detach(),
                "stage_contribution": stage_contribution.detach(),
                "logits": logits.detach(),
                "stage_bias": self.stage_bias.detach(),
            }
            return logits, 0.0
        self._last_eval_cache = None
        return logits, aux_loss

    def explain_last_batch(self) -> dict[str, torch.Tensor]:
        if self.last_explanations is None:
            raise RuntimeError("Run a forward pass before requesting explanations")
        return self.last_explanations

    @torch.no_grad()
    def deletion_sweep(
        self,
        fractions: Iterable[float] = (0.05, 0.10, 0.20),
        *,
        seed: int = 1729,
    ) -> dict[str, torch.Tensor]:
        """Compare top-attribution edge deletion with equal-count random deletion."""

        if self._last_eval_cache is None:
            raise RuntimeError("Run an evaluation forward pass before deletion_sweep")
        fraction_values = tuple(float(value) for value in fractions)
        if not fraction_values or any(
            value <= 0 or value >= 1 for value in fraction_values
        ):
            raise ValueError("deletion fractions must lie strictly between 0 and 1")

        cache = self._last_eval_cache
        score = cache["stage_contribution"].abs().sum(dim=1)
        valid_edges = (
            cache["row_valid"].unsqueeze(2)
            & cache["col_valid"].unsqueeze(1)
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        top_logits = []
        random_logits = []
        top_shift = []
        random_shift = []
        positive_target_logits = []
        negative_target_logits = []
        positive_available = []
        negative_available = []

        for fraction in fraction_values:
            top_masks = torch.zeros_like(valid_edges)
            random_masks = torch.zeros_like(valid_edges)
            for batch_index in range(score.size(0)):
                flat_valid = torch.nonzero(
                    valid_edges[batch_index].flatten(), as_tuple=False
                ).flatten()
                count = max(1, int(round(flat_valid.numel() * fraction)))
                top_order = torch.argsort(
                    score[batch_index].flatten()[flat_valid],
                    descending=True,
                )
                top_masks[batch_index].view(-1)[
                    flat_valid[top_order[:count]]
                ] = True
                random_order = torch.randperm(
                    flat_valid.numel(), generator=generator, device="cpu"
                ).to(flat_valid.device)
                random_masks[batch_index].view(-1)[
                    flat_valid[random_order[:count]]
                ] = True

            current_logits = []
            current_shifts = []
            for deletion_mask in (top_masks, random_masks):
                modified_cost = cache["stable_cost"] + (
                    deletion_mask.to(dtype=score.dtype)
                    * self.ist_deletion_penalty
                )
                plan, _, _ = self._solve(
                    modified_cost,
                    cache["row_valid"],
                    cache["col_valid"],
                )
                deleted_logits, _ = self._logits_from_plan(
                    plan, cache["edge_values"]
                )
                current_logits.append(deleted_logits)
                current_shifts.append(
                    (plan - cache["stable_plan"]).abs().sum(dim=(1, 2))
                )
            top_logits.append(current_logits[0])
            random_logits.append(current_logits[1])
            top_shift.append(current_shifts[0])
            random_shift.append(current_shifts[1])

            positive_masks = torch.zeros(
                score.size(0),
                self.num_classes,
                score.size(1),
                score.size(2),
                device=score.device,
                dtype=torch.bool,
            )
            negative_masks = torch.zeros_like(positive_masks)
            positive_present = torch.zeros(
                score.size(0),
                self.num_classes,
                device=score.device,
                dtype=torch.bool,
            )
            negative_present = torch.zeros_like(positive_present)
            for batch_index in range(score.size(0)):
                flat_valid = torch.nonzero(
                    valid_edges[batch_index].flatten(), as_tuple=False
                ).flatten()
                base_count = max(1, int(round(flat_valid.numel() * fraction)))
                for stage in range(self.num_classes):
                    stage_values = cache["stage_contribution"][
                        batch_index, stage
                    ].flatten()
                    positive = flat_valid[stage_values[flat_valid] > 0]
                    negative = flat_valid[stage_values[flat_valid] < 0]
                    if positive.numel():
                        positive_present[batch_index, stage] = True
                        order = torch.argsort(
                            stage_values[positive], descending=True
                        )
                        positive_masks[batch_index, stage].view(-1)[
                            positive[order[: min(base_count, positive.numel())]]
                        ] = True
                    if negative.numel():
                        negative_present[batch_index, stage] = True
                        order = torch.argsort(
                            stage_values[negative], descending=False
                        )
                        negative_masks[batch_index, stage].view(-1)[
                            negative[order[: min(base_count, negative.numel())]]
                        ] = True

            signed_logits = []
            for deletion_masks in (positive_masks, negative_masks):
                modified = cache["stable_cost"].unsqueeze(1) + (
                    deletion_masks.to(dtype=score.dtype)
                    * self.ist_deletion_penalty
                )
                batch_size, stages, patches, pathways = modified.shape
                signed_plan, _, _ = self._solve(
                    modified.flatten(0, 1),
                    cache["row_valid"]
                    .unsqueeze(1)
                    .expand(-1, stages, -1)
                    .flatten(0, 1),
                    cache["col_valid"]
                    .unsqueeze(1)
                    .expand(-1, stages, -1)
                    .flatten(0, 1),
                )
                signed_plan = signed_plan.view(
                    batch_size, stages, patches, pathways
                )
                target_logits = cache["stage_bias"] + torch.einsum(
                    "bsng,bngs->bs",
                    signed_plan,
                    cache["edge_values"],
                )
                signed_logits.append(target_logits)
            positive_target_logits.append(signed_logits[0])
            negative_target_logits.append(signed_logits[1])
            positive_available.append(positive_present)
            negative_available.append(negative_present)

        top_logits_tensor = torch.stack(top_logits, dim=1)
        random_logits_tensor = torch.stack(random_logits, dim=1)
        positive_tensor = torch.stack(positive_target_logits, dim=1)
        negative_tensor = torch.stack(negative_target_logits, dim=1)
        positive_available_tensor = torch.stack(positive_available, dim=1)
        negative_available_tensor = torch.stack(negative_available, dim=1)
        factual_by_fraction = cache["logits"].unsqueeze(1)
        return {
            "fractions": cache["logits"].new_tensor(fraction_values),
            "factual_logits": cache["logits"],
            "factual_risk": _survival_risk(cache["logits"]),
            "top_deleted_logits": top_logits_tensor,
            "random_deleted_logits": random_logits_tensor,
            "top_deleted_risk": _survival_risk(
                top_logits_tensor.flatten(0, 1)
            ).view(score.size(0), -1),
            "random_deleted_risk": _survival_risk(
                random_logits_tensor.flatten(0, 1)
            ).view(score.size(0), -1),
            "top_plan_shift": torch.stack(top_shift, dim=1),
            "random_plan_shift": torch.stack(random_shift, dim=1),
            "positive_deleted_target_logits": positive_tensor,
            "negative_deleted_target_logits": negative_tensor,
            "positive_direction_available": positive_available_tensor,
            "negative_direction_available": negative_available_tensor,
            "positive_direction_ok": (
                positive_tensor < factual_by_fraction
            )
            & positive_available_tensor,
            "negative_direction_ok": (
                negative_tensor > factual_by_fraction
            )
            & negative_available_tensor,
        }
