"""Cohort-Anchored Adaptive Prognostic Slot Attention.

CA-PSA gives slot index ``k`` a cohort-level identity shared by WSI and omics,
while modality-specific recurrent states retain patient-level variation.  A
hard-concrete gate chooses how many of the shared identities are active for
each patient.  The trainer supplies the survival NLL; this module adds only an
expected-L0 sparsity term and a same-identity cross-modal alignment term.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp


class CohortAnchoredSlotUpdater(nn.Module):
    """Competitive Slot Attention whose initial identities are supplied anchors."""

    def __init__(self, dim: int, heads: int = 4, iters: int = 3, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("capsa_heads must divide wsi_projection_dim")
        if iters < 1:
            raise ValueError("capsa_slot_iters must be at least 1")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.iters = iters
        self.scale = self.head_dim**-0.5

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.combine_heads = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.state_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self, tokens: torch.Tensor, anchors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"CA-PSA tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
        batch = tokens.size(0)
        anchor_batch = anchors.unsqueeze(0).expand(batch, -1, -1)
        state = torch.zeros_like(anchor_batch)

        normalized_tokens = self.norm_inputs(tokens)
        keys = self._split_heads(self.to_k(normalized_tokens))
        values = self._split_heads(self.to_v(normalized_tokens))
        attention = None

        for _ in range(self.iters):
            previous = state
            queries = self._split_heads(
                self.to_q(self.norm_queries(anchor_batch + state))
            )
            scores = torch.einsum("bhkd,bhnd->bhkn", queries, keys) * self.scale
            # Each token first chooses among stable cohort identities.  The
            # second normalization makes every slot update a weighted mean.
            attention = scores.softmax(dim=2)
            attention = attention + 1e-8
            attention = attention / attention.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bhkn,bhnd->bhkd", attention, values)
            updates = updates.permute(0, 2, 1, 3).reshape(batch, anchors.size(0), self.dim)
            updates = self.combine_heads(updates)
            state = self.gru(updates.reshape(-1, self.dim), previous.reshape(-1, self.dim))
            state = state.reshape_as(previous)
            state = state + self.state_mlp(state)

        assert attention is not None
        state = self.output_norm(state)
        slots = anchor_batch + state
        return slots, state, attention.mean(dim=1)


class CohortAnchoredAdaptivePrognosticSlotAttention(nn.Module):
    """Shared slot identity + patient state + dynamic activation for survival."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = int(args.n_classes)
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.wsi_projection_dim = int(args.wsi_projection_dim)
        self.omics_input_dim = omic_input_dim

        self.max_slots = int(getattr(args, "capsa_max_slots", 16))
        self.slot_iters = int(getattr(args, "capsa_slot_iters", getattr(args, "slot_iters", 3)))
        self.heads = int(getattr(args, "capsa_heads", 4))
        self.dropout = float(getattr(args, "capsa_dropout", 0.15))
        self.gate_temperature = float(getattr(args, "capsa_gate_temperature", 2.0 / 3.0))
        self.gate_gamma = float(getattr(args, "capsa_gate_gamma", -0.1))
        self.gate_zeta = float(getattr(args, "capsa_gate_zeta", 1.1))
        self.gate_threshold = float(getattr(args, "capsa_gate_threshold", 0.5))
        self.gate_prior_start = float(getattr(args, "capsa_gate_prior_start", -1.0))
        self.gate_prior_end = float(getattr(args, "capsa_gate_prior_end", -2.2))
        self.lambda_sparse = float(getattr(args, "capsa_lambda_sparse", 0.01))
        self.lambda_align = float(getattr(args, "capsa_lambda_align", 0.02))
        self._validate_hyperparameters()

        dim = self.wsi_projection_dim
        self.wsi_mlp = WSI_Mlp(dim_in=self.wsi_embedding_dim, feat_dim=dim)
        self._init_omics_encoder(args.rna_format)

        self.cohort_anchors = nn.Parameter(torch.empty(self.max_slots, dim))
        nn.init.normal_(self.cohort_anchors, std=0.02)
        self.wsi_slot_updater = CohortAnchoredSlotUpdater(
            dim, heads=self.heads, iters=self.slot_iters, dropout=self.dropout
        )
        self.omic_slot_updater = CohortAnchoredSlotUpdater(
            dim, heads=self.heads, iters=self.slot_iters, dropout=self.dropout
        )

        self.same_identity_fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.gate_head[-1].bias)
        # Ordered capacity priors keep the first validation pass away from the
        # degenerate all-on/all-off regimes. They are fully learnable and the
        # patient feature still determines the posterior activation.
        self.gate_prior_logits = nn.Parameter(
            torch.linspace(self.gate_prior_start, self.gate_prior_end, self.max_slots)
        )
        self.temporal_attention = nn.Linear(dim, self.num_classes)
        self.slot_hazard = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim, self.num_classes),
        )
        align_dim = max(16, dim // 2)
        self.wsi_align_projection = nn.Linear(dim, align_dim, bias=False)
        self.omic_align_projection = nn.Linear(dim, align_dim, bias=False)

        self.last_training_losses: dict[str, torch.Tensor] = {}
        self._last_explanation: dict[str, torch.Tensor] = {}
        if omic_names:
            try:
                self.all_gene_names = list(np.unique(np.concatenate(omic_names)))
            except Exception:
                pass

    def _validate_hyperparameters(self) -> None:
        if self.max_slots < 1:
            raise ValueError("capsa_max_slots must be at least 1")
        if self.slot_iters < 1:
            raise ValueError("capsa_slot_iters must be at least 1")
        if self.heads < 1 or self.wsi_projection_dim % self.heads != 0:
            raise ValueError("capsa_heads must be positive and divide wsi_projection_dim")
        if self.gate_temperature <= 0:
            raise ValueError("capsa_gate_temperature must be positive")
        if not self.gate_gamma < 0 < 1 < self.gate_zeta:
            raise ValueError("hard-concrete bounds require capsa_gate_gamma < 0 < 1 < capsa_gate_zeta")
        if not 0 < self.gate_threshold < 1:
            raise ValueError("capsa_gate_threshold must be strictly between 0 and 1")
        if self.lambda_sparse < 0 or self.lambda_align < 0:
            raise ValueError("CA-PSA loss weights must be non-negative")

    def _init_omics_encoder(self, rna_format: str) -> None:
        dim = self.wsi_projection_dim
        if rna_format == "Pathways":
            if not self.omic_sizes:
                raise ValueError("omic_sizes is required when rna_format='Pathways'")
            self.num_pathways = len(self.omic_sizes)
            self.sig_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        SNN_Block(dim1=input_dim, dim2=dim),
                        SNN_Block(dim1=dim, dim2=dim, dropout=0.25),
                    )
                    for input_dim in self.omic_sizes
                ]
            )
        elif rna_format == "GeneEmbedding":
            self.sig_networks = SNN_Block(dim1=768, dim2=dim)
        elif rna_format == "RNASeq":
            if self.omics_input_dim is None:
                raise ValueError("omic_input_dim is required when rna_format='RNASeq'")
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=dim)
        else:
            raise ValueError(f"Invalid omics_format: {rna_format}")

    def _encode_omics(self, kwargs: dict[str, Any]) -> torch.Tensor:
        if self.args.rna_format == "Pathways":
            features = [kwargs[f"x_omic{i}"] for i in range(1, self.num_pathways + 1)]
            encoded = [self.sig_networks[i](feature) for i, feature in enumerate(features)]
            return torch.stack(encoded, dim=1)
        encoded = self.sig_networks(kwargs["x_omics"])
        return encoded.unsqueeze(1) if encoded.ndim == 2 else encoded

    @staticmethod
    def _availability(kwargs: dict[str, Any], name: str, batch: int, ref: torch.Tensor) -> torch.Tensor:
        value = kwargs.get(name)
        if value is None:
            return ref.new_ones(batch)
        return value.to(device=ref.device, dtype=ref.dtype).reshape(batch)

    def _hard_concrete(
        self, log_alpha: torch.Tensor, sample: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample is None:
            sample = self.training
        expected_open = torch.sigmoid(
            log_alpha
            - self.gate_temperature
            * math.log(-self.gate_gamma / self.gate_zeta)
        )
        if sample:
            uniform = torch.rand_like(log_alpha).clamp_(1e-6, 1.0 - 1e-6)
            logistic = uniform.log() - (-uniform).log1p()
            concrete = torch.sigmoid((logistic + log_alpha) / self.gate_temperature)
            soft = (
                concrete * (self.gate_zeta - self.gate_gamma) + self.gate_gamma
            ).clamp(0, 1)
            # z > 0 is the Bernoulli event whose probability is `expected_open`.
            hard = (soft > 0).to(soft.dtype)
        else:
            soft = expected_open
            hard = (expected_open >= self.gate_threshold).to(soft.dtype)
        # A patient must retain at least one prognostic route.  This affects only
        # the hard forward pass; straight-through gradients still follow `soft`.
        empty = hard.sum(dim=1, keepdim=True) == 0
        top_one = F.one_hot(log_alpha.argmax(dim=1), num_classes=self.max_slots).to(soft.dtype)
        safe_hard = torch.where(empty, top_one, hard)
        gates = safe_hard + soft - soft.detach()

        return gates, expected_open

    def _alignment_loss(
        self,
        wsi_state: torch.Tensor,
        omic_state: torch.Tensor,
        expected_open: torch.Tensor,
        jointly_available: torch.Tensor,
    ) -> torch.Tensor:
        wsi_common = F.normalize(self.wsi_align_projection(wsi_state), dim=-1)
        omic_common = F.normalize(self.omic_align_projection(omic_state), dim=-1)
        distance = 1.0 - (wsi_common * omic_common).sum(dim=-1)
        weights = expected_open * jointly_available.unsqueeze(1)
        return (distance * weights).sum() / weights.sum().clamp_min(1e-6)

    def forward(self, **kwargs):
        wsi_tokens = self.wsi_mlp(kwargs["x_wsi"])
        omic_tokens = self._encode_omics(kwargs)
        batch = wsi_tokens.size(0)

        wsi_slots, wsi_state, wsi_attention = self.wsi_slot_updater(
            wsi_tokens, self.cohort_anchors
        )
        omic_slots, omic_state, omic_attention = self.omic_slot_updater(
            omic_tokens, self.cohort_anchors
        )

        wsi_available = self._availability(kwargs, "wsi_available", batch, wsi_tokens)
        omics_available = self._availability(kwargs, "omics_available", batch, wsi_tokens)
        anchors = self.cohort_anchors.unsqueeze(0).expand(batch, -1, -1)
        wsi_mask = wsi_available[:, None, None] > 0
        omic_mask = omics_available[:, None, None] > 0
        wsi_slots = torch.where(wsi_mask, wsi_slots, anchors)
        omic_slots = torch.where(omic_mask, omic_slots, anchors)
        wsi_state = torch.where(wsi_mask, wsi_state, torch.zeros_like(wsi_state))
        omic_state = torch.where(omic_mask, omic_state, torch.zeros_like(omic_state))

        fused = self.same_identity_fusion(
            torch.cat(
                [wsi_slots, omic_slots, wsi_slots * omic_slots, (wsi_slots - omic_slots).abs()],
                dim=-1,
            )
        )
        log_alpha = self.gate_head(fused).squeeze(-1) + self.gate_prior_logits.unsqueeze(0)
        gates, expected_open = self._hard_concrete(log_alpha)
        slot_logits = self.slot_hazard(fused)
        temporal_alpha = torch.softmax(self.temporal_attention(fused), dim=1)
        gated_alpha = temporal_alpha * gates.unsqueeze(-1)
        gated_alpha = gated_alpha / gated_alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logits = (gated_alpha * slot_logits).sum(dim=1)

        sparse_loss = expected_open.mean()
        align_loss = self._alignment_loss(
            wsi_state,
            omic_state,
            expected_open,
            (wsi_available > 0).to(wsi_tokens.dtype) * (omics_available > 0).to(wsi_tokens.dtype),
        )
        aux_loss = self.lambda_sparse * sparse_loss + self.lambda_align * align_loss
        if not self.training:
            aux_loss = logits.new_zeros(())

        self.last_training_losses = {
            "sparse": sparse_loss.detach(),
            "align": align_loss.detach(),
            "auxiliary": aux_loss.detach(),
        }
        self._last_explanation = {
            "gates": gates.detach(),
            "expected_open_probability": expected_open.detach(),
            "active_slot_count": (gates.detach() > 0.5).sum(dim=1),
            "slot_logits": slot_logits.detach(),
            "temporal_slot_weights": gated_alpha.detach(),
            "wsi_slots": wsi_slots.detach(),
            "omic_slots": omic_slots.detach(),
            "wsi_attention": wsi_attention.detach(),
            "omic_attention": omic_attention.detach(),
        }
        return logits, aux_loss

    def explain_last_batch(self) -> dict[str, torch.Tensor]:
        """Return detached slot activation and contribution diagnostics."""
        if not self._last_explanation:
            raise RuntimeError("Run a forward pass before requesting CA-PSA diagnostics")
        return dict(self._last_explanation)
