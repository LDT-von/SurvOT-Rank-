"""V70: patient-specific sparse prognostic circuits for WSI--omics survival.

The method deliberately does not use Slot Attention or fixed event queries.
WSI patches and omics/pathway tokens are read into a reusable cohort-level
module bank.  A patient-conditioned generator selects active modules and
constructs a sparse directed circuit, which is then executed before a
time-local hazard readout.

The common trainer supplies the survival NLL.  This module returns only two
structural regularizers: expected active-module mass and expected off-diagonal
edge density.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp


class CircuitPropagationLayer(nn.Module):
    """One code-conditioned message-passing step over a generated circuit."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(dim, dim, bias=False)
        self.pre_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.code_to_film = nn.Linear(dim, dim * 2)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        states: torch.Tensor,
        adjacency: torch.Tensor,
        module_codes: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        values = self.message(states)
        messages = torch.einsum("bij,bjd->bid", adjacency, values)
        mixed = self.pre_norm(states + messages)
        transformed = self.feed_forward(mixed)
        scale, shift = self.code_to_film(module_codes).chunk(2, dim=-1)
        transformed = transformed * (1.0 + torch.tanh(scale).unsqueeze(0))
        transformed = transformed + shift.unsqueeze(0)
        return self.output_norm(states + gates.unsqueeze(-1) * transformed)


class V70PatientSpecificPrognosticCircuits(nn.Module):
    """Generate and execute a sparse prognostic circuit for each patient."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = int(args.n_classes)
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.wsi_projection_dim = int(args.wsi_projection_dim)
        self.omics_input_dim = omic_input_dim

        self.max_modules = int(getattr(args, "pspc_max_modules", 16))
        self.heads = int(getattr(args, "pspc_heads", 4))
        self.layers = int(getattr(args, "pspc_layers", 3))
        self.dropout = float(getattr(args, "pspc_dropout", 0.15))
        self.gate_temperature = float(getattr(args, "pspc_gate_temperature", 2.0 / 3.0))
        self.gate_gamma = float(getattr(args, "pspc_gate_gamma", -0.1))
        self.gate_zeta = float(getattr(args, "pspc_gate_zeta", 1.1))
        self.gate_threshold = float(getattr(args, "pspc_gate_threshold", 0.5))
        self.edge_temperature = float(getattr(args, "pspc_edge_temperature", 0.75))
        self.edge_threshold = float(getattr(args, "pspc_edge_threshold", 0.5))
        self.edge_rank = int(getattr(args, "pspc_edge_rank", 4))
        self.lambda_node_sparse = float(getattr(args, "pspc_lambda_node_sparse", 0.01))
        self.lambda_edge_sparse = float(getattr(args, "pspc_lambda_edge_sparse", 0.005))
        self._validate_hyperparameters()

        dim = self.wsi_projection_dim
        self.wsi_mlp = WSI_Mlp(dim_in=self.wsi_embedding_dim, feat_dim=dim)
        self._init_omics_encoder(args.rna_format)

        self.modality_embeddings = nn.Parameter(torch.empty(2, dim))
        self.null_token = nn.Parameter(torch.empty(1, 1, dim))
        self.module_codes = nn.Parameter(torch.empty(self.max_modules, dim))
        nn.init.normal_(self.modality_embeddings, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)
        nn.init.normal_(self.module_codes, std=0.02)

        self.context_fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.availability_encoder = nn.Linear(2, dim, bias=False)
        self.query_context = nn.Linear(dim, dim)
        self.read_in = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.read_in_norm = nn.LayerNorm(dim)

        self.base_gate_logits = nn.Parameter(torch.zeros(self.max_modules))
        self.context_to_gates = nn.Linear(dim, self.max_modules)
        self.base_edge_logits = nn.Parameter(
            torch.empty(self.max_modules, self.max_modules)
        )
        self.context_to_edge_factors = nn.Linear(dim, self.edge_rank)
        self.edge_basis = nn.Parameter(
            torch.empty(self.edge_rank, self.max_modules, self.max_modules)
        )
        nn.init.normal_(self.base_edge_logits, std=0.1)
        nn.init.normal_(self.edge_basis, std=0.02)

        self.propagation_layers = nn.ModuleList(
            [CircuitPropagationLayer(dim, self.dropout) for _ in range(self.layers)]
        )
        self.module_hazard = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim, self.num_classes),
        )
        self.temporal_readout = nn.Linear(dim, self.num_classes)

        self.last_training_losses: dict[str, torch.Tensor] = {}
        self._last_explanation: dict[str, torch.Tensor] = {}
        if omic_names:
            try:
                self.all_gene_names = list(np.unique(np.concatenate(omic_names)))
            except Exception:
                pass

    def _validate_hyperparameters(self) -> None:
        if self.max_modules < 1:
            raise ValueError("pspc_max_modules must be at least 1")
        if self.heads < 1 or self.wsi_projection_dim % self.heads != 0:
            raise ValueError("pspc_heads must be positive and divide wsi_projection_dim")
        if self.layers < 1:
            raise ValueError("pspc_layers must be at least 1")
        if self.edge_rank < 1:
            raise ValueError("pspc_edge_rank must be at least 1")
        if self.gate_temperature <= 0 or self.edge_temperature <= 0:
            raise ValueError("PSPC temperatures must be positive")
        if not self.gate_gamma < 0 < 1 < self.gate_zeta:
            raise ValueError("hard-concrete bounds require pspc_gate_gamma < 0 < 1 < pspc_gate_zeta")
        if not 0 <= self.gate_threshold <= 1 or not 0 <= self.edge_threshold <= 1:
            raise ValueError("PSPC gate and edge thresholds must be in [0, 1]")
        if self.lambda_node_sparse < 0 or self.lambda_edge_sparse < 0:
            raise ValueError("PSPC structural loss weights must be non-negative")

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
    def _availability(
        kwargs: dict[str, Any], name: str, batch: int, reference: torch.Tensor
    ) -> torch.Tensor:
        value = kwargs.get(name)
        if value is None:
            return torch.ones(batch, dtype=torch.bool, device=reference.device)
        available = torch.as_tensor(value, device=reference.device).bool().reshape(-1)
        if available.numel() != batch:
            raise ValueError(f"{name} must contain {batch} values, got {available.numel()}")
        return available

    @staticmethod
    def _token_mask(
        kwargs: dict[str, Any], name: str, tokens: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        batch, count = tokens.shape[:2]
        value = kwargs.get(name)
        if value is None:
            mask = torch.ones(batch, count, dtype=torch.bool, device=tokens.device)
        else:
            mask = torch.as_tensor(value, device=tokens.device).bool()
            if mask.shape != (batch, count):
                raise ValueError(f"{name} must have shape {(batch, count)}, got {tuple(mask.shape)}")
        return mask & available.unsqueeze(1)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _hard_concrete(
        self, log_alpha: torch.Tensor, sample: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample is None:
            sample = self.training
        if sample:
            uniform = torch.rand_like(log_alpha).clamp_(1e-6, 1.0 - 1e-6)
            logistic = uniform.log() - (-uniform).log1p()
            concrete = torch.sigmoid((logistic + log_alpha) / self.gate_temperature)
        else:
            concrete = torch.sigmoid(log_alpha / self.gate_temperature)

        soft = (concrete * (self.gate_zeta - self.gate_gamma) + self.gate_gamma).clamp(0, 1)
        hard = (soft >= self.gate_threshold).to(soft.dtype)
        empty = hard.sum(dim=1, keepdim=True) == 0
        top_one = F.one_hot(log_alpha.argmax(dim=1), num_classes=self.max_modules).to(soft.dtype)
        safe_hard = torch.where(empty, top_one, hard)
        gates = safe_hard + soft - soft.detach()
        expected_open = torch.sigmoid(
            log_alpha
            - self.gate_temperature * math.log(-self.gate_gamma / self.gate_zeta)
        )
        return gates, expected_open

    def _generate_circuit(
        self, context: torch.Tensor, gates: torch.Tensor, expected_open: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        factors = torch.tanh(self.context_to_edge_factors(context))
        edge_logits = self.base_edge_logits.unsqueeze(0) + torch.einsum(
            "br,rkl->bkl", factors, self.edge_basis
        )
        edge_prob = torch.sigmoid(edge_logits / self.edge_temperature)
        hard_edges = (edge_prob >= self.edge_threshold).to(edge_prob.dtype)
        straight_through_edges = hard_edges + edge_prob - edge_prob.detach()

        eye = torch.eye(self.max_modules, device=context.device, dtype=context.dtype)
        off_diagonal = 1.0 - eye
        node_pairs = gates.unsqueeze(2) * gates.unsqueeze(1)
        adjacency = straight_through_edges * off_diagonal.unsqueeze(0) * node_pairs
        adjacency = adjacency + torch.diag_embed(gates)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expected_pairs = expected_open.unsqueeze(2) * expected_open.unsqueeze(1)
        edge_sparse_loss = (
            edge_prob * expected_pairs * off_diagonal.unsqueeze(0)
        ).sum() / max(1, context.size(0) * self.max_modules * (self.max_modules - 1))
        return adjacency, edge_prob, edge_sparse_loss

    def forward(self, **kwargs):
        wsi_tokens = self.wsi_mlp(kwargs["x_wsi"])
        omic_tokens = self._encode_omics(kwargs)
        batch = wsi_tokens.size(0)
        if omic_tokens.size(0) != batch:
            raise ValueError("WSI and omics batch sizes must match")

        has_wsi = self._availability(kwargs, "wsi_available", batch, wsi_tokens)
        has_omic = self._availability(kwargs, "omics_available", batch, wsi_tokens)
        wsi_mask = self._token_mask(kwargs, "wsi_token_mask", wsi_tokens, has_wsi)
        omic_mask = self._token_mask(kwargs, "omics_token_mask", omic_tokens, has_omic)

        wsi_tokens = wsi_tokens + self.modality_embeddings[0]
        omic_tokens = omic_tokens + self.modality_embeddings[1]
        wsi_summary = self._masked_mean(wsi_tokens, wsi_mask)
        omic_summary = self._masked_mean(omic_tokens, omic_mask)
        context = self.context_fusion(
            torch.cat(
                [
                    wsi_summary,
                    omic_summary,
                    wsi_summary * omic_summary,
                    (wsi_summary - omic_summary).abs(),
                ],
                dim=-1,
            )
        )
        availability = torch.stack([has_wsi, has_omic], dim=1).to(wsi_tokens.dtype)
        context = context + self.availability_encoder(availability)

        null_token = self.null_token.expand(batch, -1, -1)
        all_tokens = torch.cat([wsi_tokens, omic_tokens, null_token], dim=1)
        null_mask = torch.ones(batch, 1, dtype=torch.bool, device=wsi_tokens.device)
        all_mask = torch.cat([wsi_mask, omic_mask, null_mask], dim=1)
        queries = self.module_codes.unsqueeze(0) + self.query_context(context).unsqueeze(1)
        read_states, read_attention = self.read_in(
            queries,
            all_tokens,
            all_tokens,
            key_padding_mask=~all_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        states = self.read_in_norm(queries + read_states)

        gate_logits = self.base_gate_logits.unsqueeze(0) + self.context_to_gates(context)
        gates, expected_open = self._hard_concrete(gate_logits)
        adjacency, edge_prob, edge_sparse_loss = self._generate_circuit(
            context, gates, expected_open
        )
        for layer in self.propagation_layers:
            states = layer(states, adjacency, self.module_codes, gates)

        module_logits = self.module_hazard(states)
        temporal_scores = self.temporal_readout(states)
        temporal_weights = torch.softmax(temporal_scores, dim=1) * gates.unsqueeze(-1)
        temporal_weights = temporal_weights / temporal_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logits = (temporal_weights * module_logits).sum(dim=1)

        node_sparse_loss = expected_open.mean()
        aux_loss = (
            self.lambda_node_sparse * node_sparse_loss
            + self.lambda_edge_sparse * edge_sparse_loss
        )
        if not self.training:
            aux_loss = logits.new_zeros(())

        self.last_training_losses = {
            "node_sparse": node_sparse_loss.detach(),
            "edge_sparse": edge_sparse_loss.detach(),
            "auxiliary": aux_loss.detach(),
        }
        off_diagonal = 1.0 - torch.eye(
            self.max_modules, device=logits.device, dtype=logits.dtype
        )
        self._last_explanation = {
            "module_gates": gates.detach(),
            "expected_open_probability": expected_open.detach(),
            "active_module_count": (gates.detach() > 0.5).sum(dim=1),
            "adjacency": adjacency.detach(),
            "edge_probability": edge_prob.detach(),
            "active_edge_count": (
                (adjacency.detach() > 0) * off_diagonal.unsqueeze(0).bool()
            ).sum(dim=(1, 2)),
            "module_logits": module_logits.detach(),
            "temporal_module_weights": temporal_weights.detach(),
            "read_in_attention": read_attention.detach(),
            "patient_context": context.detach(),
        }
        return logits, aux_loss

    def explain_last_batch(self) -> dict[str, torch.Tensor]:
        """Return detached patient circuit and time-local contribution diagnostics."""
        if not self._last_explanation:
            raise RuntimeError("Run a forward pass before requesting PSPC diagnostics")
        return dict(self._last_explanation)


__all__ = ["CircuitPropagationLayer", "V70PatientSpecificPrognosticCircuits"]
