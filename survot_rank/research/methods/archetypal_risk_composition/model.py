"""ArcSurv: cross-modal archetypal risk composition for survival prediction.

The method keeps the established WSI and omics encoders but replaces
patient-specific event gating with a cohort-level prognostic simplex. Each
archetype is constrained to be a convex combination of a fold-local memory
bank, and every patient's hazard logits are a convex combination of
archetype-specific hazard curves.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp
from survot_rank.research.components.slot_attention import MultiHeadSlotAttention


class CohortArchetypeBank(nn.Module):
    """Learn archetypes as convex combinations of a fixed fold-local memory.

    The memory is filled once from training patients and then kept fixed.
    Learnable row-stochastic weights select boundary representatives from that
    memory. This preserves the defining archetypal-analysis constraint
    ``A = Beta @ Z`` instead of treating archetypes as unconstrained prototypes.
    """

    def __init__(self, dim: int, num_archetypes: int, bank_size: int, temperature: float):
        super().__init__()
        if num_archetypes < 2:
            raise ValueError("ArcSurv requires at least two archetypes")
        if bank_size < num_archetypes:
            raise ValueError("arc_bank_size must be at least arc_num_archetypes")
        if temperature <= 0:
            raise ValueError("arc_temperature must be positive")

        self.dim = int(dim)
        self.num_archetypes = int(num_archetypes)
        self.bank_size = int(bank_size)
        self.temperature = float(temperature)

        self.beta_logits = nn.Parameter(torch.zeros(num_archetypes, bank_size))
        self.empty_bank_seed = nn.Parameter(torch.randn(num_archetypes, dim) * 0.02)
        self.register_buffer("memory", torch.zeros(bank_size, dim))
        self.register_buffer("memory_count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, states: torch.Tensor) -> None:
        """Fill previously unused memory rows; never update during evaluation."""
        if not self.training or states.numel() == 0:
            return
        states = states.detach()
        states = states[torch.isfinite(states).all(dim=1)]
        if states.numel() == 0:
            return

        start = int(self.memory_count.item())
        if start >= self.bank_size:
            return
        take = min(states.size(0), self.bank_size - start)
        self.memory[start : start + take].copy_(states[:take])
        self.memory_count.fill_(start + take)

    def archetypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(self.memory_count.item())
        if count == 0:
            weights = torch.full(
                (self.num_archetypes, 1),
                1.0,
                device=self.empty_bank_seed.device,
                dtype=self.empty_bank_seed.dtype,
            )
            return self.empty_bank_seed, weights

        weights = torch.softmax(self.beta_logits[:, :count], dim=1)
        archetypes = weights @ self.memory[:count]
        return archetypes, weights

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        archetypes, _ = self.archetypes()
        squared_distance = (states[:, None, :] - archetypes[None, :, :]).pow(2).mean(dim=-1)
        composition = torch.softmax(-squared_distance / self.temperature, dim=1)
        reconstruction = composition @ archetypes
        return composition, reconstruction, archetypes


class ArchetypalRiskComposition(nn.Module):
    """Shared histology-omics prognostic simplex with archetype hazard curves."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = int(args.n_classes)
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.wsi_projection_dim = int(args.wsi_projection_dim)
        self.omics_input_dim = omic_input_dim

        dim = self.wsi_projection_dim
        self.num_archetypes = int(getattr(args, "arc_num_archetypes", 6))
        self.arc_lambda_recon = float(getattr(args, "arc_lambda_recon", 0.05))
        self.arc_lambda_align = float(getattr(args, "arc_lambda_align", 0.05))
        self.arc_lambda_balance = float(getattr(args, "arc_lambda_balance", 0.01))
        self.arc_lambda_rank = float(getattr(args, "arc_lambda_rank", 0.10))
        self.arc_rank_margin = float(getattr(args, "arc_rank_margin", 0.0))
        self.arc_rank_max_pairs = int(getattr(args, "arc_rank_max_pairs", 4096))

        self._init_omics_encoder(self.omic_sizes, args.rna_format)
        self.wsi_mlp = WSI_Mlp(dim_in=self.wsi_embedding_dim, feat_dim=dim)
        slot_init_mode = getattr(args, "dct_slot_init_mode", "gaussian")
        slot_eval_seed = int(getattr(args, "dct_slot_eval_seed", 1729))
        self.slot_attention_wsi = MultiHeadSlotAttention(
            dim=dim,
            num_slots=args.slot_num_wsi,
            iters=args.slot_iters,
            heads=8,
            init_mode=slot_init_mode,
            eval_seed=slot_eval_seed,
        )
        self.slot_attention_omic = MultiHeadSlotAttention(
            dim=dim,
            num_slots=args.slot_num_omics,
            iters=args.slot_iters,
            heads=8,
            init_mode=slot_init_mode,
            eval_seed=slot_eval_seed + 1,
        )

        bank_size = int(getattr(args, "arc_bank_size", 256))
        temperature = float(getattr(args, "arc_temperature", 0.25))
        self.wsi_state_norm = nn.LayerNorm(dim)
        self.omic_state_norm = nn.LayerNorm(dim)
        self.wsi_archetypes = CohortArchetypeBank(
            dim, self.num_archetypes, bank_size, temperature
        )
        self.omic_archetypes = CohortArchetypeBank(
            dim, self.num_archetypes, bank_size, temperature
        )

        self.archetype_hazard_logits = nn.Parameter(
            torch.randn(self.num_archetypes, self.num_classes) * 0.02
        )
        self.hazard_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.missing_logits = nn.Parameter(torch.zeros(self.num_classes))
        self.last_training_losses: dict[str, torch.Tensor] = {}
        self.last_composition: torch.Tensor | None = None

        if omic_names:
            try:
                self.all_gene_names = list(np.unique(np.concatenate(omic_names)))
            except Exception:
                pass

    def _init_omics_encoder(self, omic_sizes, omics_format):
        dim = self.wsi_projection_dim
        if omics_format == "Pathways":
            self.num_pathways = len(omic_sizes)
            self.sig_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        SNN_Block(dim1=input_dim, dim2=dim),
                        SNN_Block(dim1=dim, dim2=dim, dropout=0.25),
                    )
                    for input_dim in omic_sizes
                ]
            )
        elif omics_format == "GeneEmbedding":
            self.sig_networks = SNN_Block(dim1=768, dim2=dim)
        elif omics_format == "RNASeq":
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=dim)
        else:
            raise ValueError(f"Invalid omics_format: {omics_format}")

    def _encode_omics(self, kwargs):
        if self.args.rna_format == "Pathways":
            pathway_inputs = [
                kwargs[f"x_omic{index}"] for index in range(1, self.num_pathways + 1)
            ]
            pathway_states = [
                self.sig_networks[index](features)
                for index, features in enumerate(pathway_inputs)
            ]
            return torch.stack(pathway_states).permute(1, 0, 2)
        return self.sig_networks(kwargs["x_omics"])

    @staticmethod
    def _availability(kwargs, name, batch_size, device):
        value = kwargs.get(name)
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        value = torch.as_tensor(value, device=device).bool().view(-1)
        if value.numel() != batch_size:
            raise ValueError(f"{name} must contain {batch_size} values")
        return value

    @staticmethod
    def _slot_mask(kwargs, name, batch_size, slots, device):
        value = kwargs.get(name)
        if value is None:
            return torch.ones(batch_size, slots, dtype=torch.bool, device=device)
        mask = torch.as_tensor(value, device=device).bool()
        if mask.shape != (batch_size, slots):
            raise ValueError(f"{name} must have shape {(batch_size, slots)}")
        return mask

    @staticmethod
    def _masked_mean(states, mask):
        weights = mask.to(states.dtype).unsqueeze(-1)
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _js_divergence(left, right):
        left = left.clamp_min(1e-8)
        right = right.clamp_min(1e-8)
        middle = 0.5 * (left + right)
        return 0.5 * (
            (left * (left.log() - middle.log())).sum(dim=1)
            + (right * (right.log() - middle.log())).sum(dim=1)
        ).mean()

    def _ranking_loss(self, logits, y, c):
        hazards = torch.sigmoid(logits)
        risk = -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)
        times = y.float().view(-1)
        observed = (1.0 - c.float()).view(-1) > 0.5
        comparable = observed[:, None] & (times[:, None] < times[None, :])
        if risk.numel() < 2 or not comparable.any():
            return risk.sum() * 0.0
        differences = risk[:, None] - risk[None, :]
        values = F.softplus(-(differences - self.arc_rank_margin))[comparable]
        if values.numel() > self.arc_rank_max_pairs:
            keep = torch.randperm(values.numel(), device=values.device)[
                : self.arc_rank_max_pairs
            ]
            values = values[keep]
        return values.mean()

    def archetype_parameters(self):
        """Return current modality archetypes, hull weights, and hazard curves."""
        wsi_archetypes, wsi_beta = self.wsi_archetypes.archetypes()
        omic_archetypes, omic_beta = self.omic_archetypes.archetypes()
        return {
            "wsi_archetypes": wsi_archetypes,
            "wsi_beta": wsi_beta,
            "omic_archetypes": omic_archetypes,
            "omic_beta": omic_beta,
            "hazard_logits": self.archetype_hazard_logits,
        }

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"]
        batch_size = x_wsi.size(0)
        device = x_wsi.device

        wsi_tokens = self.wsi_mlp(x_wsi)
        omic_tokens = self._encode_omics(kwargs)
        wsi_slots = self.slot_attention_wsi(wsi_tokens)
        omic_slots = self.slot_attention_omic(omic_tokens)

        has_wsi = self._availability(kwargs, "wsi_available", batch_size, device)
        has_omic = self._availability(kwargs, "omics_available", batch_size, device)
        wsi_mask = self._slot_mask(
            kwargs, "wsi_slot_mask", batch_size, wsi_slots.size(1), device
        )
        omic_mask = self._slot_mask(
            kwargs, "omics_slot_mask", batch_size, omic_slots.size(1), device
        )
        if (has_wsi & ~wsi_mask.any(dim=1)).any():
            raise ValueError("each available WSI sample needs at least one valid slot")
        if (has_omic & ~omic_mask.any(dim=1)).any():
            raise ValueError("each available omics sample needs at least one valid slot")

        wsi_state = self.wsi_state_norm(self._masked_mean(wsi_slots, wsi_mask))
        omic_state = self.omic_state_norm(self._masked_mean(omic_slots, omic_mask))

        self.wsi_archetypes.update(wsi_state[has_wsi])
        self.omic_archetypes.update(omic_state[has_omic])
        wsi_composition, wsi_reconstruction, _ = self.wsi_archetypes(wsi_state)
        omic_composition, omic_reconstruction, _ = self.omic_archetypes(omic_state)

        wsi_weight = has_wsi.to(wsi_state.dtype).unsqueeze(1)
        omic_weight = has_omic.to(wsi_state.dtype).unsqueeze(1)
        denominator = (wsi_weight + omic_weight).clamp_min(1.0)
        composition = (
            wsi_weight * wsi_composition + omic_weight * omic_composition
        ) / denominator
        neither = ~(has_wsi | has_omic)
        if neither.any():
            composition = composition + neither.to(composition.dtype).unsqueeze(1) / self.num_archetypes

        logits = composition @ self.archetype_hazard_logits + self.hazard_bias
        logits = torch.where(neither.unsqueeze(1), self.missing_logits.unsqueeze(0), logits)
        self.last_composition = composition.detach()

        if not self.training:
            self.last_training_losses = {}
            return logits, logits.sum() * 0.0

        zero = logits.sum() * 0.0
        reconstruction_loss = zero
        if has_wsi.any():
            reconstruction_loss = reconstruction_loss + F.mse_loss(
                wsi_reconstruction[has_wsi], wsi_state[has_wsi]
            )
        if has_omic.any():
            reconstruction_loss = reconstruction_loss + F.mse_loss(
                omic_reconstruction[has_omic], omic_state[has_omic]
            )

        both = has_wsi & has_omic
        alignment_loss = (
            self._js_divergence(wsi_composition[both], omic_composition[both])
            if both.any()
            else zero
        )

        available_compositions = []
        if has_wsi.any():
            available_compositions.append(wsi_composition[has_wsi])
        if has_omic.any():
            available_compositions.append(omic_composition[has_omic])
        if available_compositions:
            mean_composition = torch.cat(available_compositions, dim=0).mean(dim=0)
            target = torch.full_like(mean_composition, 1.0 / self.num_archetypes)
            balance_loss = F.mse_loss(mean_composition, target)
        else:
            balance_loss = zero

        rank_loss = zero
        if kwargs.get("y") is not None and kwargs.get("c") is not None:
            rank_loss = self._ranking_loss(logits, kwargs["y"], kwargs["c"])

        aux_loss = (
            self.arc_lambda_recon * reconstruction_loss
            + self.arc_lambda_align * alignment_loss
            + self.arc_lambda_balance * balance_loss
            + self.arc_lambda_rank * rank_loss
        )
        self.last_training_losses = {
            "arc_reconstruction": reconstruction_loss.detach(),
            "arc_alignment": alignment_loss.detach(),
            "arc_balance": balance_loss.detach(),
            "arc_rank": rank_loss.detach(),
            "arc_wsi_bank_count": self.wsi_archetypes.memory_count.detach().float(),
            "arc_omic_bank_count": self.omic_archetypes.memory_count.detach().float(),
        }
        return logits, aux_loss


__all__ = ["ArchetypalRiskComposition", "CohortArchetypeBank"]
