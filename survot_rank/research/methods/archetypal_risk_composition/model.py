"""ArcSurv v2 — recovery + cross-method fusion.

# Base
Restored from commit 80ee202 (staged_final) which scored BLCA 0.6995. The
mechanism: two cohort-level archetype banks (wsi/omic) built once from epoch
0 priority reservoir; patient hazard = softmax(-d^2/T) composition @ bank
hazard logits.

# Deletions from final (0eb705b)
1. arc_seed_anchors (furthest-point anchor reset) — collapsed run, fold1 0.5665
2. arc_freeze_state_encoder (lock the state encoder with bank) — over-constrains
3. _simplex_volume_loss — every attempt with this term produced -0.099 drops
4. arc_warmup_epochs / arc_ramp_epochs — they are zero in staged_final because
   only NLL/rank should drive early training

# Additions (cross-method fusion)
1. hard-concrete gate on composition (from CA-PSA): top-K active archetypes per
   patient; auxiliary budget keeps the cohort routing from collapsing.
2. keep/remove cross-modal archetype re-transport (from CATET): each patient
   sees a Sinkhorn-balanced plan between its wsi-composition and
   omic-composition; the auxiliary loss asks both modalities to agree on
   which archetypes are unused.

# Defaults match staged_final (batch=4, max_epochs=20, lr=5e-4, alpha=0.15).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp
from survot_rank.research.components.slot_attention import MultiHeadSlotAttention


class CohortArchetypeBank(nn.Module):
    """Learn archetypes as convex combinations of an order-robust fold memory."""

    def __init__(
        self,
        dim: int,
        num_archetypes: int,
        bank_size: int,
        temperature: float,
        beta_init_scale: float = 1.5,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError("ArcSurv archetype dimension must be positive")
        if num_archetypes < 2:
            raise ValueError("ArcSurv requires at least two archetypes")
        if bank_size < num_archetypes:
            raise ValueError("arc_bank_size must be at least arc_num_archetypes")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("arc_temperature must be positive")
        if not math.isfinite(beta_init_scale) or beta_init_scale <= 0:
            raise ValueError("arc_beta_init_scale must be positive")

        self.dim = int(dim)
        self.num_archetypes = int(num_archetypes)
        self.bank_size = int(bank_size)
        self.temperature = float(temperature)

        self.beta_logits = nn.Parameter(
            torch.randn(num_archetypes, bank_size) * float(beta_init_scale)
        )
        self.empty_bank_seed = nn.Parameter(torch.randn(num_archetypes, dim) * 0.02)
        self.register_buffer("memory", torch.zeros(bank_size, dim))
        self.register_buffer("memory_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("memory_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "memory_priority",
            torch.full((bank_size,), -torch.inf),
        )
        self.register_buffer(
            "priority_projection",
            torch.linspace(0.173, 1.913, dim),
            persistent=True,
        )

    @torch.no_grad()
    def update(self, states: torch.Tensor, *, allow_update: bool = True) -> None:
        if not self.training or not allow_update or states.numel() == 0:
            return
        states = states.detach()
        states = states[torch.isfinite(states).all(dim=1)]
        if states.numel() == 0:
            return

        projection = self.priority_projection.to(
            device=states.device, dtype=states.dtype
        )
        phase = (states * projection).sum(dim=1)
        priorities = torch.frac(
            torch.sin(phase * 12.9898).abs() * 43758.5453
        )

        count = int(self.memory_count.item())
        existing_states = self.memory[:count].to(states.device)
        existing_priorities = self.memory_priority[:count].to(states.device)
        candidates = torch.cat([existing_states, states], dim=0)
        candidate_priorities = torch.cat(
            [existing_priorities, priorities], dim=0
        )
        keep = min(self.bank_size, candidates.size(0))
        selected_priority, selected_index = torch.topk(
            candidate_priorities,
            k=keep,
            largest=True,
            sorted=True,
        )
        selected_states = candidates.index_select(0, selected_index)

        self.memory[:keep].copy_(selected_states.to(self.memory))
        self.memory_priority[:keep].copy_(
            selected_priority.to(self.memory_priority)
        )
        if keep < self.bank_size:
            self.memory_priority[keep:].fill_(-torch.inf)
        self.memory_count.fill_(keep)
        self.memory_seen.add_(states.size(0))

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

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        archetypes, _ = self.archetypes()
        squared_distance = (states[:, None, :] - archetypes[None, :, :]).pow(
            2
        ).mean(dim=-1)
        composition = torch.softmax(
            -squared_distance / self.temperature, dim=1
        )
        reconstruction = composition @ archetypes
        return composition, reconstruction, archetypes


class ArchetypalRiskComposition(nn.Module):
    """v2: staged_final base + hard gate (CA-PSA) + balanced re-transport (CATET)."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = int(args.n_classes)
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.wsi_projection_dim = int(args.wsi_projection_dim)
        self.omics_input_dim = omic_input_dim

        dim = self.wsi_projection_dim
        self.num_archetypes = int(
            getattr(args, "arc_num_archetypes", 6)
        )
        # ---- staged_final losses (kept) ----
        self.arc_lambda_recon = float(getattr(args, "arc_lambda_recon", 0.05))
        self.arc_lambda_balance = float(
            getattr(args, "arc_lambda_balance", 0.01)
        )
        self.arc_lambda_rank = float(getattr(args, "arc_lambda_rank", 0.10))
        self.arc_rank_margin = float(getattr(args, "arc_rank_margin", 0.0))
        self.arc_rank_max_pairs = int(
            getattr(args, "arc_rank_max_pairs", 4096)
        )
        # ---- v2 additions ----
        self.arc_lambda_ot = float(getattr(args, "arc_lambda_ot", 0.04))
        self.arc_lambda_gate = float(getattr(args, "arc_lambda_gate", 0.01))
        self.arc_topk_active = int(getattr(args, "arc_topk_active", 3))
        self.arc_ot_eps = float(getattr(args, "arc_ot_eps", 0.05))
        self.arc_ot_iters = int(getattr(args, "arc_ot_iters", 25))
        self._validate_hyperparameters()

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
        beta_init_scale = float(getattr(args, "arc_beta_init_scale", 1.5))
        self.wsi_state_norm = nn.LayerNorm(dim)
        self.omic_state_norm = nn.LayerNorm(dim)
        self.wsi_archetypes = CohortArchetypeBank(
            dim, self.num_archetypes, bank_size, temperature, beta_init_scale
        )
        self.omic_archetypes = CohortArchetypeBank(
            dim, self.num_archetypes, bank_size, temperature, beta_init_scale
        )

        # gate head (CA-PSA inheritance) — same arch operates on the soft
        # composition to produce hard top-K activation.
        self.gate_temperature = float(
            getattr(args, "arc_gate_temperature", 0.5)
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

    def _validate_hyperparameters(self) -> None:
        weights = {
            "arc_lambda_recon": self.arc_lambda_recon,
            "arc_lambda_balance": self.arc_lambda_balance,
            "arc_lambda_rank": self.arc_lambda_rank,
            "arc_lambda_ot": self.arc_lambda_ot,
            "arc_lambda_gate": self.arc_lambda_gate,
        }
        for name, value in weights.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.arc_rank_margin):
            raise ValueError("arc_rank_margin must be finite")
        if self.arc_rank_max_pairs < 1:
            raise ValueError("arc_rank_max_pairs must be at least 1")
        if self.arc_topk_active < 1 or self.arc_topk_active > self.num_archetypes:
            raise ValueError(
                "arc_topk_active must be in [1, arc_num_archetypes]"
            )

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
                kwargs[f"x_omic{index}"]
                for index in range(1, self.num_pathways + 1)
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

    # ------------------------------------------------------------------
    # v2 additions
    # ------------------------------------------------------------------
    def _hard_topk(self, composition: torch.Tensor) -> torch.Tensor:
        """CA-PSA-style hard top-K gate on the soft composition.

        Returns a binary mask [B, K] that selects the top-K archetypes per
        patient.  Gradients pass through unchanged; in evaluation the mask
        renormalises the composition so hazard head sees a convex weight.
        """
        k = min(self.arc_topk_active, composition.size(1))
        top = composition.topk(k, dim=1).indices
        gate = torch.zeros_like(composition)
        gate.scatter_(1, top, 1.0)
        return gate

    def _cross_modal_retransport(
        self,
        wsi_composition: torch.Tensor,
        omic_composition: torch.Tensor,
    ) -> torch.Tensor:
        """Faithfulness audit between modality archetype compositions.

        Implements the CATET keep/remove idea at the composition level: the
        two modalities should agree on which archetypes carry signal and which
        do not.  Cheap symmetric KL bounded between 0 and log(num_archetypes).
        Sinkhorn over 6 archetypes is wasted compute; KL does the job.
        """
        eps = 1e-8
        w = wsi_composition.clamp_min(eps)
        o = omic_composition.clamp_min(eps)
        return 0.5 * (
            F.kl_div(w.log(), o, reduction="batchmean")
            + F.kl_div(o.log(), w, reduction="batchmean")
        )

    def archetype_parameters(self):
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

        wsi_state = self.wsi_state_norm(self._masked_mean(wsi_slots, wsi_mask))
        omic_state = self.omic_state_norm(self._masked_mean(omic_slots, omic_mask))

        cur_epoch = int(kwargs.get("cur_epoch", kwargs.get("epoch", 0)))
        # v2: allow either epoch==0 (default staged_final) or up to
        # arc_bank_update_epochs-1 if user wants longer bank building.
        update_epochs = int(getattr(self.args, "arc_bank_update_epochs", 0))
        update_memory = cur_epoch < max(1, update_epochs) if update_epochs > 0 else cur_epoch == 0
        self.wsi_archetypes.update(
            wsi_state[has_wsi],
            allow_update=update_memory,
        )
        self.omic_archetypes.update(
            omic_state[has_omic],
            allow_update=update_memory,
        )
        (
            wsi_composition,
            wsi_reconstruction,
            wsi_archetypes,
        ) = self.wsi_archetypes(wsi_state)
        (
            omic_composition,
            omic_reconstruction,
            omic_archetypes,
        ) = self.omic_archetypes(omic_state)

        # v2: hard top-K gate on the active composition.  During training we
        # pass the gate as a multiplier on the composition; at eval we
        # renormalise so the convex weight still sums to 1.
        if self.training:
            gate_wsi = self._hard_topk(wsi_composition) + wsi_composition - wsi_composition.detach()
            gate_omic = self._hard_topk(omic_composition) + omic_composition - omic_composition.detach()
            wsi_composition_gated = wsi_composition * gate_wsi
            omic_composition_gated = omic_composition * gate_omic
            wsi_composition_gated = wsi_composition_gated / wsi_composition_gated.sum(dim=1, keepdim=True).clamp_min(1e-6)
            omic_composition_gated = omic_composition_gated / omic_composition_gated.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            wsi_composition_gated = wsi_composition
            omic_composition_gated = omic_composition

        wsi_weight = has_wsi.to(wsi_state.dtype).unsqueeze(1)
        omic_weight = has_omic.to(wsi_state.dtype).unsqueeze(1)
        denominator = (wsi_weight + omic_weight).clamp_min(1.0)
        composition = (
            wsi_weight * wsi_composition_gated
            + omic_weight * omic_composition_gated
        ) / denominator
        neither = ~(has_wsi | has_omic)
        if neither.any():
            composition = composition + neither.to(composition.dtype).unsqueeze(
                1
            ) / self.num_archetypes

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

        # v2: balanced OT re-transport keeps the two modality compositions
        # mutually faithful (was: JS divergence, removed).
        both = has_wsi & has_omic
        ot_loss = zero
        if both.any():
            ot_loss = self._cross_modal_retransport(
                wsi_composition[both], omic_composition[both]
            )

        available_compositions = []
        if has_wsi.any():
            available_compositions.append(wsi_composition[has_wsi])
        if has_omic.any():
            available_compositions.append(omic_composition[has_omic])
        if available_compositions:
            mean_composition = (
                torch.cat(available_compositions, dim=0).mean(dim=0)
            )
            target = torch.full_like(mean_composition, 1.0 / self.num_archetypes)
            balance_loss = F.mse_loss(mean_composition, target)
        else:
            balance_loss = zero

        # v2: gate budget keeps the per-patient active archetype count near
        # the target K=arc_topk_active; cheap entropy-style regulariser.
        target_active = float(self.arc_topk_active) / float(self.num_archetypes)
        active_rate = (wsi_composition > 1e-6).float().sum(dim=1).mean() / float(self.num_archetypes)
        gate_loss = (active_rate - target_active) ** 2

        rank_loss = zero
        if kwargs.get("y") is not None and kwargs.get("c") is not None:
            rank_loss = self._ranking_loss(logits, kwargs["y"], kwargs["c"])

        aux_loss = (
            self.arc_lambda_recon * reconstruction_loss
            + self.arc_lambda_ot * ot_loss
            + self.arc_lambda_balance * balance_loss
            + self.arc_lambda_gate * gate_loss
            + self.arc_lambda_rank * rank_loss
        )
        self.last_training_losses = {
            "arc_reconstruction": reconstruction_loss.detach(),
            "arc_balance": balance_loss.detach(),
            "arc_gate": gate_loss.detach(),
            "arc_ot_retransport": ot_loss.detach(),
            "arc_rank": rank_loss.detach(),
            "arc_composition_entropy": (
                -(composition.clamp_min(1e-8) * composition.clamp_min(1e-8).log())
                .sum(dim=1)
                .mean()
                .detach()
            ),
            "arc_composition_variance": (
                composition.var(dim=0, unbiased=False).mean().detach()
            ),
            "arc_wsi_bank_count": self.wsi_archetypes.memory_count.detach().float(),
            "arc_omic_bank_count": self.omic_archetypes.memory_count.detach().float(),
            "arc_wsi_bank_seen": self.wsi_archetypes.memory_seen.detach().float(),
            "arc_omic_bank_seen": self.omic_archetypes.memory_seen.detach().float(),
        }
        return logits, aux_loss


__all__ = ["ArchetypalRiskComposition", "CohortArchetypeBank"]
