"""V60: missing-aware log-domain OT event ranking for survival prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    OTEventHazardV2Survival,
    cosine_cost,
    dot_cost,
    euclidean_cost,
    log_sinkhorn_plan,
)


def masked_log_sinkhorn_plan(cost, row_mask, col_mask, eps=0.05, max_iter=40):
    """Balanced log-Sinkhorn with per-sample valid-slot marginals."""
    row_mask = row_mask.bool()
    col_mask = col_mask.bool()
    if not row_mask.any(dim=1).all() or not col_mask.any(dim=1).all():
        raise ValueError("masked OT requires at least one valid slot per modality")

    dtype = cost.dtype
    row_mass = row_mask.to(dtype)
    col_mass = col_mask.to(dtype)
    log_mu = (row_mass / row_mass.sum(dim=1, keepdim=True)).clamp_min(1e-8).log()
    log_nu = (col_mass / col_mass.sum(dim=1, keepdim=True)).clamp_min(1e-8).log()

    valid_pair = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)
    safe_cost = cost.masked_fill(~valid_pair, 1e4)
    kernel = -safe_cost / max(float(eps), 1e-6)
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)
    for _ in range(max_iter):
        log_u = log_mu - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_nu - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)

    plan = (kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)).exp()
    plan = plan.masked_fill(~valid_pair, 0.0)
    distance = (plan * cost.masked_fill(~valid_pair, 0.0)).sum(dim=(1, 2))
    return plan, distance.clamp(min=0.0, max=10.0)


class V60OTEventRank(OTEventHazardV2Survival):
    """V9 OT-event backbone plus compact event NLL and censor-aware ranking.

    Optional inputs:
      ``wsi_available`` / ``omics_available``: [B] availability flags.
      ``wsi_slot_mask`` / ``omics_slot_mask``: [B, K] valid slot masks.
    Missing-modality samples bypass cross-modal OT and use a modality-specific
    fallback head. Complete samples use masked log-domain Sinkhorn plans.
    """

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        args.otehv2_num_events = int(getattr(args, "v60_num_events", getattr(args, "otehv2_num_events", 24)))
        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        dim = self.wsi_projection_dim
        self.v60_lambda_per_event = float(getattr(args, "v60_lambda_per_event", 0.15))
        self.v60_lambda_rank = float(getattr(args, "v60_lambda_rank", 0.15))
        self.v60_rank_margin = float(getattr(args, "v60_rank_margin", 0.0))
        self.v60_rank_max_pairs = int(getattr(args, "v60_rank_max_pairs", 4096))

        self.v60_wsi_fallback = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.num_classes))
        self.v60_omic_fallback = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.num_classes))
        self.v60_missing_fallback = nn.Parameter(torch.zeros(self.num_classes))

    @staticmethod
    def _availability(kwargs, name, batch_size, device):
        value = kwargs.get(name)
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return torch.as_tensor(value, device=device).bool().view(-1)

    @staticmethod
    def _slot_mask(kwargs, name, batch_size, slots, device):
        value = kwargs.get(name)
        if value is None:
            return torch.ones(batch_size, slots, dtype=torch.bool, device=device)
        mask = torch.as_tensor(value, device=device).bool()
        if mask.shape != (batch_size, slots):
            raise ValueError(f"{name} must have shape {(batch_size, slots)}, got {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _per_event_nll(event_logits, y, c):
        bsz, num_events, num_classes = event_logits.shape
        flat = event_logits.reshape(bsz * num_events, num_classes)
        y_rep = y.view(-1, 1).expand(-1, num_events).reshape(-1)
        c_rep = c.view(-1, 1).expand(-1, num_events).reshape(-1)
        hazards = torch.sigmoid(flat)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        padded = torch.cat([torch.ones_like(c_rep[:, None]), survival], dim=1)
        index = y_rep.long().clamp(0, num_classes - 1).view(-1, 1)
        s_prev = torch.gather(padded, 1, index).clamp_min(1e-7)
        h_this = torch.gather(hazards, 1, index).clamp_min(1e-7)
        s_this = torch.gather(padded, 1, index + 1).clamp_min(1e-7)
        uncensored = -(1.0 - c_rep[:, None]) * (s_prev.log() + h_this.log())
        censored = -c_rep[:, None] * s_this.log()
        return (uncensored + censored).mean()

    def _ranking_loss(self, logits, y, c):
        hazards = torch.sigmoid(logits)
        risk = -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)
        times = y.float().view(-1)
        observed = (1.0 - c.float()).view(-1) > 0.5
        comparable = observed[:, None] & (times[:, None] < times[None, :])
        if risk.numel() < 2 or not comparable.any():
            return risk.sum() * 0.0
        differences = risk[:, None] - risk[None, :]
        values = F.softplus(-(differences - self.v60_rank_margin))[comparable]
        if values.numel() > self.v60_rank_max_pairs:
            keep = torch.randperm(values.numel(), device=values.device)[: self.v60_rank_max_pairs]
            values = values[keep]
        return values.mean()

    def _masked_plans(self, slots_wsi, slots_omic, wsi_mask, omic_mask):
        costs = (
            cosine_cost(slots_wsi, slots_omic),
            euclidean_cost(slots_wsi, slots_omic),
            dot_cost(slots_wsi, slots_omic),
        )
        return tuple(
            masked_log_sinkhorn_plan(cost, wsi_mask, omic_mask, eps=self.ot_eps, max_iter=self.ot_iter)
            for cost in costs
        )

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"]
        batch_size = x_wsi.shape[0]
        device = x_wsi.device
        x_wsi_proj = self.wsi_mlp(x_wsi)
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        has_wsi = self._availability(kwargs, "wsi_available", batch_size, device)
        has_omic = self._availability(kwargs, "omics_available", batch_size, device)
        both = has_wsi & has_omic
        wsi_mask = self._slot_mask(kwargs, "wsi_slot_mask", batch_size, slots_wsi.size(1), device)
        omic_mask = self._slot_mask(kwargs, "omics_slot_mask", batch_size, slots_omic.size(1), device)

        logits = torch.zeros(batch_size, self.num_classes, device=device, dtype=slots_wsi.dtype)
        event_logits_valid = None
        valid_positions = both.nonzero(as_tuple=False).view(-1)
        ot_loss = logits.sum() * 0.0

        if valid_positions.numel() > 0:
            sw = slots_wsi[valid_positions]
            so = slots_omic[valid_positions]
            wm = wsi_mask[valid_positions]
            om = omic_mask[valid_positions]
            if not wm.any(dim=1).all() or not om.any(dim=1).all():
                raise ValueError("each OT-valid sample needs one valid WSI and omics slot")
            plans = self._masked_plans(sw, so, wm, om)
            event_tokens, _ = self.fusion(sw, so, *(plan for plan, _ in plans))
            event_tokens = self.event_norm(self.event_encoder(event_tokens))
            event_logits_valid = self.event_hazard(event_tokens)
            gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
            logits[valid_positions] = torch.einsum("be,bec->bc", gate, event_logits_valid)
            ot_loss = torch.stack([distance for _, distance in plans], dim=0).mean()

        wsi_only = has_wsi & ~has_omic
        if wsi_only.any():
            valid = wsi_mask[wsi_only].to(slots_wsi.dtype).unsqueeze(-1)
            pooled = (slots_wsi[wsi_only] * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            logits[wsi_only] = self.v60_wsi_fallback(pooled)
        omic_only = ~has_wsi & has_omic
        if omic_only.any():
            valid = omic_mask[omic_only].to(slots_omic.dtype).unsqueeze(-1)
            pooled = (slots_omic[omic_only] * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            logits[omic_only] = self.v60_omic_fallback(pooled)
        neither = ~has_wsi & ~has_omic
        if neither.any():
            logits[neither] = self.v60_missing_fallback

        if not self.training:
            return logits, 0.0

        aux_loss = self.lambda_ot * ot_loss
        if event_logits_valid is not None and "y" in kwargs and "c" in kwargs:
            y = kwargs["y"][valid_positions]
            c = kwargs["c"][valid_positions]
            aux_loss = aux_loss + self.v60_lambda_per_event * self._per_event_nll(event_logits_valid, y, c)
        if "y" in kwargs and "c" in kwargs:
            aux_loss = aux_loss + self.v60_lambda_rank * self._ranking_loss(logits, kwargs["y"], kwargs["c"])
        return logits, aux_loss


__all__ = ["V60OTEventRank", "masked_log_sinkhorn_plan"]
