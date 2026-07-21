#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OT event hazard v2 — multi-scale OT fusion with cross-modal Transformer.

Improvements over v1 (29):
- Three OT cost matrices (cosine / euclidean / dot) concatenated for richer transport.
- Cross-attention fusion replacing naive concat.
- Deeper 3-layer Transformer encoder.
- Reconstruction loss for OT plan and slots.
- Warmup ramp + clamp on OT distance (stable training).
- Auxiliary event-hazard loss (supervision on mean event logits).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.training.paths import ensure_slotspe_in_path  # noqa

ensure_slotspe_in_path()

from survot_rank.research.components.slot_attention import MultiHeadSlotAttention  # noqa
from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp  # noqa


def cosine_cost(x, y):
    x_n = F.normalize(x, dim=-1)
    y_n = F.normalize(y, dim=-1)
    return 1.0 - torch.bmm(x_n, y_n.transpose(1, 2))


def euclidean_cost(x, y):
    x_u = x.unsqueeze(2)       # [B, K_w, 1, D]
    y_u = y.unsqueeze(1)       # [B, 1, K_o, D]
    return (x_u - y_u).pow(2).sum(-1).sqrt()


def dot_cost(x, y):
    return -torch.bmm(x, y.transpose(1, 2))


def log_sinkhorn_plan(cost, eps=0.05, max_iter=40):
    """Stable balanced OT in log space with clamped distance."""
    bsz, rows, cols = cost.shape
    device, dtype = cost.device, cost.dtype
    log_mu = torch.full((bsz, rows), 1.0 / rows, device=device, dtype=dtype).log()
    log_nu = torch.full((bsz, cols), 1.0 / cols, device=device, dtype=dtype).log()
    kernel = -cost / eps
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)

    for _ in range(max_iter):
        log_u = log_mu - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_nu - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)

    log_plan = kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    plan = log_plan.exp()
    ot_dist = (plan * cost).sum(dim=(1, 2)).clamp(min=0.0, max=10.0)
    return plan, ot_dist


class MultiScaleOTFusion(nn.Module):
    """Three cost matrices → three OT plans → concat → cross-attention fusion."""

    def __init__(self, dim, num_events=16, nhead=4, dropout=0.1):
        super().__init__()
        self.num_events = num_events
        self.cost_convs = nn.ModuleDict({
            "cosine": nn.Linear(1, dim),
            "euclidean": nn.Sequential(
                nn.Linear(1, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            ),
            "dot": nn.Linear(1, dim),
        })
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

        self.event_queries = nn.Parameter(torch.randn(num_events, dim) * 0.02)

        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )

        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_pair_tokens(self, slots_wsi, slots_omic):
        bsz, sw, so, dim = slots_wsi.shape[0], slots_wsi.shape[1], slots_omic.shape[1], slots_wsi.shape[-1]
        w = slots_wsi.unsqueeze(2).expand(bsz, sw, so, dim)
        o = slots_omic.unsqueeze(1).expand(bsz, sw, so, dim)
        return torch.cat([w, o, w * o, (w - o).abs()], dim=-1)

    def forward(self, slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
        bsz, sw, dim = slots_wsi.shape
        so = slots_omic.shape[1]

        pair_tokens = self._build_pair_tokens(slots_wsi, slots_omic)  # [B, sw, so, dim*4]

        # Project each cost type
        c_cos = self.cost_convs["cosine"](plan_cos.unsqueeze(-1))
        c_euc = self.cost_convs["euclidean"](plan_euc.unsqueeze(-1))
        c_dot = self.cost_convs["dot"](plan_dot.unsqueeze(-1))
        cost_concat = torch.cat([c_cos, c_euc, c_dot], dim=-1)  # [B, sw*so, dim*3]
        pair_context = pair_tokens[..., : dim * 3]
        pair_tokens = self.proj(cost_concat + pair_context)

        # Aggregate into events via attention
        pair_tokens = pair_tokens.reshape(bsz, sw * so, dim)
        pair_mass = plan_cos.reshape(bsz, sw * so).clamp_min(1e-8).log().unsqueeze(-1)
        q = F.normalize(self.event_queries, dim=-1)
        t = F.normalize(pair_tokens, dim=-1)
        scores = torch.einsum("kd,bpd->bpk", q, t)
        scores = scores + pair_mass
        assign = torch.softmax(scores.transpose(1, 2), dim=-1)
        events = torch.bmm(assign, pair_tokens)

        # Cross-attention refinement (WSI ↔ Omic bidirectional)
        events = self.norm(self.cross_attn(events))

        return events + self.refine(events), assign


class OTEventHazardV2Survival(nn.Module):
    """Multi-scale OT event hazard model with stable training."""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = args.n_classes
        self.wsi_embedding_dim = args.encoding_dim
        self.wsi_projection_dim = args.wsi_projection_dim
        self.omics_input_dim = omic_input_dim

        dim = self.wsi_projection_dim
        self.ot_eps = getattr(args, "otehv2_eps", 0.05)
        self.ot_iter = getattr(args, "otehv2_iter", 40)
        self.ot_warmup = getattr(args, "otehv2_warmup", 8)
        self.lambda_ot = getattr(args, "lambda_otehv2_ot", 0.02)
        self.lambda_div = getattr(args, "lambda_otehv2_div", 0.005)
        self.lambda_event_surv = getattr(args, "lambda_otehv2_event_surv", 0.15)
        self.lambda_recon = getattr(args, "lambda_otehv2_recon", 0.1)

        self._init_per_path_model(self.omic_sizes, args.rna_format)
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

        self.fusion = MultiScaleOTFusion(
            dim=dim,
            num_events=getattr(args, "otehv2_num_events", 16),
            nhead=getattr(args, "otehv2_heads", 4),
            dropout=getattr(args, "otehv2_dropout", 0.1),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=getattr(args, "otehv2_heads", 4),
            dim_feedforward=dim * 2,
            dropout=getattr(args, "otehv2_dropout", 0.1),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=getattr(args, "otehv2_layers", 3)
        )
        self.event_norm = nn.LayerNorm(dim)
        self.event_hazard = nn.Linear(dim, self.num_classes)
        self.event_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(getattr(args, "otehv2_dropout", 0.1)),
            nn.Linear(dim // 2, 1),
        )

        # Reconstruction heads
        self.recon_wsi = nn.Linear(dim, dim)
        self.recon_omic = nn.Linear(dim, dim)

        if omic_names:
            try:
                self.all_gene_names = list(np.unique(np.concatenate(omic_names)))
            except Exception:
                pass

    def _init_per_path_model(self, omic_sizes, omics_format):
        dim = self.wsi_projection_dim
        if omics_format == "Pathways":
            self.num_pathways = len(omic_sizes)
            self.sig_networks = nn.ModuleList([
                nn.Sequential(
                    SNN_Block(dim1=idim, dim2=dim),
                    SNN_Block(dim1=dim, dim2=dim, dropout=0.25),
                )
                for idim in omic_sizes
            ])
        elif omics_format == "GeneEmbedding":
            self.sig_networks = SNN_Block(dim1=768, dim2=dim)
        elif omics_format == "RNASeq":
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=dim)
        else:
            raise ValueError(f"Invalid omics_format: {omics_format}")

    def _encode_omics(self, kwargs):
        if self.args.rna_format == "Pathways":
            x_omic = [kwargs[f"x_omic{i}"] for i in range(1, self.num_pathways + 1)]
            h_omic = [self.sig_networks[i](feat) for i, feat in enumerate(x_omic)]
            return torch.stack(h_omic).permute(1, 0, 2)
        return self.sig_networks(kwargs["x_omics"])

    @staticmethod
    def _diversity_loss(tokens):
        tokens = F.normalize(tokens, dim=-1)
        sim = torch.bmm(tokens, tokens.transpose(1, 2))
        eye = torch.eye(sim.size(1), device=sim.device, dtype=sim.dtype).unsqueeze(0)
        return ((sim - eye) ** 2).mean()

    def _warmup_ramp(self, kwargs):
        epoch = kwargs.get("cur_epoch", 0)
        if epoch < self.ot_warmup:
            return 0.0
        return min(1.0, (epoch - self.ot_warmup) / max(1, self.ot_warmup))

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"]
        x_wsi_proj = self.wsi_mlp(x_wsi)
        x_omics = self._encode_omics(kwargs)

        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        cost_cos = cosine_cost(slots_wsi, slots_omic)
        cost_euc = euclidean_cost(slots_wsi, slots_omic)
        cost_dot = dot_cost(slots_wsi, slots_omic)

        plan_cos, ot_dist_cos = log_sinkhorn_plan(cost_cos, eps=self.ot_eps, max_iter=self.ot_iter)
        plan_euc, ot_dist_euc = log_sinkhorn_plan(cost_euc, eps=self.ot_eps, max_iter=self.ot_iter)
        plan_dot, ot_dist_dot = log_sinkhorn_plan(cost_dot, eps=self.ot_eps, max_iter=self.ot_iter)

        event_tokens, _ = self.fusion(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot)
        event_tokens = self.event_norm(self.event_encoder(event_tokens))
        event_logits = self.event_hazard(event_tokens)
        gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)

        if self.training:
            ramp = self._warmup_ramp(kwargs)
            ot_mean = (ot_dist_cos + ot_dist_euc + ot_dist_dot).mean() / 3.0

            # Reconstruction loss: predict each other's slots
            recon_wsi = self.recon_wsi(slots_wsi)     # [B, K_w, D]
            recon_omic = self.recon_omic(slots_omic)  # [B, K_o, D]
            recon_loss = F.mse_loss(recon_wsi, slots_omic) + F.mse_loss(recon_omic, slots_wsi)

            aux_loss = (
                ramp * self.lambda_ot * ot_mean
                + self.lambda_div * self._diversity_loss(event_tokens)
                + self.lambda_recon * recon_loss
            )
            if "y" in kwargs and "c" in kwargs and self.lambda_event_surv > 0:
                y, c = kwargs["y"], kwargs["c"]
                event_mean_logits = event_logits.mean(dim=1)
                from utils.loss_func import NLLSurvLoss
                loss_fn = NLLSurvLoss(alpha=getattr(self.args, "alpha_surv", 0.0))
                aux_loss = aux_loss + self.lambda_event_surv * loss_fn(
                    event_mean_logits, y=y, c=c, t=None
                )
        else:
            aux_loss = 0.0
        return logits, aux_loss
