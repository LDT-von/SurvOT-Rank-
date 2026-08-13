"""ArcSurv: cross-modal archetypal risk composition for survival prediction.

The method keeps the established WSI and omics encoders but replaces
patient-specific event gating with a cohort-level prognostic simplex. Each
archetype is constrained to be a convex combination of a fold-local memory
bank, and every patient's hazard logits are a convex combination of
archetype-specific hazard curves.
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
    """Learn archetypes as convex combinations of an order-robust fold memory.

    During the first training epoch, a deterministic priority reservoir selects
    a fixed-size subset from every patient state it sees.  Keeping the highest
    priorities and sorting them canonically makes the selected memory invariant
    to dataloader order, unlike a first-come buffer.  Later epochs and all
    evaluation forwards leave the memory fixed.  Learnable row-stochastic
    weights then preserve the defining archetypal-analysis constraint
    ``A = Beta @ Z`` instead of becoming unconstrained prototypes.
    """

    def __init__(
        self,
        dim: int,
        num_archetypes: int,
        bank_size: int,
        temperature: float,
        beta_init_scale: float = 1.5,
        distance_reduction: str = "scaled",
        anchor_logit: float = 6.0,
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
        self.distance_reduction = str(distance_reduction)
        if self.distance_reduction not in {"mean", "scaled"}:
            raise ValueError("arc_distance_reduction must be 'mean' or 'scaled'")
        self.anchor_logit = float(anchor_logit)

        # Zero rows make every archetype exactly identical. That symmetry is
        # fatal here: all patients then receive the same uniform composition,
        # while the log-volume loss has zero first derivative at the fully
        # collapsed point. A moderately sparse random convex initialization
        # gives each row a different fold-local anchor set without relaxing
        # the row-stochastic archetypal-analysis constraint.
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
        self.register_buffer("anchors_seeded", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def seed_anchors_once(self) -> bool:
        """冻结 memory 后，把每个原型锚定到互相最远的队列成员上。

        塌缩根因：``archetypes = softmax(beta_logits) @ memory``，softmax 摊在
        整个 bank（256 项）上，``randn * beta_init_scale`` 在这个宽度上不足以
        打破对称，于是 K 行全部收敛到「队列的加权均值」附近——彼此几乎重合。
        患者到各原型的距离因此近乎相等，composition 退化为均匀分布
        （实测组合熵 ≈ ln(6) = 1.7918，患者间方差 ≈ 1e-4）。

        这里在 memory 冻结的那一刻做一次 furthest-point 采样，让每行有一个
        不同的强锚点，同时保留小幅噪声，使 ``A = Beta @ Z`` 的行随机约束不变。
        """
        if bool(self.anchors_seeded.item()):
            return False
        count = int(self.memory_count.item())
        if count < self.num_archetypes:
            return False

        memory = self.memory[:count]
        centre = memory.mean(dim=0, keepdim=True)
        selected = [int(torch.cdist(memory, centre).squeeze(1).argmax().item())]
        while len(selected) < self.num_archetypes:
            chosen = memory.index_select(
                0, torch.tensor(selected, device=memory.device)
            )
            spread = torch.cdist(memory, chosen).min(dim=1).values
            spread[torch.tensor(selected, device=memory.device)] = -1.0
            selected.append(int(spread.argmax().item()))

        logits = torch.empty_like(self.beta_logits).normal_(0.0, 0.1)
        for row, index in enumerate(selected):
            logits[row, index] += self.anchor_logit
        self.beta_logits.data.copy_(logits)
        self.anchors_seeded.fill_(True)
        return True

    @torch.no_grad()
    def update(self, states: torch.Tensor, *, allow_update: bool = True) -> None:
        """Update the first-epoch priority reservoir without gradient leakage."""
        if not self.training or not allow_update or states.numel() == 0:
            return
        states = states.detach()
        states = states[torch.isfinite(states).all(dim=1)]
        if states.numel() == 0:
            return

        projection = self.priority_projection.to(device=states.device, dtype=states.dtype)
        phase = (states * projection).sum(dim=1)
        priorities = torch.frac(
            torch.sin(phase * 12.9898).abs() * 43758.5453
        )

        count = int(self.memory_count.item())
        existing_states = self.memory[:count].to(states.device)
        existing_priorities = self.memory_priority[:count].to(states.device)
        candidates = torch.cat([existing_states, states], dim=0)
        candidate_priorities = torch.cat([existing_priorities, priorities], dim=0)
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

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        archetypes, _ = self.archetypes()
        squared = (states[:, None, :] - archetypes[None, :, :]).pow(2)
        if self.distance_reduction == "mean":
            # 旧行为：对 dim 取均值，把距离量级压掉 dim 倍（dim=256），
            # 再除以 temperature 也无法产生有区分度的 softmax。
            squared_distance = squared.mean(dim=-1)
        else:
            # 按 sqrt(dim) 归一，使距离尺度不随投影维度塌缩。
            squared_distance = squared.sum(dim=-1) / math.sqrt(self.dim)
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
        self.arc_lambda_volume = float(getattr(args, "arc_lambda_volume", 0.01))
        self.arc_lambda_rank = float(getattr(args, "arc_lambda_rank", 0.10))
        # 分阶段激活原型结构类损失。BLCA fold1 的最佳验证 C-index 出现在
        # epoch 29（预算边界）且最后 5 轮仍在上升：recon/align/balance/volume
        # 四项从第一轮就与生存目标竞争，拖慢收敛。
        # warmup 期间只保留 NLL + rank，之后线性拉起四项结构损失。
        self.arc_warmup_epochs = int(getattr(args, "arc_warmup_epochs", 5))
        self.arc_ramp_epochs = int(getattr(args, "arc_ramp_epochs", 10))
        # 原型库原先只在 epoch 0 建立并冻结，此时编码器尚未被生存目标塑形。
        # 默认改为在 warmup 期间持续更新，warmup 结束后冻结。
        bank_epochs = int(getattr(args, "arc_bank_update_epochs", -1))
        self.arc_bank_update_epochs = (
            self.arc_warmup_epochs if bank_epochs < 0 else bank_epochs
        )
        for name in ("arc_warmup_epochs", "arc_ramp_epochs", "arc_bank_update_epochs"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        self.arc_rank_margin = float(getattr(args, "arc_rank_margin", 0.0))
        self.arc_rank_max_pairs = int(getattr(args, "arc_rank_max_pairs", 4096))
        # 个体 composition 的锐度项。balance 只把**批次平均**推向均匀，
        # 而此前没有任何一项奖励**单个患者**的组合变尖，因此「所有患者都
        # 平均使用全部原型」是一个可行解（实测组合熵 ≈ ln(K)、方差 ≈ 0）。
        # 两者组合才是目标形态：批次均匀 + 个体集中 = 不同患者用不同原型。
        self.arc_lambda_sharpness = float(
            getattr(args, "arc_lambda_sharpness", 0.0)
        )
        self.arc_seed_anchors = bool(getattr(args, "arc_seed_anchors", False))
        self.arc_freeze_state_encoder = bool(
            getattr(args, "arc_freeze_state_encoder", True)
        )
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
        distance_reduction = str(
            getattr(args, "arc_distance_reduction", "scaled")
        )
        anchor_logit = float(getattr(args, "arc_anchor_logit", 6.0))
        self.wsi_state_norm = nn.LayerNorm(dim)
        self.omic_state_norm = nn.LayerNorm(dim)
        # One cohort hull is the central final-version invariant.  Separate
        # WSI and omics banks silently gave the same archetype index two
        # unrelated meanings and made the JS composition loss ill-posed.
        self.shared_archetypes = CohortArchetypeBank(
            dim,
            self.num_archetypes,
            bank_size,
            temperature,
            beta_init_scale,
            distance_reduction,
            anchor_logit,
        )
        self.register_buffer(
            "state_encoder_frozen",
            torch.zeros((), dtype=torch.bool),
        )

        self.archetype_hazard_logits = nn.Parameter(
            torch.randn(self.num_archetypes, self.num_classes) * 0.02
        )
        self.hazard_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.missing_logits = nn.Parameter(torch.zeros(self.num_classes))
        self.last_training_losses: dict[str, torch.Tensor] = {}
        self.last_composition: torch.Tensor | None = None
        self._last_explanation: dict[str, torch.Tensor] = {}

        if omic_names:
            try:
                self.all_gene_names = list(np.unique(np.concatenate(omic_names)))
            except Exception:
                pass

    @property
    def wsi_archetypes(self) -> CohortArchetypeBank:
        """Compatibility alias: both modalities use the same final bank."""
        return self.shared_archetypes

    @property
    def omic_archetypes(self) -> CohortArchetypeBank:
        """Compatibility alias: both modalities use the same final bank."""
        return self.shared_archetypes

    def _structure_scale(self, epoch) -> float:
        """原型结构损失的分阶段权重：warmup 内为 0，随后线性拉到 1。

        评估时始终返回 1.0，使验证指标反映方法的完整形态。
        """
        if not self.training:
            return 1.0
        epoch = int(epoch)
        if epoch < self.arc_warmup_epochs:
            return 0.0
        if self.arc_ramp_epochs <= 0:
            return 1.0
        post_warmup = epoch - self.arc_warmup_epochs + 1
        return min(1.0, post_warmup / self.arc_ramp_epochs)

    def _validate_hyperparameters(self) -> None:
        """Reject ArcSurv settings that would silently poison the objective."""
        weights = {
            "arc_lambda_recon": self.arc_lambda_recon,
            "arc_lambda_align": self.arc_lambda_align,
            "arc_lambda_balance": self.arc_lambda_balance,
            "arc_lambda_volume": self.arc_lambda_volume,
            "arc_lambda_rank": self.arc_lambda_rank,
            "arc_lambda_sharpness": self.arc_lambda_sharpness,
        }
        for name, value in weights.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.arc_rank_margin):
            raise ValueError("arc_rank_margin must be finite")
        if self.arc_rank_max_pairs < 1:
            raise ValueError("arc_rank_max_pairs must be at least 1")

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

    @staticmethod
    def _simplex_volume_loss(archetypes: torch.Tensor) -> torch.Tensor:
        """Penalize a collapsed prognostic simplex using its edge Gram volume."""
        if archetypes.size(0) < 2:
            return archetypes.sum() * 0.0
        edges = archetypes[1:] - archetypes[:1]
        gram = edges @ edges.transpose(0, 1)
        gram = gram / float(max(1, archetypes.size(1)))
        identity = torch.eye(
            gram.size(0),
            device=gram.device,
            dtype=gram.dtype,
        )
        _, log_volume_squared = torch.linalg.slogdet(gram + 1e-4 * identity)
        return -0.5 * log_volume_squared / float(gram.size(0))

    @staticmethod
    def _mean_pairwise_cosine(archetypes: torch.Tensor) -> torch.Tensor:
        """原型两两余弦均值。接近 1 表示原型已塌缩成同一个方向。"""
        if archetypes.size(0) < 2:
            return archetypes.new_zeros(())
        normalised = F.normalize(archetypes, dim=-1)
        gram = normalised @ normalised.transpose(0, 1)
        upper = torch.triu_indices(
            gram.size(0), gram.size(1), offset=1, device=gram.device
        )
        return gram[upper[0], upper[1]].mean()

    def _archetype_hazard_spread(self) -> torch.Tensor:
        """各原型风险分数的标准差。接近 0 表示原型在预测上不可区分。"""
        hazards = torch.sigmoid(self.archetype_hazard_logits + self.hazard_bias)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risk = -survival.sum(dim=1)
        if risk.numel() < 2:
            return risk.new_zeros(())
        return risk.std(unbiased=False)

    def _archetype_diagnostics(
        self,
        archetypes: torch.Tensor,
        composition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """原型是否真的分化开的只读诊断。

        这些量不参与反传，只用于判断 ArcSurv / v4.2 的「凸组合」前提是否成立：
        原型若既在特征空间塌缩（cosine→1）又在风险上不可区分
        （hazard_spread→0），那么 composition 就退化成了一个几乎恒定的向量，
        分数与可解释性两个卖点都不成立。
        """
        with torch.no_grad():
            dominant = composition.argmax(dim=1)
            active = torch.unique(dominant).numel()
            return {
                "arc_shared_archetype_cosine": self._mean_pairwise_cosine(
                    archetypes
                ).detach(),
                # Historical names remain aliases in result collectors.
                "arc_wsi_archetype_cosine": self._mean_pairwise_cosine(
                    archetypes
                ).detach(),
                "arc_omic_archetype_cosine": self._mean_pairwise_cosine(
                    archetypes
                ).detach(),
                "arc_hazard_spread": self._archetype_hazard_spread().detach(),
                "arc_active_archetype_fraction": composition.new_tensor(
                    active / float(self.num_archetypes)
                ),
                "arc_max_composition_weight": composition.max(dim=1)
                .values.mean()
                .detach(),
            }

    def archetype_parameters(self):
        """Return the shared hull, support weights, and hazard curves."""
        archetypes, beta = self.shared_archetypes.archetypes()
        return {
            "shared_archetypes": archetypes,
            "shared_beta": beta,
            "wsi_archetypes": archetypes,
            "wsi_beta": beta,
            "omic_archetypes": archetypes,
            "omic_beta": beta,
            "hazard_logits": self.archetype_hazard_logits,
        }

    def _freeze_state_encoder_once(self) -> None:
        """Freeze the coordinate map exactly when its cohort bank freezes."""
        if bool(self.state_encoder_frozen.item()):
            return
        modules = (
            self.wsi_mlp,
            self.sig_networks,
            self.slot_attention_wsi,
            self.slot_attention_omic,
            self.wsi_state_norm,
            self.omic_state_norm,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.state_encoder_frozen.fill_(True)

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"]
        batch_size = x_wsi.size(0)
        device = x_wsi.device

        current_epoch = int(
            getattr(
                self.args,
                "cur_epoch",
                kwargs.get("cur_epoch", kwargs.get("epoch", 0)),
            )
        )
        freeze_epoch = max(1, self.arc_bank_update_epochs)
        if (
            self.training
            and self.arc_freeze_state_encoder
            and current_epoch >= freeze_epoch
        ):
            self._freeze_state_encoder_once()

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

        # 至少更新 1 轮，否则原型库永远为空。
        update_memory = current_epoch < max(1, self.arc_bank_update_epochs)
        structure_scale = self._structure_scale(current_epoch)
        wsi_weight = has_wsi.to(wsi_state.dtype).unsqueeze(1)
        omic_weight = has_omic.to(wsi_state.dtype).unsqueeze(1)
        denominator = (wsi_weight + omic_weight).clamp_min(1.0)
        shared_state = (
            wsi_weight * wsi_state + omic_weight * omic_state
        ) / denominator
        has_any = has_wsi | has_omic
        self.shared_archetypes.update(
            shared_state[has_any],
            allow_update=update_memory,
        )
        # memory 冻结后立刻把原型锚定到互相最远的成员上。必须在冻结之后做：
        # 此时队列内容不再变化，furthest-point 选出的锚点才是稳定的。
        if self.training and not update_memory and self.arc_seed_anchors:
            self.shared_archetypes.seed_anchors_once()
        (
            wsi_composition,
            wsi_reconstruction,
            archetypes,
        ) = self.shared_archetypes(wsi_state)
        (
            omic_composition,
            omic_reconstruction,
            _,
        ) = self.shared_archetypes(omic_state)

        composition = (
            wsi_weight * wsi_composition + omic_weight * omic_composition
        ) / denominator
        neither = ~(has_wsi | has_omic)
        if neither.any():
            composition = composition + neither.to(composition.dtype).unsqueeze(1) / self.num_archetypes

        logits = composition @ self.archetype_hazard_logits + self.hazard_bias
        logits = torch.where(neither.unsqueeze(1), self.missing_logits.unsqueeze(0), logits)
        self.last_composition = composition.detach()
        archetype_hazards = torch.sigmoid(
            self.archetype_hazard_logits + self.hazard_bias
        )
        archetype_survival = torch.cumprod(1.0 - archetype_hazards, dim=1)
        self._last_explanation = {
            "composition": composition.detach(),
            "wsi_composition": wsi_composition.detach(),
            "omic_composition": omic_composition.detach(),
            "archetypes": archetypes.detach(),
            "archetype_hazards": archetype_hazards.detach(),
            "archetype_survival": archetype_survival.detach(),
            "archetype_logit_contribution": (
                composition.unsqueeze(-1)
                * self.archetype_hazard_logits.unsqueeze(0)
            ).detach(),
            "bank_support_weights": self.shared_archetypes.archetypes()[1].detach(),
        }

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

        volume_loss = self._simplex_volume_loss(archetypes)

        # 个体锐度：最小化每个患者 composition 的熵。与 balance（批次平均趋于
        # 均匀）互补，共同排除「所有患者都均匀使用全部原型」这个退化解。
        sharpness_loss = (
            -(composition.clamp_min(1e-8) * composition.clamp_min(1e-8).log())
            .sum(dim=1)
            .mean()
        )

        rank_loss = zero
        if kwargs.get("y") is not None and kwargs.get("c") is not None:
            rank_loss = self._ranking_loss(logits, kwargs["y"], kwargs["c"])

        # rank 损失从第一轮即生效（与 NLL 同期），只有四项结构损失走 ramp。
        aux_loss = structure_scale * (
            self.arc_lambda_recon * reconstruction_loss
            + self.arc_lambda_align * alignment_loss
            + self.arc_lambda_balance * balance_loss
            + self.arc_lambda_volume * volume_loss
            + self.arc_lambda_sharpness * sharpness_loss
        ) + self.arc_lambda_rank * rank_loss
        self.last_training_losses = {
            "arc_structure_scale": logits.new_tensor(structure_scale).detach(),
            "arc_bank_updating": logits.new_tensor(
                float(update_memory)
            ).detach(),
            "arc_reconstruction": reconstruction_loss.detach(),
            "arc_alignment": alignment_loss.detach(),
            "arc_balance": balance_loss.detach(),
            "arc_simplex_volume": volume_loss.detach(),
            "arc_sharpness": sharpness_loss.detach(),
            "arc_anchors_seeded": logits.new_tensor(
                float(bool(self.shared_archetypes.anchors_seeded.item()))
            ).detach(),
            "arc_state_encoder_frozen": self.state_encoder_frozen.detach().float(),
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
            "arc_shared_bank_count": self.shared_archetypes.memory_count.detach().float(),
            "arc_shared_bank_seen": self.shared_archetypes.memory_seen.detach().float(),
            "arc_wsi_bank_count": self.shared_archetypes.memory_count.detach().float(),
            "arc_omic_bank_count": self.shared_archetypes.memory_count.detach().float(),
            "arc_wsi_bank_seen": self.shared_archetypes.memory_seen.detach().float(),
            "arc_omic_bank_seen": self.shared_archetypes.memory_seen.detach().float(),
            **self._archetype_diagnostics(
                archetypes, composition
            ),
        }
        return logits, aux_loss

    def explain_last_batch(self) -> dict[str, torch.Tensor]:
        """Return the exact shared-simplex quantities used for prediction."""
        if not self._last_explanation:
            raise RuntimeError("Run a forward pass before requesting ArcSurv diagnostics")
        return dict(self._last_explanation)


__all__ = ["ArchetypalRiskComposition", "CohortArchetypeBank"]
