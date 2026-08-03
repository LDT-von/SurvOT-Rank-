"""v4.2 ACT-Surv：原型运输组合（ArcSurv 与 IST-Surv 的数学合并）。

核心主张
--------
患者风险是**队列预后极端态（archetype）hazard 曲线的凸组合**，而凸组合的坐标
不是自由回归出来的，而是患者自身 patch/pathway 到 archetype 的**运输质量**。

由此，可加归因不是额外模块，而是凸组合的**直接推论**：

    logit_t = Σ_k β_k · h_{k,t}          （archetype 级）
            = Σ_k Σ_i P_{i,k} · h_{k,t}  （token × archetype 级，精确相等）

其中 β_k = Σ_i P_{i,k}，而 P 的行边缘被约束为 token 质量 a_i，因此
Σ_k β_k = Σ_i a_i = 1 且 β_k ≥ 0——**凸组合由运输边缘条件保证，而非 softmax 凑出**。

三个免费得到的性质（都是推论，不是新增损失）
------------------------------------------------
1. **精确完备性**：不设自由 bias，故归因残差恒为 0（仅受浮点精度限制）。
   IST-Surv 需要用 completeness 指标去监控的东西，这里是构造性的。
2. **闭式删除反事实**：行 softmax 相互独立，删掉 token i 后

       logit'_t = (logit_t − Σ_k P_{i,k} h_{k,t}) / (1 − a_i)

   无需重解 Sinkhorn。IST-Surv 每次删除都要重解一次运输问题。
3. **有界外推**：任何预测都落在 K 条 archetype hazard 曲线的凸包内。

相对两个来源方法的取舍
----------------------
- 相对 ArcSurv：β 由 token 级运输给出而非池化向量的 softmax，因此归因能下到
  patch/pathway；archetype 改为可学习预后顶点，**移除了「第一轮建库后冻结」**
  这一已知脆弱点；不再需要 recon/align/volume 三个结构损失。
- 相对 IST-Surv：可加分解与删除反事实变成闭式推论，**不再需要重解运输**，
  也不需要 deletion penalty 与 attribution stability 损失。

辅助损失只有一项
----------------
唯一的结构性风险是 archetype 塌缩（所有患者用同一个顶点）。用批内平均 β 对
均匀分布的 KL 抑制，权重很小。其余全部交给 NLL（+ 可选 IPCW 排序）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.components.omics_encoder import SNN_Block, WSI_Mlp
from survot_rank.research.components.slot_attention import MultiHeadSlotAttention


class ArchetypalTransportComposition(nn.Module):
    """v4.2：hazard = archetype hazard 曲线在运输质量坐标下的凸组合。"""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        super().__init__()
        self.args = args
        self.omic_sizes = args.omic_sizes
        self.num_classes = int(args.n_classes)
        self.wsi_embedding_dim = int(args.encoding_dim)
        self.wsi_projection_dim = int(args.wsi_projection_dim)
        self.omics_input_dim = omic_input_dim

        dim = self.wsi_projection_dim
        self.num_archetypes = int(getattr(args, "act_num_archetypes", 6))
        self.act_epsilon = float(getattr(args, "act_epsilon", 0.10))
        self.act_lambda_balance = float(getattr(args, "act_lambda_balance", 0.01))
        self.act_lambda_rank = float(getattr(args, "act_lambda_rank", 0.10))
        self.act_rank_margin = float(getattr(args, "act_rank_margin", 0.0))
        self.act_rank_max_pairs = int(getattr(args, "act_rank_max_pairs", 4096))
        self.act_hazard_scale = float(getattr(args, "act_hazard_scale", 1.0))
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
            eval_seed=slot_eval_seed,
        )
        self.token_norm = nn.LayerNorm(dim)

        # 预后顶点：可学习的 archetype 方向 + 每个 archetype 自己的 hazard 曲线。
        # 不再用「第一轮建库后冻结」的记忆库，从根上去掉那个脆弱点。
        self.archetype_embedding = nn.Parameter(torch.randn(self.num_archetypes, dim))
        nn.init.orthogonal_(self.archetype_embedding)
        self.archetype_hazard_logits = nn.Parameter(
            torch.zeros(self.num_archetypes, self.num_classes)
        )
        nn.init.normal_(self.archetype_hazard_logits, std=0.5)
        # 无模态患者的兜底：等权使用全部 archetype（仍落在凸包内）。
        self.last_training_losses: dict[str, torch.Tensor] = {}
        self.last_explanations: dict[str, torch.Tensor] | None = None

    def _validate_hyperparameters(self) -> None:
        if self.num_archetypes < 2:
            raise ValueError("act_num_archetypes must be at least 2")
        if self.act_epsilon <= 0.0:
            raise ValueError("act_epsilon must be positive")
        if self.act_hazard_scale <= 0.0:
            raise ValueError("act_hazard_scale must be positive")
        for name in ("act_lambda_balance", "act_lambda_rank", "act_rank_margin"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.act_rank_max_pairs < 1:
            raise ValueError("act_rank_max_pairs must be at least 1")

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


    def _token_masses(self, token_mask):
        """行边缘 a：每位患者的有效 token 质量之和恰为 1。

        某个模态缺失时其 token 质量为 0，自动被排除，无需特例分支。
        """
        weights = token_mask.to(self.archetype_embedding.dtype)
        total = weights.sum(dim=1, keepdim=True)
        degenerate = total.squeeze(1) <= 0
        # 无任何模态的患者退化为等权使用全部 archetype，仍落在凸包内。
        safe_total = total.clamp_min(1.0)
        return weights / safe_total, degenerate

    def _transport_plan(self, tokens, token_mask):
        """行约束的熵正则最优运输（列边缘自由），有闭式解。

        P_{i,k} = a_i · softmax_k(−C_{i,k} / ε)，因此 Σ_k P_{i,k} = a_i 精确成立，
        总质量 Σ_{i,k} P_{i,k} = 1 精确成立。凸组合由此为构造性结果。
        """
        archetypes = F.normalize(self.archetype_embedding, dim=-1)
        directions = F.normalize(tokens, dim=-1)
        # 代价用余弦距离，范围 [0, 2]，与 ε 的量纲匹配。
        cost = 1.0 - directions @ archetypes.t()
        assignment = torch.softmax(-cost / self.act_epsilon, dim=-1)
        masses, degenerate = self._token_masses(token_mask)
        plan = masses.unsqueeze(-1) * assignment
        if degenerate.any():
            uniform = plan.new_full(
                (1, plan.size(1), self.num_archetypes),
                1.0 / (plan.size(1) * self.num_archetypes),
            )
            plan = torch.where(
                degenerate[:, None, None], uniform.expand_as(plan), plan
            )
        return plan, cost, degenerate

    def _ranking_loss(self, logits, y, c):
        hazards = torch.sigmoid(logits)
        risk = -torch.cumprod(1.0 - hazards, dim=1).sum(dim=1)
        y = y.view(-1).to(risk.dtype)
        events = (1.0 - c.view(-1).to(risk.dtype)).bool()
        if events.sum() == 0:
            return risk.sum() * 0.0
        comparable = events[:, None] & (y[:, None] < y[None, :])
        pairs = comparable.nonzero(as_tuple=False)
        if pairs.numel() == 0:
            return risk.sum() * 0.0
        if pairs.size(0) > self.act_rank_max_pairs:
            keep = torch.randperm(pairs.size(0), device=pairs.device)
            pairs = pairs[keep[: self.act_rank_max_pairs]]
        margin = self.act_rank_margin - (risk[pairs[:, 0]] - risk[pairs[:, 1]])
        return F.relu(margin).mean()

    def forward(self, **kwargs):
        x_wsi = kwargs["x_wsi"].float()
        device = x_wsi.device
        batch_size = x_wsi.size(0)

        wsi_tokens = self.slot_attention_wsi(self.wsi_mlp(torch.nan_to_num(x_wsi)))
        omic_tokens = self.slot_attention_omic(
            torch.nan_to_num(self._encode_omics(kwargs).float())
        )

        has_wsi = self._availability(kwargs, "wsi_available", batch_size, device)
        has_omic = self._availability(kwargs, "omics_available", batch_size, device)
        wsi_mask = self._slot_mask(
            kwargs, "wsi_slot_mask", batch_size, wsi_tokens.size(1), device
        ) & has_wsi[:, None]
        omic_mask = self._slot_mask(
            kwargs, "omics_slot_mask", batch_size, omic_tokens.size(1), device
        ) & has_omic[:, None]

        num_wsi_tokens = wsi_tokens.size(1)
        tokens = self.token_norm(torch.cat([wsi_tokens, omic_tokens], dim=1))
        token_mask = torch.cat([wsi_mask, omic_mask], dim=1)

        plan, cost, degenerate = self._transport_plan(tokens, token_mask)
        # 凸组合坐标 = 列边缘。
        composition = plan.sum(dim=1)

        hazard_curves = self.act_hazard_scale * self.archetype_hazard_logits
        # 不加自由 bias：这是「归因残差恒为 0」的前提。
        logits = composition @ hazard_curves

        # token × archetype 级的精确贡献：三者求和逐级相等。
        edge_contribution = plan.unsqueeze(-1) * hazard_curves[None, None, :, :]
        token_contribution = edge_contribution.sum(dim=2)
        archetype_contribution = composition.unsqueeze(-1) * hazard_curves.unsqueeze(0)
        completeness_error = (
            logits - archetype_contribution.sum(dim=1)
        ).abs().amax(dim=1)

        self.last_explanations = {
            "transport_plan": plan.detach(),
            "transport_cost": cost.detach(),
            "composition": composition.detach(),
            "archetype_hazards": hazard_curves.detach(),
            "archetype_contribution": archetype_contribution.detach(),
            "token_contribution": token_contribution.detach(),
            "wsi_token_contribution": token_contribution[:, :num_wsi_tokens].detach(),
            "omic_token_contribution": token_contribution[:, num_wsi_tokens:].detach(),
            "completeness_error": completeness_error.detach(),
            "token_mass": plan.sum(dim=2).detach(),
            "degenerate_patients": degenerate.detach(),
            "hazards": torch.sigmoid(logits).detach(),
            "survival": torch.cumprod(1.0 - torch.sigmoid(logits), dim=1).detach(),
        }
        self.last_explanations.update(self._archetype_diagnostics(composition))

        if not self.training:
            self.last_training_losses = {}
            return logits, logits.sum() * 0.0

        zero = logits.sum() * 0.0
        # 唯一的辅助损失：抑制 archetype 塌缩（批内平均 β 对均匀分布的 KL）。
        mean_composition = composition.mean(dim=0).clamp_min(1e-8)
        balance_loss = (
            mean_composition * (mean_composition.log() + torch.log(
                torch.tensor(float(self.num_archetypes), device=device)
            ))
        ).sum()
        rank_loss = zero
        if kwargs.get("y") is not None and kwargs.get("c") is not None:
            rank_loss = self._ranking_loss(logits, kwargs["y"], kwargs["c"])

        aux_loss = (
            self.act_lambda_balance * balance_loss
            + self.act_lambda_rank * rank_loss
        )
        self.last_training_losses = {
            "act_balance": balance_loss.detach(),
            "act_rank": rank_loss.detach(),
            "act_total": aux_loss.detach(),
            "act_completeness_error": completeness_error.max().detach(),
            **{
                key: value
                for key, value in self._archetype_diagnostics(composition).items()
            },
        }
        return logits, aux_loss


    @torch.no_grad()
    def _archetype_diagnostics(self, composition):
        """archetype 是否真的分化开——这是接可加归因这一卖点的前置条件。

        - ``act_archetype_cosine``：顶点方向的最大两两余弦，接近 1 = 方向塌缩
        - ``act_hazard_spread``：hazard 曲线的最小两两 L1 距离，接近 0 = 预后无区分
        - ``act_effective_archetypes``：exp(H(mean β))，接近 1 = 全队列只用一个顶点
        - ``act_composition_dispersion``：患者间 β 的标准差均值，接近 0 = 无个体差异
        """
        directions = F.normalize(self.archetype_embedding, dim=-1)
        cosine = directions @ directions.t()
        offdiag = ~torch.eye(
            self.num_archetypes, dtype=torch.bool, device=cosine.device
        )
        hazards = self.act_hazard_scale * self.archetype_hazard_logits
        pairwise_hazard = (
            hazards[:, None, :] - hazards[None, :, :]
        ).abs().sum(dim=-1)
        mean_composition = composition.mean(dim=0).clamp_min(1e-12)
        entropy = -(mean_composition * mean_composition.log()).sum()
        return {
            "act_archetype_cosine": cosine[offdiag].max().detach(),
            "act_hazard_spread": pairwise_hazard[offdiag].min().detach(),
            "act_effective_archetypes": entropy.exp().detach(),
            "act_composition_dispersion": composition.std(dim=0).mean().detach(),
        }

    @torch.no_grad()
    def deletion_counterfactual(self, token_index: int):
        """闭式删除反事实，无需重解运输问题。

        行 softmax 相互独立，故删除 token i 后剩余行只需按 1/(1 − a_i) 重标定：

            logit'_t = (logit_t − Σ_k P_{i,k} h_{k,t}) / (1 − a_i)

        这是 IST-Surv 需要对每个删除重解一次 Sinkhorn 才能得到的量。
        """
        if self.last_explanations is None:
            raise RuntimeError("run a forward pass before requesting counterfactuals")
        plan = self.last_explanations["transport_plan"]
        if not 0 <= token_index < plan.size(1):
            raise IndexError(f"token_index out of range: {token_index}")
        hazards = self.last_explanations["archetype_hazards"]
        composition = self.last_explanations["composition"]
        factual = composition @ hazards
        removed = plan[:, token_index] @ hazards
        remaining_mass = (1.0 - plan[:, token_index].sum(dim=1)).clamp_min(1e-8)
        return (factual - removed) / remaining_mass.unsqueeze(1)
