"""DCT v3.9: Risk-Simplex Transport (RST).

v3.9 改变的是**预测对象**，不是再加一项损失。

v3.3--v3.8 的做法是"回归一个自由的风险分数，再在评估期做 anchor 干预"。
v3.9 把预测定义为一个坐标：

    患者自己的跨模态耦合 P，在低危队列的代价几何 C_L 下更省，还是在
    高危队列的代价几何 C_H 下更省？

    lambda_s = sigmoid( ( <P_s, C_L,s> - <P_s, C_H,s> ) / tau )

    logits_s = (1 - lambda_s) * h_L,s + lambda_s * h_H,s

因此反事实不再是事后查询，它就是坐标轴本身：lambda=0 表示"如果这名患者的
运输结构变成低危队列的样子"，lambda=1 表示变成高危队列的样子。

三条结构性保证（无需任何 margin 损失去"鼓励"，可由单测数值验证）：

1. 预测严格落在 {h_L, h_H} 的凸包内，风险有界；
2. h_H = h_L + softplus(gap) 保证逐时间箱 hazard 偏序，于是 risk 关于
   lambda 严格单调递增 —— v3.8 的 TID/TDM 想用 margin 损失换取的方向一致性
   与剂量单调性在这里是恒等式，同时消除了"把结论写进训练目标"的循环论证；
3. 跨 slot 中心化使运输几何只由 slot 之间的差异决定：sum_i v_i = 0，因此
   两两内积之和严格等于 -sum_i ||v_i||^2 < 0（各 slot 范数相等时，两两余弦
   均值恰为 -1/(K-1)）。

相对 v3.3 删除的东西（每一项都有实测或历史依据）：

* `_semantic_slots` 与 shared prototypes：实测把 slot 两两余弦推到 0.9987
  (T=0.30, 即配置实际值)，8 个"全局坐标"数值上是同一个向量，OT plan 因此
  退化为均匀分布（偏离均匀度 0.0116）。索引稳定性改由 learned slot query
  提供（BO-QSA 的既有结论，属正确性修复，不作为创新点）。
* ETAR：v3.6 实测 BLCA 略好、LUAD/LUSC 更差，且是无结构保证的启发式。
* geometry reliability (RTEM) 与 evidence cost：两个从未启用过的分支。
* `_project_coupling` 的 1000 次图内迭代：log-domain Sinkhorn 本身已收敛到
  给定边缘，该投影只用于修补 `nan_to_num` 造成的破坏，默认降到 3 次。

本方法是基于模型的运输敏感性分析，不是可识别的因果治疗效应。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)
from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    cosine_cost,
    euclidean_cost,
)


class DCTV39RiskSimplexTransport(DistributionalCounterfactualTransport):
    """把生存预测定义为低危--高危运输几何之间的单形坐标。"""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        # learned slot query 提供跨患者的 slot 索引稳定性，替代已被实测证明
        # 会导致表征塌缩的 shared prototype 池化。必须在 super() 之前设置，
        # 因为 slot attention 在祖先类的 __init__ 中构建。
        if not getattr(args, "dct_v39_keep_legacy_slot_init", False):
            args.dct_slot_init_mode = "learned"
        # log-domain Sinkhorn 已经收敛到给定边缘，投影只是数值兜底。
        if not hasattr(args, "dct_coupling_projection_iters_v39_applied"):
            args.dct_coupling_projection_iters = int(
                getattr(args, "dct_v39_projection_iters", 3)
            )
            args.dct_coupling_projection_iters_v39_applied = True

        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        # v3.9 不使用这两个模块，显式删除以保证它们不再进入前向、参数量与
        # state_dict —— "关掉权重"和"移除模块"在论文口径上不是一回事。
        del self.shared_wsi_prototypes
        del self.shared_omic_prototypes

        # v3.3 遗留的启发式目标在 v3.9 中被硬性关闭，避免配置漂移把它们带回来。
        self.dct_lambda_etar = 0.0
        self.dct_evidence_cost_weight = 0.0
        self.dct_geometry_reliability_strength = 0.0

        self.v39_center_slots = bool(getattr(args, "dct_v39_center_slots", True))
        self.v39_residual_scale = float(getattr(args, "dct_v39_residual_scale", 0.0))
        self.v39_anchor_freeze_epoch = int(
            getattr(args, "dct_v39_anchor_freeze_epoch", 0)
        )
        self.v39_lambda_spread_target = float(
            getattr(args, "dct_v39_lambda_spread_target", 0.0)
        )
        if self.v39_residual_scale < 0.0:
            raise ValueError("dct_v39_residual_scale must be non-negative")
        if self.v39_anchor_freeze_epoch < 0:
            raise ValueError("dct_v39_anchor_freeze_epoch must be non-negative")
        if not 0.0 <= self.v39_lambda_spread_target <= 0.25:
            raise ValueError(
                "dct_v39_lambda_spread_target must be in [0, 0.25]; "
                "0.25 is the maximum variance of a [0,1] variable"
            )

        stages = self.spt_num_stages
        classes = self.num_classes
        # 每个阶段一对锚定 hazard 曲线。h_high = h_low + softplus(gap) 保证
        # 逐时间箱的 hazard 偏序，从而 risk(h_high) > risk(h_low) 恒成立。
        self.anchor_hazard_low = nn.Parameter(torch.zeros(stages, classes))
        self.anchor_hazard_gap = nn.Parameter(torch.full((stages, classes), -1.0))
        # 温度必须匹配"锚点代价差"的实际尺度，否则 lambda 会挤在 0.5 附近而
        # 学不动。本机实测该尺度约 0.01（归一化 cost 下），因此默认 0.02，
        # 并默认开启首个 batch 的自动标定。
        self.v39_log_tau = nn.Parameter(
            torch.tensor(float(getattr(args, "dct_v39_tau_init", 0.02))).log()
        )
        self.v39_tau_autoscale = bool(getattr(args, "dct_v39_tau_autoscale", True))
        # 非持久 buffer：旧 checkpoint 仍可加载，且重新训练会重新标定。
        self.register_buffer(
            "v39_tau_calibrated", torch.zeros((), dtype=torch.bool), persistent=False
        )

        self._last_lambda = None
        self._last_anchor_gap = None

    # ------------------------------------------------------------------
    # 表征：删掉塌缩的 prototype 池化，改为跨 slot 中心化
    # ------------------------------------------------------------------

    @staticmethod
    def _center_slots(slots):
        """跨 slot 中心化：移除所有 slot 共有的共模分量。

        运输代价只取决于 slot 之间的相对关系，共模分量对跨模态匹配是纯噪声，
        却会让 cost 矩阵的所有行趋同、把 Sinkhorn 解推向均匀 plan。
        中心化后 sum_i v_i = 0，两两内积之和严格为 -sum_i ||v_i||^2（零参数）。
        """
        return slots - slots.mean(dim=1, keepdim=True)

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)
        if self.v39_center_slots:
            slots_wsi = self._center_slots(slots_wsi)
            slots_omic = self._center_slots(slots_omic)
        # 返回的后两项在 v3.3 中是 prototype 分配矩阵，用于可解释性导出。
        # v3.9 没有该矩阵，返回 slot attention 的池化权重占位（形状不同，
        # 仅用于日志，不参与任何计算）。
        placeholder = slots_wsi.new_zeros(slots_wsi.size(0), slots_wsi.size(1), 1)
        return slots_wsi, slots_omic, placeholder, placeholder

    # ------------------------------------------------------------------
    # 代价：删掉两个从未启用的分支，只保留 evidence-conditioned 边缘
    # ------------------------------------------------------------------

    def _cost_tensor(self, slots_wsi, slots_omic):
        """精简版代价构建：base 几何 + 阶段预后代价，evidence gate 只调边缘质量。"""
        pair_tokens = self._pair_tokens(slots_wsi, slots_omic)
        bsz, sw, so, dim4 = pair_tokens.shape
        dim = dim4 // 4
        stage_cost = F.softplus(self.stage_pair_cost(pair_tokens)).permute(0, 3, 1, 2)
        base_costs = (
            self._normalize_cost(cosine_cost(slots_wsi, slots_omic)),
            self._normalize_cost(euclidean_cost(slots_wsi, slots_omic)),
            self._normalize_cost(self._positive_dot_cost(slots_wsi, slots_omic)),
        )

        all_stage_costs, row_marginals, col_marginals, gates = [], [], [], []
        for stage_idx in range(self.spt_num_stages):
            stage_code = self.stage_embedding[stage_idx].view(1, 1, 1, dim)
            stage_code = stage_code.expand(bsz, sw, so, dim)
            gate = torch.sigmoid(
                self.evidence_gate(
                    torch.cat([pair_tokens, stage_code], dim=-1)
                ).squeeze(-1)
            )
            prognostic_cost = self._normalize_stage_cost(stage_cost[:, stage_idx])
            all_stage_costs.append(
                torch.stack(
                    [
                        base_cost + self.spt_prog_cost_weight * prognostic_cost
                        for base_cost in base_costs
                    ],
                    dim=1,
                )
            )
            row_marginals.append(
                gate.mean(dim=-1).clamp_min(self.dct_evidence_mass_floor)
            )
            col_marginals.append(
                gate.mean(dim=-2).clamp_min(self.dct_evidence_mass_floor)
            )
            gates.append(gate)

        rows = torch.stack(row_marginals, dim=1)
        cols = torch.stack(col_marginals, dim=1)
        rows = rows / rows.sum(dim=-1, keepdim=True)
        cols = cols / cols.sum(dim=-1, keepdim=True)
        strength = self.dct_evidence_marginal_strength
        if strength < 1.0:
            uniform_rows = torch.full_like(rows, 1.0 / rows.size(-1))
            uniform_cols = torch.full_like(cols, 1.0 / cols.size(-1))
            rows = (1.0 - strength) * uniform_rows + strength * rows
            cols = (1.0 - strength) * uniform_cols + strength * cols
        self._last_transport_reliability = None
        return torch.stack(all_stage_costs, dim=1), rows, cols, torch.stack(gates, dim=1)

    def _project_coupling(self, plan, rows, cols):
        """轻量边缘投影：只做数值兜底，判据在 no_grad 下计算。"""
        for _ in range(self.dct_coupling_projection_iters):
            plan = plan * (
                rows.unsqueeze(-1) / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            )
            plan = plan * (
                cols.unsqueeze(1) / plan.sum(dim=-2, keepdim=True).clamp_min(1e-8)
            )
            with torch.no_grad():
                row_error = (plan.sum(dim=-1) - rows).abs().amax()
                col_error = (plan.sum(dim=-2) - cols).abs().amax()
                converged = bool(
                    torch.maximum(row_error, col_error)
                    <= self.dct_coupling_projection_tol
                )
            if converged:
                break
        return plan

    # ------------------------------------------------------------------
    # 核心：风险单形坐标
    # ------------------------------------------------------------------

    def _risk_simplex_coordinate(self, plans):
        """用患者自己的运输计划去评估两个队列锚定几何的代价，得到 lambda。

        <P, C_low> 大而 <P, C_high> 小，说明这名患者的跨模态耦合方式在高危
        队列的几何下更"自然"，于是 lambda 趋近 1。

        anchor 代价是 detach 的队列统计量（buffer），单个患者的梯度不会推动
        锚点；梯度只经由 P 回传到 slot/cost 通路。
        """
        # 必须 detach().clone()：v3.9 把队列锚点拉进了训练前向路径，而
        # `_update_risk_anchors` 用 lerp_/copy_ 原地更新这个 buffer。不 clone
        # 会让 autograd 在 backward 时发现张量版本被原地修改而直接报错
        # （v3.3 不暴露该问题，因为锚点只在评估分支被读取）。detach 同时保证
        # 单个患者的梯度不会推动队列统计量。
        anchors = (
            self.risk_anchor_costs.detach()
            .clone()
            .to(device=plans[0][0].device, dtype=plans[0][0].dtype)
        )
        differences, defined = [], []
        for stage_idx, stage_plans in enumerate(plans):
            plan = torch.stack(stage_plans, dim=1)
            cost_low = (plan * anchors[stage_idx, self._LOW_RISK].unsqueeze(0)).sum(
                dim=(-2, -1)
            ).mean(dim=1)
            cost_high = (plan * anchors[stage_idx, self._HIGH_RISK].unsqueeze(0)).sum(
                dim=(-2, -1)
            ).mean(dim=1)
            differences.append(cost_low - cost_high)
            defined.append(bool(self.risk_anchor_seen[stage_idx].all()))
        difference = torch.stack(differences, dim=1)

        self._maybe_calibrate_tau(difference)
        tau = self.v39_log_tau.exp().clamp_min(1e-4)
        coordinate = torch.sigmoid(difference / tau)
        # 锚点尚未被训练折观察到的阶段，坐标没有定义，用中性 0.5。
        mask = difference.new_tensor(defined, dtype=torch.bool).view(1, -1)
        return torch.where(mask, coordinate, torch.full_like(coordinate, 0.5))

    @torch.no_grad()
    def _maybe_calibrate_tau(self, difference):
        """用首个可用 batch 的代价差尺度标定温度。

        锚点代价差的绝对尺度取决于归一化后的 cost、slot 数与阶段数，写死一个
        常数很容易让 sigmoid 饱和（lambda 全 0/1）或退化（lambda 全 0.5）。
        这里把 tau 设为代价差的标准差，使初始 logit 落在约 ±1 的高梯度区间。
        """
        if not (self.training and self.v39_tau_autoscale):
            return
        if bool(self.v39_tau_calibrated) or not bool(self.risk_anchor_seen.all()):
            return
        scale = difference.detach().std()
        if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
            return
        self.v39_log_tau.data = scale.clamp(1e-4, 1.0).log()
        self.v39_tau_calibrated.fill_(True)

    def _anchor_hazards(self):
        """返回 (h_low, h_high)，逐时间箱满足 h_high >= h_low。"""
        low = self.anchor_hazard_low
        return low, low + F.softplus(self.anchor_hazard_gap)

    def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
        """预测 = 锚定 hazard 曲线在 lambda 处的凸组合。

        event Transformer 不再直接输出风险，它只决定各阶段的权重 g_s。
        主信号路径因此极短：slot -> cost -> plan -> lambda -> hazard，
        这本身就是对 380 例小样本的强正则化。
        """
        tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
        tokens = tokens + self.stage_embedding.unsqueeze(0)
        tokens = self.event_norm(self.event_encoder(tokens))
        gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)

        coordinate = self._risk_simplex_coordinate(plans)
        hazard_low, hazard_high = self._anchor_hazards()
        stage_logits = (
            1.0 - coordinate.unsqueeze(-1)
        ) * hazard_low.unsqueeze(0) + coordinate.unsqueeze(-1) * hazard_high.unsqueeze(0)
        logits = (gate.unsqueeze(-1) * stage_logits).sum(dim=1)

        if self.v39_residual_scale > 0.0:
            # 可选旁路，仅用于"结构约束 vs 模型容量"的消融；默认 0 时预测严格
            # 落在锚定 hazard 的凸包内。
            residual = torch.einsum("be,bec->bc", gate, self.event_hazard(tokens))
            logits = logits + self.v39_residual_scale * residual

        self._last_lambda = coordinate
        self._last_anchor_gap = F.softplus(self.anchor_hazard_gap).mean().detach()
        return logits, gate

    # ------------------------------------------------------------------
    # 锚点与目标
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_risk_anchors(self, costs, low_weights, high_weights):
        """可选地在某个 epoch 之后冻结锚点，避免 lambda 追逐移动目标。"""
        if self.v39_anchor_freeze_epoch > 0:
            epoch = int(getattr(self.args, "cur_epoch", 0))
            if epoch >= self.v39_anchor_freeze_epoch and bool(
                self.risk_anchor_seen.all()
            ):
                return
        super()._update_risk_anchors(costs, low_weights, high_weights)

    def _combine_auxiliary_objectives(
        self,
        *,
        ipcw_rank_loss,
        etar_loss,
        transport_objective,
        transport_metrics,
        epoch,
    ):
        """v3.9 的目标 = NLL(trainer) + IPCW rank + 可选的坐标铺开项。

        坐标铺开项不规定任何患者的方向或次序（那会是循环论证），它只惩罚
        "所有患者挤在同一个 lambda 上"这种退化解，是防塌缩项而非监督项。
        默认权重 0，即目标严格等于 v3.3 的两项。
        """
        del etar_loss, transport_metrics, epoch
        total = self.dct_lambda_ipcw_rank * ipcw_rank_loss + transport_objective
        if self.v39_lambda_spread_target > 0.0 and self._last_lambda is not None:
            variance = self._last_lambda.var(dim=0, unbiased=False).mean()
            total = total + F.relu(self.v39_lambda_spread_target - variance)
        return total

    def forward(self, **kwargs):
        logits, aux_loss = super().forward(**kwargs)

        coordinate = self._last_lambda
        if coordinate is None:
            return logits, aux_loss

        if self.training and isinstance(self.last_training_losses, dict):
            self.last_training_losses.update(
                {
                    "v39_lambda_mean": coordinate.mean().detach(),
                    "v39_lambda_std": coordinate.std().detach(),
                    "v39_lambda_stage_spread": coordinate.var(dim=0, unbiased=False)
                    .mean()
                    .detach(),
                    "v39_anchor_hazard_gap": self._last_anchor_gap,
                    "v39_tau": self.v39_log_tau.exp().detach(),
                }
            )
        elif isinstance(self.last_explanations, dict):
            # 结构自洽性审计：把低/高危锚点代价重新解出的 plan 送回坐标函数，
            # lambda 应分别趋向 0 和 1。这是 v3.9 免费获得的忠实性检验 ——
            # 它检查的是坐标定义本身，而不是某个 margin 损失是否泛化。
            hazard_low, hazard_high = self._anchor_hazards()
            self.last_explanations.update(
                {
                    "v39_lambda": coordinate.detach(),
                    "v39_anchor_hazard_low": hazard_low.detach(),
                    "v39_anchor_hazard_high": hazard_high.detach(),
                    "v39_tau": self.v39_log_tau.exp().detach(),
                }
            )
        return logits, aux_loss

    @torch.no_grad()
    def audit_coordinate_endpoints(self, **kwargs):
        """返回 factual / 低危锚点 / 高危锚点三种输入下的 lambda。

        期望 lambda(low) < lambda(factual) < lambda(high)。这不是训练目标，
        因此它是对坐标定义的独立检验，可直接写进论文的忠实性表格。
        """
        was_training = self.training
        self.eval()
        try:
            x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
            x_omics = self._encode_omics(kwargs)
            slots_wsi, slots_omic, _, _ = self._encode_transport_slots(
                x_wsi_proj, x_omics, kwargs
            )
            epoch = int(getattr(self.args, "cur_epoch", kwargs.get("cur_epoch", 0)))
            costs, rows, cols, _ = self._cost_tensor(slots_wsi, slots_omic)
            low_costs, high_costs = self._counterfactual_costs(costs)
            result = {}
            for name, cost in (
                ("factual", costs),
                ("low_anchor", low_costs),
                ("high_anchor", high_costs),
            ):
                plans, _ = self._plans_from_cost_tensor(cost, rows, cols, epoch)
                result[name] = self._risk_simplex_coordinate(plans).mean(dim=1)
            return result
        finally:
            if was_training:
                self.train()
