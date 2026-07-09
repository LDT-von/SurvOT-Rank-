#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`UnifiedSurvivalObjective` 单元测试。

对应需求：`.kiro/specs/survot-rank-enhancements/requirements.md` 需求 3
（统一的 NLL 与排序目标），验证以下验收标准：
- AC1: 返回 0 维张量，dtype 与输入 logits 一致。
- AC2: 全删失 batch -> 有限值，反向无 NaN/Inf 梯度。
- AC3: 无可比对样本对 -> 排序项退化为 0 且不抛异常。
- AC4: 合法 batch/事件数/类别数组合下，反向同时为 event_logits 与
  risk_logits 产生非 NaN/Inf 梯度。
- AC5: 不一致可比对数量为 0/1/5 时，排序贡献单调非减。

均使用合成小张量（batch<=6, num_events<=4, num_classes in {2,4}），不依赖
真实 WSI/Omics 数据。
"""

import torch

from survot_rank.training.paths import ensure_slotspe_in_path  # noqa

ensure_slotspe_in_path()
from utils.loss_func import UnifiedSurvivalObjective  # noqa


def _make_event_logits(batch, num_events, num_classes, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, num_events, num_classes, generator=g).to(dtype)


def _make_risk_logits(batch, num_classes, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, num_classes, generator=g).to(dtype)


# ---------------------------------------------------------------------------
# AC1: 返回 0 维张量，dtype 与输入 logits 一致
# ---------------------------------------------------------------------------


class TestForwardOutputContract:
    def test_returns_scalar_tensor_float32(self):
        obj = UnifiedSurvivalObjective()
        event_logits = _make_event_logits(4, 2, 4, dtype=torch.float32)
        risk_logits = _make_risk_logits(4, 4, dtype=torch.float32)
        y = torch.tensor([0, 1, 2, 3])
        c = torch.tensor([0, 1, 0, 1])

        result = obj(event_logits=event_logits, risk_logits=risk_logits, y=y, c=c)

        assert result.dim() == 0
        assert result.dtype == event_logits.dtype

    def test_returns_scalar_tensor_float64(self):
        obj = UnifiedSurvivalObjective()
        event_logits = _make_event_logits(4, 2, 4, dtype=torch.float64)
        risk_logits = _make_risk_logits(4, 4, dtype=torch.float64)
        y = torch.tensor([0, 1, 2, 3])
        c = torch.tensor([0, 1, 0, 1])

        result = obj(event_logits=event_logits, risk_logits=risk_logits, y=y, c=c)

        assert result.dim() == 0
        assert result.dtype == torch.float64
        assert result.dtype == event_logits.dtype

    def test_dtype_follows_risk_logits_when_no_event_logits(self):
        obj = UnifiedSurvivalObjective()
        risk_logits = _make_risk_logits(3, 2, dtype=torch.float64)
        y = torch.tensor([0, 1, 2])
        c = torch.tensor([0, 0, 1])

        result = obj(risk_logits=risk_logits, y=y, c=c)

        assert result.dim() == 0
        assert result.dtype == risk_logits.dtype


# ---------------------------------------------------------------------------
# AC2: 全删失 batch -> 有限值，反向无 NaN/Inf 梯度
# ---------------------------------------------------------------------------


class TestAllCensoredBatch:
    def test_all_censored_forward_is_finite_and_grad_is_finite(self):
        obj = UnifiedSurvivalObjective()
        event_logits = _make_event_logits(5, 2, 4, dtype=torch.float32).requires_grad_(True)
        risk_logits = _make_risk_logits(5, 4, dtype=torch.float32).requires_grad_(True)
        y = torch.tensor([0, 1, 2, 3, 1])
        c = torch.ones(5)  # 全部删失

        result = obj(event_logits=event_logits, risk_logits=risk_logits, y=y, c=c)

        assert torch.isfinite(result).all()

        result.backward()

        assert event_logits.grad is not None
        assert torch.isfinite(event_logits.grad).all()
        assert risk_logits.grad is not None
        assert torch.isfinite(risk_logits.grad).all()


# ---------------------------------------------------------------------------
# AC3: 无可比对样本对 -> 排序项退化为 0 且不抛异常
# ---------------------------------------------------------------------------


class TestNoComparablePairs:
    def test_all_same_time_bin_ranking_term_is_zero(self):
        """所有样本的时间标签相同时，不存在 t_i < t_j 的可比对。"""
        obj = UnifiedSurvivalObjective()
        event_logits = _make_event_logits(4, 1, 4, dtype=torch.float32)
        y = torch.tensor([1, 1, 1, 1])  # 时间标签全部相同 -> 无 ti<tj
        c = torch.tensor([0, 0, 0, 0])  # 全部未删失，排除删失导致的退化

        # 仅提供 event_logits，无 risk_logits：排序项应完全退化为 0，
        # forward 输出等于 per-event NLL 基线。
        result = obj(event_logits=event_logits, y=y, c=c)
        baseline = obj._per_event_nll(event_logits, y, c)

        assert torch.isfinite(result).all()
        assert torch.allclose(result, baseline.to(result.dtype))

    def test_single_sample_no_exception(self):
        """单样本 batch：不存在任何样本对，排序项应退化为 0 且不抛异常。"""
        obj = UnifiedSurvivalObjective()
        event_logits = _make_event_logits(1, 2, 4, dtype=torch.float32)
        y = torch.tensor([2])
        c = torch.tensor([0])

        result = obj(event_logits=event_logits, y=y, c=c)

        assert torch.isfinite(result).all()

    def test_pairwise_margin_penalty_zero_when_no_comparable_pairs(self):
        """直接白盒测试 `_pairwise_margin_penalty`：无可比对样本对时返回 0。"""
        obj = UnifiedSurvivalObjective()
        risk = torch.randn(4)
        y = torch.tensor([1, 1, 1, 1])
        c = torch.tensor([0, 0, 0, 0])

        penalty = obj._pairwise_margin_penalty(risk, y, c)

        assert torch.isfinite(penalty).all()
        assert penalty.item() == 0.0


# ---------------------------------------------------------------------------
# AC4: 合法 batch/事件数/类别数组合下，反向同时为 event_logits 与
# risk_logits 产生非 NaN/Inf 梯度
# ---------------------------------------------------------------------------


class TestBackwardGradientsForBothLogits:
    COMBOS = [
        (2, 1, 2),
        (3, 2, 4),
        (6, 4, 4),
    ]

    def test_backward_produces_finite_grads_for_both_inputs(self):
        obj = UnifiedSurvivalObjective()
        for batch, num_events, num_classes in self.COMBOS:
            event_logits = _make_event_logits(
                batch, num_events, num_classes, dtype=torch.float32, seed=batch + num_events
            ).requires_grad_(True)
            risk_logits = _make_risk_logits(
                batch, num_classes, dtype=torch.float32, seed=batch * num_classes
            ).requires_grad_(True)
            # 构造有变化的时间标签与部分未删失样本，确保存在可比对样本对。
            y = torch.arange(batch) % num_classes
            c = torch.tensor([0 if i % 2 == 0 else 1 for i in range(batch)])

            result = obj(event_logits=event_logits, risk_logits=risk_logits, y=y, c=c)
            result.backward()

            assert event_logits.grad is not None, f"combo={ (batch, num_events, num_classes) }"
            assert torch.isfinite(event_logits.grad).all()
            assert risk_logits.grad is not None
            assert torch.isfinite(risk_logits.grad).all()

            # 清理，避免下一个 combo 的梯度累积互相影响（这里用的是不同张量，
            # 但仍显式置 None 以保持每次迭代独立)。
            event_logits.grad = None
            risk_logits.grad = None


# ---------------------------------------------------------------------------
# AC5: 不一致可比对数量为 0/1/5 时，排序贡献单调非减
# ---------------------------------------------------------------------------


class TestRankingMonotonicity:
    """白盒测试 `_pairwise_margin_penalty`：构造不一致（discordant）可比对
    数量分别为 0、1、5 的风险/时间标签输入，断言排序贡献单调非减。

    构造方式：固定 6 个样本、时间标签严格递增（0..5）、全部未删失（c=0），
    此时全部 C(6,2)=15 对样本两两可比对。风险值取 {10,20,...,60} 的不同
    排列：
    - 完全按时间递减排列（早发生事件风险更高）-> 0 个不一致对；
    - 交换排列尾部一对 -> 恰好 1 个不一致对；
    - 特定排列 -> 恰好 5 个不一致对（已手工验证）。
    """

    y = torch.tensor([0, 1, 2, 3, 4, 5])
    c = torch.zeros(6)

    # 完全一致（0 个不一致可比对）：风险严格随时间递减。
    risk_0 = torch.tensor([60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
    # 恰好 1 个不一致可比对：交换最后两个样本的风险值。
    risk_1 = torch.tensor([60.0, 50.0, 40.0, 30.0, 10.0, 20.0])
    # 恰好 5 个不一致可比对（手工验证的排列）。
    risk_5 = torch.tensor([60.0, 50.0, 20.0, 10.0, 30.0, 40.0])

    @staticmethod
    def _count_discordant_pairs(risk, y, c):
        t = y.float()
        e = 1.0 - c.float()
        ti = t.view(-1, 1)
        tj = t.view(1, -1)
        comparable = (e.view(-1, 1) > 0.5) & (ti < tj)
        discordant = comparable & (risk.view(-1, 1) < risk.view(1, -1))
        return int(discordant.sum().item())

    def test_discordant_pair_counts_match_construction(self):
        assert self._count_discordant_pairs(self.risk_0, self.y, self.c) == 0
        assert self._count_discordant_pairs(self.risk_1, self.y, self.c) == 1
        assert self._count_discordant_pairs(self.risk_5, self.y, self.c) == 5

    def test_ranking_contribution_is_non_decreasing(self):
        obj = UnifiedSurvivalObjective()

        penalty_0 = obj._pairwise_margin_penalty(self.risk_0, self.y, self.c)
        penalty_1 = obj._pairwise_margin_penalty(self.risk_1, self.y, self.c)
        penalty_5 = obj._pairwise_margin_penalty(self.risk_5, self.y, self.c)

        assert torch.isfinite(penalty_0).all()
        assert torch.isfinite(penalty_1).all()
        assert torch.isfinite(penalty_5).all()

        assert penalty_0.item() <= penalty_1.item()
        assert penalty_1.item() <= penalty_5.item()
