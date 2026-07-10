#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AdaptiveLossWeighter（可学习自适应损失加权）单元测试。

用合成小张量验证：
1. 输出为 0 维标量，dtype 与输入一致。
2. 只传入注册项的子集也能正常合并。
3. 反向传播能同时为各损失项对应的 log_var 参数产生非 NaN/Inf 梯度。
4. 初始 log_var=0 时初始权重为 1（未加权起点）。
5. 传入未注册的损失项名称时抛出 KeyError（避免静默忽略拼写错误）。
"""

import math

import pytest
import torch

from survot_rank.research.components.adaptive_loss_weighter import AdaptiveLossWeighter


def test_output_is_scalar_and_dtype_matches():
    weighter = AdaptiveLossWeighter(["nll", "rank", "div"])
    losses = {
        "nll": torch.tensor(2.5),
        "rank": torch.tensor(0.3),
        "div": torch.tensor(0.1),
    }
    out = weighter(losses)
    assert out.dim() == 0
    assert out.dtype == torch.float32


def test_accepts_subset_of_registered_names():
    weighter = AdaptiveLossWeighter(["nll", "rank", "div", "unified"])
    # 只传两个已注册项，应正常合并。
    out = weighter({"nll": torch.tensor(1.0), "unified": torch.tensor(0.5)})
    assert torch.isfinite(out)


def test_backward_produces_finite_grads_for_log_vars():
    weighter = AdaptiveLossWeighter(["nll", "rank", "div"])
    # 让每个损失项都带梯度来源。
    a = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(0.4, requires_grad=True)
    c = torch.tensor(0.2, requires_grad=True)
    out = weighter({"nll": a, "rank": b, "div": c})
    out.backward()
    for name, param in weighter.log_vars.items():
        if name in ("nll", "rank", "div"):
            assert param.grad is not None, f"{name} log_var 无梯度"
            assert torch.isfinite(param.grad).all(), f"{name} log_var 梯度非有限"
    for t in (a, b, c):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_initial_weight_is_one():
    weighter = AdaptiveLossWeighter(["nll"], init_log_var=0.0)
    weights = weighter.current_weights()
    assert math.isclose(weights["nll"], 1.0, rel_tol=1e-6)


def test_unregistered_name_raises():
    weighter = AdaptiveLossWeighter(["nll"])
    with pytest.raises(KeyError):
        weighter({"typo_name": torch.tensor(1.0)})


def test_empty_losses_raises():
    weighter = AdaptiveLossWeighter(["nll"])
    with pytest.raises(ValueError):
        weighter({})


def test_empty_names_raises():
    with pytest.raises(ValueError):
        AdaptiveLossWeighter([])
