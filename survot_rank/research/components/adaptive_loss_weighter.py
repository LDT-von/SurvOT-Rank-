#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可学习的自适应损失加权模块（Adaptive / Uncertainty-based Loss Weighting）。

动机
----
`OTEHV2RankEvent`（V45）与 `OTEHV2RankEventV2` 的总损失是多项加权和：
per-event NLL、Cox 排序、global consistency、gate entropy、OT、diversity、recon
等。这些权重（`lambda_*`）都是**人工固定**的，是在双模态 V45 场景下手工调出来的。
一旦引入新能力（临床三模态、统一目标、slot 路由重设计），损失地形改变，旧的固定
lambda 往往不再是最优配平——历史上 V44“一次性叠加多个 trick”就因为各损失项互相
打架（loss 反复震荡、fold 崩）而从 0.7105 跌到 0.6760。

本模块用 Kendall et al. (2018, CVPR) 的**同方差不确定性加权**（homoscedastic
uncertainty weighting）思路，为每个损失项引入一个可学习的对数方差参数 `s_i =
log(sigma_i^2)`，把总损失写成：

    L_total = sum_i [ exp(-s_i) * L_i + s_i ]

- `exp(-s_i)` 是第 i 项的自适应精度（权重）：模型可以自己学到把噪声大/帮助小的
  损失项权重压低（`s_i` 变大 → `exp(-s_i)` 变小）。
- `+ s_i` 是正则项，防止 `s_i` 无限增大把所有损失都压到 0（当 `L_i > 0` 时，
  过大的 `s_i` 会让 `+ s_i` 这一项本身变大，形成制衡）。

这样“损失函数太多、手工配不平”的问题就交给梯度下降自动解决，而不是靠人工试参。
默认在 `OTEHV2RankEventV2` 中**关闭**（`otehv2v2_learnable_loss_weights=False`），
开启后作为一个可单独消融的能力，与 V45 baseline 数值行为互不影响。

参考
----
Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty
to weigh losses for scene geometry and semantics. CVPR.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn


class AdaptiveLossWeighter(nn.Module):
    """按可学习对数方差自适应加权一组命名损失项。

    Parameters
    ----------
    loss_names: Iterable[str]
        参与自适应加权的损失项名称集合（超集）。为每个名称注册一个可学习的
        标量参数 `log_var`（初值 0，对应 sigma^2 = 1，即初始权重为 1）。
    init_log_var: float
        `log_var` 参数的初始值，默认 0.0（初始权重 exp(0)=1，与“未加权”起点一致，
        保证训练早期不会突然改变损失尺度）。
    """

    def __init__(self, loss_names: Iterable[str], init_log_var: float = 0.0):
        super().__init__()
        names = list(dict.fromkeys(loss_names))  # 去重且保持顺序
        if len(names) == 0:
            raise ValueError("AdaptiveLossWeighter 至少需要一个损失项名称")
        self.loss_names = names
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(float(init_log_var))) for name in names}
        )

    def forward(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """把一组命名损失项按可学习对数方差合并为单个标量。

        Parameters
        ----------
        losses: Mapping[str, Tensor]
            `{名称: 标量损失张量}`。允许只传入 `loss_names` 的子集（例如某些能力
            关闭时对应损失项不存在）；每个未在 `__init__` 注册的名称会被拒绝，
            以避免静默忽略拼写错误的键。

        Returns
        -------
        Tensor
            0 维标量，dtype/device 与输入损失一致。
        """
        if len(losses) == 0:
            raise ValueError("AdaptiveLossWeighter.forward 收到空的 losses 字典")

        total: torch.Tensor | None = None
        for name, value in losses.items():
            if name not in self.log_vars:
                raise KeyError(
                    f"损失项 '{name}' 未在 AdaptiveLossWeighter 注册；"
                    f"已注册项为 {self.loss_names}"
                )
            s = self.log_vars[name]
            term = torch.exp(-s) * value + s
            total = term if total is None else total + term

        assert total is not None  # 上面已保证 losses 非空
        return total

    def current_weights(self) -> Dict[str, float]:
        """返回各损失项当前的自适应权重 `exp(-s_i)`（供日志/调试用，不参与反向）。"""
        with torch.no_grad():
            return {name: float(torch.exp(-p).item()) for name, p in self.log_vars.items()}


__all__ = ["AdaptiveLossWeighter"]
