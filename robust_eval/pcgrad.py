#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PCGrad 梯度手术：消解多损失之间的梯度方向冲突。

要解决的问题
------------
本项目的结构性病根是「损失太多、梯度互相拉扯」：v45 有 8 个损失，训练集都拟合
不了（train C-index < 0.5）；即便砍到 3 个（RG-ET / CA-TET），OT / risk-set /
intervention 三项的梯度方向仍可能互相抵消。

手工调 lambda 或用不确定性加权（`AdaptiveLossWeighter`）只改变各损失的**尺度**，
不改变**方向**——两个梯度夹角 > 90° 时，再怎么缩放都还是在互相抵消。

PCGrad（Yu et al., NeurIPS 2020, "Gradient Surgery for Multi-Task Learning"）
直接处理方向冲突：对每一对任务梯度，若其点积为负（冲突），就把其中一个投影到
另一个的**法平面**上，去掉互相抵消的分量，再把处理后的梯度求和。

本模块是**独立工具**，不修改任何模型代码。它需要每个损失单独求一次梯度，因此
与「梯度裁剪」不同，不能做成零改动的 monkey-patch——需要在训练循环里用
`pcgrad_backward([...])` 替换原来的单次 `loss.backward()`（见文末集成说明）。

算法（对 N 个任务梯度 g_1..g_N）
--------------------------------
    for i in tasks:
        g_i^PC = g_i
        for j in random_order(other tasks):
            if <g_i^PC, g_j> < 0:                       # 方向冲突
                g_i^PC -= (<g_i^PC, g_j> / ||g_j||^2) g_j  # 投影掉冲突分量
    merged = sum_i g_i^PC
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import torch


def _flatten_grads(grads: Sequence, params: Sequence[torch.Tensor]) -> torch.Tensor:
    """把与 params 对齐的梯度列表拉平成一维向量；None（该损失不依赖此参数）填 0。"""
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(p.new_zeros(p.numel()))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat)


def _project_conflicting(task_grads: list[torch.Tensor], reduction: str = "sum") -> torch.Tensor:
    """对一组已拉平的任务梯度做 PCGrad 投影，返回合并后的单个梯度向量。"""
    num_tasks = len(task_grads)
    projected = [g.clone() for g in task_grads]
    for i in range(num_tasks):
        order = list(range(num_tasks))
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            g_j = task_grads[j]  # 投影始终针对**原始**的 g_j
            inner = torch.dot(projected[i], g_j)
            if inner < 0:
                projected[i] = projected[i] - (inner / g_j.dot(g_j).clamp_min(1e-12)) * g_j
    stacked = torch.stack(projected, dim=0)
    if reduction == "mean":
        return stacked.mean(dim=0)
    return stacked.sum(dim=0)


def pcgrad_backward(
    losses: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    reduction: str = "sum",
) -> torch.Tensor:
    """对多个损失做 PCGrad 梯度手术，并把合并后的梯度写入各参数的 ``.grad``。

    用法上等价于「多损失版的 ``loss.backward()``」：调用后各参数的 ``.grad`` 已就绪，
    调用方直接 ``optimizer.step()`` 即可。**调用前请先 ``optimizer.zero_grad()``**，
    因为本函数是**覆盖**写入 ``.grad``（不累加），因此不兼容梯度累积
    （accumulation_steps 需为 1，本项目 batch_size != 1 时正好满足）。

    参数
    ----
    losses : 一组标量损失张量（如 ``[loss_surv, slot_loss]`` 或拆得更细的各辅助项）。
    params : 模型参数（会被物化成 list；仅处理 ``requires_grad`` 的参数）。
    reduction : "sum"（默认，PCGrad 原论文）或 "mean"。

    返回
    ----
    合并后的一维梯度向量（便于记录范数 / 调试）。
    """
    param_list = [p for p in params if p.requires_grad]
    if len(losses) == 0:
        raise ValueError("losses 不能为空")

    # 单损失时退化为普通反传，省去投影开销。
    if len(losses) == 1:
        grads = torch.autograd.grad(losses[0], param_list, retain_graph=False, allow_unused=True)
        merged = _flatten_grads(grads, param_list)
    else:
        task_grads = []
        for k, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss,
                param_list,
                retain_graph=(k < len(losses) - 1),
                allow_unused=True,
            )
            task_grads.append(_flatten_grads(grads, param_list))
        merged = _project_conflicting(task_grads, reduction=reduction)

    # 写回 .grad
    offset = 0
    for p in param_list:
        n = p.numel()
        p.grad = merged[offset:offset + n].view_as(p).clone()
        offset += n
    return merged


# ======================================================================
# 集成说明（不改模型，只在训练循环里替换一次反传）
# ----------------------------------------------------------------------
# survot_rank/training/train_runner.py 的 train_one_epoch 里，原本是：
#
#     loss = (loss_surv + slot_loss) / accumulation_steps
#     loss.backward()
#     ... optimizer.step(); optimizer.zero_grad()
#
# 改为（batch_size != 1，即 accumulation_steps == 1 时）：
#
#     from robust_eval.pcgrad import pcgrad_backward
#     optimizer.zero_grad()
#     pcgrad_backward([loss_surv, slot_loss], model.parameters())
#     optimizer.step()
#
# 若想更细粒度地消解冲突，可让模型 forward 额外返回各辅助项（如 ot / rank /
# intervention 分开），传成 [loss_surv, ot, rank, intervention]，PCGrad 会两两
# 消解它们之间的方向冲突。传入的项越细，冲突消解越彻底，但每项要多一次 backward。
# ======================================================================


def _selftest() -> None:
    """自测：(1) 投影几何正确；(2) 端到端能在参数上写出与朴素求和不同的梯度。"""
    # --- 1. 投影几何：g1=[1,0], g2=[-1,1] 冲突，投影后应两两正交，合并 != 朴素和 ---
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 1.0])
    merged = _project_conflicting([g1.clone(), g2.clone()], reduction="sum")
    naive = g1 + g2
    # 手算：g1'=[0.5,0.5], g2'=[0,1], merged=[0.5,1.5]
    assert torch.allclose(merged, torch.tensor([0.5, 1.5]), atol=1e-6), merged
    assert not torch.allclose(merged, naive), "PCGrad 结果不应等于朴素求和"
    print(f"[selftest] projection ok: naive={naive.tolist()} -> pcgrad={merged.tolist()}")

    # --- 2. 无冲突时（点积 >= 0）应等于朴素求和 ---
    a = torch.tensor([1.0, 1.0])
    b = torch.tensor([0.5, 2.0])  # dot = 2.5 > 0
    merged2 = _project_conflicting([a.clone(), b.clone()], reduction="sum")
    assert torch.allclose(merged2, a + b, atol=1e-6), "无冲突时应退化为朴素求和"
    print("[selftest] no-conflict passthrough ok")

    # --- 3. 端到端：小模型 + 两个损失，验证 .grad 被正确写入且有限 ---
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4)
    target_a = torch.randn(8, 2)
    target_b = torch.randn(8, 2)
    out = model(x)
    loss_a = torch.nn.functional.mse_loss(out, target_a)
    loss_b = torch.nn.functional.mse_loss(out, target_b)
    model.zero_grad()
    merged3 = pcgrad_backward([loss_a, loss_b], model.parameters())
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), "梯度应已写入且有限"
    assert torch.isfinite(merged3).all()
    print(f"[selftest] end-to-end ok: merged grad dim={merged3.numel()}")
    print("[selftest] OK")


if __name__ == "__main__":
    _selftest()
