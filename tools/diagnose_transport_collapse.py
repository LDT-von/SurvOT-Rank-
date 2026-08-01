#!/usr/bin/env python3
"""运输表征塌缩诊断器。

用途：任何基于 slot + OT 的方法都可能出现"slot 全部塌成同一个向量 → cost
矩阵各行趋同 → Sinkhorn 解退化为均匀 plan → 运输机制不携带患者特异信息"
这条失效链。它不会报错、不会 NaN，只会让所有加在运输上的机制静默失效。

本工具用两个数量化指标定位它：

* ``slot 余弦``：slot 两两余弦相似度均值。接近 1 = 塌缩。
* ``plan 偏离均匀``：OT plan 相对均匀分布的总变差。接近 0 = 运输无信息。

用法::

    python tools/diagnose_transport_collapse.py                 # 全部诊断
    python tools/diagnose_transport_collapse.py --section pooling
    python tools/diagnose_transport_collapse.py --section methods
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from survot_rank.research.components.slot_attention import MultiHeadSlotAttention
from survot_rank.research.methods.ot_event_hazard_v2.model_v2 import (
    cosine_cost,
    log_sinkhorn_plan,
)
from survot_rank.training.model_factory import get_model


def offdiag_cosine(slots: torch.Tensor) -> float:
    """slot 两两余弦相似度均值（排除对角）。接近 1 表示塌缩。"""
    normalized = F.normalize(slots, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    k = similarity.size(1)
    mask = ~torch.eye(k, dtype=torch.bool, device=similarity.device)
    return similarity[:, mask].mean().item()


def plan_deviation(plan: torch.Tensor) -> float:
    """OT plan 相对均匀分布的总变差。0 = 完全均匀 = 运输不携带信息。"""
    uniform = 1.0 / (plan.size(-1) * plan.size(-2))
    return (plan - uniform).abs().sum(dim=(-2, -1)).mean().item() / 2.0


def normalize_cost(cost: torch.Tensor) -> torch.Tensor:
    cost = cost - cost.amin(dim=(1, 2), keepdim=True)
    return cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)


def legacy_prototype_pooling(
    tokens: torch.Tensor, prototypes: torch.Tensor, temperature: float
) -> torch.Tensor:
    """复现 DCT v3.3 `_semantic_slots` 的 shared-prototype 池化。"""
    scores = torch.einsum(
        "bnd,kd->bkn", F.normalize(tokens, dim=-1), F.normalize(prototypes, dim=-1)
    )
    assignment = torch.softmax(scores / temperature, dim=1)
    weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bkn,bnd->bkd", weights, tokens)


def center_slots(slots: torch.Tensor) -> torch.Tensor:
    return slots - slots.mean(dim=1, keepdim=True)


def diagnose_pooling(dim: int = 256, num_slots: int = 8) -> None:
    """诊断一：shared-prototype 池化本身是否塌缩，以及三种修法的效果。"""
    print("=" * 78)
    print("诊断一：slot 池化方式对运输信息量的影响")
    print("=" * 78)
    torch.manual_seed(0)
    patches = torch.randn(8, 512, dim)
    pathways = torch.randn(8, 331, dim)
    prototypes_wsi = torch.randn(num_slots, dim) * 0.02
    prototypes_omic = torch.randn(num_slots, dim) * 0.02

    for init_mode in ("gaussian", "learned"):
        wsi = MultiHeadSlotAttention(
            num_slots=num_slots, dim=dim, heads=8, iters=3, init_mode=init_mode
        ).eval()
        omic = MultiHeadSlotAttention(
            num_slots=num_slots, dim=dim, heads=8, iters=3, init_mode=init_mode
        ).eval()
        with torch.no_grad():
            local_w, local_o = wsi(patches), omic(pathways)
            variants = (
                (
                    "v3.3 shared-prototype 池化 (T=0.30)",
                    legacy_prototype_pooling(local_w, prototypes_wsi, 0.30),
                    legacy_prototype_pooling(local_o, prototypes_omic, 0.30),
                ),
                ("删掉池化（纯 local slot）", local_w, local_o),
                (
                    "删掉池化 + 跨 slot 中心化 (v3.9)",
                    center_slots(local_w),
                    center_slots(local_o),
                ),
            )
            print(f"\n  slot_init = {init_mode}")
            for label, slots_w, slots_o in variants:
                cost = normalize_cost(cosine_cost(slots_w, slots_o))
                plan, _ = log_sinkhorn_plan(cost, eps=0.05, max_iter=40)
                print(
                    f"    {label:<38} slot余弦={offdiag_cosine(slots_w):+.4f}  "
                    f"plan偏离均匀={plan_deviation(plan):.4f}"
                )
    print(
        f"\n  参考：K={num_slots} 时中心化后余弦的理论值（各 slot 范数相等）"
        f" = {-1.0 / (num_slots - 1):+.4f}"
    )


def _base_args(**overrides) -> SimpleNamespace:
    values = dict(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=1024,
        wsi_projection_dim=256,
        rna_format="RNASeq",
        slot_num_wsi=8,
        slot_num_omics=8,
        slot_iters=3,
        otehv2_eps=0.05,
        otehv2_iter=40,
        otehv2_heads=4,
        otehv2_layers=2,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.10,
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.50,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=0,
        dct_lambda_etar=0.0,
        dct_anchor_momentum=0.90,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_coupling_projection_iters=1000,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.50,
        dct_slot_init_mode="gaussian",
        dct_slot_eval_seed=1729,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.20,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.10,
        rg_eps_anneal=12,
        cur_epoch=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(batch_size: int = 16) -> dict:
    torch.manual_seed(0)
    half = batch_size // 2
    return dict(
        x_wsi=torch.randn(batch_size, 512, 1024),
        x_omics=torch.randn(batch_size, 331, 1024),
        y=torch.randint(0, 4, (batch_size,)),
        c=torch.tensor([0.0] * half + [1.0] * (batch_size - half)),
        event_time=torch.arange(1.0, batch_size + 1.0),
        cur_epoch=0,
    )


def diagnose_methods() -> None:
    """诊断二：端到端对比已注册方法的运输信息量与温度标定健康度。"""
    print("\n" + "=" * 78)
    print("诊断二：端到端方法对比")
    print("=" * 78)
    batch = _batch()

    candidates = (
        ("v3.3 DCT", "distributional_counterfactual_transport", {}),
        ("v3.9 RST", "dct_v39_risk_simplex_transport", {}),
        (
            "v3.9 关闭中心化",
            "dct_v39_risk_simplex_transport",
            {"dct_v39_center_slots": False},
        ),
    )
    for label, method, overrides in candidates:
        model = get_model(method, _base_args(**overrides), omic_input_dim=1024)
        model.configure_train_reference(batch["event_time"], batch["c"])
        model.train()
        model(**batch)  # 填充队列锚点（v3.9 还会触发温度标定）
        model(**batch)

        with torch.no_grad():
            slots_wsi, slots_omic, _, _ = model._encode_transport_slots(
                model.wsi_mlp(batch["x_wsi"]), model._encode_omics(batch), batch
            )
            costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
            plans, _ = model._plans_from_cost_tensor(costs, rows, cols, 0)
            deviations = [
                plan_deviation(plan) for stage in plans for plan in stage
            ]
            model.eval()
            logits, _ = model(**batch)
            risk_spread = model._risk(logits).std().item()

        parameters = sum(p.numel() for p in model.parameters()) / 1e6
        extra = ""
        coordinate = getattr(model, "_last_lambda", None)
        if coordinate is not None:
            extra = (
                f"  lambda_std={coordinate.std():.4f}"
                f"  tau={model.v39_log_tau.exp().item():.5f}"
            )
        print(
            f"  {label:<16} 参数={parameters:5.2f}M  "
            f"plan偏离均匀={sum(deviations) / len(deviations):.4f}  "
            f"风险离散度={risk_spread:.4f}{extra}"
        )

    print(
        "\n  判读：plan 偏离均匀 < 0.10 说明运输机制几乎不携带患者特异信息，"
        "\n  此时任何加在运输上的损失或干预都会静默失效。"
        "\n  v3.9 还应监控 lambda_std：持续 < 0.02 说明温度过大或锚点未分化。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["all", "pooling", "methods"],
        default="all",
        help="只运行其中一项诊断。",
    )
    args = parser.parse_args()
    if args.section in {"all", "pooling"}:
        diagnose_pooling()
    if args.section in {"all", "methods"}:
        diagnose_methods()


if __name__ == "__main__":
    main()
