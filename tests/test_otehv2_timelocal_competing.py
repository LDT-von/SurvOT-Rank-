#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OTEHTimeLocalCompeting（V50）的单元测试。

V50 原为独立文件夹 `50_otehv2_timelocal_competing/` 中的游离实现（单独训练、
单独出结果），现合并进主项目 `prognostic_event_transport/model.py` 并接入
`METHOD_REGISTRY`（方法名 `otehv2_timelocal_competing`，别名 `50`）统一管理。

本测试验证：
1. 前向输出形状正确，训练模式下 aux_loss 有限，反向传播产生非 NaN/Inf 梯度。
2. eval 模式下 aux_loss 恒为 0.0（与 V45/V45v2 约定一致）。
3. `beta_protect` 经 softplus 后非负（保护通路竞争强度约束）。
4. 时间局部特化/覆盖正则、竞争稳定正则均为有限标量。
5. `METHOD_REGISTRY`/`METHOD_ALIASES` 正确注册，`get_model` 能构建出该模型。

使用 `rna_format="RNASeq"` 简化组学编码路径，所有张量为小尺寸合成数据，不依赖
GPU、不依赖真实 WSI `.pt` 特征文件。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from survot_rank.research.methods.prognostic_event_transport.model import (
    OTEHTimeLocalCompeting,
)
from survot_rank.training.model_factory import METHOD_ALIASES, METHOD_REGISTRY, get_model


class _Args:
    """一个简单的可变属性对象，用作合成的 args Namespace。"""


def make_args(**overrides) -> _Args:
    args = _Args()
    args.omic_sizes = None
    args.n_classes = 4
    args.encoding_dim = 16
    args.wsi_projection_dim = 16
    args.rna_format = "RNASeq"
    args.slot_num_wsi = 3
    args.slot_num_omics = 3
    args.slot_iters = 2
    args.otehv2_num_events = 4
    args.otehv2_heads = 2
    args.otehv2_layers = 1
    args.otehv2_dropout = 0.1
    args.otehv2_eps = 0.05
    args.otehv2_iter = 5
    args.otehv2_warmup = 0
    args.lambda_otehv2_ot = 0.0
    args.lambda_otehv2_div = 0.0
    args.lambda_otehv2_event_surv = 0.0
    args.lambda_otehv2_recon = 0.0
    args.alpha_surv = 0.0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


DIM = 16
OMIC_INPUT_DIM = 20
NUM_PATCHES = 6
NUM_OMIC_TOKENS = 5


def make_inputs(batch: int):
    x_wsi = torch.randn(batch, NUM_PATCHES, DIM)
    x_omics = torch.randn(batch, NUM_OMIC_TOKENS, OMIC_INPUT_DIM)
    y = torch.randint(0, 4, (batch,)).long()
    c = torch.randint(0, 2, (batch,)).float()
    return {"x_wsi": x_wsi, "x_omics": x_omics, "y": y, "c": c}


class TestOTEHTimeLocalCompetingForward:
    def test_forward_shape_and_backward_finite_grads(self):
        torch.manual_seed(0)
        args = make_args()
        model = OTEHTimeLocalCompeting(args, omic_input_dim=OMIC_INPUT_DIM)
        model.train()

        inputs = make_inputs(batch=2)
        logits, aux_loss = model(**inputs)

        assert logits.shape == (2, 4)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux_loss)

        (logits.sum() + aux_loss).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"

    def test_eval_mode_returns_zero_aux_loss(self):
        torch.manual_seed(0)
        args = make_args()
        model = OTEHTimeLocalCompeting(args, omic_input_dim=OMIC_INPUT_DIM)
        model.eval()

        inputs = make_inputs(batch=2)
        with torch.no_grad():
            logits, aux_loss = model(**inputs)

        assert logits.shape == (2, 4)
        assert torch.isfinite(logits).all()
        assert aux_loss == 0.0

    def test_beta_protect_is_nonnegative_after_softplus(self):
        torch.manual_seed(0)
        args = make_args(compete_beta_init=-2.0)
        model = OTEHTimeLocalCompeting(args, omic_input_dim=OMIC_INPUT_DIM)
        beta = F.softplus(model.beta_protect)
        assert beta.item() >= 0.0

    def test_timelocal_regularizers_are_finite_scalars(self):
        torch.manual_seed(0)
        args = make_args()
        model = OTEHTimeLocalCompeting(args, omic_input_dim=OMIC_INPUT_DIM)
        model.train()

        inputs = make_inputs(batch=3)
        x_wsi_proj = model.wsi_mlp(inputs["x_wsi"])
        x_omics = model._encode_omics(inputs)
        slots_wsi = model.slot_attention_wsi(x_wsi_proj)
        slots_omic = model.slot_attention_omic(x_omics)

        import survot_rank.research.methods.prognostic_event_transport.model as model_module

        cost_cos = model_module._parent.cosine_cost(slots_wsi, slots_omic)
        plan_cos, _ = model_module._parent.log_sinkhorn_plan(cost_cos, eps=model.ot_eps, max_iter=model.ot_iter)
        cost_euc = model_module._parent.euclidean_cost(slots_wsi, slots_omic)
        plan_euc, _ = model_module._parent.log_sinkhorn_plan(cost_euc, eps=model.ot_eps, max_iter=model.ot_iter)
        cost_dot = model_module._parent.dot_cost(slots_wsi, slots_omic)
        plan_dot, _ = model_module._parent.log_sinkhorn_plan(cost_dot, eps=model.ot_eps, max_iter=model.ot_iter)

        event_tokens, _ = model.fusion(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot)
        event_tokens = model.event_norm(model.event_encoder(event_tokens))

        _, _, _, _, extra = model._make_timelocal_logits(event_tokens)
        for name in ("spec", "cover", "compete"):
            assert torch.isfinite(extra[name]), f"{name} 正则项非有限"
            assert extra[name].dim() == 0


class TestMethodRegistration:
    def test_registered_in_method_registry(self):
        assert "otehv2_timelocal_competing" in METHOD_REGISTRY
        method_dir, cls_name = METHOD_REGISTRY["otehv2_timelocal_competing"]
        assert cls_name == "OTEHTimeLocalCompeting"

    def test_alias_50_resolves_correctly(self):
        assert METHOD_ALIASES.get("50") == "otehv2_timelocal_competing"

    def test_get_model_builds_instance_via_alias(self):
        # get_model 通过 importlib 动态从磁盘重新加载模块，构建出的类对象与本文件
        # 顶部 import 的 OTEHTimeLocalCompeting 并非同一个 Python 类对象（尽管是
        # 同一份源码），因此按类名字符串比较而非 isinstance（其余方法测试同理）。
        torch.manual_seed(0)
        args = make_args()
        model = get_model("50", args, omic_input_dim=OMIC_INPUT_DIM)
        assert type(model).__name__ == "OTEHTimeLocalCompeting"
        logits, aux_loss = model(**make_inputs(batch=2))
        assert logits.shape == (2, 4)

    def test_existing_entries_not_modified(self):
        # 需求6 AC5 同类约束：新增条目不应影响/修改已有注册条目。
        assert METHOD_REGISTRY["otehv2_rankevent"][1] == "OTEHV2RankEvent"
        assert METHOD_REGISTRY["otehv2_rankevent_v2"][1] == "OTEHV2RankEventV2"
        assert METHOD_REGISTRY["ot_event_hazard_v2"][1] == "OTEventHazardV2Survival"
