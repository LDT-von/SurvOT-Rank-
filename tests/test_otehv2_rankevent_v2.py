"""OTEHV2RankEventV2 的单元测试。

覆盖任务 8.4（骨架向后兼容性）、9.5（Clinical 分支与三模态融合）、
11.3（跨能力组合集成测试），对应
`.kiro/specs/survot-rank-enhancements/requirements.md` 需求 1/3/4/5/6。

使用 `rna_format="RNASeq"` 简化组学编码路径（避免构造 Pathways 签名数据），
所有张量维度均为小尺寸合成数据，不依赖 GPU、不依赖真实 WSI `.pt` 特征文件。
"""

from __future__ import annotations

import time

import pytest
import torch

from survot_rank.research.methods.prognostic_event_transport.model import (
    OTEHV2RankEvent,
    OTEHV2RankEventV2,
)


class _Args:
    """一个简单的可变属性对象，用作合成的 args Namespace。"""


def make_args(**overrides) -> _Args:
    args = _Args()
    # ---> OTEventHazardV2Survival / OTEHV2RankEvent 所需的基础字段
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


def make_inputs(batch: int, clinical_feature_dim: int | None = None, requires_grad_clinical: bool = False):
    x_wsi = torch.randn(batch, NUM_PATCHES, DIM)
    x_omics = torch.randn(batch, NUM_OMIC_TOKENS, OMIC_INPUT_DIM)
    y = torch.randint(0, 4, (batch,)).long()
    c = torch.randint(0, 2, (batch,)).float()
    kwargs = {"x_wsi": x_wsi, "x_omics": x_omics, "y": y, "c": c}
    if clinical_feature_dim is not None:
        x_clinical = torch.randn(batch, clinical_feature_dim)
        if requires_grad_clinical:
            x_clinical.requires_grad_(True)
        kwargs["x_clinical"] = x_clinical
    return kwargs


# ---------------------------------------------------------------------------
# 任务 8.4：骨架向后兼容性
# ---------------------------------------------------------------------------


class TestSkeletonBackwardCompatibility:
    def test_logits_match_parent_when_all_new_flags_off(self):
        # 需求6 AC7：新开关全部关闭时，OTEHV2RankEventV2 与 OTEHV2RankEvent
        # 在相同种子/相同输入下输出逐元素误差不超过 1e-5。
        batch = 2
        torch.manual_seed(0)
        args_v1 = make_args()
        model_v1 = OTEHV2RankEvent(args_v1, omic_input_dim=OMIC_INPUT_DIM)
        model_v1.train()

        torch.manual_seed(0)
        args_v2 = make_args()
        model_v2 = OTEHV2RankEventV2(args_v2, omic_input_dim=OMIC_INPUT_DIM)
        model_v2.train()

        torch.manual_seed(123)
        inputs = make_inputs(batch)

        torch.manual_seed(999)
        logits_v1, _ = model_v1(**inputs)
        torch.manual_seed(999)
        logits_v2, _ = model_v2(**inputs)

        assert torch.allclose(logits_v1, logits_v2, atol=1e-5)

    def test_state_dict_keys_and_shapes_identical(self):
        # 需求6 AC4：新开关全部关闭时，两者 state_dict 键集合与参数形状完全一致。
        torch.manual_seed(0)
        model_v1 = OTEHV2RankEvent(make_args(), omic_input_dim=OMIC_INPUT_DIM)
        torch.manual_seed(0)
        model_v2 = OTEHV2RankEventV2(make_args(), omic_input_dim=OMIC_INPUT_DIM)

        sd1 = model_v1.state_dict()
        sd2 = model_v2.state_dict()

        assert set(sd1.keys()) == set(sd2.keys())
        for key in sd1:
            assert sd1[key].shape == sd2[key].shape, f"参数 {key} 形状不一致"

    def test_state_dict_cross_loadable_with_strict_true(self):
        # 需求6 AC4：新开关全部关闭时，OTEHV2RankEventV2 可用 strict=True
        # 加载 OTEHV2RankEvent 的 state_dict，且不出现形状不匹配/缺失/多余键错误。
        torch.manual_seed(0)
        model_v1 = OTEHV2RankEvent(make_args(), omic_input_dim=OMIC_INPUT_DIM)
        torch.manual_seed(1)
        model_v2 = OTEHV2RankEventV2(make_args(), omic_input_dim=OMIC_INPUT_DIM)

        # 加载前两者权重不同（不同种子）；加载后应完全一致，且不抛异常。
        model_v2.load_state_dict(model_v1.state_dict(), strict=True)
        for (name1, p1), (name2, p2) in zip(
            model_v1.named_parameters(), model_v2.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# 任务 9.5：Clinical 分支与三模态融合
# ---------------------------------------------------------------------------


class TestClinicalBranch:
    def test_missing_x_clinical_raises_value_error(self):
        # 需求1 AC3：三模态开关启用但未提供 x_clinical -> 抛出明确异常。
        torch.manual_seed(0)
        args = make_args(otehv2v2_use_clinical=True, otehv2v2_clinical_feature_dim=5)
        model = OTEHV2RankEventV2(args, omic_input_dim=OMIC_INPUT_DIM)
        model.train()

        inputs = make_inputs(batch=2)  # 不含 x_clinical
        with pytest.raises(ValueError, match="Clinical"):
            model(**inputs)

    def test_wrong_shape_x_clinical_raises_value_error(self):
        # 需求1 AC3：x_clinical 形状与配置声明不一致 -> 抛出明确异常。
        torch.manual_seed(0)
        args = make_args(otehv2v2_use_clinical=True, otehv2v2_clinical_feature_dim=5)
        model = OTEHV2RankEventV2(args, omic_input_dim=OMIC_INPUT_DIM)
        model.train()

        inputs = make_inputs(batch=2)
        inputs["x_clinical"] = torch.randn(2, 3)  # 错误维度（期望 5）
        with pytest.raises(ValueError, match="Clinical"):
            model(**inputs)

    def test_clinical_disabled_no_clinical_modules_in_state_dict(self):
        # 需求1 AC4：关闭时不实例化 Clinical 相关模块，state_dict 中不含相关键。
        torch.manual_seed(0)
        args = make_args()  # otehv2v2_use_clinical 默认 False
        model = OTEHV2RankEventV2(args, omic_input_dim=OMIC_INPUT_DIM)

        keys = list(model.state_dict().keys())
        assert not any("clinical_encoder" in k for k in keys)
        assert not any("slot_attention_clinical" in k for k in keys)
        assert not any("three_way_fusion" in k for k in keys)

    def test_clinical_enabled_training_forward_succeeds(self):
        # 需求1 AC2/AC7：启用三模态融合，训练模式下前向成功，logits/aux_loss
        # 形状正确、无 NaN/Inf。
        torch.manual_seed(0)
        args = make_args(otehv2v2_use_clinical=True, otehv2v2_clinical_feature_dim=5)
        model = OTEHV2RankEventV2(args, omic_input_dim=OMIC_INPUT_DIM)
        model.train()

        inputs = make_inputs(batch=2, clinical_feature_dim=5)
        logits, aux_loss = model(**inputs)

        assert logits.shape == (2, 4)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux_loss)

    def test_clinical_enabled_eval_mode_returns_zero_aux_loss(self):
        # eval 模式下不计算 aux_loss，返回 (logits, 0.0)。
        torch.manual_seed(0)
        args = make_args(otehv2v2_use_clinical=True, otehv2v2_clinical_feature_dim=5)
        model = OTEHV2RankEventV2(args, omic_input_dim=OMIC_INPUT_DIM)
        model.eval()

        inputs = make_inputs(batch=2, clinical_feature_dim=5)
        with torch.no_grad():
            logits, aux_loss = model(**inputs)

        assert logits.shape == (2, 4)
        assert torch.isfinite(logits).all()
        assert aux_loss == 0.0


# ---------------------------------------------------------------------------
# 任务 11.3：跨能力组合集成测试
# ---------------------------------------------------------------------------


class TestFullCapabilityIntegration:
    @pytest.mark.parametrize("batch", [1, 2])
    def test_all_capabilities_enabled_forward_backward(self, batch):
        # 需求6 AC1/AC6：启用全部新增能力（clinical + unified objective +
        # slot 解耦 + sinkhorn 路由 + 跨模态条件化 + 自适应迭代）后，完整前向
        # + 损失计算 + 反向传播在 10 秒内完成，logits 形状正确、无 NaN/Inf。
        torch.manual_seed(0)
        args = make_args(
            otehv2v2_use_clinical=True,
            otehv2v2_clinical_feature_dim=5,
            otehv2v2_num_slots_clinical=3,
            otehv2v2_use_unified_objective=True,
            lambda_unified_rank=0.15,
            otehv2v2_slot_disentangled=True,
            otehv2v2_slot_router="sinkhorn",
            otehv2v2_sinkhorn_max_iters=10,
            otehv2v2_slot_cross_modal_cond=True,
            otehv2v2_slot_adaptive_iters=True,
            otehv2v2_convergence_threshold=0.0,
        )

        start = time.time()

        model = OTEHV2RankEventV2(args, omic_input_dim=OMIC_INPUT_DIM)
        model.train()

        inputs = make_inputs(batch=batch, clinical_feature_dim=5, requires_grad_clinical=True)
        logits, aux_loss = model(**inputs)

        assert logits.shape == (batch, 4)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux_loss)

        total_loss = logits.sum() + aux_loss
        total_loss.backward()

        elapsed = time.time() - start
        assert elapsed < 10.0, f"完整前向+反向耗时 {elapsed:.2f}s，超过 10 秒预算"

        # 需求6 AC2：反向传播后，Clinical 输入张量所接收梯度的 L2 范数应大于 0，
        # 证明梯度确实经由 slot attention 路由与统一目标回传到 Clinical 输入。
        x_clinical = inputs["x_clinical"]
        assert x_clinical.grad is not None
        assert x_clinical.grad.norm(p=2).item() > 0.0
