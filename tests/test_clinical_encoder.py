"""ClinicalEncoder 单元测试。

对应需求文档（requirements.md）需求 1（临床模态编码器与三模态融合架构）：
- AC1: 输出形状为 [batch, dim]，dim 等于传入的 wsi_projection_dim。
- AC5: 输入含 NaN 占位符时前向不产生 NaN/Inf，输出形状与不含 NaN 时一致。
- AC6: 对 batch>=1、clinical_feature_dim>=1、dim>=1 的合法取值组合，反向传播为
  所有可训练参数产生非 NaN/Inf 梯度。
"""

import pytest
import torch

from survot_rank.research.components.clinical_encoder import ClinicalEncoder

# 用于多组合法取值组合覆盖的参数网格
BATCH_SIZES = [1, 2, 4]
CLINICAL_FEATURE_DIMS = [1, 3, 5]
DIMS = [1, 4, 8]


def _combos():
    """生成 (batch, clinical_feature_dim, dim) 三元组合。"""
    combos = []
    for batch in BATCH_SIZES:
        for cdim in CLINICAL_FEATURE_DIMS:
            for dim in DIMS:
                combos.append((batch, cdim, dim))
    return combos


@pytest.mark.parametrize("batch,clinical_feature_dim,dim", _combos())
def test_output_shape_no_nan(batch, clinical_feature_dim, dim):
    """需求1 AC1：输出形状为 [batch, dim]，dim 等于传入的 wsi_projection_dim。"""
    torch.manual_seed(0)
    model = ClinicalEncoder(clinical_feature_dim=clinical_feature_dim, dim=dim)
    model.eval()  # 关闭 dropout 随机性，聚焦形状/数值有效性断言
    x = torch.randn(batch, clinical_feature_dim)

    out = model(x)

    assert out.shape == (batch, dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("batch,clinical_feature_dim,dim", _combos())
def test_output_finite_with_nan_placeholder_and_shape_matches(
    batch, clinical_feature_dim, dim
):
    """需求1 AC5：输入含 NaN 占位符时前向不产生 NaN/Inf，输出形状与不含 NaN 时一致。"""
    torch.manual_seed(0)
    model = ClinicalEncoder(clinical_feature_dim=clinical_feature_dim, dim=dim)
    model.eval()

    x_clean = torch.randn(batch, clinical_feature_dim)
    out_clean = model(x_clean)

    # 构造含 NaN 占位符的输入：每个样本至少将第一个特征置为 NaN
    x_nan = x_clean.clone()
    x_nan[:, 0] = float("nan")

    out_nan = model(x_nan)

    assert out_nan.shape == out_clean.shape == (batch, dim)
    assert torch.isfinite(out_nan).all()


@pytest.mark.parametrize("batch,clinical_feature_dim,dim", _combos())
def test_backward_produces_finite_gradients_for_all_trainable_params(
    batch, clinical_feature_dim, dim
):
    """需求1 AC6：反向传播为所有可训练参数产生非 NaN、非 Inf 的梯度。

    使用含 NaN 占位符的输入，确保 self.impute 参数也参与梯度计算路径。
    """
    torch.manual_seed(0)
    model = ClinicalEncoder(clinical_feature_dim=clinical_feature_dim, dim=dim)
    model.train()

    x = torch.randn(batch, clinical_feature_dim)
    # 至少一个特征列含 NaN，触发 impute 参数的使用
    x[:, 0] = float("nan")

    out = model(x)
    out.sum().backward()

    params = list(model.named_parameters())
    assert len(params) > 0
    for name, param in params:
        assert param.grad is not None, f"参数 {name} 未产生梯度"
        assert torch.isfinite(param.grad).all(), f"参数 {name} 的梯度包含 NaN/Inf"
