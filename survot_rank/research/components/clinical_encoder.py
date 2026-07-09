"""临床模态（Clinical_Modality）编码器组件。

对应 requirements.md 需求 1（临床模态编码器与三模态融合架构）与
design.md Step C 的设计：`ClinicalEncoder` 把形状为
`[batch, clinical_feature_dim]` 的临床特征张量映射为形状为
`[batch, dim]` 的表示，其中 `dim` 与 WSI/Omics 的投影维度一致。
"""

import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    """临床特征编码器：``[batch, clinical_feature_dim] -> [batch, dim]``。

    缺失值约定用 NaN 占位符表示（而不是 0，避免 0 恰好是合法取值时产生
    歧义）。前向传播时先用可学习的填充向量 ``self.impute`` 替换 NaN，
    再经过 ``LayerNorm -> Linear -> GELU -> Dropout -> Linear`` 得到输出。

    Args:
        clinical_feature_dim: 输入临床特征的维度。
        dim: 输出表示的维度（与 WSI/Omics 投影维度一致）。
        dropout: Dropout 概率，默认 0.1。
    """

    def __init__(self, clinical_feature_dim, dim, dropout=0.1):
        super().__init__()
        # 可学习的缺失值填充向量，形状为 [clinical_feature_dim]
        self.impute = nn.Parameter(torch.zeros(clinical_feature_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(clinical_feature_dim),
            nn.Linear(clinical_feature_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x: 形状为 ``[batch, clinical_feature_dim]`` 的张量，缺失值用
                NaN 占位符表示。

        Returns:
            形状为 ``[batch, dim]`` 的张量。
        """
        # 用可学习填充向量替换 NaN 占位符缺失值
        mask = torch.isnan(x)
        x = torch.where(mask, self.impute.expand_as(x), x)
        return self.net(x)
