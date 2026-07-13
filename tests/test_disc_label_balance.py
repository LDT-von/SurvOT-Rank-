#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SurvivalDatasetFactory._disc_label 的回归测试。

历史 bug：原实现先用 `pd.qcut` 在未删失病人身上算出正确的等频分位数边界，但
紧接着又用 `pd.cut`（等宽分箱）对全部病人重新分箱并覆盖掉前者，导致 qcut 的
计算完全是死代码。等宽分箱在右删失比例高、时间分布长尾的生存数据上会把绝大多
数病人塞进同一个箱（历史上 BLCA 的真实分布是 {0:310, 1:50, 2:16, 3:4}，箱3
仅占全体的 1.1%），5-fold 切分后某些折几乎分不到该类别的样本，导致 C-index 在
不同 fold 间出现系统性的大幅波动（"前三折还行，后两折崩"）。

本测试用合成小数据集验证修复：
1. `_disc_label` 使用的分箱边界应来自未删失病人的等频分位数（而不是全体病人的
   等宽分箱），因此各类别样本数应大致均衡，不应出现某一类别占比 <5% 的极端情况。
2. 所有病人（含删失）都应被分配到合法类别（0 ~ n_bins-1），不产生 NaN。
3. 用真实 BLCA 数据验证类别分布不再是历史上的病态分布（箱3占比 <5%）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_SLOTSPE_RUNTIME = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "survot_rank", "research", "legacy", "slotspe_runtime",
)
if _SLOTSPE_RUNTIME not in sys.path:
    sys.path.insert(0, _SLOTSPE_RUNTIME)

from dataset.dataset_survival import SurvivalDatasetFactory  # noqa: E402


class _FakeFactory(SurvivalDatasetFactory):
    """跳过真实文件 IO（基因/信号通路/临床 CSV），只测 `_disc_label` 的分箱逻辑。"""

    def __init__(self, clinical_df: pd.DataFrame, censorship_var: str, label_col: str,
                 n_bins: int = 4, eps: float = 1e-6):
        self.study = "synthetic"
        self.data_path = ""
        self.signature = "all"
        self.rna_format = "RNASeq"
        self.n_bins = n_bins
        self.num_patches = None
        self.num_genes = None
        self.eps = eps
        self.label_col = label_col
        self.clinical_feature_cols = None
        self.use_clinical_modality = False
        self.censorship_var = censorship_var

        self.clinical_df = clinical_df.reset_index(drop=True)
        uncensored_df = self._get_uncensored_data()
        self._disc_label(uncensored_df)


def _make_synthetic_survival_df(n=380, censor_rate=0.66, seed=0) -> pd.DataFrame:
    """构造一个右删失比例高、时间分布长尾的合成生存数据集（模拟 TCGA 场景）。"""
    rng = np.random.RandomState(seed)
    # 指数分布制造长尾（多数病人早期删失/早期事件，少数病人存活很久）。
    times = rng.exponential(scale=15.0, size=n)
    censor = (rng.rand(n) < censor_rate).astype(int)  # 1=censored, 0=event
    df = pd.DataFrame({
        "case id": [f"case_{i}" for i in range(n)],
        "wsi": [f"slide_{i}" for i in range(n)],
        "survival_months_dss": times,
        "censorship_dss": censor,
    })
    return df


class TestDiscLabelBalance:
    def test_no_nan_after_binning(self):
        df = _make_synthetic_survival_df()
        factory = _FakeFactory(df, censorship_var="censorship_dss", label_col="survival_months_dss")
        assert factory.clinical_df["label"].isna().sum() == 0

    def test_fold_bins_use_training_cases_only(self):
        df = pd.DataFrame({
            "case id": ["train_0", "train_1", "train_2", "train_3", "val_tail"],
            "wsi": ["a", "b", "c", "d", "e"],
            "survival_months_dss": [1.0, 2.0, 3.0, 4.0, 1000.0],
            "censorship_dss": [0, 0, 0, 0, 1],
        })
        factory = _FakeFactory(df, censorship_var="censorship_dss", label_col="survival_months_dss")
        factory.fit_label_bins(["train_0", "train_1", "train_2", "train_3"])

        assert factory.bins[-1] == np.inf
        assert factory.clinical_df.loc[4, "label"] == factory.n_bins - 1
        assert np.allclose(factory.bins[1:-1], [1.75, 2.5, 3.25])

    def test_all_labels_within_valid_range(self):
        df = _make_synthetic_survival_df()
        factory = _FakeFactory(df, censorship_var="censorship_dss", label_col="survival_months_dss")
        labels = factory.clinical_df["label"]
        assert labels.min() >= 0
        assert labels.max() <= factory.n_bins - 1

    def test_label_distribution_is_not_degenerate(self):
        # 回归测试核心断言：修复前，长尾右删失数据会让等宽分箱把 >80% 病人塞进
        # 第一个箱、最后一个箱占比 <2%。修复后应恢复到与未删失病人等频分箱
        # 大致一致的均衡分布：每个箱至少占全体 10%。
        df = _make_synthetic_survival_df(n=380, censor_rate=0.66, seed=0)
        factory = _FakeFactory(df, censorship_var="censorship_dss", label_col="survival_months_dss")
        counts = factory.clinical_df["label"].value_counts()
        min_frac = counts.min() / len(factory.clinical_df)
        assert min_frac >= 0.10, (
            f"最小类别占比 {min_frac:.3f} 过低，疑似退化为历史等宽分箱 bug: "
            f"{counts.to_dict()}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_stable_across_random_seeds(self, seed):
        df = _make_synthetic_survival_df(n=500, censor_rate=0.7, seed=seed)
        factory = _FakeFactory(df, censorship_var="censorship_dss", label_col="survival_months_dss")
        counts = factory.clinical_df["label"].value_counts()
        assert len(counts) == factory.n_bins, "应产生全部 n_bins 个类别，不应有类别完全为空"
        min_frac = counts.min() / len(factory.clinical_df)
        assert min_frac >= 0.05


@pytest.mark.parametrize("study", ["blca", "brca", "coadread", "hnsc", "stad"])
def test_real_tcga_studies_label_distribution_not_degenerate(study):
    """用真实 TCGA 临床 CSV 验证修复后的分布不再是历史病态分布（某箱占比 <5%）。

    若 dataset_csv 中缺少对应 study 的临床文件（例如精简过的测试环境），跳过。
    """
    data_path = os.path.join(_SLOTSPE_RUNTIME, "dataset_csv")
    clinical_csv = os.path.join(data_path, "clinical", "all", f"{study}.csv")
    if not os.path.isfile(clinical_csv):
        pytest.skip(f"未找到 {clinical_csv}，跳过真实数据回归测试")

    factory = SurvivalDatasetFactory(
        study=study,
        data_path=data_path,
        rna_format="Pathways",
        label_col="survival_months_dss",
        signature="combine",
        n_bins=4,
        num_patches=2048,
    )
    counts = factory.clinical_df["label"].value_counts()
    assert factory.clinical_df["label"].isna().sum() == 0
    min_frac = counts.min() / len(factory.clinical_df)
    assert min_frac >= 0.05, (
        f"{study} 最小类别占比 {min_frac:.3f} 过低（历史 bug 曾产生 <2% 的病态占比）: "
        f"{counts.to_dict()}"
    )
