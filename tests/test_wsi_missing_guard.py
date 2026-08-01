"""WSI 特征缺失守卫（on_missing_wsi）的行为验证。

默认 error：缺特征时拒绝运行，防止零填充静默污染结果（例如 BRCA 在 UNI2-h
下仅 74% 覆盖，271 名患者会被零填充）。zero：保留旧的零填充行为。
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
    SurvivalDataset,
)


def _make_dataset(tmp_path, present_slides, label_slides, mode):
    """构造一个绕过 __init__ 文件依赖的最小 SurvivalDataset 实例。"""
    for slide_id in present_slides:
        torch.save(torch.zeros((4, 8)), tmp_path / f"{slide_id}.pt")

    ds = object.__new__(SurvivalDataset)
    ds.wsi_path = str(tmp_path)
    ds.encoding_dim = 8
    ds.on_missing_wsi = mode
    ds._wsi_feature_index = None
    ds.split_key = "train"
    ds.fold = 0
    ds.dataset_factory = SimpleNamespace(study="blca", num_patches=4)
    ds.label_df = pd.DataFrame(
        {
            "case id": [f"P{i}" for i in range(len(label_slides))],
            "wsi": label_slides,
        }
    )
    return ds


def test_preflight_raises_on_missing(tmp_path):
    ds = _make_dataset(
        tmp_path,
        present_slides=["TCGA-AA-0001"],
        label_slides=["TCGA-AA-0001", "TCGA-BB-0002"],  # 第二个缺特征
        mode="error",
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        ds._preflight_wsi_features()
    message = str(excinfo.value)
    assert "1/2" in message  # 2 名患者中 1 名缺失
    assert "TCGA-BB-0002" in message


def test_preflight_passes_when_complete(tmp_path):
    ds = _make_dataset(
        tmp_path,
        present_slides=["TCGA-AA-0001", "TCGA-BB-0002"],
        label_slides=["TCGA-AA-0001", "TCGA-BB-0002"],
        mode="error",
    )
    ds._preflight_wsi_features()  # 不应抛异常


def test_nan_wsi_is_not_treated_as_missing(tmp_path):
    """wsi=nan 表示该患者本就无 WSI 模态，属合法情况，不算缺失。"""
    ds = _make_dataset(
        tmp_path,
        present_slides=["TCGA-AA-0001"],
        label_slides=["TCGA-AA-0001", "nan"],
        mode="error",
    )
    ds._preflight_wsi_features()  # 不应抛异常


def test_load_wsi_error_mode_raises(tmp_path):
    ds = _make_dataset(
        tmp_path, present_slides=[], label_slides=["TCGA-AA-0001"], mode="error"
    )
    with pytest.raises(FileNotFoundError):
        ds.load_wsi("TCGA-CC-9999")


def test_load_wsi_zero_mode_fills_zeros(tmp_path):
    ds = _make_dataset(
        tmp_path, present_slides=[], label_slides=["TCGA-AA-0001"], mode="zero"
    )
    wsi = ds.load_wsi("TCGA-CC-9999")
    assert wsi.shape == (4, 8)
    assert bool((wsi == 0).all())


def test_invalid_mode_rejected(tmp_path):
    """__init__ 在读取任何 split 文件之前就拒绝非法的 on_missing_wsi 取值。"""
    factory = SimpleNamespace(study="blca")
    with pytest.raises(ValueError, match="on_missing_wsi"):
        SurvivalDataset(
            factory, str(tmp_path), "train", 0, 8, on_missing_wsi="bogus"
        )
