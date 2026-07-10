#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/gen_ablation_configs.py 的单元测试。

验证：
1. 生成的每个消融配置都切到 otehv2_rankevent_v2，且 results_dir / specific_simple 唯一。
2. abl_00_baseline 不含任何新能力开关（纯 V45 基线）。
3. 单能力消融只开启对应的一个开关。
4. 临床消融自动同步 clinical_feature_cols 与 otehv2v2_clinical_feature_dim。
5. 除被消融的开关与输出目录外，其余字段与模板逐字一致（对照公平）。
"""

import os

import yaml

from tools.gen_ablation_configs import (
    ABLATION_MATRIX,
    DEFAULT_CLINICAL_COLS,
    generate,
)


TEMPLATE = {
    "name": "v45_blca",
    "data": {
        "study": "blca",
        "data_root_dir": "/data/CPathPatchFeature",
        "num_patches": 2048,
    },
    "split": {"k_start": 0, "k_end": 5},
    "train": {
        "survot_method": "otehv2_rankevent",
        "max_epochs": 30,
        "seed": 3,
        "lr": 0.0005,
    },
    "model": {
        "otehv2_num_events": 24,
        "lambda_rankevent_rank": 0.15,
    },
}


def _write_template(tmp_path):
    path = os.path.join(tmp_path, "template.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(TEMPLATE, fh, allow_unicode=True, sort_keys=False)
    return path


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_generates_all_ablations(tmp_path):
    template_path = _write_template(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "ablation")
    written = generate(template_path, out_dir, DEFAULT_CLINICAL_COLS)
    assert len(written) == len(ABLATION_MATRIX)
    # 文件名按字母序。
    names = [os.path.basename(p) for p in written]
    assert names == sorted(names)


def test_all_use_v2_method_and_unique_dirs(tmp_path):
    template_path = _write_template(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "ablation")
    written = generate(template_path, out_dir, DEFAULT_CLINICAL_COLS)
    seen_dirs = set()
    for path in written:
        cfg = _load(path)
        assert cfg["train"]["survot_method"] == "otehv2_rankevent_v2"
        rd = cfg["train"]["results_dir"]
        assert rd not in seen_dirs, f"results_dir 重复: {rd}"
        seen_dirs.add(rd)


def test_baseline_has_no_new_flags(tmp_path):
    template_path = _write_template(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "ablation")
    generate(template_path, out_dir, DEFAULT_CLINICAL_COLS)
    cfg = _load(os.path.join(out_dir, "abl_00_baseline.yaml"))
    model = cfg["model"]
    for key in model:
        assert not key.startswith("otehv2v2_"), f"基线不应包含新能力开关 {key}"


def test_single_feature_ablation_enables_only_one(tmp_path):
    template_path = _write_template(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "ablation")
    generate(template_path, out_dir, DEFAULT_CLINICAL_COLS)
    cfg = _load(os.path.join(out_dir, "abl_02_unified.yaml"))
    model = cfg["model"]
    assert model.get("otehv2v2_use_unified_objective") is True
    # 其他新开关不应出现（未启用则不写入）。
    assert "otehv2v2_slot_disentangled" not in model
    assert "otehv2v2_use_clinical" not in model


def test_clinical_ablation_syncs_feature_dim(tmp_path):
    template_path = _write_template(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "ablation")
    generate(template_path, out_dir, DEFAULT_CLINICAL_COLS)
    cfg = _load(os.path.join(out_dir, "abl_01_clinical.yaml"))
    assert cfg["data"]["clinical_feature_cols"] == DEFAULT_CLINICAL_COLS
    assert cfg["model"]["otehv2v2_clinical_feature_dim"] == len(DEFAULT_CLINICAL_COLS)


def test_non_ablated_fields_match_template(tmp_path):
    template_path = _write_template(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "ablation")
    generate(template_path, out_dir, DEFAULT_CLINICAL_COLS)
    cfg = _load(os.path.join(out_dir, "abl_00_baseline.yaml"))
    # 数据与 slot/model 的原有字段应与模板一致。
    assert cfg["data"]["num_patches"] == TEMPLATE["data"]["num_patches"]
    assert cfg["data"]["data_root_dir"] == TEMPLATE["data"]["data_root_dir"]
    assert cfg["model"]["otehv2_num_events"] == TEMPLATE["model"]["otehv2_num_events"]
    assert cfg["split"] == TEMPLATE["split"]
