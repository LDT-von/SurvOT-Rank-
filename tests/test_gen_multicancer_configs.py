"""
tests/test_gen_multicancer_configs.py

针对 tools/gen_multicancer_configs.py 的 pytest 测试。

覆盖点：
1. 对 STUDIES 中的每个癌种，build_config_for_study(template, study) 生成的
   配置字典仅 data.study 与 train.results_dir 与模板不同，其余字段逐字段一致。
2. 使用 pytest tmp_path fixture，配合 dump_config 将生成的配置写出为
   v45_{study}.yaml，再用 yaml.safe_load 重新读取，验证往返（round-trip）
   结果与内存中的配置字典一致。

注意：所有写文件操作都限定在 tmp_path 下，不会触碰真实的 configs/ 目录。
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tools.gen_multicancer_configs import (
    STUDIES,
    build_config_for_study,
    dump_config,
    load_template,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "configs" / "v45_blca.yaml"


@pytest.fixture(scope="module")
def template() -> dict:
    """加载真实模板配置（只读，不修改）。"""
    return load_template(TEMPLATE_PATH)


def _diff_keys(a: dict, b: dict, prefix: str = "") -> set:
    """递归比较两个字典，返回值不同的“路径”集合，例如 {"data.study"}。"""
    diffs: set = set()
    keys = set(a.keys()) | set(b.keys())
    for key in keys:
        path = f"{prefix}.{key}" if prefix else key
        a_val = a.get(key, object())
        b_val = b.get(key, object())
        if isinstance(a_val, dict) and isinstance(b_val, dict):
            diffs |= _diff_keys(a_val, b_val, path)
        else:
            if a_val != b_val:
                diffs.add(path)
    return diffs


@pytest.mark.parametrize("study", STUDIES)
def test_build_config_only_changes_study_and_results_dir(template: dict, study: str) -> None:
    """build_config_for_study 只应改变 data.study 与 train.results_dir。"""
    template_copy = copy.deepcopy(template)  # 防止被测函数意外修改模板
    generated = build_config_for_study(template_copy, study)

    diffs = _diff_keys(template, generated)

    # 只允许 data.study 与 train.results_dir 两个字段不同；当 study 恰好等于
    # 模板自身的癌种（blca）时，两者也可能与模板相同，diffs 为空集也合法。
    assert diffs.issubset({"data.study", "train.results_dir"})
    assert generated["data"]["study"] == study
    assert template["data"]["study"] in template["train"]["results_dir"]
    expected_results_dir = template["train"]["results_dir"].replace(
        template["data"]["study"], study
    )
    assert generated["train"]["results_dir"] == expected_results_dir


def test_template_is_not_mutated_by_build_config(template: dict) -> None:
    """确保多次调用 build_config_for_study 不会污染模板本身。"""
    template_before = copy.deepcopy(template)
    for study in STUDIES:
        build_config_for_study(template, study)
    assert template == template_before


@pytest.mark.parametrize("study", STUDIES)
def test_dump_and_reload_round_trip(template: dict, study: str, tmp_path: Path) -> None:
    """dump_config 写出的 YAML 文件应能被 yaml.safe_load 完整还原。"""
    generated = build_config_for_study(copy.deepcopy(template), study)

    output_path = tmp_path / f"v45_{study}.yaml"
    dump_config(generated, output_path)

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file_obj:
        reloaded = yaml.safe_load(file_obj)

    assert reloaded == generated
    assert reloaded["data"]["study"] == study


def test_dump_config_does_not_touch_real_configs_dir(template: dict, tmp_path: Path) -> None:
    """回归保护：确认测试过程中未在真实 configs/ 目录下新增任何文件。

    v45_blca.yaml 本身就是模板文件，测试前后都应存在，因此这里比较的是
    “测试前后 configs/ 目录内的文件集合是否完全一致”，而不是简单判断
    某个文件是否存在。
    """
    real_configs_dir = REPO_ROOT / "configs"
    files_before = set(real_configs_dir.iterdir()) if real_configs_dir.exists() else set()

    for study in STUDIES:
        generated = build_config_for_study(copy.deepcopy(template), study)
        dump_config(generated, tmp_path / f"v45_{study}.yaml")

    files_after = set(real_configs_dir.iterdir()) if real_configs_dir.exists() else set()
    assert files_before == files_after
