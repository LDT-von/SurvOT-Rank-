#!/usr/bin/env python3
"""
gen_multicancer_configs.py —— 生成多癌种验证配置文件

以 `configs/v45_blca.yaml` 为模板，为 `STUDIES` 中的每个癌种生成一份
`configs/v45_{study}.yaml`，仅替换 `data.study` 与 `train.results_dir`
两个字段，其余字段（包括 `data.data_root_dir`、`data.data_path`）逐字段
保持与模板完全一致。

对应 Requirement 2 AC1/AC2（详见
.kiro/specs/survot-rank-enhancements/requirements.md）。

用法：
    python tools/gen_multicancer_configs.py
（从仓库根目录运行，直接写入 configs/ 目录下的 5 个文件）
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# 仓库根目录（本文件位于 tools/ 下，父目录的父目录即仓库根目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
TEMPLATE_PATH = CONFIG_DIR / "v45_blca.yaml"

# 按字母序排列，对应需求2 AC7 的行序约束（生成文件顺序与汇总行序保持一致的习惯）
STUDIES = ["blca", "brca", "coadread", "hnsc", "stad"]


def load_template(template_path: Path) -> dict:
    """读取模板 YAML 配置文件为字典。"""
    with open(template_path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"模板配置文件格式错误，期望为映射类型: {template_path}")
    return data


def build_config_for_study(template: dict, study: str) -> dict:
    """基于模板生成指定癌种的配置字典，仅替换 data.study 与 train.results_dir。"""
    # 深拷贝：模板中的嵌套字典（data/split/train/slot/model）都需要独立副本，
    # 避免多次调用之间相互污染。
    import copy

    config = copy.deepcopy(template)

    if "data" not in config or "study" not in config["data"]:
        raise ValueError("模板配置缺少 data.study 字段，无法替换")
    if "train" not in config or "results_dir" not in config["train"]:
        raise ValueError("模板配置缺少 train.results_dir 字段，无法替换")

    template_study = template["data"]["study"]
    template_results_dir = template["train"]["results_dir"]

    # 按模板中 results_dir 的命名模式（例如 results/v45_blca）替换癌种部分，
    # 保证仅 study 对应的那一段变化，其余路径结构保持一致。
    if template_study not in template_results_dir:
        raise ValueError(
            f"模板 train.results_dir='{template_results_dir}' 中未出现 "
            f"data.study='{template_study}'，无法安全推导癌种专属路径"
        )
    new_results_dir = template_results_dir.replace(template_study, study)

    config["data"]["study"] = study
    config["train"]["results_dir"] = new_results_dir

    # 注意：除 data.study 与 train.results_dir 外，其余字段（包括 name、
    # description）必须与模板逐字段保持一致（需求2 AC2），因此这里不对
    # name/description 做任何改动。
    return config


def dump_config(config: dict, output_path: Path) -> None:
    """将配置字典写出为 YAML 文件，保持字段插入顺序（不按字母排序）。"""
    with open(output_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(
            config,
            file_obj,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def main() -> int:
    if not TEMPLATE_PATH.exists():
        print(f"错误：模板配置文件不存在: {TEMPLATE_PATH}", file=sys.stderr)
        return 1

    template = load_template(TEMPLATE_PATH)

    for study in STUDIES:
        config = build_config_for_study(template, study)
        output_path = CONFIG_DIR / f"v45_{study}.yaml"
        dump_config(config, output_path)
        print(f"已生成: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
