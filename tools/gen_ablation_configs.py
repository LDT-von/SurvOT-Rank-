#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 OTEHV2RankEventV2 的系统消融实验配置。

背景
----
`OTEHV2RankEventV2`（V45v2）在 V45（`OTEHV2RankEvent`，C-index 0.7105）基础上引入了
多项可选能力：临床三模态融合、统一生存目标、slot 身份/状态解耦、Sinkhorn 路由、
跨模态条件化、自适应迭代，以及可学习自适应损失加权。历史经验（V44）表明**一次性
叠加多个改动会导致损失项互相打架、协同崩塌**（0.7105 → 0.6760）。因此正确的做法是
**逐个隔离验证**每个能力单独开启时的效果，再决定哪些组合值得保留。

本脚本以一个真实数据训练配置（默认 `configs/v45_blca.yaml`）为模板，生成一套消融
配置，写到 `configs/ablation/` 下。每个配置只改动 `survot_method`（切到 v2 子类）、
`specific_simple`/`results_dir`（区分输出目录）以及被消融的那几个新能力开关，其余
字段与模板逐字保持一致，保证对照公平。

消融矩阵
--------
- ``abl_00_baseline``：所有新能力关闭（等价 V45，作为复现/对照基线，目标 ~0.7105）
- ``abl_01_clinical``：仅临床三模态融合
- ``abl_02_unified``：仅统一生存目标
- ``abl_03_disentangle``：仅 slot 身份/状态解耦
- ``abl_04_sinkhorn``：仅 Sinkhorn 路由
- ``abl_05_crossmodal``：仅跨模态条件化
- ``abl_06_adaptive_iters``：仅自适应迭代次数
- ``abl_07_learnable_weights``：仅可学习自适应损失加权（基线 + 该开关）
- ``abl_08_all_on``：全部能力开启（复现“全开”场景，对照 V44 教训）
- ``abl_09_all_on_learnable``：全部能力开启 + 可学习损失加权（检验加权能否救回全开）

用法
----
    python tools/gen_ablation_configs.py \
        --template configs/v45_blca.yaml \
        --out-dir configs/ablation

生成后在服务器上逐个运行，例如：
    python -m survot_rank.cli train --config configs/ablation/abl_00_baseline.yaml
再用 tools/aggregate_cross_cancer.py 或自有脚本汇总各消融的 summary.csv 对比 C-index。
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict, List

import yaml


# 临床特征列（与 configs/smoke_local_v45v2_full_enhancements.yaml 保持一致）。
# 若你的临床 CSV 用了不同列名，用 --clinical-cols 覆盖，脚本会自动同步
# otehv2v2_clinical_feature_dim。
DEFAULT_CLINICAL_COLS = [
    "age_at_diagnosis",
    "pathologic_stage",
    "histological_grade",
]

# 消融名称 -> 该消融相对基线要额外开启的新能力开关（写入 model: 段）。
# 空 dict 表示基线（所有新能力关闭）。
ABLATION_MATRIX: Dict[str, Dict[str, Any]] = {
    "abl_00_baseline": {},
    "abl_01_clinical": {"otehv2v2_use_clinical": True},
    "abl_02_unified": {"otehv2v2_use_unified_objective": True},
    "abl_03_disentangle": {"otehv2v2_slot_disentangled": True},
    "abl_04_sinkhorn": {"otehv2v2_slot_router": "sinkhorn"},
    "abl_05_crossmodal": {"otehv2v2_slot_cross_modal_cond": True},
    "abl_06_adaptive_iters": {"otehv2v2_slot_adaptive_iters": True},
    "abl_07_learnable_weights": {"otehv2v2_learnable_loss_weights": True},
    "abl_08_all_on": {
        "otehv2v2_use_clinical": True,
        "otehv2v2_use_unified_objective": True,
        "otehv2v2_slot_disentangled": True,
        "otehv2v2_slot_router": "sinkhorn",
        "otehv2v2_slot_cross_modal_cond": True,
        "otehv2v2_slot_adaptive_iters": True,
    },
    "abl_09_all_on_learnable": {
        "otehv2v2_use_clinical": True,
        "otehv2v2_use_unified_objective": True,
        "otehv2v2_slot_disentangled": True,
        "otehv2v2_slot_router": "sinkhorn",
        "otehv2v2_slot_cross_modal_cond": True,
        "otehv2v2_slot_adaptive_iters": True,
        "otehv2v2_learnable_loss_weights": True,
    },
}


def _requires_clinical(features: Dict[str, Any]) -> bool:
    return bool(features.get("otehv2v2_use_clinical", False))


def build_ablation_config(
    template: Dict[str, Any],
    name: str,
    features: Dict[str, Any],
    clinical_cols: List[str],
) -> Dict[str, Any]:
    """基于模板构造单个消融配置（深拷贝，不修改模板）。"""
    cfg = copy.deepcopy(template)

    cfg["name"] = name
    cfg["description"] = (
        f"Ablation '{name}' for OTEHV2RankEventV2. "
        f"Extra enabled capabilities relative to V45 baseline: "
        f"{features if features else '(none — pure V45 baseline)'}"
    )

    train = cfg.setdefault("train", {})
    # 切换到 v2 子类；基线消融同样走 v2 子类（新开关全关闭时数值等价 V45），
    # 这样 10 个消融走的是同一份代码路径，只有开关不同，对照最干净。
    train["survot_method"] = "otehv2_rankevent_v2"
    train["specific_simple"] = name
    train["results_dir"] = f"results/ablation/{name}"

    model = cfg.setdefault("model", {})
    # 先清掉模板里可能残留的所有新能力字段，保证“基线=全关闭”。
    for key in (
        "otehv2v2_use_clinical",
        "otehv2v2_clinical_feature_dim",
        "otehv2v2_num_slots_clinical",
        "otehv2v2_use_unified_objective",
        "otehv2v2_slot_disentangled",
        "otehv2v2_slot_router",
        "otehv2v2_slot_cross_modal_cond",
        "otehv2v2_slot_adaptive_iters",
        "otehv2v2_learnable_loss_weights",
    ):
        model.pop(key, None)

    # 写入本消融要开启的能力。
    for key, value in features.items():
        model[key] = value

    # 临床模态需要同步声明特征列与维度。
    if _requires_clinical(features):
        data = cfg.setdefault("data", {})
        data["clinical_feature_cols"] = list(clinical_cols)
        model["otehv2v2_clinical_feature_dim"] = len(clinical_cols)
        model["otehv2v2_num_slots_clinical"] = 4

    return cfg


def generate(template_path: str, out_dir: str, clinical_cols: List[str]) -> List[str]:
    """生成全部消融配置文件，返回写出的文件路径列表（按消融名称字母序）。"""
    with open(template_path, "r", encoding="utf-8") as fh:
        template = yaml.safe_load(fh) or {}
    if not isinstance(template, dict):
        raise ValueError(f"模板配置必须是映射类型: {template_path}")

    os.makedirs(out_dir, exist_ok=True)

    written: List[str] = []
    for name in sorted(ABLATION_MATRIX.keys()):
        features = ABLATION_MATRIX[name]
        cfg = build_ablation_config(template, name, features, clinical_cols)
        out_path = os.path.join(out_dir, f"{name}.yaml")
        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        written.append(out_path)
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default=os.path.join("configs", "v45_blca.yaml"),
        help="作为模板的真实数据训练配置（默认 configs/v45_blca.yaml）",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("configs", "ablation"),
        help="消融配置输出目录（默认 configs/ablation）",
    )
    parser.add_argument(
        "--clinical-cols",
        nargs="+",
        default=DEFAULT_CLINICAL_COLS,
        help="临床特征列名（用于临床消融），维度自动同步",
    )
    args = parser.parse_args(argv)

    written = generate(args.template, args.out_dir, args.clinical_cols)
    print(f"已生成 {len(written)} 个消融配置到 {args.out_dir}：")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
