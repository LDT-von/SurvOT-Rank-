#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@file: paths.py（50_otehv2_timelocal_competing 独立方法文件夹）
@desc: 本文件夹只放"方法代码"（backbone / model / args / train），
       共享基础设施（models/、utils/、dataset/、dataset_csv/）复用兄弟目录
       SurvOT-Rank- 基座，不再复制一份。

       查找优先级：
         1. 环境变量 SURVOT_BASE（用户显式指定）
         2. 兄弟目录 SurvOT-Rank-
         3. 兄弟目录 SurvOT-Rank
         4. 本文件夹内部（如果把 models/ 等也拷进来）

       解析成功后会把 本文件夹 与 基座目录 都加入 sys.path：
         - 本文件夹在前 -> import backbone / model_v45 / model 命中本文件夹自己的代码
         - 基座目录在后 -> from models.xxx / utils.xxx / dataset.xxx 命中基座共享代码
"""

import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_base_dir():
    """按优先级查找 SurvOT-Rank- 基座目录（内含 models/utils/dataset/dataset_csv）。"""
    env = os.environ.get("SURVOT_BASE")
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    parent = os.path.dirname(THIS_DIR)  # 上一级目录
    candidates = [
        parent,  # 嵌套布局：本方法是基座仓库的子文件夹，基座即直接父目录
        os.path.join(parent, "SurvOT-Rank-"),
        os.path.join(parent, "SurvOT-Rank"),
        THIS_DIR,  # 兜底：基础设施被拷进本文件夹内部
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "models")) and \
           os.path.isfile(os.path.join(c, "models", "slot_attention.py")):
            return os.path.abspath(c)
    return None


BASE_DIR = _find_base_dir()


def ensure_slotspe_in_path():
    """把 基座目录 与 本文件夹 加入 sys.path（本文件夹优先）。"""
    if BASE_DIR is None:
        raise FileNotFoundError(
            "找不到 SurvOT-Rank- 基座目录（需内含 models/utils/dataset/dataset_csv）。\n"
            "请通过以下方式之一指定:\n"
            "  1. 设置环境变量 SURVOT_BASE=/path/to/SurvOT-Rank-\n"
            "  2. 把本方法文件夹与 SurvOT-Rank- 放到同一父目录下\n"
            "     例如: SurvOT-Rank/50_otehv2_timelocal_competing/  和  SurvOT-Rank/SurvOT-Rank-/"
        )
    # 基座先入，本文件夹后入 -> 本文件夹在 sys.path 更靠前，方法代码优先命中
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)


def get_base_dir():
    return BASE_DIR
