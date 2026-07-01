#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@file: paths.py（独立版本）
@desc: SurvOT-Rank 完全独立版本，不需要外部 SlotSPE 目录。

   所有依赖文件（models/、utils/、dataset/、dataset_csv/）已复制到本文件夹内。
   本脚本会优先查找本文件夹内的依赖。
"""

import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_slotspe_in_path():
    """把本文件夹的 models/、utils/、dataset/ 加到 sys.path"""
    # 本文件夹根目录
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)


def get_slotspe_dir():
    """返回本文件夹路径"""
    return THIS_DIR
