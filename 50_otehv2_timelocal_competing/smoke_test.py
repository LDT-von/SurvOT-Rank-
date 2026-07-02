#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成数据冒烟测试：验证 V50 时间局部竞争事件模型前向/反向不报错。

不需要真实数据、不需要 GPU。用法：
    python smoke_test.py

预期输出：
    PASS timelocal_competing (standalone): logits=(2, 4) aux_loss=... eval_ok=True
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from args import process_args
from model import OTEHTimeLocalCompeting


def build_args():
    args = process_args([])
    args.rna_format = "Pathways"
    args.n_classes = 4
    args.encoding_dim = 1024
    args.wsi_projection_dim = 256
    args.slot_num_wsi = 8
    args.slot_num_omics = 8
    args.slot_iters = 3
    args.omic_sizes = [40, 60, 80]   # 3 条通路，维度各不同
    args.omic_names = None
    args.pathway_names = None
    return args


def make_batch(args, bsz=2, num_patches=64):
    dim = args.encoding_dim
    kwargs = {
        "x_wsi": torch.randn(bsz, num_patches, dim),
        "y": torch.randint(0, args.n_classes, (bsz,)),
        "c": torch.randint(0, 2, (bsz,)).float(),
        "cur_epoch": 3,
    }
    for i, sz in enumerate(args.omic_sizes, start=1):
        kwargs[f"x_omic{i}"] = torch.randn(bsz, sz)
    return kwargs


def main():
    torch.manual_seed(0)
    args = build_args()
    model = OTEHTimeLocalCompeting(args)
    kwargs = make_batch(args)

    # train 路径：前向 + 反向
    model.train()
    logits, aux = model(**kwargs)
    assert logits.shape == (2, args.n_classes), f"logits shape {logits.shape}"
    assert torch.is_tensor(aux) and aux.requires_grad, "aux_loss 应为可反向张量"
    (logits.pow(2).mean() + aux).backward()

    # eval 路径
    model.eval()
    with torch.no_grad():
        logits_eval, aux_eval = model(**kwargs)
    assert logits_eval.shape == (2, args.n_classes)
    assert aux_eval == 0.0

    print(f"PASS timelocal_competing (standalone): logits={tuple(logits.shape)} "
          f"aux_loss={float(aux):.4f} eval_ok=True")


if __name__ == "__main__":
    main()
