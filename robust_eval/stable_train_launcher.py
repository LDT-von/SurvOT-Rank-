#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定训练启动器：在**不修改任何源码**的前提下，为现有训练器叠加两项能力。

要解决的问题
------------
诊断显示当前训练极不稳定（验证 C-index 末期在 0.35~0.55 大幅抖动），
根因之一是多辅助损失 + 重型 Transformer + 小数据下缺乏梯度约束，且单 seed
结论不可复现。本启动器叠加：

1. 梯度裁剪 (grad clipping)
   通过运行时包装 ``train_runner.init_optimizer`` 返回的优化器，在每次
   ``optimizer.step()`` 前对全部参数做 ``clip_grad_norm_``。这是抑制训练
   抖动最直接有效的手段，且完全不侵入源码。

2. 多 seed 循环
   同一 config 用多个 seed 依次训练，每个 seed 写到独立
   ``<results_dir>/seed<k>/`` 子目录。配合 ``honest_report.py`` 就能得到
   跨 seed 的稳健 mean ± std，而不是单 seed 的运气结果。

实现方式全部是运行时注入（monkey-patch + 直接调用 ``run``），
``survot_rank`` 包内的文件一行都不改。

用法
----
    python robust_eval/stable_train_launcher.py \
        --config configs/v45v2_blca_clinical.yaml \
        --seeds 3 5 7 --grad-clip 1.0 --gpu 0

    # 单参数覆盖（透传给现有 config 机制）
    python robust_eval/stable_train_launcher.py \
        --config configs/rank_guided_event_transport_blca.yaml \
        --seeds 1 2 3 --set batch_size=16 --set max_epochs=40
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

# 让本文件既能作为脚本运行，也能被导入：把项目根加入 sys.path。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _install_grad_clip_patch() -> None:
    """包装 train_runner.init_optimizer，为其返回的优化器注入梯度裁剪。

    优化器的 ``step`` 被替换为「先 clip_grad_norm_ 再原始 step」。裁剪阈值从
    ``args.grad_clip_norm`` 读取（由启动器在解析后手动挂到 args 上，因此不需要
    改 extended_args）。阈值 <=0 时不裁剪，行为与原始一致。
    """
    import torch
    import survot_rank.training.train_runner as tr

    if getattr(tr, "_stable_grad_clip_installed", False):
        return

    original_init_optimizer = tr.init_optimizer

    def init_optimizer_with_clip(args, model):
        optimizer = original_init_optimizer(args, model)
        max_norm = float(getattr(args, "grad_clip_norm", 0.0) or 0.0)
        if max_norm > 0.0:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            original_step = optimizer.step

            def step_with_clip(*step_args, **step_kwargs):
                torch.nn.utils.clip_grad_norm_(params, max_norm)
                return original_step(*step_args, **step_kwargs)

            optimizer.step = step_with_clip
            print(f"[stable] gradient clipping enabled: max_norm={max_norm}")
        return optimizer

    tr.init_optimizer = init_optimizer_with_clip
    tr._stable_grad_clip_installed = True


def _build_parsed_args(config_path: str, overrides: list[str]):
    """复用现有 YAML -> argv -> argparse 机制，得到一个 args namespace。"""
    from survot_rank.config import apply_overrides, config_to_argv, load_config
    from survot_rank.project import add_project_paths
    from survot_rank.training.extended_args import process_args_extended

    add_project_paths()
    config = load_config(config_path)
    config = apply_overrides(config, overrides or [])
    argv = config_to_argv(config)
    return process_args_extended(argv)


def run_multi_seed(
    config_path: str,
    seeds: list[int],
    grad_clip: float,
    gpu: str,
    overrides: list[str],
) -> list[str]:
    """对给定 config 跑多个 seed，返回每个 seed 的结果目录。"""
    _install_grad_clip_patch()
    import survot_rank.training.train_runner as tr

    base_parsed = _build_parsed_args(config_path, overrides)
    base_results_dir = base_parsed.results_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    seed_dirs: list[str] = []
    for seed in seeds:
        parsed = copy.deepcopy(base_parsed)
        parsed.seed = int(seed)
        parsed.gpu = str(gpu)
        parsed.grad_clip_norm = float(grad_clip)
        parsed.results_dir = os.path.join(base_results_dir, f"seed{seed}")
        os.makedirs(parsed.results_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"[stable] seed={seed}  grad_clip={grad_clip}  -> {parsed.results_dir}")
        print("=" * 70)
        tr.run(parsed)
        seed_dirs.append(parsed.results_dir)

    print("\n[stable] 全部 seed 完成。结果目录：")
    for d in seed_dirs:
        print("  ", d)
    print(
        "\n[stable] 下一步用诚实汇总：\n"
        f"  python robust_eval/honest_report.py --dirs {base_results_dir} "
        "--strategy last_k_mean"
    )
    return seed_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="不改源码地为 SurvOT-Rank 叠加梯度裁剪与多 seed 训练"
    )
    parser.add_argument("--config", required=True, help="YAML 实验配置路径")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[3, 5, 7], help="要跑的 seed 列表"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="梯度裁剪 max_norm，<=0 关闭（默认 1.0）",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides",
        help="透传的单参数覆盖，如 --set batch_size=16",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_multi_seed(
        config_path=args.config,
        seeds=args.seeds,
        grad_clip=args.grad_clip,
        gpu=args.gpu,
        overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
