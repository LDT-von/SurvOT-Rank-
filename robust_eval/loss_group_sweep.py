#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""损失子集扫描器：约束「同时激活的辅助损失只能是 3 个或 4 个」并逐组合训练。

动机
----
V45(8 项)/ V45v2(默认 8)/ V50(11 项)的辅助损失太多、梯度互相拉扯，导致训练集都
拟合不了、fold 间崩溃、难复现。与其手工猜 8~11 个 lambda，不如**系统枚举**：
每次只放开 3 个或 4 个损失（权重可变），其余强制置 0，各跑一次，看哪一小组损失
组合最稳/最好。禁止 5 个及以上同时激活。

它怎么工作（不改任何模型源码）
------------------------------
完全通过现有 CLI 的 `--set key=value` 覆盖机制驱动：对方法的**全部**辅助损失
lambda 逐一设值——组合内的设为其默认权重（各不相同=“权重可变”），组合外的设为
0。外层的 NLL 生存损失（bag_loss=nll_surv）始终在，因此每个组合都有预测监督。

    组合 {ot, rank, stage} -> --set lambda_ot=0.06 --set lambda_rank=0.15 ...
                              其余 lambda_* 全部 --set =0

默认只跑 fold2（最难折，用作快速筛子；结论需再上 5-fold 确认——见 honest_report）。

用法
----
    # 先看计划有多少组合、命令长什么样（不真正训练）
    python robust_eval/loss_group_sweep.py --method otehv2_rankevent --dry-run

    # V45：枚举 3 组和 4 组，fold2，每组 30ep
    python robust_eval/loss_group_sweep.py --method otehv2_rankevent \
        --python /home/ubuntu/.conda/envs/trisurv/bin/python --gpu 0

    # V50（11 项，组合很多），只枚举 3 组、缩短到 15ep 做初筛
    python robust_eval/loss_group_sweep.py --method otehv2_timelocal_competing \
        --sizes 3 --epochs 15 --python .../python --gpu 0

    # 让权重也变：每个激活项在默认值上乘 {0.5,1,2}（组合数会暴涨，需 --yes 确认）
    python robust_eval/loss_group_sweep.py --method otehv2_rankevent \
        --weight-grid 0.5,1,2 --yes --python .../python
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# 每个方法：辅助损失名 -> (对应的 lambda CLI 参数名, 默认权重)。
# 默认权重取自 extended_args / 各 config，"权重可变"即指组合内各项用这些不同的默认值
# （也可用 --weight-grid 在其上再乘系数）。
LOSS_REGISTRY: dict[str, dict[str, tuple[str, float]]] = {
    # V45：OTEHV2RankEvent.forward 的 8 个辅助项
    "otehv2_rankevent": {
        "ot": ("lambda_otehv2_ot", 0.06),
        "div": ("lambda_otehv2_div", 0.01),
        "recon": ("lambda_otehv2_recon", 0.20),
        "gate_ent": ("lambda_rankevent_gate_ent", 0.005),
        "event_surv": ("lambda_otehv2_event_surv", 0.25),
        "per_event": ("lambda_rankevent_per_event", 0.15),
        "rank": ("lambda_rankevent_rank", 0.15),
        "global_cons": ("lambda_rankevent_global_cons", 0.02),
    },
    # V50：与 V45 相同 8 项 + spec/cover/compete = 11 项
    "otehv2_timelocal_competing": {
        "ot": ("lambda_otehv2_ot", 0.06),
        "div": ("lambda_otehv2_div", 0.01),
        "recon": ("lambda_otehv2_recon", 0.20),
        "gate_ent": ("lambda_rankevent_gate_ent", 0.005),
        "event_surv": ("lambda_otehv2_event_surv", 0.25),
        "per_event": ("lambda_rankevent_per_event", 0.15),
        "rank": ("lambda_rankevent_rank", 0.15),
        "global_cons": ("lambda_rankevent_global_cons", 0.02),
        "spec": ("lambda_timelocal_spec", 0.01),
        "cover": ("lambda_timelocal_cover", 0.01),
        "compete": ("lambda_compete_reg", 0.001),
    },
}
# V45v2 默认路径与 V45 完全一致（未开 clinical/unified/learnable 时委托父类）。
LOSS_REGISTRY["otehv2_rankevent_v2"] = dict(LOSS_REGISTRY["otehv2_rankevent"])

DEFAULT_CONFIG = {
    "otehv2_rankevent": "configs/v45_blca.yaml",
    "otehv2_rankevent_v2": "configs/v45_blca.yaml",
    "otehv2_timelocal_competing": "configs/v50_blca.yaml",
}


def build_run_plan(method: str, sizes, weight_grid=None):
    """枚举 size 属于 sizes 的所有损失子集，返回每个组合的 override 字典列表。

    weight_grid: None 或系数列表。None -> 每组合一次(默认权重)。给列表 -> 组合内
    每个激活项在默认权重上遍历这些系数的笛卡尔积。
    """
    if method not in LOSS_REGISTRY:
        raise KeyError(f"未知方法 {method!r}，可选: {list(LOSS_REGISTRY)}")
    registry = LOSS_REGISTRY[method]
    all_losses = list(registry.keys())

    for s in sizes:
        if s >= 5:
            raise ValueError(f"不允许同时激活 5 个及以上损失（收到 size={s}）")
        if s < 1:
            raise ValueError(f"size 必须 >=1（收到 {s}）")

    plans = []
    for size in sizes:
        for combo in itertools.combinations(all_losses, size):
            if weight_grid:
                factor_sets = itertools.product(weight_grid, repeat=size)
            else:
                factor_sets = [tuple(1.0 for _ in combo)]
            for factors in factor_sets:
                # 全部 lambda：组合内=默认*系数，组合外=0
                lambda_overrides = {}
                for name in all_losses:
                    arg, default = registry[name]
                    if name in combo:
                        idx = combo.index(name)
                        lambda_overrides[arg] = round(default * factors[idx], 6)
                    else:
                        lambda_overrides[arg] = 0.0
                if weight_grid:
                    tag = "+".join(f"{n}x{f:g}" for n, f in zip(combo, factors))
                else:
                    tag = "+".join(combo)
                combo_id = f"n{size}_{tag}"
                plans.append({
                    "size": size,
                    "losses": list(combo),
                    "combo_id": combo_id,
                    "lambda_overrides": lambda_overrides,
                })
    return plans


def build_command(plan, method, config, python, gpu, seed, epochs, fold, results_root):
    results_dir = f"{results_root}/{method}/{plan['combo_id']}"
    cmd = [
        python, "-m", "survot_rank.cli", "train",
        "--config", config,
        "--set", f"survot_method={method}",
        "--set", f"gpu={gpu}",
        "--set", "num_workers=0",
        "--set", f"max_epochs={epochs}",
        "--set", f"seed={seed}",
        "--set", f"k_start={fold}",
        "--set", f"k_end={fold + 1}",
        "--set", f"results_dir={results_dir}",
        "--set", f"specific_simple=lsweep_{plan['combo_id']}",
    ]
    for arg, value in plan["lambda_overrides"].items():
        cmd += ["--set", f"{arg}={value}"]
    return cmd, results_dir


def main():
    parser = argparse.ArgumentParser(
        description="损失子集扫描器：只放开 3 或 4 个辅助损失，逐组合训练"
    )
    parser.add_argument("--method", required=True, choices=list(LOSS_REGISTRY))
    parser.add_argument("--config", default=None, help="不填则用该方法默认 config")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4],
                        help="要枚举的激活损失个数（默认 3 4；禁止 >=5）")
    parser.add_argument("--weight-grid", default=None,
                        help="逗号分隔的权重系数，如 '0.5,1,2'；不填=每组合一次用默认权重")
    parser.add_argument("--fold", type=int, default=2, help="用哪一折做筛选（默认 fold2）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--python", default=sys.executable, help="训练用的 Python 解释器")
    parser.add_argument("--results-root", default="results/loss_sweep")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划与命令，不训练")
    parser.add_argument("--yes", action="store_true", help="组合数很大时跳过确认")
    parser.add_argument("--manifest", default=None, help="清单 CSV 路径（默认写到 results-root 下）")
    args = parser.parse_args()

    config = args.config or DEFAULT_CONFIG[args.method]
    weight_grid = None
    if args.weight_grid:
        weight_grid = [float(x) for x in args.weight_grid.split(",") if x.strip()]

    plans = build_run_plan(args.method, args.sizes, weight_grid)
    print(f"[sweep] method={args.method} sizes={args.sizes} "
          f"weight_grid={weight_grid} -> 共 {len(plans)} 个组合运行")
    print(f"[sweep] config={config} fold={args.fold} epochs={args.epochs} seed={args.seed}")

    manifest_path = args.manifest or os.path.join(args.results_root, f"manifest_{args.method}.csv")

    if args.dry_run:
        for p in plans[:5]:
            cmd, _ = build_command(p, args.method, config, args.python,
                                   args.gpu, args.seed, args.epochs, args.fold, args.results_root)
            print("  例:", p["combo_id"])
            print("     ", " ".join(cmd))
        if len(plans) > 5:
            print(f"  ... 其余 {len(plans) - 5} 个组合省略")
        print(f"[sweep] dry-run 结束（未训练）。清单将写到 {manifest_path}")
        return

    if len(plans) > 200 and not args.yes:
        raise SystemExit(
            f"[sweep] 计划 {len(plans)} 个运行，数量较大。确认请加 --yes，"
            f"或用 --sizes / --weight-grid 收窄。"
        )

    os.makedirs(args.results_root, exist_ok=True)
    rows = []
    for i, p in enumerate(plans, 1):
        cmd, results_dir = build_command(
            p, args.method, config, args.python, args.gpu,
            args.seed, args.epochs, args.fold, args.results_root
        )
        print(f"\n[sweep] ({i}/{len(plans)}) {p['combo_id']} -> {results_dir}")
        print("       ", " ".join(cmd))
        try:
            ret = subprocess.run(cmd, cwd=str(_PROJECT_ROOT)).returncode
            status = "ok" if ret == 0 else f"exit_{ret}"
        except Exception as exc:  # noqa: BLE001
            status = f"error:{exc}"
        rows.append({
            "combo_id": p["combo_id"],
            "size": p["size"],
            "losses": "+".join(p["losses"]),
            "results_dir": results_dir,
            "status": status,
        })
        # 每一步都覆盖写清单，防中断丢失
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["combo_id", "size", "losses", "results_dir", "status"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[sweep] 全部完成。清单: {manifest_path}")
    print(f"[sweep] 汇总各组合 fold{args.fold} 分数：\n"
          f"  python robust_eval/honest_report.py --dirs {args.results_root}/{args.method}/* "
          f"--strategy last_k_mean")


def _selftest():
    """本地自测：枚举计数正确、组合内外 lambda 置值正确、size>=5 被拒。"""
    from math import comb

    plans3 = build_run_plan("otehv2_rankevent", [3])
    assert len(plans3) == comb(8, 3) == 56, len(plans3)
    plans34 = build_run_plan("otehv2_rankevent", [3, 4])
    assert len(plans34) == comb(8, 3) + comb(8, 4) == 126, len(plans34)
    plans_v50 = build_run_plan("otehv2_timelocal_competing", [3, 4])
    assert len(plans_v50) == comb(11, 3) + comb(11, 4) == 495, len(plans_v50)

    # 组合内=默认权重、组合外=0
    p = plans3[0]
    active = set(p["losses"])
    reg = LOSS_REGISTRY["otehv2_rankevent"]
    for name, (arg, default) in reg.items():
        expected = default if name in active else 0.0
        assert p["lambda_overrides"][arg] == expected, (name, p["lambda_overrides"][arg], expected)
    nonzero = sum(1 for v in p["lambda_overrides"].values() if v != 0.0)
    assert nonzero == 3, nonzero

    # 权重网格：3 项 * 2 系数 -> 每组合 2^3=8 次
    pg = build_run_plan("otehv2_rankevent", [3], weight_grid=[1.0, 2.0])
    assert len(pg) == comb(8, 3) * (2 ** 3), len(pg)

    # size>=5 必须被拒
    try:
        build_run_plan("otehv2_rankevent", [5])
        raise AssertionError("size=5 应当被拒绝")
    except ValueError:
        pass

    print("[selftest] 组合枚举/置零/拒 5 全部通过")
    print(f"[selftest] V45: 3组={comb(8,3)}, 4组={comb(8,4)}, 合计=126 运行")
    print(f"[selftest] V50: 3组={comb(11,3)}, 4组={comb(11,4)}, 合计=495 运行")
    print("[selftest] OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
