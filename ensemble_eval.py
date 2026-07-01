#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V45 多 seed 集成评估。

对若干个 seed 的 V45 训练结果做验证集集成：按 fold 读取各 seed 的
split_{fold}_results_final.pkl（内含每个病人的 risk / censor / time / logits），
对同一 fold 内同一病人的预测跨 seed 平均后重算 C-index，报告 5-fold mean±std，
并与单 seed 结果对照。

集成方式两种，均输出：
  - risk 平均：直接对 risk 分数跨 seed 取均值（最简单，适合 C-index）
  - logits 平均：对离散危险率 logits 跨 seed 取均值后重算 risk（更规范）

fold split 由预生成 split 文件决定，不受 --seed 影响，因此同一 fold 各 seed 的
验证集病人一致，按 case_id 平均严格有效。脚本会取各 seed 的 case_id 交集以稳健处理。

用法:
  python ensemble_eval.py --dirs DIR1 DIR2 ... [--n_classes 4]
"""

import argparse
import glob
import os
import pickle
import re

import numpy as np

try:
    from sksurv.metrics import concordance_index_censored
except Exception as e:  # pragma: no cover
    raise SystemExit(f"需要 scikit-survival: {e}")


_FOLD_RE = re.compile(r"split_(\d+)_results_final\.pkl$")


def _load_seed_dir(root):
    """在一个 seed 目录下递归查找 split_{fold}_results_final.pkl。

    返回 {fold: patient_results_dict}
    """
    pkls = glob.glob(os.path.join(root, "**", "split_*_results_final.pkl"), recursive=True)
    pkls += glob.glob(os.path.join(root, "split_*_results_final.pkl"))
    folds = {}
    for p in sorted(set(pkls)):
        m = _FOLD_RE.search(os.path.basename(p))
        if not m:
            continue
        fold = int(m.group(1))
        with open(p, "rb") as f:
            folds[fold] = pickle.load(f)
    return folds


def _risk_from_logits(logits):
    """从离散时间危险率 logits 重算 risk（越大风险越高）。

    logits: [num_classes]  -> hazards=sigmoid -> S=cumprod(1-h) -> risk=-sum(S)
    """
    logits = np.asarray(logits, dtype=np.float64)
    hazards = 1.0 / (1.0 + np.exp(-logits))
    surv = np.cumprod(1.0 - hazards)
    return -float(surv.sum())


def _cindex(risks, censors, times):
    event = (1 - np.asarray(censors)).astype(bool)
    return concordance_index_censored(
        event, np.asarray(times, dtype=np.float64),
        np.asarray(risks, dtype=np.float64), tied_tol=1e-8
    )[0]


def _fold_cindex_single(pr):
    """单 seed 单 fold 的 C-index（用 pkl 内已存 risk）。"""
    cids = list(pr.keys())
    risks = [pr[c]["risk"] for c in cids]
    censors = [pr[c]["censor"] for c in cids]
    times = [pr[c]["time"] for c in cids]
    return _cindex(risks, censors, times)


def _fold_cindex_ensemble(pr_list, mode="risk"):
    """多 seed 单 fold 集成 C-index。

    pr_list: [patient_results_dict, ...]（同一 fold 的多个 seed）
    mode: 'risk' 直接平均 risk；'logits' 平均 logits 后重算 risk
    """
    # 取 case_id 交集，保证对齐
    common = set(pr_list[0].keys())
    for pr in pr_list[1:]:
        common &= set(pr.keys())
    common = sorted(common)
    if not common:
        return None, 0

    risks, censors, times = [], [], []
    for cid in common:
        if mode == "risk":
            r = np.mean([pr[cid]["risk"] for pr in pr_list])
        else:  # logits
            logit_stack = np.stack([np.asarray(pr[cid]["logits"], dtype=np.float64)
                                    for pr in pr_list], axis=0)
            r = _risk_from_logits(logit_stack.mean(axis=0))
        risks.append(r)
        # censor/time 各 seed 相同，取第一个
        censors.append(pr_list[0][cid]["censor"])
        times.append(pr_list[0][cid]["time"])
    return _cindex(risks, censors, times), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="各 seed 的结果根目录")
    ap.add_argument("--n_classes", type=int, default=4)
    args = ap.parse_args()

    seed_folds = {}
    for d in args.dirs:
        name = os.path.basename(os.path.normpath(d))
        folds = _load_seed_dir(d)
        if folds:
            seed_folds[name] = folds
            print(f"[load] {name}: folds={sorted(folds.keys())}")
        else:
            print(f"[warn] {name}: 未找到 split_*_results_final.pkl，跳过")

    if len(seed_folds) < 2:
        raise SystemExit("至少需要 2 个有效 seed 目录才能集成")

    # 所有 seed 共同拥有的 fold
    all_folds = None
    for folds in seed_folds.values():
        fs = set(folds.keys())
        all_folds = fs if all_folds is None else (all_folds & fs)
    all_folds = sorted(all_folds)
    print(f"\n[folds] 参与集成的 fold: {all_folds}\n")

    # ---- 单 seed 逐 fold C-index ----
    print("=" * 64)
    print("单 seed 结果（各自 5-fold mean±std）")
    print("=" * 64)
    for name, folds in seed_folds.items():
        cs = [_fold_cindex_single(folds[f]) for f in all_folds]
        print(f"  {name:32s} mean={np.mean(cs):.4f}  std={np.std(cs):.4f}  "
              f"folds={[f'{c:.4f}' for c in cs]}")

    # ---- 集成逐 fold C-index ----
    for mode in ("risk", "logits"):
        print("\n" + "=" * 64)
        print(f"集成结果（{mode} 平均，{len(seed_folds)} seeds）")
        print("=" * 64)
        cs = []
        for f in all_folds:
            pr_list = [seed_folds[name][f] for name in seed_folds]
            c, n = _fold_cindex_ensemble(pr_list, mode=mode)
            cs.append(c)
            print(f"  fold {f}: C-index={c:.4f}  (n={n})")
        print(f"  --> ensemble mean={np.mean(cs):.4f}  std={np.std(cs):.4f}")

    print("\n对照：V45 单 seed(=3) 记录值 = 0.7105 ±0.0181")
    print("baseline v9 = 0.7078 ；目标 ≥0.72")


if __name__ == "__main__":
    main()
