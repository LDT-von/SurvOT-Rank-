#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳健的逐-epoch 分数选择策略。

背景 / 要解决的问题
--------------------
现有训练器 (``survot_rank/training/train_runner.py``) 用
``if val_c >= args.max_cindex:`` 挑选「验证集 C-index 最高的那个 epoch」，
并把该 epoch 的验证指标当作最终结果上报。

在每 fold 只有 ~76 个验证样本、训练曲线又剧烈抖动 (0.35~0.55) 的情况下，
「挑验证峰值」等价于在噪声上取最大值 —— 这是一种乐观偏差 (optimism bias)，
会同时导致：
  * fold 之间分数方差极大 (某 fold 峰值恰好对上 = 0.74，对不上 = 0.62)；
  * 上报的 mean 系统性偏高，换个 seed 结论就变。

本模块**不修改任何源码**，只对训练器已经落盘的 ``epoch_curve_fold*.csv``
做后处理，提供几种不会「偷看验证峰值」的稳健选择策略，并量化乐观偏差。

epoch_curve_fold*.csv 的列
--------------------------
    epoch, val_cindex, val_cindex_ipcw, val_IBS, val_iauc, val_loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# 指标方向：True = 越大越好，False = 越小越好。
METRIC_HIGHER_IS_BETTER = {
    "val_cindex": True,
    "val_cindex_ipcw": True,
    "val_iauc": True,
    "val_IBS": False,
    "val_loss": False,
}

# 可选的选择策略。
STRATEGIES = ("best", "last", "last_k_mean", "smoothed_peak", "plateau")


@dataclass
class FoldScore:
    """单个 fold 在某指标下、按某策略得到的代表分数。"""

    fold: int
    metric: str
    strategy: str
    value: float
    chosen_epoch: int  # -1 表示由多 epoch 聚合得到，无单一 epoch


def _higher_is_better(metric: str) -> bool:
    if metric not in METRIC_HIGHER_IS_BETTER:
        raise KeyError(
            f"未知指标 {metric!r}，已知: {sorted(METRIC_HIGHER_IS_BETTER)}"
        )
    return METRIC_HIGHER_IS_BETTER[metric]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """居中滑动平均，边界用可用窗口收缩，保持长度不变。"""
    if window <= 1 or values.size <= 1:
        return values.astype(float)
    window = min(window, values.size)
    out = np.empty_like(values, dtype=float)
    half = window // 2
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = values[lo:hi].mean()
    return out


def select_fold_score(
    curve: pd.DataFrame,
    metric: str = "val_cindex",
    strategy: str = "last_k_mean",
    k: int = 5,
    smooth_window: int = 3,
) -> FoldScore:
    """从一个 fold 的 epoch 曲线里取出稳健代表分数。

    策略说明
    --------
    best
        逐 epoch 取最优值。**这是会泄漏的乐观基线**，仅用于和稳健策略对比、
        量化偏差，不应作为最终上报值。
    last
        最后一个 epoch 的值。最诚实、但对训练末期抖动敏感。
    last_k_mean
        最后 k 个 epoch 的均值（默认 k=5）。推荐默认：既反映收敛区表现，
        又平滑掉单点抖动。
    smoothed_peak
        先做滑动平均再取峰值。允许「选峰」但要求峰是持续的、而非单点尖刺，
        比 raw best 稳健得多。
    plateau
        取平滑曲线达到最优的 epoch，再回读该 epoch 的**原始**值。近似
        「用内部准则选 epoch」。
    """
    if metric not in curve.columns:
        raise KeyError(f"曲线缺少列 {metric!r}，现有列: {list(curve.columns)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}，可选: {STRATEGIES}")

    df = curve.sort_values("epoch").reset_index(drop=True)
    raw = df[metric].to_numpy(dtype=float)
    epochs = df["epoch"].to_numpy()
    higher = _higher_is_better(metric)
    fold = int(df["fold"].iloc[0]) if "fold" in df.columns else -1

    if strategy == "best":
        idx = int(np.argmax(raw) if higher else np.argmin(raw))
        return FoldScore(fold, metric, strategy, float(raw[idx]), int(epochs[idx]))

    if strategy == "last":
        return FoldScore(fold, metric, strategy, float(raw[-1]), int(epochs[-1]))

    if strategy == "last_k_mean":
        kk = min(max(1, k), raw.size)
        return FoldScore(fold, metric, strategy, float(raw[-kk:].mean()), -1)

    smoothed = _moving_average(raw, smooth_window)
    idx = int(np.argmax(smoothed) if higher else np.argmin(smoothed))
    if strategy == "smoothed_peak":
        return FoldScore(fold, metric, strategy, float(smoothed[idx]), int(epochs[idx]))
    # plateau: 在平滑最优 epoch 处回读原始值
    return FoldScore(fold, metric, strategy, float(raw[idx]), int(epochs[idx]))


@dataclass
class Aggregate:
    metric: str
    strategy: str
    n_folds: int
    mean: float
    std: float
    values: list[float]
    folds: list[int]


def aggregate_folds(scores: Sequence[FoldScore]) -> Aggregate:
    """把多个 fold 的代表分数聚合成 mean ± std。"""
    if not scores:
        raise ValueError("没有可聚合的 fold 分数")
    metric = scores[0].metric
    strategy = scores[0].strategy
    vals = np.array([s.value for s in scores], dtype=float)
    return Aggregate(
        metric=metric,
        strategy=strategy,
        n_folds=len(scores),
        mean=float(vals.mean()),
        std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        values=[float(v) for v in vals],
        folds=[s.fold for s in scores],
    )


def optimism_gap(
    curves: Iterable[pd.DataFrame],
    metric: str = "val_cindex",
    robust_strategy: str = "last_k_mean",
    k: int = 5,
    smooth_window: int = 3,
) -> dict:
    """量化「挑验证峰值」相对稳健策略高估了多少。

    返回 best 聚合、robust 聚合，以及二者均值之差 (gap)。gap 越大，说明原始
    上报方式越乐观、越不可信。
    """
    curves = list(curves)
    best_scores, robust_scores = [], []
    for c in curves:
        best_scores.append(select_fold_score(c, metric, "best"))
        robust_scores.append(
            select_fold_score(c, metric, robust_strategy, k=k, smooth_window=smooth_window)
        )
    best_agg = aggregate_folds(best_scores)
    robust_agg = aggregate_folds(robust_scores)
    higher = _higher_is_better(metric)
    gap = best_agg.mean - robust_agg.mean
    if not higher:
        gap = robust_agg.mean - best_agg.mean  # 对 lower-better 指标，best 更小
    return {
        "metric": metric,
        "best": best_agg,
        "robust": robust_agg,
        "optimism_gap": gap,
    }


def _selftest() -> None:
    """用合成曲线自测策略逻辑（无需 GPU / 数据）。"""
    rng = np.random.default_rng(0)
    curves = []
    for fold in range(5):
        n = 30
        # 缓慢上升到 ~0.68 的真实趋势 + 强噪声（模拟抖动），偶发单点尖刺。
        trend = 0.55 + 0.13 * (1 - np.exp(-np.arange(n) / 8))
        noise = rng.normal(0, 0.05, n)
        vals = np.clip(trend + noise, 0, 1)
        vals[rng.integers(5, n)] += 0.08  # 单点乐观尖刺
        curves.append(
            pd.DataFrame(
                {
                    "epoch": np.arange(n),
                    "fold": fold,
                    "val_cindex": np.clip(vals, 0, 1),
                    "val_IBS": np.clip(0.3 + noise, 0, 1),
                }
            )
        )

    best = aggregate_folds([select_fold_score(c, "val_cindex", "best") for c in curves])
    robust = aggregate_folds(
        [select_fold_score(c, "val_cindex", "last_k_mean", k=5) for c in curves]
    )
    gap = optimism_gap(curves, "val_cindex")

    print("[selftest] best   mean±std = %.4f ± %.4f" % (best.mean, best.std))
    print("[selftest] robust mean±std = %.4f ± %.4f" % (robust.mean, robust.std))
    print("[selftest] optimism_gap    = %.4f" % gap["optimism_gap"])
    assert best.mean > robust.mean, "best 策略应当高于稳健策略（存在乐观偏差）"
    assert gap["optimism_gap"] > 0, "乐观偏差应为正"
    # lower-better 指标方向检查
    fs = select_fold_score(curves[0], "val_IBS", "best")
    assert fs.value <= curves[0]["val_IBS"].mean() + 1e-6
    print("[selftest] OK")


if __name__ == "__main__":
    _selftest()
