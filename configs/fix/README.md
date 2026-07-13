# 修复版 configs（configs/fix/）

针对 `EXPERIMENT_SUMMARY.md` 里 fold2 30ep 结果暴露出的方法级问题，
按方法逐一给出**只改 config、不动模型源码**的修复版本。所有修复的共同处方
来自 V4a 已被数据证实的配方：`opt=adamW + reg=5e-4 + 早停 + 适度加大 dropout`，
每个方法再叠加针对自身病灶的特定改动。

默认都只跑 **fold2**（`k_start=2, k_end=3`）用于快速迭代。修复有效后
再上完整 5-fold × 多 seed。

## 病灶 → 修法对照

### A. 训练不稳定类（正则化/早停修复）

| 方法 | 原版病灶 | 修复要点（除通用配方外的额外改动） |
|---|---|---|
| **RG-ET** `rank_guided_event_transport_fix_blca.yaml` | 过拟合 + IBS 从 0.25 爆到 0.75、生存分布崩溃 | `rg_lambda_rank` 0.15→0.05（削弱 rank 主导） |
| **CATE-T** `censoring_aware_temporal_evidence_transport_fix_blca.yaml` | 标准过拟合 (train 0.89 / val 0.64)，IBS 稳 | 只补正则化，不动 catet_lambda_* |
| **DCT** `distributional_counterfactual_transport_fix_blca.yaml` | 未收敛，best@ep29 还在爬 | `max_epochs` 30→60，`early_stop_warmup` 25 |
| **Faithful** `faithful_evidence_transport_fix_blca.yaml` | train 欠拟合 + val 单峰 + IBS 剧烈震荡 | `fet_lambda_sparse/faith` 各减半（弱化 keep/removed 拉扯），激进早停 warmup=3 |

### B. 损失黑名单类（rankevent 4 项证伪，全设 0）

详见 [`docs/LOSS_BLACKLIST.md`](../../docs/LOSS_BLACKLIST.md)。V4a/V4b 5-fold 证明 4 项 rankevent
损失为负贡献（V45 0.6013 → V4a/V4b 0.7007-0.7095，涨 +0.10）。

| 方法 | 原损失数 | 修复后损失数 | Config |
|---|---|---|---|
| **V45** | 8 | 4 (OT + Div + Recon + event_surv) | `v45_norank_blca.yaml` |
| **V45v2** | 8 | 4（默认路径同 V45） | `v45v2_norank_blca.yaml` |
| **V50** | 11 | 7（保留 spec/cover/compete 三项时间局部机制） | `v50_norank_blca.yaml` |

**注意**：这三份用**完整 5-fold**（`k_start:0 k_end:5`），因为 V4a/V4b 5-fold 已跑过、
可直接对比；不像 A 组是 fold2 快速迭代。

## 用法

```bash
# 单个修复 config
python -m survot_rank.cli train --config configs/fix/rank_guided_event_transport_fix_blca.yaml

# 依次跑完四个
for CFG in configs/fix/rank_guided_event_transport_fix_blca.yaml \
           configs/fix/censoring_aware_temporal_evidence_transport_fix_blca.yaml \
           configs/fix/distributional_counterfactual_transport_fix_blca.yaml \
           configs/fix/faithful_evidence_transport_fix_blca.yaml ; do
  python -m survot_rank.cli train --config "$CFG"
done

# 全部跑完后诚实汇总（last5 + 乐观偏差 + 校准告警）
python robust_eval/honest_report.py --dirs \
  results/rank_guided_event_transport_fix_blca \
  results/censoring_aware_temporal_evidence_transport_fix_blca \
  results/distributional_counterfactual_transport_fix_blca \
  results/faithful_evidence_transport_fix_blca \
  --labels rg-et catet dct faithful \
  --strategy last_k_mean --out results/fix_report.md
```

## 判据

不看 `val_best`（peak-picking 泄漏）；改看：

1. **`val_last5` 是否比原版高**——真实收敛质量。
2. **IBS 是否稳定在 <0.30**——概率校准没崩。
3. **train 与 val 的 gap 是否 < 0.15**——过拟合被抑制。

若某个修复三项都改善 → 上完整 5-fold × 3 seed 确认；
若某项恶化 → 该方法的病灶不在正则/早停，需重新诊断。
