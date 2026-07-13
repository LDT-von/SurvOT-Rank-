# 分数结果 — 唯一数据源

> **此文件是 SurvOT-Rank 所有实验分数的唯一权威来源。所有新实验结果只在此追加，不再使用其他 SUMMARY/REPORT 文件。**
>
> 最后更新: 2026-07-13 | 代码版本: `71fd010` (main, +PCGrad #7 +LossSweep #8 +CuratedPresets #9)
> ## 排队中: #1 RG-ET ✅ | #2 v50 ✅ | #3 CATE-T ✅ | #4 DCT ✅ | #5 RG-ET+PCGrad 🔄 | #6–#10 ⬜

---

## 方法一览

| # | 方法 | 状态 | val_cidx best | 来源 config |
|---|------|------|--------------|-------------|
| 1 | v45 — 8-loss baseline | ✅ 完成 | 0.6814 ± 0.0491 | `v45_blca.yaml` |
| 2 | v45v2 — 8-loss + clinical | ✅ 完成 | 0.6919 ± 0.0499 | `v45v2_blca_clinical.yaml` |
| 3 | Rank-Guided Event Transport — 3-loss | ✅ 完成 (10ep/30ep fold2) | 0.6389 (fold2 30ep) | `rank_guided_event_transport_blca.yaml` |
| 4 | Stagewise Prognostic Transport | ✅ 完成 (fold2, 30ep) | 0.6741 | `stagewise_prognostic_transport_blca.yaml` |
| 5 | Faithful Evidence Transport | ✅ 完成 (fold2, 30ep) | — | `faithful_evidence_transport_blca.yaml` |
| 6 | v50 (Time-Local Competing) | ⚠️ 部分 (fold2) | **0.6749** | `v50_blca.yaml` |
| 7 | CATE-T (Censoring-Aware) | ⚠️ 部分 (fold2) | 0.6405 | `censoring_aware_temporal_evidence_transport_blca.yaml` |
| 8 | DCT (Distributional Counterfactual) | ⚠️ 部分 (fold2) | 0.6237 | `distributional_counterfactual_transport_blca.yaml` |
| 9 | RG-ET + PCGrad | 🔄 进行中 (#5) | — | `rank_guided_event_transport_blca.yaml` + `pcgrad.py` |
| 10 | V2 — 关 rankevent (newSlotSPE 0.7100) | ⬜ 排队 #6 | — | `v2_norank_blca.yaml` |
| 11 | V4a — 关 rankevent + AdamW | ⬜ 排队 #7 | — | `v2_norank_blca.yaml` + `--set opt=adamW` |
| 12 | ot_v3 (newSlotSPE #1, 0.7282) | ⬜ 排队 #8 | 0.7282 (newSlotSPE) | `--newslot_method ot_v3` |
| 13 | V45 损失子集 curated | ⬜ 排队 #9 | — | 10 组, ~12.5h |
| 14 | V50 损失子集 curated | ⬜ 排队 #10 | — | 10 组, ~12.5h |

> 排队脚本：`bash scripts/queue_fold2.sh`（依次 10 个，fold2 only, 30ep）
> #1–#8 各 ~1h15m, #9–#10 各 ~12.5h, 总计 ~35h

### 损失子集扫描

> 工具：`robust_eval/loss_group_sweep.py` — 枚举 V45/V50 损失组合，三档规模

| preset | V45 | V50 | 说明 |
|--------|:---:|:---:|------|
| **`curated`（默认）** | **10** | **10** | 手工精选，必含 OT + 至少一个预测监督项 |
| `pruned` | 52 | 130 | 全枚举 + 剪枝（必含 OT + 至少一个预测监督项） |
| `full` | 126 | 495 | 原始全枚举（兜底） |

> 用法：`--preset curated`（默认）| `--preset pruned` | `--preset full`
> dry-run：`--dry-run` | 权重网格：`--weight-grid 0.5,1,2`

---

## 1. v45 — 8-loss baseline

**Config**: `v45_blca.yaml` | **Method**: `otehv2_rankevent` | **Losses**: OT + Div + Recon + GateEnt + NLL + PerEvent + CoxRank + GlobalCons

### seed=3, 30 epochs, 5-fold (alpha_surv=0.15, fold-aware bins)

| Fold | Ep | train_cidx | val_cidx | best @ | val_cidx last5 | val_ipcw | val_IBS | val_iauc |
|------|----|-----------|---------|--------|---------------|---------|--------|---------|
| 0 | 30 | 0.5370 | 0.7120 | 15 | 0.6887 | 0.6753 | 0.3478 | 0.8601 |
| 1 | 30 | 0.6315 | 0.7411 | 3 | 0.7080 | 0.7114 | 0.1544 | 0.7806 |
| 2 | 30 | 0.5365 | 0.6013 | 5 | 0.5026 | 0.6855 | 0.2534 | 0.8632 |
| 3 | 30 | 0.5705 | 0.6995 | 6 | 0.6654 | 0.6363 | 0.1500 | 0.6380 |
| 4* | 5 | 0.5152 | 0.6530 | 4 | 0.5988 | 0.6110 | 0.1488 | 0.8056 |

> \* Fold 4 仅 5 epoch 后手动 kill（train_cidx=0.5152 < 0.5，bug 已确认）

| Aggregate | Score |
|-----------|-------|
| val_cidx best | **0.6814 ± 0.0491** |
| val_cidx last5 | 0.6327 ± 0.0748 |
| train_cidx (mean of means) | **0.48** — 不如随机猜 |

### seed=None, 30 epochs, fold2 only (grad_clip=1.0)

| Fold | Ep | val_cidx | best @ | val_cidx last5 | train_cidx @29 | val_ipcw | val_IBS | val_iauc |
|------|----|---------|--------|---------------|----------------|---------|--------|---------|
| 2 | 30 | **0.6389** | **9** | 0.5844 | 0.8597 | 0.5541 | 0.7521 | 0.3875 |

> **分析 — 重度过拟合 + 后期崩溃：**
> - train_cidx 0.86 vs val 0.58: 差异 0.28，10 epoch 后梯度拉扯开始破坏泛化。
> - epoch 9 达到峰值 0.6389 后持续下降，epoch 20 后 IBS 从 0.25 爆炸到 0.75，iAUC 从 0.84 崩塌到 0.39——**生存分布彻底崩溃**。
> - 3-loss (OT+Rank+StageOrder) 方向冲突未消解，batch=4 下信号本就弱，30ep 反而让冲突累积到不可逆。
> - 结论：grad_clip 只缓解梯度爆炸，不解决方向冲突。PCGrad（#5）是冲着这个问题来的。

---

## 2. v45v2 — 8-loss + age/gender clinical features

**Config**: `v45v2_blca_clinical.yaml` | **Method**: `otehv2_rankevent_v2`

### seed=3, 30 epochs, 5-fold

| Fold | Ep | val_cidx | best @ | val_ipcw | val_IBS | val_iauc |
|------|----|---------|--------|---------|--------|---------|
| 0 | 30 | 0.7417 | 14 | 0.4581 | 0.8828 | 0.8302 |
| 1 | 30 | 0.7453 | 24 | 0.6801 | 0.4261 | 0.6910 |
| 2 | 30 | 0.6237 | 8 | 0.5581 | 0.8884 | 0.4965 |
| 3 | 30 | 0.7049 | 20 | 0.6313 | 0.4543 | 0.5897 |
| 4 | 30 | 0.6441 | 15 | 0.6074 | 0.3131 | 0.6253 |

| Aggregate | Score |
|-----------|-------|
| val_cidx | **0.6919 ± 0.0499** |

---

## 3. Rank-Guided Event Transport — 3-loss

**Config**: `rank_guided_event_transport_blca.yaml` | **Method**: `rank_guided_event_transport`
**Losses**: OT + Ranking + StageOrder (砍掉 5 个辅助损失)

### seed=1, 10 epochs, 5-fold, grad_clip=1.0 (quick_robust_eval)

| Fold | Ep | train_cidx | val_cidx | best @ | val_cidx last5 | val_ipcw | val_IBS | val_iauc |
|------|----|-----------|---------|--------|---------------|---------|--------|---------|
| 0 | 10 | 0.5296 | 0.6965 | 5 | 0.6487 | 0.7790 | 0.3499 | 0.8604 |
| 1 | 10 | 0.7501 | 0.7403 | 6 | 0.6335 | 0.7204 | 0.1572 | 0.8641 |
| 2 | 10 | 0.6358 | 0.6037 | 8 | 0.5955 | 0.5899 | 0.2313 | 0.8932 |
| 3 | 10 | 0.8136 | 0.6470 | 6 | 0.6350 | 0.5974 | 0.1451 | 0.9285 |
| 4 | 10 | 0.7252 | 0.6859 | 8 | 0.6407 | 0.6439 | 0.1493 | 0.9326 |

| Aggregate | Score |
|-----------|-------|
| val_cidx best | 0.6747 ± 0.0463 |
| val_cidx last5 | **0.6307 ± 0.0205** |
| train_cidx max | 0.53–0.81 (能拟合了) |

#### 诚实报告 (last_k_mean, k=5)

| Metric | robust | best (leak) | optimism gap |
|--------|--------|-------------|--------------|
| val_cindex ↑ | 0.6307 ± 0.0205 | 0.6747 ± 0.0518 | **+0.0440** |
| val_cindex_ipcw ↑ | 0.5917 ± 0.0782 | 0.6661 ± 0.0817 | +0.0744 |
| val_iauc ↑ | 0.6300 ± 0.1353 | 0.8958 ± 0.0343 | **+0.2658** |
| val_IBS ↓ | 0.2289 ± 0.0805 | 0.2065 ± 0.0875 | +0.0223 |

### seed=3, fold2 only (no grad_clip)

| Fold | Ep | val_cidx | best @ | train_cidx max |
|------|----|---------|--------|---------------|
| 2 (25ep) | 25 | 0.6341 | 29 | 0.7285 |
| 2 (3ep smoke) | 3 | 0.5592 | 0 | — |

### seed=1, fold0 only, 30 epochs, grad_clip=1.0

| Fold | Ep | val_cidx | best @ | val_cidx last5 |
|------|----|---------|--------|---------------|
| 0 | 30 | 0.6743 | 7 | 0.6520 |

---

## 4. Stagewise Prognostic Transport

**Config**: `stagewise_prognostic_transport_blca.yaml` | **Method**: `stagewise_prognostic_transport`

### seed=3, fold2 only

| Fold | Ep | val_cidx best | best @ | val_cidx last5 | train_cidx max | val_ipcw | val_IBS | val_iauc |
|------|----|--------------|--------|---------------|---------------|---------|--------|---------|
| 2 (old, early kill) | 8 | 0.5909 | 7 | — | 0.6026 | — | — | — |
| **2 (rerun, 30ep)** | 30 | **0.6741** | 13 | — | **0.7886** | 0.6948 | 0.2198 | 0.7848 |

> 完整 30 epoch 后 train_cidx 飙到 0.7886，模型能拟合了；best val 0.6741 @epoch 13，早停于 epoch 29

---

## 5. Faithful Evidence Transport

**Config**: `faithful_evidence_transport_blca.yaml` | **Method**: `faithful_evidence_transport`

### seed=None, 30 epochs, fold2 only

| Fold | Ep | val_cidx | best @ | val_cidx last5 | train_cidx @29 | val_ipcw | val_IBS | val_iauc |
|------|----|---------|--------|---------------|----------------|---------|--------|---------|
| 2 | 30 | **0.6837** | **7** | 0.5301 | 0.6165 | 0.5840 | 0.3203 | 0.7973 |

> **分析 — 峰值最高但极不稳定：**
> - best 0.6837 @ epoch 7，是目前 fold2 的绝对最高值。
> - 但 IBS 在 epoch 20–28 剧烈震荡（0.21→0.53→0.56→0.45→0.60→0.36），模式完全无序。
> - train_cidx 仅 0.5167→0.6165（缓慢爬升），没有过拟合迹象——不稳定似乎来自损失内部而非 train/val 差异。
> - epoch 7 的 0.6837 是单点峰值，可信度低于 v50 的连续平台.

---

## 6. Censoring-Aware Temporal Evidence Transport (CATE-T)

**Config**: `censoring_aware_temporal_evidence_transport_blca.yaml` | **Method**: `censoring_aware_temporal_evidence_transport`

### seed=None, 30 epochs, fold2 only

| Fold | Ep | val_cidx | best @ | val_cidx last5 | train_cidx @29 | val_ipcw | val_IBS | val_iauc |
|------|----|---------|--------|---------------|----------------|---------|--------|---------|
| 2 | 30 | **0.6405** | **8** | 0.5950 | 0.8926 | 0.6918 | 0.1839 | 0.8163 |

> **分析 — 标准过拟合模式，但比 RG-ET 干净：**
> - train 爬到 0.89（21 个 epoch 0.73→0.89），val 从 peak 0.64 退化到 0.60。
> - train/val 差异 0.30——信息泄漏量级和 RG-ET 一致。
> - 但 IBS 全程健康（0.16–0.24），iAUC 0.57–0.87，**生存分布未崩塌**。
> - 这意味着 CATE-T 的过拟合是"预测不准"而非"输出爆炸"——和 RG-ET 的 IBS→0.75 崩溃是本质不同的现象。

---

## 7. Distributional Counterfactual Transport (DCT)

**Config**: `distributional_counterfactual_transport_blca.yaml` | **Method**: `distributional_counterfactual_transport`

### seed=None, 30 epochs, fold2 only

| Fold | Ep | val_cidx | best @ | val_cidx last5 | train_cidx @29 | val_ipcw | val_IBS | val_iauc |
|------|----|---------|--------|---------------|----------------|---------|--------|---------|
| 2 | 30 | **0.6237** | **29** | 0.5936 | 0.8642 | 0.7101 | 0.2008 | 0.8539 |

> **分析 — 未收敛，仍有上升趋势：**
> - best @ epoch 29（最后 epoch），说明模型仍在学习，30 epoch 不够。
> - train 0.86 vs val 0.62: 差异 0.24，标准过拟合但未崩塌。
> - epoch 22–29 呈持续上升趋势（0.58→0.62），尚未进入平台期。
> - IBS 全程健康（0.19–0.26），iAUC 0.71–0.88——生存分布稳定。
> - 和 CATE-T 类似：过拟合是"预测不准"而非"输出爆炸"。
> - 需更多 epoch（如 50–60）确认是否还能继续爬升，还是进平台期后开始退化。

---

## 8. v50 — Time-Local Competing Event Hazards

**Config**: `v50_blca.yaml` | **Method**: `otehv2_timelocal_competing` | **Losses**: 11 项 (含 OT + time-local 特有项)

### seed=None, 30 epochs, fold2 only

| Fold | Ep | val_cidx | best @ | val_cidx last5 | train_cidx @29 | val_ipcw | val_IBS | val_iauc |
|------|----|---------|--------|---------------|----------------|---------|--------|---------|
| 2 | 30 | **0.6749** | **12** | 0.6198 | 0.7758 | 0.5699 | 0.2387 | 0.8698 |

> **分析 — 目前 fold2 最佳，且学习曲线极不寻常：**
> - val 在 epoch 8–12 就达到峰值 0.64–0.67，但此时 train_cidx 仅 0.43–0.49（近乎随机）。
> - train 直到 epoch 22 后才爬到 0.70+，val 反而从峰值退化到 0.62 震荡。
> - 退化幅度轻微（IBS 全程 0.24–0.26 稳定，iAUC 全程 0.65–0.88），不存在 RG-ET 式的生存崩塌。
> - epoch 23 有一次单 epoch spike（val 掉到 0.44），下一 epoch 恢复——可能是某个 batch 梯度伪逆。
> - **关键假设**：time-local 机制提供了隐式正则化。model 在训练早期不过拟合时泛化最优（0.67+），后期 train 攀升反而稀释了这种正则。需验证早停（~12 epoch, train 未完） vs 完整过拟合（30 epoch）时 5-fold 是否有差异。

---

## 9. newSlotSPE 消融 (config: `v45_blca.yaml`, seed=3)

> 数据来自 `/home/ubuntu/newSlotSPE`，CSV 未同步到 SurvOT-Rank。

| 配置 | val_cidx | Fold 3 | 说明 |
|------|---------|--------|------|
| V0 baseline — 8-loss | 0.6993 ± 0.0218 | — | 等频分箱, alpha_surv=0.15 |
| V1 wd=1e-4 + rank 降权 | 0.6991 ± 0.0323 | 0.6426 | 降权反而更差 |
| V2 wd=5e-4 + **关 rankevent** | **0.7100 ± 0.0186** | 0.6896 | 关 4 个辅助损失的因果验证 |
| V3 wd=1e-3 + 关 rankevent | =V2 (md5 一致) | — | wd 被 Adam 忽略 |
| V4a AdamW + wd=5e-4 | =V2 (fold3 only) | — | AdamW wd 也未生效 |

> `--reg` + `--opt adam` 时 bridge 未传 `weight_decay`，V1–V4 的 wd 实际均为 0

---

## 10. Fold 2 全量对比 (最难的 fold)

| 模型 | Loss | Ep | Seed | grad_clip | val_cidx best | val last5 | train_cidx max | 能拟合训练集? |
|------|------|----|------|----------|--------------|----------|---------------|-------------|
| v45 (batch) | 8 | 30 | 3 | no | 0.6013 | 0.5026 | 0.5365 | ❌ |
| v45 (fold2) | 8 | 5 | None | yes | 0.6253 | 0.5768 | 0.5002 | ❌ |
| v45v2+clinical | 8 | 30 | 3 | no | 0.6237 | — | — | ❌ |
| RG-ET 25ep | 3 | 25 | 3 | no | 0.6341 | 0.5616 | 0.7285 | ✅ |
| RG-ET 10ep | 3 | 10 | 1 | yes | 0.6037 | 0.5955 | 0.6358 | ✅ |
| **RG-ET 30ep no seed** | 3 | 30 | None | yes | **0.6389** | 0.5844 | 0.8597 | ✅ → 💥 (IBS 崩塌) |
| Stagewise (rerun) | ? | 30 | 3 | no | 0.6741 | — | 0.7886 | ✅ |
| **Faithful 30ep no seed** | ? | 30 | None | no | **0.6837** | 0.5301 | 0.6165 | ✅ → ⚠️ (不稳定) |
| **v50 30ep no seed** | 11 | 30 | None | no | **0.6749** | 0.6198 | 0.7758 | ⚠️ (train 晚于 val 爬升) |
| **CATE-T 30ep no seed** | ? | 30 | None | no | 0.6405 | 0.5950 | 0.8926 | ✅ (标准过拟合) |
| **DCT 30ep no seed** | ? | 30 | None | no | **0.6237** | 0.5936 | 0.8642 | ✅ (未收敛,持续上升) |
| RG-ET+PCGrad | 3+PC | 30 | None | yes | 🔄 | 🔄 | 🔄 | 🔄 |

---

## 11. 运行环境

- **SurvOT-Rank**: `/home/ubuntu/SurvOT-Rank` (commit `71fd010`, main)
- **newSlotSPE**: `/home/ubuntu/newSlotSPE` (commit `0eaadc6`, 等宽分箱旧版)
- **Python**: conda env `trisurv`
- **GPU**: gpu=0
- **数据**: 5-fold BLCA, `survival_months_dss`, Pathways, n=380
