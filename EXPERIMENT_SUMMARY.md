# 分数结果 — 唯一数据源

> **此文件是 SurvOT-Rank 所有实验分数的唯一权威来源。所有新实验结果只在此追加，不再使用其他 SUMMARY/REPORT 文件。**
>
> 最后更新: 2026-07-14 | 代码版本: `983265a` (main, +全局分箱修复 +queue_fix_5fold)

---

## 关键发现: 等频分箱修复验证 (2026-07-14)

**两个 bug 已修复并验证：**

| Bug | 仓库 | 根因 | 影响 | 修复 |
|---|---|---|---|---|
| fold-aware 分箱 | SurvOT-Rank | `fit_label_bins()` 每折重新算分位 bins | fold2 -0.127 (0.60 vs newSlotSPE 0.73) | 移除调用 → 全局分箱 |
| 等宽分箱 | newSlotSPE | `pd.cut(bins=4)` 覆盖 `pd.qcut` 结果 | 全折被等宽 bins 虚高 (imbalanced {310,50,16,4}) | `bins=q_bins` + ±inf |

**验证方法：** 用 V51 SlimBridge seed3 和 v45_norank 两个方法对比 (均使用修复后的代码)。

---

### V51 SlimBridge seed3 (newSlotSPE, 修复后: 全局 + 等频分箱)

| Fold | Best epoch | val_cidx (best) | val_cidx (last5) |
|:----:|:----------:|:---------------:|:----------------:|
| 0 | 5 | **0.7433** | 0.7118 |
| 1 | 4 | 0.7191 | 0.6552 |
| 2 | 16 | 0.5821 | 0.5560 |
| 3 | 11 | 0.6650 | 0.6157 |
| 4 | 11 | 0.6837 | 0.5645 |

**5-fold last5 mean: 0.6207 ± 0.0581**
5-fold best peak: 0.6786 ± 0.0554 (peak bias +0.06)

> ❌ **fold2 仍然崩溃** (0.5821, last5=0.5560)。全局等频分箱修复了 V45 的 fold2 但没有修复 SlimBridge。
> **结论：SlimBridge 的 fold2 崩溃是模型架构问题**（SlotBridge/Modality Dropout 机制），与分箱无关。

---

### v45_norank (SurvOT-Rank, 修复后: 全局 + 等频分箱, seed=random, 5-fold)

| Fold | Best epoch | val_cidx (best) | val_cidx (last5) | train_cidx (last) |
|:----:|:----------:|:---------------:|:----------------:|:-----------------:|
| 0 | 2 | **0.7361** | 0.6536 | 0.4126 |
| 1 | 0 | 0.7094 | 0.6643 | 0.6486 |
| 2 | 14 | 0.6765 | 0.6637 | 0.6830 |
| 3 | 8 | 0.6273 | 0.6021 | 0.6113 |
| 4 | 6 | 0.6744 | 0.6195 | 0.5882 |

**5-fold last5 mean: 0.6406 ± 0.0253**
5-fold best peak: 0.6848 ± 0.0367 (peak bias +0.04)
train-val gap: **-0.0519** (train < val → 模型正则化有效)

> ✅ **fold2 恢复正常！** 从旧的 0.6013 (fold-aware) 提升到 0.6637 (全局分箱)，+0.062。
> 但 fold0 出现 0.7361 @ epoch 2 的极端 early peak，表明 30ep 的峰值选择不可靠。
> train < val 全部 fold 出现负 gap，说明 AdamW + wd=5e-4 正则化有效抑制了过拟合。

---

## 旧记录 (fold2-only, 2026-07-13)
>
> ## 队列状态: #1–#8 ✅ (ot_v3 失败) | #9、#10 ❌ 已终止

---

## 方法一览

| # | 方法 | 状态 | val_cidx best (fold2) | 来源 config |
|---|------|------|----------------------|-------------|
| 1 | v45 — 8-loss baseline | ✅ 5-fold 完成 | 0.6013 | `v45_blca.yaml` |
| 2 | v45v2 — 8-loss + clinical | ✅ 5-fold 完成 | 0.6237 | `v45v2_blca_clinical.yaml` |
| 3 | Rank-Guided Event Transport — 3-loss | ✅ fold2 完成 | 0.6341 | `rank_guided_event_transport_blca.yaml` |
| 4 | Stagewise Prognostic Transport | ✅ fold2 完成 | **0.6741** | `stagewise_prognostic_transport_blca.yaml` |
| 5 | Faithful Evidence Transport | ✅ fold2 完成 | **0.6837** | `faithful_evidence_transport_blca.yaml` |
| 6 | v50 (Time-Local Competing) | ✅ fold2 完成 | **0.6749** | `v50_blca.yaml` |
| 7 | CATE-T (Censoring-Aware) | ✅ fold2 完成 | 0.6405 | `censoring_aware_temporal_evidence_transport_blca.yaml` |
| 8 | DCT (Distributional Counterfactual) | ✅ fold2 完成 | 0.6237 | `distributional_counterfactual_transport_blca.yaml` |
| 9 | RG-ET + PCGrad | ✅ 完成 (PCGrad 未集成, 实际=RG-ET rerun) | 0.6341 | `rank_guided_event_transport_blca.yaml` |
| 10 | V2 — 关 rankevent | ✅ fold2 完成 (#6) | 0.7174 | `v2_norank_blca.yaml` |
| 11 | **V4a — 关 rankevent + AdamW wd=5e-4** | ✅ fold2 完成 (#7) | **🏆 0.7254** | `v2_norank_blca.yaml` + `--set opt=adamW` |
| 12 | ot_v3 (newSlotSPE #1, 0.7282) | ❌ 失败 | — | 缺 `04_optimal_transport_align/model_v3.py` |
| 13 | V45 损失子集 curated | ❌ 已终止 (2/10 完成) #9 | 0.6165 / 0.6125 | 随机组合无意义，且 #1–#8 已提供明确答案 |
| 14 | V50 损失子集 curated | ❌ 已终止 (未开始) #10 | — | V50 fold2 0.6749 不如 V4a 0.7254，无需扫描 |

> 排队脚本：`bash scripts/queue_fold2.sh`（依次 10 个，fold2 only, 30ep）
> #1–#8, #10 各 ~1h15m, #9, #13–#14 各 ~12.5h, 总计 ~35h
> **🏆 V4a (关 rankevent + AdamW) 是 fold2 目前最高分 (0.7254 @ ep12)**，超过 v45 8-loss 的 0.6013 达 +0.1241
> 注意: V2 和 V4a 共用同一 results 目录 (`configs/v2_norank_blca.yaml`)，V2 先跑 (0.7174)，V4a 后覆盖 (0.7254)

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

## 12. V2 — 关 rankevent (4-loss: OT + Div + Recon + NLL)

**Config**: `v2_norank_blca.yaml` | **Method**: `ot_event_hazard_v2` | **Losses**: OT + Div + Recon + NLL (砍掉全部 4 个 rankevent 辅助项)

### seed=3, 30 epochs, fold2 only

| Fold | Ep | val_cidx best | best @ | val_cidx last5 | val_ipcw best | val_IBS best | val_iauc best |
|------|----|--------------|--------|---------------|--------------|-------------|--------------|
| 2 | 30 | **0.7174** | **12** | — | — | — | — |

> **注意**: 当前磁盘上的 epoch_curve CSV 已被 V4a 后续运行覆盖（两者共用同一个 `results_dir`），原始 V2 的逐 epoch 数据已丢失。以上 0.7174 来自队列日志。
>
> **分析:**
> - best 0.7174 @ epoch 12 — fold2 第二高。
> - 仅 4 loss 砍掉全部 rankevent 项，与 newSlotSPE 的 5-fold V2 结论（0.7100）方向一致。
> - 对比 V4a 的 0.7254，AdamW 的 weight_decay=5e-4 提供了约 +0.008 的增量。

---

## 12b. V4a — 关 rankevent + AdamW wd=5e-4

**Config**: `v2_norank_blca.yaml` + `--set opt=adamW --set reg=0.0005` | **Method**: `ot_event_hazard_v2`

### seed=3, 30 epochs, fold2 only

| Fold | Ep | val_cidx best | best @ | val_cidx last5 | val_ipcw best | val_IBS best | val_iauc best |
|------|----|--------------|--------|---------------|--------------|-------------|--------------|
| 2 | 30 | **🏆 0.7254** | **12** | 0.6581 | 0.6863 | 0.2713 | 0.9378 |

**逐 epoch 曲线 (cidx ≥ 0.62):**

| Ep | val_cidx | ipcw | IBS | iAUC |
|----|---------|------|-----|------|
| 6 | 0.6197 | 0.7016 | 0.2540 | 0.8857 |
| 7 | 0.6853 | 0.7579 | 0.2637 | 0.9111 |
| 10 | 0.6493 | 0.7090 | 0.2675 | 0.8187 |
| 11 | 0.6882 | 0.7543 | 0.2699 | 0.7474 |
| **12** | **0.7254** | 0.6863 | 0.2713 | 0.9378 |
| 13 | 0.6926 | 0.7419 | 0.2650 | 0.8862 |
| 14 | 0.6966 | 0.7055 | 0.2604 | 0.9299 |
| 18 | 0.6769 | 0.7245 | 0.2704 | 0.9555 |
| 19 | 0.6886 | 0.7335 | 0.2695 | 0.9126 |
| 20 | 0.6749 | 0.7252 | 0.2640 | 0.9568 |
| 21–29 | 0.648–0.660 | 0.705–0.714 | 0.263–0.265 | 0.902–0.956 |

> **分析 — fold2 绝对冠军:**
> - **best 0.7254 @ epoch 12** — fold2 历史最高，比 v45 8-loss (0.6013) 高 +0.1241，比 Faithful (0.6837) 高 +0.0417。
> - epoch 6–14 区间出现多个 0.68–0.72 的尖峰，之后在 0.65–0.66 区间稳定平台。
> - AdamW wd=5e-4 相比 V2 纯 Adam 提升了约 +0.008。
> - 砍掉 rankevent 的 4 项 + AdamW 正则化在 fold2 上表现最优。

---

## 9. newSlotSPE 消融 (config: `v45_blca.yaml`, seed=3)

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
| RG-ET 30ep no seed | 3 | 30 | None | yes | 0.6389 | 0.5844 | 0.8597 | ✅ → 💥 (IBS 崩塌) |
| Stagewise (rerun) | ? | 30 | 3 | no | 0.6741 | — | 0.7886 | ✅ |
| Faithful 30ep no seed | ? | 30 | None | no | 0.6837 | 0.5301 | 0.6165 | ✅ → ⚠️ (不稳定) |
| v50 30ep no seed | 11 | 30 | None | no | 0.6749 | 0.6198 | 0.7758 | ⚠️ (train 晚于 val 爬升) |
| CATE-T 30ep no seed | ? | 30 | None | no | 0.6405 | 0.5950 | 0.8926 | ✅ (标准过拟合) |
| DCT 30ep no seed | ? | 30 | None | no | 0.6237 | 0.5936 | 0.8642 | ✅ (未收敛,持续上升) |
| **V2 关 rankevent** | **4** | **30** | **3** | **no** | **0.7174** | — | — | ✅ |
| **V4a norank+AdamW** | **4** | **30** | **3** | **no** | **🏆 0.7254** | **0.6581** | **0.5488** | **⚠️ (单峰, 后期平台)** |
| RG-ET+PCGrad | 3 | 30 | 3 | no | 0.6341 | — | — | ✅ (实际=RG-ET rerun) |
| ot_v3 (newSlotSPE) | 5 | — | — | — | ❌ (file not found) | ❌ | ❌ | ❌ |

> **🏆 V4a 以 0.7254 创 fold2 新高** — 仅用 4 loss + AdamW wd=5e-4；V2 同配置但纯 Adam 得 0.7174。

---

## 12. V45 损失子集扫描 (curated, fold2, seed=3, 30ep) — ❌ 已终止

> 原计划 10 组枚举 V45 损失组合，2 组完成后判定无继续价值，手动 kill。

| 组合 | val_cidx best | 说明 |
|------|-------------|------|
| n3: OT + event_surv + rank | 0.6165 | 远低于 V4a 0.7254 |
| n3: OT + event_surv + per_event | 0.6125 | 远低于 V4a 0.7254 |
| 剩余 8 组 | ❌ 终止 | — |

**终止原因:**
- 已完成 2 组的 cidx (0.61–0.62) 远不如 V4a 的 0.7254 + V2 的 0.7174，差距达 0.10+。
- 损失子集扫描本质是随机组合穷举，而 #1–#8 的方法对比 + #10–#11 的消融已明确指向最佳策略：**关 rankevent + AdamW + 4-loss**。
- 继续跑 10 组 V45 / 10 组 V50 只是在已有结论上加噪声，不产生新信息。

---
## 14. 当前运行队列 (2026-07-14 凌晨)

| 状态 | 方法 | 备注 |
|:--:|---|------|
| ✅ | V51 SlimBridge seed3 | 5-fold done, fold2 崩溃 (0.58) |
| 🔄 | V51 SlimBridge seed5 | fold0-1 done, fold2-4 进行中 |
| ⏳ | V60 OT Event Rank | 等 V51 完成后启动 |
| ✅ | v45_norank (5-fold seed=random) | 完成, last5=0.6406 |
| ⏳ | v45v2_norank, v50_norank | 等 V51/V60 完成后 |
| ⏳ | 批次2 (rg_et/catet/dct/faithful fix) | 等批次1完成后 |

## 15. 运行环境

- **SurvOT-Rank**: `/home/ubuntu/SurvOT-Rank` (commit `983265a`, main, +全局分箱修复)
- **newSlotSPE**: `/home/ubuntu/newSlotSPE` (commit `6b0091c`, feat/v51-slimbridge, +等频分箱修复)
- **Python**: conda env `trisurv`
- **GPU**: gpu=0
- **数据**: 5-fold BLCA, `survival_months_dss`, Pathways, n=380
