# SurvOT-Rank 多癌种实验结果汇总

> 更新时间: 2026-07-21 | Seed: 3 | DCT v3.3 Score-First + v3.5R fold0 筛选

---

## 🆕 多癌种 DCT v3.3 Score-First 实验结果 (2026-07-21)

> 本轮实验：BRCA、LUAD、LUSC 三个癌种各完成 5 折交叉验证
> 参数配置: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15, dct_lambda_ipcw_rank=0.10
> 日志目录: `logs/{dataset}_fold{0-4}.log`
- BRCA: [configs/distributional_counterfactual_transport_brca.yaml](file:///home/ubuntu/SurvOT-Rank/configs/distributional_counterfactual_transport_brca.yaml)
- LUAD: [configs/distributional_counterfactual_transport_luad.yaml](file:///home/ubuntu/SurvOT-Rank/configs/distributional_counterfactual_transport_luad.yaml)
- LUSC: [configs/distributional_counterfactual_transport_lusc.yaml](file:///home/ubuntu/SurvOT-Rank/configs/distributional_counterfactual_transport_lusc.yaml)

### BRCA (5/5 folds)
| Fold | Best C-Index | Best Epoch |
|:----:|:------------:|:----------:|
| 0 | 0.6639 | 24 |
| 1 | 0.7432 | 3 |
| 2 | **0.7510** | 17 |
| 3 | 0.6486 | 10 |
| 4 | 0.7245 | 29 |
| **Mean±Std** | **0.7062±0.0420** | — |

### LUAD (5/5 folds)
| Fold | Best C-Index | Best Epoch |
|:----:|:------------:|:----------:|
| 0 | **0.7662** | 32 |
| 1 | 0.6987 | 13 |
| 2 | 0.7297 | 2 |
| 3 | 0.6899 | 22 |
| 4 | 0.6656 | 17 |
| **Mean±Std** | **0.7100±0.0348** | — |

### LUSC (5/5 folds)
| Fold | Best C-Index | Best Epoch |
|:----:|:------------:|:----------:|
| 0 | 0.6407 | 0 |
| 1 | 0.5837 | 3 |
| 2 | 0.5800 | 22 |
| 3 | **0.6631** | 0 |
| 4 | 0.6596 | 8 |
| **Mean±Std** | **0.6254±0.0364** | — |

### 多癌种性能对比

| 排名 | 癌种 | 样本数 | Best mean±std | 最佳单折 |
|:----:|:----:|:------:|:-------------:|:--------:|
| 1 | **LUAD** | 467 | **0.7100±0.0348** | 0.7662 |
| 2 | **BRCA** | 418 | **0.7062±0.0420** | 0.7510 |
| 3 | **LUSC** | 460 | **0.6254±0.0364** | 0.6631 |

> 观察:
> - LUAD 和 BRCA 性能接近，LUSC 明显偏弱（约低 8-9 个百分点）
> - LUSC 的 fold1 出现 IPCW/IBS/IAUC 全 0 的异常（详见 [logs/lusc_fold1.log](file:///home/ubuntu/SurvOT-Rank/logs/lusc_fold1.log)）
> - LUSC fold0/fold3 在 epoch 0 即达到最佳，提示可能存在训练不稳定
> - 三个癌种的 Best Epoch 分布差异较大，过拟合趋势明显

### 与历史 BLCA 结果对比

| 癌种 | Best mean±std | Last mean | Best-Last Gap | 备注 |
|:----:|:-------------:|:---------:|:-------------:|------|
| **BLCA** | 0.7311±0.0293 | 0.6589 | 9.9% | 历史最优 |
| **LUAD** | 0.7100±0.0348 | — | — | 新结果 |
| **BRCA** | 0.7062±0.0420 | — | — | 新结果（超过历史 BRCA stable 0.6659）|
| **LUSC** | 0.6254±0.0364 | — | — | 新结果 |

> BRCA 新结果 (0.7062) 显著优于历史 BRCA stable (0.6659)，提升 +0.0403

---

## 🧪 DCT v3.5R Fold0 结果 (2026-07-21)

> 运行入口：`scripts/run_dct_v35_screen.py --variants r`
> 参数: alpha_surv=0.15, event_stratified_batches=True, slot_init_mode=deterministic, evidence_marginal_strength=1.0
> 状态: **5/5 有WSI癌种 fold0 完成**（4成功 + 1中断），IPCW/IBS/iAUC 全零 Bug 待修复

| 癌种 | Fold0 C-Index | Best Epoch | IPCW | IBS | iAUC | 状态 |
|:----:|:-------------:|:----------:|:----:|:---:|:----:|:----:|
| **LUAD** | **0.7828** | 17 | 0.0000 | 0.0000 | 0.0000 | ✅ |
| **SKCM** | **0.6686** | 4 | 0.0000 | 0.0000 | 0.0000 | ✅ |
| **BRCA** | **0.6026** | 2 | 0.0000 | 0.0000 | 0.0000 | ✅ |
| **LUSC** | **0.5962** | 3 | 0.0000 | 0.0000 | 0.0000 | ✅ |
| **BLCA** | ❌ 中断 | — | — | — | — | ⚠️ E17/50 |
| COADREAD | — | — | — | — | — | 无WSI |
| KIRC | — | — | — | — | — | 无WSI |
| UCEC | — | — | — | — | — | 无WSI |
| HNSC | — | — | — | — | — | 无WSI |
| STAD | — | — | — | — | — | 无WSI |

### 对比 v3.3 Score-First

| 癌种 | v3.3 Fold0 | v3.5R Fold0 | 差异 |
|:----:|:----------:|:-----------:|:----:|
| LUAD | 0.7662 | **0.7828** | +1.7% |
| BRCA | 0.6639 | 0.6026 | -6.1% |
| LUSC | 0.6407 | 0.5962 | -4.5% |
| BLCA | 0.7552 | ❌ | — |
| SKCM | — | 0.6686 | 新 |

### 已知 Bug
1. **IPCW/IBS/iAUC 全零** — 训练日志诊断文件存在但 `_final.pkl` 中指标为 0
2. **BLCA fold0 未生成 final.pkl** — 训练在 epoch 17 中断
3. **Fold2 全部未跑** — 脚本在处理 fold2 前退出
4. **5 癌种无 WSI 数据** — COADREAD/KIRC/UCEC/HNSC/STAD 暂无法运行

### 数据完整性
全部 10 癌种基因数据和生存标签完整，临床无缺失 RNA。WSI 缺失：BRCA 2 个 (DX2)、LUAD 1 个。

---

## 🧪 DCT v3.5 R/Q/G/L 受控筛选（计划）

> 运行入口：`scripts/run_dct_v35_screen.py`
>
> 开发协议：十癌种仅 fold0/2，batch=8，train-only bins，患者无放回分层批次，
> `alpha_surv=0.15`，IPCW rank memory 关闭。

| 版本 | 单一变量 | 正式结果目录 |
|---|---|---|
| v3.5R | 确定性验证 slots，修复基线 | `results/dct_v3.5_screen/r/<cancer>` |
| v3.5Q | 每个 slot 独立 learned query | `results/dct_v3.5_screen/q/<cancer>` |
| v3.5G | evidence marginal strength=0.25 | `results/dct_v3.5_screen/g/<cancer>` |
| v3.5L | projection=128、Transformer=1 层 | `results/dct_v3.5_screen/l/<cancer>` |

完整运行顺序、命令和入选规则见 `docs/DCT_V35_SCREENING.md`。fold0/2 只用于筛选，
最终候选仍须补齐固定 5-fold。

---

## ⚠️ DCT v3.4 事件感知实验（已暂停）

> 训练脚本: [run_dct_multicancer_formal.py](file:///home/ubuntu/SurvOT-Rank/scripts/run_dct_multicancer_formal.py)
> 日志: [logs/multicancer_formal.log](file:///home/ubuntu/SurvOT-Rank/logs/multicancer_formal.log)
> 状态: 已暂停（BRCA Fold 0, Epoch 21/50）
> 参数配置: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.6667, dct_lambda_ipcw_rank=0.10
> 事件感知采样: target=0.250, expected_events_per_batch=2.00

### BRCA Fold 0（v3.4 事件感知）详细记录

| Epoch | train_loss | train_cindex | val_cindex | ipcw | IBS | iAUC |
|:-----:|:----------:|:------------:|:----------:|:----:|:---:|:----:|
| 0 | 0.6635 | 0.4505 | **0.6189** | 0.5427 | 0.0836 | 0.5356 |
| 1 | 0.6264 | 0.4977 | 0.5114 | 0.5141 | 0.2380 | 0.5028 |
| 2 | 0.3508 | 0.7930 | 0.5552 | 0.4076 | 0.1040 | 0.6007 |
| 3 | 0.3974 | 0.8005 | 0.3793 | 0.3171 | 0.0538 | 0.5923 |
| 4 | 0.2199 | 0.9148 | 0.4342 | 0.3356 | 0.1290 | 0.5382 |
| 5 | 0.2105 | 0.9354 | 0.5605 | 0.5223 | 0.0388 | 0.5925 |
| 6 | 0.0911 | 0.9544 | 0.4769 | 0.3448 | 0.1755 | 0.5511 |
| 7 | 0.1196 | 0.8826 | 0.6055 | 0.3175 | 0.0449 | 0.7025 |
| 8 | 0.1142 | 0.9587 | 0.5365 | 0.5091 | 0.0404 | 0.5564 |
| 9 | 0.1225 | 0.9363 | 0.5932 | 0.4689 | 0.0324 | 0.6127 |
| 10 | 0.0824 | 0.9602 | 0.4874 | 0.4407 | 0.0300 | 0.5459 |
| 11 | 0.2244 | 0.8740 | 0.6067 | 0.3587 | 0.0610 | 0.6990 |
| 12 | 0.2134 | 0.9566 | 0.4348 | 0.3548 | 0.0772 | 0.4695 |
| 13 | 0.0758 | 0.9618 | 0.5733 | 0.5642 | 0.0281 | 0.4972 |
| 14 | 0.0317 | 0.9513 | 0.5546 | 0.4095 | 0.0337 | 0.5689 |
| 15 | 0.0145 | 0.9686 | 0.5769 | 0.5145 | 0.0272 | 0.6248 |
| 16 | 0.0075 | 0.9721 | 0.5038 | 0.3512 | 0.0366 | 0.6420 |
| 17 | 0.0060 | 0.9707 | 0.5625 | 0.4339 | 0.0473 | 0.6029 |
| 18 | 0.0061 | 0.9774 | 0.4863 | 0.3453 | 0.0308 | 0.5584 |
| 19 | 0.0059 | 0.9775 | 0.5961 | 0.3849 | 0.0367 | 0.6723 |
| 20 | 0.0054 | 0.9836 | 0.5663 | 0.3695 | 0.0301 | 0.5878 |
| 21 | 0.0050 | 0.9775 | 0.5456 | 0.4345 | 0.0314 | 0.5623 |

### 关键指标汇总

| 指标 | 值 | 备注 |
|------|:---:|------|
| **最佳 val C-Index** | **0.6189** | @Epoch 0 |
| **最终 train C-Index** | **0.9775** | @Epoch 21 |
| **最终 val C-Index** | **0.5456** | @Epoch 21 |
| **train/val 差距** | **0.4319** | 严重过拟合 |
| **ipcw 范围** | 0.31-0.56 | 偏低且波动 |
| **IBS 范围** | 0.027-0.238 | 不稳定 |
| **iAUC 范围** | 0.47-0.70 | 中等 |

### 问题分析

1. **严重过拟合**: 训练集 C-Index 从 0.45 快速升至 0.98，但验证集始终在 0.44-0.62 之间波动
2. **验证集不收敛**: val C-Index 在 Epoch 0 达到峰值后持续震荡，无明显上升趋势
3. **alpha_surv=0.6667**: 可能设置过高，导致生存损失权重过大，模型过度拟合训练集
4. **缺失 WSI 文件**: 2个样本缺少 UNI 特征（TCGA-A7-A6VX, TCGA-A7-A0CD），但影响有限
5. **与 v3.3 对比**: v3.3 BRCA fold0 最佳 C-Index 为 0.6639，v3.4 仅 0.6189，下降约 4.5%

### 建议

- **降低 alpha_surv**: 从 0.6667 降至 0.3-0.5
- **增加正则化**: 添加 dropout 或 weight decay
- **降低学习率**: 从 5e-4 降至 1e-4
- **早停机制**: 设置 patience=5-10，避免过度训练

---

## 历史实验记录

---

> 历史更新: 2026-07-20 | Seed: 3 | Max Epochs: 35 | Batch: 8 | 5-Fold CV | DCT v3.3 Score-First

---

## 排名总览 (按 Best C-Index mean)

| 排名 | 方法 | Folds | Best mean±std | Last mean±std | Last5 mean±std |
|:----:|------|:-----:|:-------------:|:-------------:|:--------------:|
| 1 | **DCT v3.3 — BLCA** | 5/5 | **0.7311±0.0293** | 0.6589±0.0794 | 0.6453±0.0706 |
| 2 | **V60 CA-PSA** | 5/5 | 0.7217±0.0383 | 0.6369±0.0771 | 0.6338±0.0800 |
| 3 | **dct_v3_score/no_stage_risk** | 3/5 | 0.7306±0.0301 | 0.6032±0.0299 | — |
| 4 | **dct_v3_score/no_anchor** | 3/5 | 0.6993±0.0155 | 0.6422±0.0456 | — |
| 5 | **dct_v3_score/full** | 3/5 | 0.6925±0.0196 | 0.5907±0.0337 | — |
| 6 | **DCT v3.3 — BRCA stable** | 5/5 | 0.6659±0.0445 | 0.5562±0.0676 | — |
| 7 | **DCT v3.3 — BRCA norank** | 5/5 | 0.6630±0.0501 | 0.5377±0.0660 | — |
| 8 | **dct_v3_score/evidence_cost** | 3/5 | 0.6864±0.0213 | 0.5852±0.0260 | — |
| 9 | **V70 PSPC** | 5/5 | 0.6786±0.0335 | 0.6167±0.0277 | 0.6168±0.0283 |

---

## 1. DCT v3.3 Score-First (Distributional Counterfactual Transport)

- **Config**: `configs/distributional_counterfactual_transport_blca.yaml`
- **方法**: score-first ranking + IPCW rank + anchor loss + stage risk + coordinate loss
- **Results dir**: `results/dct_v3_score_first_diagnostics/full` (folds 0,1,3,4) + `results/dct_v3_3_fold2_nan_fix` (fold 2)

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index | Last5 Mean | Source |
|:----:|:------:|:------------:|:----------:|:------------:|:----------:|--------|
| 0 | 50 | **0.7552** | 5 | 0.6482 | 0.6569 | dct_v3_score_first_diagnostics/full |
| 1 | 50 | **0.7157** | 6 | 0.5431 | 0.5314 | dct_v3_score_first_diagnostics/full |
| 2 | 50 | **0.7046** | 19 | 0.6429 | 0.6474 | dct_v3_3_fold2_nan_fix |
| 3 | 50 | **0.7104** | 35 | 0.7049 | 0.6656 | dct_v3_score_first_diagnostics/full |
| 4 | 50 | **0.7696** | 37 | 0.7553 | 0.7253 | dct_v3_score_first_diagnostics/full |
| **Mean±Std** | | **0.7311±0.0293** | | **0.6589±0.0794** | **0.6453±0.0706** | |

---

## 2. DCT v3.3 Score-First — BRCA Stable (IPCW rank enabled)

- **Config**: `configs/distributional_counterfactual_transport_brca_stable.yaml`
- **Results dir**: `results/dct_v3.3_score_first_brca_stable`
- **改进**: train-fold binning + sparse-event rank memory (64) + conservative LR (0.0002) + early stop (patience=6)
- **alpha_surv**: 0.50 (BRCA ~9% DSS events)

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index | IPCW Pairs |
|:----:|:------:|:------------:|:----------:|:------------:|:----------:|
| 0 | 7* | **0.7253** | 1 | 0.4109 | 33.2 |
| 1 | 9* | **0.6925** | 3 | 0.6603 | 34.7 |
| 2 | 12* | **0.6635** | 6 | 0.5889 | 26.4 |
| 3 | 6* | **0.6333** | 0 | 0.5404 | 23.5 |
| 4 | 14* | **0.6148** | 10 | 0.5804 | 26.5 |
| **Mean±Std** | | **0.6659±0.0445** | | **0.5562±0.0676** | |

> *Early stopped. Stable vs norank gap: +0.0029. IPCW rank 有微弱正向效果，方差略小，但提升不显著。

### 2.1 DCT v3.3 Score-First — BRCA Norank Control

- **Config**: `configs/distributional_counterfactual_transport_brca_norank_control.yaml`
- **Results dir**: `results/dct_v3.3_score_first_brca_norank_control`
- **变化**: dct_lambda_ipcw_rank=0.0, dct_ipcw_rank_memory_size=0, 其余与 stable 相同

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index |
|:----:|:------:|:------------:|:----------:|:------------:|
| 0 | 7* | **0.7300** | 1 | 0.5430 |
| 1 | 9* | **0.6931** | 3 | 0.6009 |
| 2 | 12* | **0.6595** | 6 | 0.4737 |
| 3 | 6* | **0.6281** | 0 | 0.6162 |
| 4 | 11* | **0.6045** | 5 | 0.4546 |
| **Mean±Std** | | **0.6630±0.0501** | | **0.5377±0.0660** |

> 对照组。与 stable 相比差距极小 (+0.0029)，IPCW rank 在 BRCA 上贡献有限。

---

### BLCA vs BRCA 对比

| 指标 | BLCA (381) | BRCA stable (418) | BRCA norank (418) |
|------|:----------:|:-----------------:|:-----------------:|
| Best Mean | **0.7311** | 0.6659 | 0.6630 |
| Last Mean | **0.6589** | 0.5562 | 0.5377 |
| Best-Last Gap | 9.9% | 16.5% | 18.9% |
| Best Std | ±0.0293 | ±0.0445 | ±0.0501 |

> BRCA 效果远低于 BLCA，过拟合也更严重。IPCW rank 几乎无贡献（+0.0029）。

---

## 4. V60 CA-PSA (Cohort-Anchored Adaptive Prognostic Slot Attention)

- **Config**: `configs/cohort_anchored_adaptive_prognostic_slot_attention_blca.yaml`
- **Results dir**: `results/v60_caapsa_dct_matched_blca`

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index | Last5 Mean |
|:----:|:------:|:------------:|:----------:|:------------:|:----------:|
| 0 | 50 | **0.7274** | 7 | 0.5571 | 0.5479 |
| 1 | 50 | **0.7623** | 2 | 0.6134 | 0.6105 |
| 2 | 50 | **0.6605** | 19 | 0.5773 | 0.5747 |
| 3 | 50 | **0.7421** | 36 | 0.7224 | 0.7220 |
| 4 | 50 | **0.7162** | 44 | 0.7144 | 0.7139 |
| **Mean±Std** | | **0.7217±0.0383** | | **0.6369±0.0771** | **0.6338±0.0800** |

> Fold 0/1 早期过拟合严重 (Best@epoch 7/2); fold 2 首次运行卡死在 epoch 24，重跑完成; fold 3/4 相对稳定

---

## 5. V70 PSPC (Patient-Specific Prognostic Circuits)

- **Config**: `configs/v70_pspc_blca.yaml`
- **Results dir**: `results/v70_pspc_dct_matched_blca`

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index | Last5 Mean |
|:----:|:------:|:------------:|:----------:|:------------:|:----------:|
| 0 | 50 | **0.6648** | 4 | 0.6054 | 0.6048 |
| 1 | 50 | **0.6701** | 6 | 0.6058 | 0.6051 |
| 2 | 50 | **0.6373** | 15 | 0.5821 | 0.5821 |
| 3 | 50 | **0.6951** | 14 | 0.6470 | 0.6466 |
| 4 | 50 | **0.7260** | 34 | 0.6432 | 0.6457 |
| **Mean±Std** | | **0.6786±0.0335** | | **0.6167±0.0277** | **0.6168±0.0283** |

> 整体偏弱，过拟合明显 (fold 0/1 best@epoch 4/6)

---

## 6. dct_v3_score ablated variants (仅 3/5 fold)

### 6.1 dct_v3_score / no_stage_risk

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index |
|:----:|:------:|:------------:|:----------:|:------------:|
| 0 | 50 | **0.7599** | 10 | 0.5697 |
| 2 | 50 | **0.6998** | 40 | 0.6125 |
| 3 | 50 | **0.7322** | 27 | 0.6273 |
| **Mean (3/5)** | | **0.7306** | | **0.6032** |

### 6.2 dct_v3_score / no_anchor

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index |
|:----:|:------:|:------------:|:----------:|:------------:|
| 0 | 50 | **0.6933** | 4 | 0.6743 |
| 2 | 50 | **0.6878** | 19 | 0.5901 |
| 3 | 50 | **0.7169** | 32 | 0.6623 |
| **Mean (3/5)** | | **0.6993** | | **0.6422** |

### 6.3 dct_v3_score / full

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index |
|:----:|:------:|:------------:|:----------:|:------------:|
| 0 | 50 | **0.7068** | 9 | 0.5523 |
| 2 | 50 | **0.6701** | 28 | 0.6045 |
| 3 | 50 | **0.7005** | 44 | 0.6153 |
| **Mean (3/5)** | | **0.6925** | | **0.5907** |

### 6.4 dct_v3_score / evidence_cost

| Fold | Epochs | Best C-Index | Best Epoch | Last C-Index |
|:----:|:------:|:------------:|:----------:|:------------:|
| 0 | 50 | **0.7044** | 7 | 0.6149 |
| 2 | 50 | **0.6629** | 8 | 0.5669 |
| 3 | 50 | **0.6918** | 13 | 0.5738 |
| **Mean (3/5)** | | **0.6864** | | **0.5852** |

---

## 与 SlotSPE 基准对比

| 方法 | Best mean | Last mean | 备注 |
|------|:---------:|:---------:|------|
| ot_v3 (SlotSPE 最高) | 0.7282 | 0.6013 | |
| otehv2_capacity (最稳定) | 0.7075 | **0.6708** | |
| otehv2_rankevent_seed5 | 0.7158 | 0.6604 | |
| **DCT v3.3 Score-First (BLCA)** | **0.7311** | 0.6589 | |
| **DCT v3.3 — BRCA stable** | 0.6659 | 0.5562 | IPCW rank +0.0029, 几乎无效果 |
| **DCT v3.3 — BRCA norank** | 0.6630 | 0.5377 | 对照组 |
| V60 CA-PSA | 0.7217 | 0.6369 | |
| V70 PSPC | 0.6786 | 0.6167 | |

---

## 结论

1. **DCT v3.3 BLCA (0.7311)** 达到 SlotSPE ot_v3 (0.7282) 水平，且 Last mean (0.6589) 优于 ot_v3 (0.6013)
2. **DCT v3.3 BRCA stable (0.6659)** vs norank (0.6630)：IPCW rank 提升仅 +0.0029，几乎无效
3. BRCA 效果远低于 BLCA，原因待分析（数据异质性、事件率低等）
4. 所有方法存在不同程度的过拟合，Last/Best 差距约 0.07-0.13
5. dct_v3_score 消融实验中 no_anchor 变体 Last mean 最高 (0.6422)，去除 anchor 对稳定性有益
6. V70 PSPC 整体偏弱 (0.6786)，不推荐继续
7. 归档文件: `reproducibility_archives/` (summary CSV + epoch curves + manifest)

---

## 多癌种数据集目录 (10 个)

| 癌种 | 样本 | Clinical | Omics | 5-fold | WSI | DCT 3.3 | 缺失项 |
|:----:|:----:|:--------:|:-----:|:------:|:---:|:-------:|--------|
| BLCA | 381 | Y | Y | Y | 457 | **Done: 0.7311** | — |
| BRCA | 418 | Y | Y | Y | 1131 | **Done: 0.6659** | — |
| UCEC | 488 | Y | Y | Y | 0 | 不运行 | **无 WSI** |
| LUAD | 467 | Y | Y | Y | Y | v3.5R fold0 已有 | — |
| COADREAD | 573 | Y | Y | Y | 0 | 不运行 | **无 WSI** |
| KIRC | 488 | Y | Y | Y | 0 | 不运行 | **无 WSI** |
| LUSC | 460 | Y | Y | Y | Y | v3.5R fold0 已有 | — |
| HNSC | 438 | Y | Y | Y | 0 | 不运行 | **无 WSI** |
| SKCM | 409 | Y | Y | Y | Y | v3.5R fold0 已有 | — |
| STAD | 366 | Y | Y | Y | 0 | 不运行 | **无 WSI** |

### CPTAC 数据集

| 癌种 | 样本 | Clinical | Omics | 5-fold | WSI | 缺失项 |
|:----:|:----:|:--------:|:-----:|:------:|:---:|--------|
| CPTAC-LUAD | 57 | Y | **空** | Y | — | **RNA pathway data** |
| CPTAC-LUSC | 33 | Y | **空** | Y | — | **RNA pathway data** |

### 数据准备清单

**当前可运行 WSI patches（5 个）：** BRCA, LUAD, LUSC, BLCA, SKCM
> 放置路径: `/data/CPathPatchFeature/{study}/uni/pt_files/*.pt`

**当前无 WSI、不能进入 WSI+RNA 主实验（5 个）：** COADREAD, KIRC, UCEC, HNSC, STAD

**只需 RNA 数据（2 个）：** CPTAC-LUAD, CPTAC-LUSC
> 从 cBioPortal 下载 mRNA expression，处理后放入 `raw_rna_data_inter/`

**配置文件**: 每个癌种需要 `configs/distributional_counterfactual_transport_{study}.yaml`，与 BLCA 完全相同参数（仅替换 study 名和结果目录）。
