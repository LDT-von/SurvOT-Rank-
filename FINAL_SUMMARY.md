# SurvOT-Rank 多癌种实验结果汇总

> 更新时间: 2026-07-20 | Seed: 3 | Max Epochs: 35 | Batch: 8 | 5-Fold CV | DCT v3.3 Score-First

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
| UCEC | 488 | Y | Y | Y | 0 | 待跑 | **WSI** |
| LUAD | 467 | Y | Y | Y | 0 | 待跑 | **WSI** |
| COADREAD | 573 | Y | Y | Y | 0 | 待跑 | **WSI** |
| KIRC | 488 | Y | Y | Y | 0 | 待跑 | **WSI** |
| LUSC | 460 | Y | Y | Y | 0 | 待跑 | **WSI** |
| HNSC | 438 | Y | Y | Y | 0 | 待跑 | **WSI** |
| SKCM | 409 | Y | Y | Y | 0 | 待跑 | **WSI** |
| STAD | 366 | Y | Y | Y | 0 | 待跑 | **WSI** |

### CPTAC 数据集

| 癌种 | 样本 | Clinical | Omics | 5-fold | WSI | 缺失项 |
|:----:|:----:|:--------:|:-----:|:------:|:---:|--------|
| CPTAC-LUAD | 57 | Y | **空** | Y | — | **RNA pathway data** |
| CPTAC-LUSC | 33 | Y | **空** | Y | — | **RNA pathway data** |

### 数据准备清单

**只需 WSI patches（8 个）：** UCEC, LUAD, COADREAD, KIRC, LUSC, HNSC, SKCM, STAD
> 放置路径: `/data/CPathPatchFeature/{study}/uni/pt_files/*.pt`

**只需 RNA 数据（2 个）：** CPTAC-LUAD, CPTAC-LUSC
> 从 cBioPortal 下载 mRNA expression，处理后放入 `raw_rna_data_inter/`

**配置文件**: 每个癌种需要 `configs/distributional_counterfactual_transport_{study}.yaml`，与 BLCA 完全相同参数（仅替换 study 名和结果目录）。
