# SurvOT-Rank 多癌种实验结果汇总

> 更新时间: 2026-07-30 | Seed: 3 | 版本: v3.3 / v3.4 / v3.5 / v3.6 / v3.7 / v3.8 / v4.0 / v4.1

---

## Split 数据修复记录 (2026-07-27)

> `tools/gen_splits_5fold.py` 中 `zip(train, val)` 按短列截断长列的 bug 已在 commit `8054f57` 修复。

| 癌种 | clinical | Split有效 | 缺失原因 | Train/Val重叠 |
|:----:|:--------:|:--------:|----------|:-----------:|
| BLCA | 381 | 380 | 1人缺DSS标签 | 无 |
| BRCA | 1046 | **1045** | 1人缺DSS标签 | 无 |
| COADREAD | 573 | 570 | 3人缺DSS标签 | 无 |
| HNSC | 438 | 437 | 1人缺DSS标签 | 无 |
| KIRC | 488 | 488 | — | 无 |
| LUAD | 467 | 458 | 9人缺DSS标签 | 无 |
| LUSC | 460 | 454 | 6人缺DSS标签 | 无 |
| SKCM | 403 | 403 | — (已修复) | 无 |
| STAD | 366 | 362 | 4人缺DSS标签 | 无 |
| UCEC | 488 | 487 | 1人缺DSS标签 | 无 |

> 所有癌种每人恰好进入一次验证集，无 train/val 重叠。

---

## ⚠️ BRCA 全部结果已删除 (2026-07-27 18:00)

> v3.3/v3.4/v3.6/v3.7 所有 BRCA 实验结果使用的 split 仅含 **775 人**（正确应为 1045 人），原因是旧 split 文件生成时临床数据仅 775 人。
> 已删除以下目录：
> - `results/dct_v3.3_score_first_brca` (v3.3)
> - `results/dct_brca_recovery` (v3.4)
> - `results/dct_v3.6_listwise/*/brca` (v3.6 × 4变体)
> - `results/dct_v3.7_uni2h/highscore/brca` (v3.7)
> - 所有 smoke test 和旧实验的 brca 目录
>
> **BRCA 需用正确 split (1045人) 全部重跑。** v3.8 和 v4.0 会使用当前磁盘上正确的 split。

---

## 版本总览

| 版本 | 名称 | 方法 | WSI编码器 | 损失函数 | 状态 |
|:----:|------|------|:--------:|----------|:----:|
| **v3.3** | Score-First | distributional_counterfactual_transport | UNI v1 (1024d) | NLL + IPCW rank | 🔄 6/10 |
| **v3.4** | BRCA Recovery | 多种消融实验 | UNI v1 | NLL 等 | ❌ 已删除 |
| **v3.5** | Smoke Tests | 烟雾测试 | UNI v1 | — | ⚠️ 非正式 |
| **v3.6** | Listwise | 6变体: NLL/IPCW/ETAR/IPCW+ETAR/GPL/TCL | UNI v1 | NLL + listwise | ⚠️ Listwise暂停 |
| **v3.7** | UNI2-h HighScore | distributional_counterfactual_transport | UNI2-h (1536d) | NLL + IPCW rank | 🔄 6/10 (LUAD完成) |
| **v3.8** | Transport Consistency | dct_transport_intervention_consistency | UNI2-h (1536d) | direction+dose+reconfiguration | 🔄 2/4 (BLCA+BRCA完成) |
| **v4.0** | IST-Surv | intervention_stable_survival_transport | UNI2-h (1536d) | 干预稳定性 | ⏸ 已中断 |
| **v4.1** | Survival Evidence Ledger | dct_v41_survival_evidence_ledger | UNI v1 (1024d) | 证据账本 | ❌ 已暂停 |

---

## DCT v3.3 Score-First (UNI v1)

> survot_method: distributional_counterfactual_transport
> 参数: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15, dct_lambda_ipcw_rank=0.10
> dct_lambda_etar=0.0, dct_lambda_listwise=0.0, dct_lambda_ot=0.0
> 结果目录: `results/dct_v3.3_score_first_<cancer>/`

### UCEC (487 样本) — Mean: 0.7964 ± 0.0342 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.7565 | 7 |
| 1 | 0.8104 | 49 |
| 2 | 0.7548 | 36 |
| 3 | **0.8340** | 34 |
| 4 | 0.8265 | 47 |

### KIRC (488 样本) — Mean: 0.7958 ± 0.0168 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.8164** | 24 |
| 1 | 0.7886 | 6 |
| 2 | 0.7842 | 6 |
| 3 | 0.7750 | 3 |
| 4 | 0.8149 | 24 |

### BLCA (380 样本) — Mean: 0.7311 ± 0.0262 ✅

> Fold 2 曾在 epoch 7 因 NaN 崩溃，从 `dct_v3_3_fold2_nan_fix` 单独重跑

| Fold | C-Index | Epoch | 来源 |
|:----:|:------:|:-----:|:-----|
| 0 | 0.7552 | 4 | diagnostics/full |
| 1 | 0.7157 | 5 | diagnostics/full |
| 2 | 0.7046 | 18 | fold2_nan_fix |
| 3 | 0.7104 | 34 | diagnostics/full |
| 4 | **0.7696** | 36 | diagnostics/full |

### COADREAD (570 样本) — Mean: 0.6774 ± 0.0515 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.7306** | 41 |
| 1 | 0.6017 | 7 |
| 2 | 0.6321 | 46 |
| 3 | 0.7241 | 41 |
| 4 | 0.6985 | 29 |

### SKCM (403 样本) — Mean: 0.6770 ± 0.0513 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.6988 | 6 |
| 1 | 0.7155 | 11 |
| 2 | 0.6466 | 11 |
| 3 | 0.5920 | 2 |
| 4 | **0.7321** | 8 |

### STAD (362 样本) — Mean: 0.6596 ± 0.0216 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.6713 | 28 |
| 1 | 0.6185 | 7 |
| 2 | 0.6683 | 47 |
| 3 | 0.6592 | 41 |
| 4 | **0.6806** | 34 |

### 待跑 — ⏳

| 癌种 | 样本 | 状态 |
|:----:|:---:|:----:|
| BRCA | 1045 | ❌ 已删除，待重跑 |
| HNSC | 437 | ⏳ |
| LUAD | 458 | ⏳ |
| LUSC | 454 | ⏳ |

### v3.3 汇总

| 排名 | 癌种 | 样本 | Mean ± Std | 最佳 | 最差 | Δ | 状态 |
|:----:|:----:|:---:|:----------:|:----:|:----:|:--:|:----:|
| 1 | UCEC | 487 | 0.7964 ± 0.0342 | 0.8340 | 0.7548 | 0.079 | ✅ |
| 2 | KIRC | 488 | 0.7958 ± 0.0168 | 0.8164 | 0.7750 | 0.041 | ✅ |
| 3 | BLCA | 380 | 0.7311 ± 0.0262 | 0.7696 | 0.7046 | 0.065 | ✅ |
| 4 | COADREAD | 570 | 0.6774 ± 0.0515 | 0.7306 | 0.6017 | 0.129 | ✅ |
| 5 | SKCM | 403 | 0.6770 ± 0.0513 | 0.7321 | 0.5920 | 0.140 | ✅ |
| 6 | STAD | 362 | 0.6596 ± 0.0216 | 0.6806 | 0.6185 | 0.062 | ✅ |
| 7 | BRCA | 1045 | — | — | — | — | ⏳ 待重跑 |
| 8-10 | HNSC/LUAD/LUSC | — | — | — | — | — | ⏳ |

---

## DCT v3.4 — BRCA Recovery 消融实验

> ❌ 已删除。所有 BRCA Recovery 数据因 split 错误（775人）被删除，待重跑。

---

## DCT v3.5 — Smoke Tests

> 烟雾测试版本，仅用于验证 pipeline。**非正式结果，不纳入对比。**

---

## DCT v3.6_listwise (UNI v1)

> 结果目录: `results/dct_v3.6_listwise/<variant>/<cancer>/`
> 参数: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15
> 目标: 4 癌种 (blca/brca/luad/lusc), 6 变体, 5 折
> **当前仅完成 fold0 和 fold2。BRCA 已删除待重跑。**

---

### 类别 A: 消融实验变体 (survot_method=distributional_counterfactual_transport)

#### A1. NLL (纯 NLL 生存损失)

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.7314 | 8 | 0.6693 | 12 | 0.7004 |
| LUAD | 0.7828 | 15 | 0.7466 | 9 | **0.7647** |
| LUSC | 0.6525 | 8 | 0.5757 | 0 | **0.6141** |
| BRCA | ❌ | — | ❌ | — | 待重跑 |

#### A2. IPCW (NLL + IPCW rank)

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.6989 | 17 | 0.6589 | 6 | 0.6789 |
| LUAD | 0.7828 | 17 | 0.7017 | 3 | 0.7423 |
| LUSC | 0.5962 | 3 | 0.5739 | 0 | 0.5850 |
| BRCA | ❌ | — | ❌ | — | 待重跑 |

#### A3. ETAR (NLL + ETAR)

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.7282 | 14 | 0.6765 | 9 | 0.7024 |
| LUAD | 0.8104 | 12 | 0.7051 | 3 | 0.7577 |
| LUSC | 0.5795 | 0 | 0.6096 | 10 | 0.5946 |
| BRCA | ❌ | — | ❌ | — | 待重跑 |

#### A4. IPCW+ETAR ✅ 全部完成

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.7163 | 3 | 0.6589 | 8 | 0.6876 |
| LUAD | 0.7654 | 16 | 0.6975 | 3 | 0.7314 |
| LUSC | 0.5755 | 0 | 0.5661 | 0 | 0.5708 |
| BRCA | ❌ | — | ❌ | — | 待重跑 |

#### 消融实验对比汇总

| 癌种 | NLL | IPCW | ETAR | IPCW+ETAR | 最佳 |
|:----:|:----:|:----:|:----:|:--------:|:----:|
| BLCA | 0.7004 | 0.6789 | **0.7024** | 0.6876 | ETAR |
| LUAD | **0.7647** | 0.7423 | 0.7577 | 0.7314 | NLL |
| LUSC | **0.6141** | 0.5850 | 0.5946 | 0.5708 | NLL |
| BRCA | — | — | — | — | 待重跑 |

---

### 类别 B: Listwise Transport 变体 (survot_method=dct_listwise_transport)

#### B1. GPL (Global Plackett-Luce) ❌ 已暂停

> GPL fold0=0.6410 @epoch22，低于消融变体（ETAR=0.7024），已终止。

| 癌种 | Fold0 | Fold2 | 状态 |
|:----:|:-----:|:-----:|:----:|
| BLCA | 0.6410 @epoch22 | — | ❌ |
| BRCA | ⏳ | ⏳ | 待重跑 |
| LUAD | ⏳ | ⏳ | ❌ 不跑 |
| LUSC | ⏳ | ⏳ | ❌ 不跑 |

#### B2. TCL (Transport-Conditioned Listwise) ❌ 已暂停

> survot_method: dct_listwise_transport, dct_listwise_mode=stage_transport, dct_lambda_listwise=0.10
> TCL fold0=0.6473 @epoch15，低于消融变体（ETAR=0.7024），已终止。

| 癌种 | Fold0 | Fold2 | 状态 |
|:----:|:-----:|:-----:|:----:|
| BLCA | 0.6473 @epoch15 | — | ❌ |
| BRCA | ⏳ | ⏳ | 待重跑 |
| LUAD | ⏳ | ⏳ | ❌ 不跑 |
| LUSC | ⏳ | ⏳ | ❌ 不跑 |

> **结论**: GPL=0.6410, TCL=0.6473，两变体均低于消融变体（ETAR=0.7024），listwise 全局排序在此规模上效果不佳。**不建议继续。**

---

### v3.6 待办清单

| 类别 | 变体 | 癌种 | 已完成 | 剩余 |
|:----:|:----:|:----:|:-----:|:---:|
| A | NLL | blca/luad/lusc | fold0+2 (6次) | fold1/3/4 (9次) |
| A | IPCW | blca/luad/lusc | fold0+2 (6次) | fold1/3/4 (9次) |
| A | ETAR | blca/luad/lusc | fold0+2 (6次) | fold1/3/4 (9次) |
| A | IPCW+ETAR | blca/luad/lusc | fold0+2 (6次) | fold1/3/4 (9次) |
| B | GPL | blca(+3癌种) | 2次 | 全5折 (18次) |
| B | TCL | blca(+3癌种) | 0次 | 全5折 (20次) |
| 🔴 | **全部BRCA** | blca→brca | 0次 | v3.6全变体 (24次) |

---

## DCT v3.7_uni2h_highscore (UNI2-h)

> survot_method: distributional_counterfactual_transport
> 参数: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15, dct_lambda_ipcw_rank=0.10
> wsi_encoder=uni2-h, encoding_dim=1536, fit_bins_on_train=false, slot_init=gaussian

### KIRC (488 样本) — Mean: 0.8149 ± 0.0169 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.8323** | 10 |
| 1 | 0.8152 | 27 |
| 2 | 0.7987 | 30 |
| 3 | 0.7971 | 8 |
| 4 | 0.8311 | 34 |

### BLCA (380 样本) — Mean: 0.7249 ± 0.0359 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.7448 | 5 |
| 1 | **0.7555** | 9 |
| 2 | 0.6597 | 13 |
| 3 | 0.7126 | 17 |
| 4 | 0.7518 | 40 |

### COADREAD (570 样本) — Mean: 0.7384 ± 0.0405 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.7803** | 17 |
| 1 | 0.7764 | 31 |
| 2 | 0.6692 | 1 |
| 3 | 0.7423 | 22 |
| 4 | 0.7240 | 1 |

### HNSC (437 样本) — Mean: 0.6406 ± 0.0244 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.6652 | 22 |
| 1 | 0.6359 | 6 |
| 2 | **0.6722** | 13 |
| 3 | 0.6193 | 7 |
| 4 | 0.6105 | 14 |

### LUAD (458 样本) — Mean: 0.6662 ± 0.0397 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.6791 | 7 |
| 1 | 0.6307 | 39 |
| 2 | 0.6864 | 2 |
| 3 | 0.6202 | 3 |
| 4 | **0.7147** | 1 |

### LUSC (454 样本) — 🔄 仅 fold0

| Fold | C-Index | Epoch | 状态 |
|:----:|:------:|:-----:|:----:|
| 0 | 0.4575 | 1 | ✅ |
| 1 | — | — | ⏳ |
| 2 | — | — | ⏳ |
| 3 | — | — | ⏳ |
| 4 | — | — | ⏳ |

> fold0=0.4575 极低，可能存在问题，需排查

### 待跑/待重跑

| 癌种 | 样本 | 状态 |
|:----:|:---:|:----:|
| BRCA | 1045 | ❌ 已删除，待重跑 |
| LUSC fold1-4 | 454 | ⏳ |
| SKCM | 403 | ⏳ |
| STAD | 362 | ⏳ |
| UCEC | 487 | ⏳ |

### v3.7 汇总

| 排名 | 癌种 | 样本 | Mean ± Std | 最佳 | 最差 | Δ | 状态 |
|:----:|:----:|:---:|:----------:|:----:|:----:|:--:|:----:|
| 1 | KIRC | 488 | 0.8149 ± 0.0169 | 0.8323 | 0.7971 | 0.035 | ✅ |
| 2 | COADREAD | 570 | 0.7384 ± 0.0405 | 0.7803 | 0.6692 | 0.111 | ✅ |
| 3 | BLCA | 380 | 0.7249 ± 0.0359 | 0.7555 | 0.6597 | 0.096 | ✅ |
| 4 | LUAD | 458 | 0.6662 ± 0.0397 | 0.7147 | 0.6202 | 0.095 | ✅ |
| 5 | HNSC | 437 | 0.6406 ± 0.0244 | 0.6722 | 0.6105 | 0.062 | ✅ |
| 6 | LUSC | 454 | — | — | — | — | 🔄 fold0=0.4575 |
| 7 | BRCA | 1045 | ❌ | — | — | — | 待重跑 |

---

## DCT v3.8 Transport Consistency (UNI2-h)

> survot_method: dct_transport_intervention_consistency
> 参数: max_epochs=50, batch_size=8, UNI2-h (1536d), ipcw_rank=0.10
> v38_lambda_direction=0.05, v38_lambda_dose=0.03, v38_lambda_reconfiguration=0.02
> target: 4 癌种 × 5 折 (blca/brca/luad/lusc)

### BLCA (380 样本) — Mean: 0.7274 ± 0.0351 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.7591** | 7 |
| 1 | 0.7343 | 5 |
| 2 | 0.7199 | 5 |
| 3 | 0.7528 | 6 |
| 4 | 0.6710 | 11 |

> 5折完成！fold0=0.7591 为单折最高，fold3=0.7528 @epoch6 冲到很高后震荡

### BRCA (1045 样本) — Mean: 0.6750 ± 0.0573 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.7358** | 20 |
| 1 | 0.6399 | 15 |
| 2 | 0.7205 | 4 |
| 3 | 0.6817 | 2 |
| 4 | 0.5970 | 0 |

> 5折完成！fold0=0.7358 最佳，fold4=0.5970 偏低 (epoch 0 即最佳)，方差较大

### 待跑

| 癌种 | 状态 |
|:----:|:----:|
| LUAD fold0 | 🔄 运行中 (epoch 24/50, best 0.7369 @epoch16) |
| LUAD fold1-4 | ⏳ |
| LUSC | ⏳ |

> BLCA 5折均值 0.7274，与 v3.3 (0.7311) 和 v3.7 (0.7249) 持平，单折最高 0.7591 超过两者

---

## DCT v4.0 — IST-Surv 🔄 运行中

> survot_method: intervention_stable_survival_transport
> 参数: UNI2-h, ist_eps=0.05, ist_num_interventions=3, ist_deletion_penalty=8.0
> 目标: 2 癌种 (blca/brca), 5 折

### BLCA (380 样本) — 🔄

| Fold | C-Index | Epoch | 状态 |
|:----:|:------:|:-----:|:----:|
| 0 | 0.6582 | 4 | 🔄 epoch 38/50 |

> fold0 最佳 0.6582 @epoch4，训练极慢 (~53s/epoch)，仍在早期阶段，后续被调度器中断

---
 
## DCT v4.1 — Survival Evidence Ledger ❌ 已暂停

> survot_method: dct_v41_survival_evidence_ledger
> 配置: `dct_v41_survival_evidence_ledger_*.yaml`
> 目标: 4 癌种 (blca/brca/hnsc/stad), 仅 folds 0/2/4

### BLCA (380 样本, UNI v1 1024d) — ❌ 分数偏低，已暂停

| Fold | C-Index | Epoch | 状态 |
|:----:|:------:|:-----:|:----:|
| 0 | 0.6201 | 20 | ✅ |
| 2 | **0.6985** | 25 (best so far) | ❌ stopped @epoch 37 |
| 4 | — | — | ❌ 未跑 |

> fold0=0.6201 偏低，fold2=0.6985 回升但低于 v3.8 BLCA (0.7415)。已终止运行。

---

## 📈 版本对比

| 癌种 | v3.3 (UNIv1) | v3.6 最佳 (UNIv1) | v3.7 (UNI2-h) | v3.8 (UNI2-h) |
|:----:|:------------:|:-----------------:|:-------------:|:-------------:|
| UCEC | **0.7964** | — | ⏳ | — |
| KIRC | 0.7958 | — | **0.8149** | — |
| COADREAD | 0.6774 | — | **0.7384** | — |
| BLCA | 0.7311 | 0.7024 (ETAR) | 0.7249 | 0.7274 |
| HNSC | ⏳ | — | 0.6406 | — |
| LUAD | ⏳ | 0.7647 (NLL) | 0.6662 | ⏳ |
| LUSC | ⏳ | 0.6141 (NLL) | 🔄 fold0=0.4575 | ⏳ |
| SKCM | 0.6770 | — | ⏳ | — |
| STAD | 0.6596 | — | ⏳ | — |
| BRCA | ⏳ 待重跑 | ⏳ 待重跑 | ⏳ 待重跑 | 0.6750 ✅ |

### v3.3 → v3.7 提升

| 癌种 | v3.3 | v3.7 | Δ |
|:----:|:----:|:----:|:--:|
| KIRC | 0.7958 | 0.8149 | **+1.91%** |
| COADREAD | 0.6774 | 0.7384 | **+6.10%** |
| BLCA | 0.7311 | 0.7249 | −0.62% |
| LUAD | ⏳ | 0.6662 | — |

### v3.7 → v3.8 提升

| 癌种 | v3.7 | v3.8 | Δ |
|:----:|:----:|:----:|:--:|
| BLCA | 0.7249 | **0.7274** | **+0.25%** |
| BRCA | — | **0.6750** | 新结果 |

---

## 🏆 排名总览 (按 Best C-Index)

| 排名 | 版本 | 癌种 | 分数 | 类型 | 状态 |
|:----:|:----:|:----:|:----:|:----:|:----:|
| 1 | v3.3 | UCEC | **0.8340** | Fold3 单折 | ✅ |
| 2 | v3.7 | KIRC | **0.8323** | Fold0 单折 | ✅ |
| 3 | v3.7 | KIRC | 0.8311 | Fold4 单折 | ✅ |
| 4 | v3.3 | UCEC | 0.8265 | Fold4 单折 | ✅ |
| 5 | v3.3 | KIRC | 0.8164 | Fold0 单折 | ✅ |
| 6 | v3.7 | KIRC | 0.8152 | Fold1 单折 | ✅ |
| 7 | v3.7 | KIRC | **0.8149** | **5折均值** | ✅ |
| 8 | v3.6 | LUAD | 0.8104 | Fold0 单折 | ✅ |
| 9 | v3.7 | KIRC | 0.7987 | Fold2 单折 | ✅ |
| 10 | v3.3 | UCEC | **0.7964** | **5折均值** | ✅ |
| — | v3.8 | BLCA | 0.7591 | Fold0 单折 | ✅ |

---

## 🔄 当前运行状态 (2026-07-30 02:10)

> 🔄 **v3.8 LUAD fold0 正在训练**，调度器已跑完 BRCA 5 折，当前处理 LUAD

| 版本 | 癌种 | Fold | 进度 | 最佳 C-Index | 状态 |
|:----:|:----:|:----:|------|:----------:|:----:|
| v3.8 | BLCA | 全部5折 | ✅ 完成 | 0.7274 mean | ✅ |
| v3.8 | BRCA | 全部5折 | ✅ 完成 | 0.6750 mean | ✅ |
| v3.8 | LUAD | fold0 | epoch 24/50 | 0.7369 @epoch16 | 🔄 运行中 |
| v3.8 | LUAD | fold1-4 | — | — | ⏳ |
| v3.8 | LUSC | 全部5折 | — | — | ⏳ |
| v3.7 | LUAD | 全部5折 | ✅ 完成 | 0.6662 mean | ✅ |
| v3.7 | LUSC | fold0 | ✅ 完成 | 0.4575 | ⚠️ |
| v4.0 | BLCA | fold0 | epoch 38/50 | 0.6582 @epoch4 | ⏸ 中断 |
| v3.6 GPL/TCL | — | — | — | — | ❌ 已暂停 |
| v4.1 | — | — | — | — | ❌ 已暂停 |

### 待办

| 优先级 | 任务 |
|:------:|:-----|
| 🔴 | **v3.8 LUAD/LUSC** — 等调度器跑完剩下癌种 |
| 🟡 | **BRCA 其他版本重跑** (v3.3/3.6/3.7) — 需用正确 1045 人 split |
|  | v3.6 消融实验 fold1/3/4 |
| 🟡 | v3.7 剩余癌种 (LUSC fold1-4/SKCM/STAD/UCEC) |
| ⬜ | v4.0 继续 (BLCA fold0→4 + BRCA) |
| ⬜ | v4.1 不继续，分数低 |
| ⚠️ | **v3.7 LUSC fold0=0.4575 异常**，需排查数据/配置 |
| ⚠️ | **v3.8 5个LUAD fold0进程同时运行**，疑似调度器重复启动，需确认 |

---

## 🏥 多癌种数据集总览 (10 个)

| 癌种 | clinical | Split | v3.3 | v3.6 最佳 | v3.7 | v3.8 |
|:----:|:--------:|:-----:|:-----:|:---------:|:----:|:----:|
| BLCA | 381 | 380 | 0.7311 ✅ | 0.7024 ETAR | 0.7249 ✅ | 0.7274 ✅ |
| BRCA | 1046 | 1045 | ⏳ 待重跑 | ⏳ 待重跑 | ⏳ 待重跑 | 0.6750 ✅ |
| COADREAD | 573 | 570 | 0.6774 ✅ | — | 0.7384 ✅ | — |
| HNSC | 438 | 437 | ⏳ | — | 0.6406 ✅ | — |
| KIRC | 488 | 488 | 0.7958 ✅ | — | 0.8149 ✅ | — |
| LUAD | 467 | 458 | ⏳ | 0.7647 NLL | 0.6662 ✅ | ⏳ |
| LUSC | 460 | 454 | ⏳ | 0.6141 NLL | 🔄 fold0=0.4575 | ⏳ |
| SKCM | 403 | 403 | 0.6770 ✅ | — | ⏳ | — |
| STAD | 366 | 362 | 0.6596 ✅ | — | ⏳ | — |
| UCEC | 488 | 487 | 0.7964 ✅ | — | ⏳ | — |
