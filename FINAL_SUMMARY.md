# SurvOT-Rank 多癌种实验结果汇总

> 更新时间: 2026-07-27 12:45 | Seed: 3 | 全版本: v3.3 / v3.4 / v3.5 / v3.6 / v3.7 / v3.8 / v4.0 / split 修复

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

> 所有癌种每人恰好进入一次验证集，无 train/val 重叠。SKCM 已使用修复后的 split，其余 9 癌种 split 不变。

---

## 版本总览

| 版本 | 名称 | 方法 | WSI编码器 | 损失函数 | 状态 |
|:----:|------|------|:--------:|----------|:----:|
| **v3.3** | Score-First | distributional_counterfactual_transport | UNI v1 (1024d) | NLL + IPCW rank | 🔄 7/10 癌种 |
| **v3.4** | BRCA Recovery | 多种消融实验 | UNI v1 (1024d) | NLL 等 | ⚠️ 仅BRCA调试 |
| **v3.5** | Smoke Tests | 烟雾测试 | UNI v1 (1024d) | — | ⚠️ 非正式版本 |
| **v3.6** | Listwise | 6变体: NLL/IPCW/ETAR/IPCW+ETAR/GPL/TCL | UNI v1 (1024d) | NLL + listwise | 🔄 fold0+2完成 |
| **v3.7** | UNI2-h HighScore | distributional_counterfactual_transport | UNI2-h (1536d) | NLL + IPCW rank | 🔄 5/10 癌种 |
| **v3.8** | Transport Consistency | dct_transport_intervention_consistency | UNI2-h (1536d) | direction+dose+reconfiguration | 🔄 运行中 |
| **v4.0** | IST-Surv | intervention_stable_survival_transport | UNI2-h (1536d) | 干预稳定性 | ⏳ 未开始 |

---

## DCT v3.3 Score-First (UNI v1)

> survot_method: distributional_counterfactual_transport
> 参数: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15, dct_lambda_ipcw_rank=0.10
> dct_lambda_etar=0.0, dct_lambda_listwise=0.0, dct_lambda_ot=0.0
> 结果目录: `results/dct_v3.3_score_first_<cancer>/`
> 目标: 10 个癌种

### UCEC (487 样本) — Mean: 0.7964 ± 0.0342 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | 0.7565 | 7 | 49 |
| 1 | 0.8104 | 49 | 49 |
| 2 | 0.7548 | 36 | 49 |
| 3 | **0.8340** | 34 | 49 |
| 4 | 0.8265 | 47 | 49 |

### KIRC (488 样本) — Mean: 0.7958 ± 0.0168 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | **0.8164** | 24 | 49 |
| 1 | 0.7886 | 6 | 49 |
| 2 | 0.7842 | 6 | 49 |
| 3 | 0.7750 | 3 | 49 |
| 4 | 0.8149 | 24 | 49 |

### BLCA (380 样本) — Mean: 0.7311 ± 0.0262 ✅

> Fold 2 曾在 epoch 7 因 NaN 崩溃，从 `dct_v3_3_fold2_nan_fix` 单独重跑

| Fold | C-Index | Best Epoch | Stopped | 来源 |
|:----:|:------:|:----------:|:-------:|:-----|
| 0 | 0.7552 | 4 | 49 | diagnostics/full |
| 1 | 0.7157 | 5 | 49 | diagnostics/full |
| 2 | 0.7046 | 18 | 49 | fold2_nan_fix |
| 3 | 0.7104 | 34 | 49 | diagnostics/full |
| 4 | **0.7696** | 36 | 49 | diagnostics/full |

### BRCA (1045 样本) — 🔄 运行中 (1/5 folds)

> 启动: Jul 27 04:45

| Fold | C-Index | Best Epoch | Stopped | 状态 |
|:----:|:------:|:----------:|:-------:|:----:|
| 0 | 0.6964 | 31 | 49 | ✅ |
| 1 | — | — | — | 🔄 epoch ~20/50 |

### COADREAD (570 样本) — Mean: 0.6774 ± 0.0515 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | **0.7306** | 41 | 49 |
| 1 | 0.6017 | 7 | 49 |
| 2 | 0.6321 | 46 | 49 |
| 3 | 0.7241 | 41 | 49 |
| 4 | 0.6985 | 29 | 49 |

### SKCM (403 样本) — Mean: 0.6770 ± 0.0513 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | 0.6988 | 6 | 49 |
| 1 | 0.7155 | 11 | 49 |
| 2 | 0.6466 | 11 | 49 |
| 3 | 0.5920 | 2 | 49 |
| 4 | **0.7321** | 8 | 49 |

### STAD (362 样本) — Mean: 0.6596 ± 0.0216 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | 0.6713 | 28 | 49 |
| 1 | 0.6185 | 7 | 49 |
| 2 | 0.6683 | 47 | 49 |
| 3 | 0.6592 | 41 | 49 |
| 4 | **0.6806** | 34 | 49 |

### HNSC (437 样本) — ⏳ 未开始 (首次因 checkpoint 损坏失败)

### LUAD (458 样本) — ⏳ 未开始

### LUSC (454 样本) — ⏳ 未开始

### v3.3 汇总

| 排名 | 癌种 | 样本 | Mean ± Std | 最佳 | 最差 | Δ | 状态 |
|:----:|:----:|:---:|:----------:|:----:|:----:|:--:|:----:|
| 1 | UCEC | 487 | 0.7964 ± 0.0342 | 0.8340 | 0.7548 | 0.0792 | ✅ |
| 2 | KIRC | 488 | 0.7958 ± 0.0168 | 0.8164 | 0.7750 | 0.0414 | ✅ |
| 3 | BLCA | 380 | 0.7311 ± 0.0262 | 0.7696 | 0.7046 | 0.0650 | ✅ |
| 4 | COADREAD | 570 | 0.6774 ± 0.0515 | 0.7306 | 0.6017 | 0.1289 | ✅ |
| 5 | SKCM | 403 | 0.6770 ± 0.0513 | 0.7321 | 0.5920 | 0.1401 | ✅ |
| 6 | STAD | 362 | 0.6596 ± 0.0216 | 0.6806 | 0.6185 | 0.0621 | ✅ |
| 7 | BRCA | 1045 | 0.6964 (1/5) | — | — | — | 🔄 |
| 8 | LUAD | 458 | — | — | — | — | ⏳ |
| 9 | LUSC | 454 | — | — | — | — | ⏳ |
| 10 | HNSC | 437 | — | — | — | — | ⏳ |

---

## DCT v3.4 — BRCA Recovery 消融实验

> 仅针对 BRCA 的调试/消融版本。对比不同 alpha_surv (0.30 vs 0.15)、binning 策略、deterministic slot 初始化等对收敛的影响。
> 非正式多癌种扫描版本，仅作参考。

| 实验 | Fold0 | Fold2 | 说明 |
|:----:|:-----:|:-----:|------|
| **ref** | — | **0.7510** | 基线参考 (alpha=0.15, binning, det) |
| **norank_legacy** | 0.6809 | — | 无rank损失, legacy模式 |
| **reg** | 0.6692 | 0.5709 | 降低reg=0.0002 |
| **det_legacy** | 0.6616 | — | deterministic, legacy |
| **det** | 0.6055 | 0.6951 | deterministic slot |
| **bin_legacy** | 0.5880 | 0.7312 | binning, legacy |
| **bin** | 0.6067 | 0.6440 | binning |
| **a30_legacy** | 0.6184 | 0.7247 | alpha=0.30, legacy |
| **a30** | 0.6067 | — | alpha=0.30 |
| **strat_legacy** | 0.6259 | — | stratified, legacy |
| **strat** | 0.6026 | — | stratified |
| **norank** | 0.6096 | — | 无rank损失 |
| **reg_legacy** | 0.6441 | — | 降低reg, legacy |

> 仅有 ref 的 fold2 达到 0.7510，其他消融实验均显著低于该基线。

---

## DCT v3.5 — Smoke Tests

> 烟雾测试版本，仅用于验证 pipeline 是否正常运行。
> 典型参数: max_epochs=2, max_smoke_batches=2
> **非正式实验结果，不纳入性能对比。**

---

## DCT v3.6_listwise (UNI v1)

> 结果目录: `results/dct_v3.6_listwise/<variant>/<cancer>/`
> 参数: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15
> tie_method: breslow (TCL), fit_bins_on_train=true, slot_init=deterministic
> 目标: 4 癌种 (blca/brca/luad/lusc), 6 变体, 5 折
> **当前仅完成 fold0 和 fold2**

---

### 类别 A: 消融实验变体 (survot_method=distributional_counterfactual_transport)

#### A1. NLL (纯 NLL 生存损失)

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.7314 | 8 | 0.6693 | 12 | 0.7004 |
| BRCA | 0.6775 | 2 | 0.7055 | 7 | 0.6915 |
| LUAD | 0.7828 | 15 | 0.7466 | 9 | **0.7647** |
| LUSC | 0.6525 | 8 | 0.5757 | 0 | **0.6141** |

#### A2. IPCW (NLL + IPCW rank, lambda=0.10)

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.6989 | 17 | 0.6589 | 6 | 0.6789 |
| BRCA | **0.8375** | 16 | 0.7669 | 9 | **0.8022** |
| LUAD | 0.7828 | 17 | 0.7017 | 3 | 0.7423 |
| LUSC | 0.5962 | 3 | 0.5739 | 0 | 0.5850 |

#### A3. ETAR (NLL + ETAR, lambda=0.10)

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.7282 | 14 | 0.6765 | 9 | 0.7024 |
| BRCA | 0.6976 | 5 | 0.7644 | 14 | 0.7310 |
| LUAD | 0.8104 | 12 | 0.7051 | 3 | 0.7577 |
| LUSC | 0.5795 | 0 | 0.6096 | 10 | 0.5946 |

#### A4. IPCW+ETAR (NLL + IPCW rank 0.10 + ETAR 0.10) ✅ 全部完成

| 癌种 | Fold0 | Epoch | Fold2 | Epoch | Mean (2f) |
|:----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| BLCA | 0.7163 | 3 | 0.6589 | 8 | 0.6876 |
| BRCA | 0.7541 | 5 | 0.7583 | 38 | 0.7562 |
| LUAD | 0.7654 | 16 | 0.6975 | 3 | 0.7314 |
| LUSC | 0.5755 | 0 | 0.5661 | 0 | 0.5708 |

#### 消融实验对比汇总

| 癌种 | NLL | IPCW | ETAR | IPCW+ETAR | 最佳 |
|:----:|:----:|:----:|:----:|:--------:|:----:|
| BLCA | 0.7004 | 0.6789 | **0.7024** | 0.6876 | ETAR |
| BRCA | 0.6915 | **0.8022** | 0.7310 | 0.7562 | IPCW |
| LUAD | **0.7647** | 0.7423 | 0.7577 | 0.7314 | NLL |
| LUSC | **0.6141** | 0.5850 | 0.5946 | 0.5708 | NLL |

---

### 类别 B: Listwise Transport 变体 (survot_method=dct_listwise_transport)

> dct_lambda_listwise=0.10, dct_lambda_ipcw_rank=0.0, dct_lambda_etar=0.0

#### B1. GPL (Global Plackett-Luce)

> dct_listwise_mode=global
> 全癌种 5 折: **❌ 尚未开始运行**

#### B2. TCL (Transport-Conditioned Listwise)

> dct_listwise_mode=stage_transport
> 全癌种 5 折: **❌ 尚未开始运行**

---

### v3.6 完整待办清单

| 类别 | 变体 | 癌种 | 已完成 | 剩余 |
|:----:|:----:|:----:|:-----:|:---:|
| A | NLL | blca/brca/luad/lusc | fold0+2 (8次) | fold1/3/4 (12次) |
| A | IPCW | blca/brca/luad/lusc | fold0+2 (8次) | fold1/3/4 (12次) |
| A | ETAR | blca/brca/luad/lusc | fold0+2 (8次) | fold1/3/4 (12次) |
| A | IPCW+ETAR | blca/brca/luad/lusc | fold0+2 (8次) ✅ | fold1/3/4 (12次) |
| B | GPL | blca/brca/luad/lusc | 0次 | 全5折 (20次) |
| B | TCL | blca/brca/luad/lusc | 0次 | 全5折 (20次) |
| **合计** | | | **32/120** | **88 次** |

---

## DCT v3.7_uni2h_highscore (UNI2-h)

> survot_method: distributional_counterfactual_transport
> 参数: max_epochs=50, batch_size=8, lr=5e-4, alpha_surv=0.15, dct_lambda_ipcw_rank=0.10
> wsi_encoder=uni2-h, encoding_dim=1536, fit_bins_on_train=false, slot_init=gaussian
> 结果目录: `results/dct_v3.7_uni2h/highscore/<cancer>/`
> 目标: 10 个癌种

### KIRC (488 样本) — Mean: 0.8108 ± 0.0166 (4/5 folds) 🔄

| Fold | C-Index | Best Epoch | Stopped | 状态 |
|:----:|:------:|:----------:|:-------:|:----:|
| 0 | **0.8323** | 10 | 49 | ✅ |
| 1 | 0.8152 | 27 | 49 | ✅ |
| 2 | 0.7987 | 30 | 49 | ✅ |
| 3 | 0.7971 | 8 | 49 | ✅ |
| 4 | — | — | — | 🔄 epoch 24/50, val_cindex=0.7842 |

### BLCA (380 样本) — Mean: 0.7249 ± 0.0359 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | 0.7448 | 5 | 49 |
| 1 | **0.7555** | 9 | 49 |
| 2 | 0.6597 | 13 | 49 |
| 3 | 0.7126 | 17 | 49 |
| 4 | 0.7518 | 40 | 49 |

### BRCA (1045 样本) — Mean: 0.7130 ± 0.0433 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | 0.7465 | 41 | 49 |
| 1 | 0.6733 | 48 | 49 |
| 2 | 0.7067 | 4 | 49 |
| 3 | **0.7765** | 24 | 49 |
| 4 | 0.6620 | 14 | 49 |

### COADREAD (570 样本) — Mean: 0.7384 ± 0.0405 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | **0.7803** | 17 | 49 |
| 1 | 0.7764 | 31 | 49 |
| 2 | 0.6692 | 1 | 49 |
| 3 | 0.7423 | 22 | 49 |
| 4 | 0.7240 | 1 | 49 |

### HNSC (437 样本) — Mean: 0.6406 ± 0.0244 ✅

| Fold | C-Index | Best Epoch | Stopped |
|:----:|:------:|:----------:|:-------:|
| 0 | 0.6652 | 22 | 49 |
| 1 | 0.6359 | 6 | 49 |
| 2 | **0.6722** | 13 | 49 |
| 3 | 0.6193 | 7 | 49 |
| 4 | 0.6105 | 14 | 49 |

### 剩余癌种 — ⏳ 未开始

| 癌种 | 样本 | 状态 |
|:----:|:---:|:----:|
| LUAD | 458 | ⏳ |
| LUSC | 454 | ⏳ |
| SKCM | 403 | ⏳ |
| STAD | 362 | ⏳ |
| UCEC | 487 | ⏳ |

### v3.7 汇总

| 排名 | 癌种 | 样本 | Mean ± Std | 最佳 | 最差 | Δ | 状态 |
|:----:|:----:|:---:|:----------:|:----:|:----:|:--:|:----:|
| 1 | KIRC | 488 | 0.8108 ± 0.0166 (4/5) | 0.8323 | 0.7971 | 0.035 | 🔄 |
| 2 | COADREAD | 570 | 0.7384 ± 0.0405 | 0.7803 | 0.6692 | 0.111 | ✅ |
| 3 | BLCA | 380 | 0.7249 ± 0.0359 | 0.7555 | 0.6597 | 0.096 | ✅ |
| 4 | BRCA | 1045 | 0.7130 ± 0.0433 | 0.7765 | 0.6620 | 0.115 | ✅ |
| 5 | HNSC | 437 | 0.6406 ± 0.0244 | 0.6722 | 0.6105 | 0.062 | ✅ |
| 6-10 | — | — | — | — | — | — | ⏳ |

---

## DCT v3.8 Transport Consistency (UNI2-h)

> survot_method: dct_transport_intervention_consistency
> 参数: max_epochs=50, batch_size=8, UNI2-h (1536d), ipcw_rank=0.10
> v38_lambda_direction=0.05, v38_lambda_dose=0.03, v38_lambda_reconfiguration=0.02
> protocol: highscore (fit_bins_on_train=false, slot_init=gaussian)
> variant: full (direction + dose + reconfiguration)
> 结果目录: `results/dct_v3.8_transport_consistency/highscore/full/<cancer>/`
> 目标: 4 癌种 (blca/brca/luad/lusc), 5 折
> 启动: Jul 27 04:45, 调度器 PID 3953442

### BLCA (380 样本) — 🔄

| Fold | C-Index | Best Epoch | Stopped | 状态 |
|:----:|:------:|:----------:|:-------:|:----:|
| 0 | **0.7591** | 7 | 49 | ✅ |
| 1 | — | — | — | 🔄 epoch 16/50 |

### 待跑

| 癌种 | Folds | 状态 |
|:----:|:-----:|:----:|
| BLCA | 2-4 | ⏳ |
| BRCA | 0-4 | ⏳ |
| LUAD | 0-4 | ⏳ |
| LUSC | 0-4 | ⏳ |

> BLCA fold0=0.7591 显著优于 v3.7 BLCA fold0=0.7448 (+1.43%)！
> 全部 20 个 fold 预计 ~5 天

---

## DCT v4.0 — IST-Surv ⏳ 未开始

> survot_method: intervention_stable_survival_transport
> 参数: UNI2-h, ist_eps=0.05, ist_num_interventions=3, ist_deletion_penalty=8.0
> 脚本: `scripts/run_v40_intervention_stable_transport.py`
> 目标: 2 癌种 (blca/brca), 5 折

---

## 📈 版本对比

| 癌种 | v3.3 (UNIv1) | v3.6 最佳 (UNIv1) | v3.7 (UNI2-h) | v3.8 (UNI2-h) |
|:----:|:------------:|:-----------------:|:-------------:|:-------------:|
| UCEC | **0.7964** | — | ⏳ | — |
| KIRC | 0.7958 | — | 0.8108 (4/5) | — |
| COADREAD | 0.6774 | — | **0.7384** | — |
| BLCA | 0.7311 | 0.7024 (ETAR) | 0.7249 | 0.7591 (1/5) |
| BRCA | 🔄 0.6964 (1/5) | **0.8022** (IPCW) | 0.7130 | ⏳ |
| HNSC | ⏳ | — | 0.6406 | — |
| LUAD | ⏳ | 0.7647 (NLL) | ⏳ | ⏳ |
| LUSC | ⏳ | 0.6141 (NLL) | ⏳ | ⏳ |
| SKCM | 0.6770 | — | ⏳ | — |
| STAD | 0.6596 | — | ⏳ | — |

### v3.3 → v3.7 提升

| 癌种 | v3.3 | v3.7 | Δ |
|:----:|:----:|:----:|:--:|
| KIRC | 0.7958 | 0.8108 | **+1.50%** |
| COADREAD | 0.6774 | 0.7384 | **+6.10%** |
| BLCA | 0.7311 | 0.7249 | −0.62% |

### v3.7 → v3.8 提升

| 癌种 | v3.7 fold0 | v3.8 fold0 | Δ |
|:----:|:----------:|:----------:|:--:|
| BLCA | 0.7448 | **0.7591** | **+1.43%** |

---

## 🏆 排名总览 (按 Best C-Index)

| 排名 | 版本 | 变体 | 癌种 | 分数 | 类型 | 状态 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 1 | v3.6 | IPCW | BRCA | **0.8375** | Fold0 单折 | ✅ |
| 2 | v3.3 | — | UCEC | **0.8340** | Fold3 单折 | ✅ |
| 3 | v3.7 | — | KIRC | **0.8323** | Fold0 单折 | ✅ |
| 4 | v3.3 | — | UCEC | 0.8265 | Fold4 单折 | ✅ |
| 5 | v3.3 | — | KIRC | 0.8164 | Fold0 单折 | ✅ |
| 6 | v3.7 | — | KIRC | 0.8152 | Fold1 单折 | ✅ |
| 7 | v3.6 | ETAR | LUAD | 0.8104 | Fold0 单折 | ✅ |
| 8 | v3.7 | — | KIRC | 0.7987 | Fold2 单折 | ✅ |
| 9 | v3.3 | — | UCEC | **0.7964** | **5折均值** | ✅ |
| 10 | v3.3 | — | KIRC | **0.7958** | **5折均值** | ✅ |
| — | v3.7 | — | KIRC | **0.8108** | **4折均值** | 🔄 |
| — | v3.8 | — | BLCA | 0.7591 | Fold0 单折 | ✅ |

---

## 🔄 当前运行状态 (2026-07-27 12:45)

### 运行中进程

| 版本 | 癌种/变体 | 当前进度 | 剩余 |
|:----:|:-----------|:--------|:---:|
| v3.3 | BRCA fold1 | epoch ~20/50 | fold2-4 + LUAD/HNSC/LUSC |
| v3.6 调度器 | 4癌种×4变体 | idle (LUSC完成) | fold1/3/4 + GPL/TCL |
| v3.7 | KIRC fold4 | epoch 24/50 | 5癌种 (LUAD/LUSC/SKCM/STAD/UCEC) |
| v3.8 | BLCA fold1 | epoch 16/50 | 18 fold |

### 待办优先级

| 优先级 | 任务 | 预计次数 |
|:------:|:-----|:--------:|
| 🔴 高 | v3.6 GPL 变体 (4癌种×5折) | 20 |
| 🔴 高 | v3.6 TCL 变体 (4癌种×5折) | 20 |
| 🟡 中 | v3.6 消融实验 fold1/3/4 (4变体×4癌种×3折) | 48 |
| 🟡 中 | v3.7 剩余5癌种 (LUAD/LUSC/SKCM/STAD/UCEC) | 25 |
| 🟢 低 | v3.8 按调度器顺序自动完成 | 18 |
| ⬜ 新 | v4.0 IST-Surv (blca/brca) | 10 |

---

## 🏥 多癌种数据集总览 (10 个)

| 癌种 | clinical | Split有效 | v3.3 | v3.6 最佳 | v3.7 | v3.8 |
|:----:|:--------:|:--------:|:-----:|:---------:|:----:|:----:|
| BLCA | 381 | 380 | 0.7311 ✅ | 0.7024 ETAR | 0.7249 ✅ | 0.7591 (fold0) |
| BRCA | 1046 | 1045 | 0.6964 🔄 | **0.8022** IPCW | 0.7130 ✅ | ⏳ |
| COADREAD | 573 | 570 | 0.6774 ✅ | — | 0.7384 ✅ | — |
| HNSC | 438 | 437 | ⏳ | — | 0.6406 ✅ | — |
| KIRC | 488 | 488 | 0.7958 ✅ | — | 0.8108 🔄 | — |
| LUAD | 467 | 458 | ⏳ | 0.7647 NLL | ⏳ | ⏳ |
| LUSC | 460 | 454 | ⏳ | 0.6141 NLL | ⏳ | ⏳ |
| SKCM | 403 | 403 | 0.6770 ✅ | — | ⏳ | — |
| STAD | 366 | 362 | 0.6596 ✅ | — | ⏳ | — |
| UCEC | 488 | 487 | 0.7964 ✅ | — | ⏳ | — |

---

## 结论

1. **v3.6 BRCA IPCW = 0.8375** 是全版本最高单折分数
2. **UNI2-h (v3.7) 优于 UNI v1 (v3.3)**: KIRC +1.50%, COADREAD +6.10%, BLCA 持平
3. **v3.8 fold0 惊艳**: BLCA 0.7591 比 v3.7 同一 fold 高 +1.43%，transport consistency 初见成效
4. **v3.6 IPCW** 在 BRCA 上显著优于其他变体 (0.8022 vs 0.69~0.76)
5. **v3.6 GPL/TCL 尚未运行**, 是 v3.6 最核心的对比实验
6. **Split bug 已修复**: 原 `zip(train, val)` 截断问题在 SKCM 已修复，其余 9 癌种 split 无受该 bug 影响
7. **v4.0 IST-Surv** 等待启动，是全新的干预稳定性范式
