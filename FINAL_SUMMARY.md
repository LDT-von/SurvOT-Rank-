# SurvOT-Rank 多癌种实验结果汇总

> 更新时间: 2026-08-03 | Seed: 3 | 版本: v3.3 / v3.4 / v3.5 / v3.6 / v3.7 / v3.8 / v3.9 / v4.0 / v4.1 / ArcSurv

---

## ⚠️ 划分分界：2026-07-30 `bee66a2` 之后 BLCA `5fold` 已重划

**跨越这个日期的结果不可比较，禁止放入同一张对比表。**

提交 `bee66a2 feat: harden survival experiments and rebalance splits`（2026-07-30）
重写了全部五个 `splits/5fold/blca/fold_*.csv`。划分由
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 生成，
重划后同一 fold 编号对应的患者集合完全改变（新旧每折仅重叠 13–17 人）。

同一个 v3.3、同一份冻结 YAML、同一 seed=3、同一编码器，**只换划分**的实测影响：

| Fold | 旧划分 | 新划分 | Δ |
|:---:|:---:|:---:|:---:|
| 0 | 0.7552 | 0.7154 | −0.0398 |
| 1 | 0.7157 | 0.6918 | −0.0239 |
| 2 | 0.7046 | 0.6735 | −0.0311 |
| 3 | 0.7104 | 0.7340 | **+0.0236** |
| 4 | 0.7696 | 0.7222 | −0.0474 |
| **均值** | **0.7311** | **0.7074** | **−0.0237** |

> 换划分使五折均值移动 0.0237、单折最大移动 0.0474，**大于当前任何方法声称的增益
> （最大 +0.0201）**。原因是每折验证仅约 76 人、DSS 事件约 20 例，重划改变了
> 「哪 20 个事件进入验证集」，rebalance 还同时改了分层结构。

**归属划分：**

| 划分 | 适用结果 |
|---|---|
| **旧划分**（< 2026-07-30） | v3.3 六癌种（UCEC/KIRC/BLCA 0.7311/COADREAD/SKCM/STAD）、v3.6 全部消融、v3.7、v3.8 50ep 与 20ep、BRCA 全部 |
| **新划分**（≥ 2026-07-30） | v3.3 BLCA rep（0.7074）、v3.8.2、v3.8.3、v3.9、v4.0、v4.1、ArcSurv、v3.8 LUSC 8 变体 |

> **fold 编号跨划分无意义**：旧 fold2 与新 fold2 是不同患者集合。
> 因此「fold2 偏难」这一结论仅在新划分内成立。

---

## Split 数据修复记录 (2026-07-30)

> `zip(train, val)` 截断 bug 虽已修复，但磁盘上的 10 癌种 split 仍是旧分配：
> 患者总数基本完整，事件却没有按折均衡。现已统一改为“事件状态 ×
> 状态内生存时间分位数”分层，并加入可执行 split 审计；所有 split
> 文件均由同一 seed=42 规则重新生成。

| 癌种 | clinical | Split有效 | 缺失原因 | Train/Val重叠 |
|:----:|:--------:|:--------:|----------|:-----------:|
| BLCA | 381 | 380 | 1人缺DSS标签 | 无 |
| BRCA | 1046 | **1045** | 1人缺DSS标签 | 无 |
| COADREAD | 573 | 570 | 3人缺DSS标签 | 无 |
| HNSC | 438 | 437 | 1人缺DSS标签 | 无 |
| KIRC | 488 | 488 | — | 无 |
| LUAD | 467 | 458 | 9人缺DSS标签 | 无 |
| LUSC | 460 | 454 | 6人缺DSS标签 | 无 |
| SKCM | 403 | 388 | 15人缺DSS标签 | 无 |
| STAD | 366 | 362 | 4人缺DSS标签 | 无 |
| UCEC | 488 | 487 | 1人缺DSS标签 | 无 |

> 所有符合 DSS 训练条件的患者都被保留，每人恰好进入一次验证集，无
> train/val 重叠。BRCA 每折事件数由 `[28,21,18,21,10]` 修正为
> `[19,19,20,20,20]`；UCEC 由 `[15,5,8,13,11]` 修正为
> `[10,11,10,11,10]`。

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
| **v3.9** | Risk-Simplex Transport | dct_v39_risk_simplex_transport | UNI2-h (1536d) | Risk Simplex约束 | ✅ BLCA完成 |
| **v4.0** | IST-Surv | intervention_stable_survival_transport | UNI2-h (1536d) | 干预稳定性 | ✅ BLCA完成 |
| **v4.1** | Survival Evidence Ledger | dct_v41_survival_evidence_ledger | UNI v1 (1024d) | 证据账本 | 🔄 BLCA fold2重跑中 |
| **ArcSurv** | Archetypal Risk Composition | archetypal_risk_composition | UNI v1 (1024d) | 原型风险组合 | ✅ BLCA完成 |

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

### BLCA — 两套划分，不可混用 ⚠️

#### 旧划分 (< 2026-07-30) — Mean: 0.7311 ± 0.0262

> Fold 2 曾在 epoch 7 因 NaN 崩溃，从 `dct_v3_3_fold2_nan_fix` 单独重跑
> **此结果已不能作为任何新方法的基线**（划分已于 `bee66a2` 重写）。

| Fold | C-Index | Epoch | 来源 |
|:----:|:------:|:-----:|:-----|
| 0 | 0.7552 | 4 | diagnostics/full |
| 1 | 0.7157 | 5 | diagnostics/full |
| 2 | 0.7046 | 18 | fold2_nan_fix |
| 3 | 0.7104 | 34 | diagnostics/full |
| 4 | **0.7696** | 36 | diagnostics/full |

#### 新划分 (≥ 2026-07-30，当前基线) — 5 折 Mean: **0.7074**

> 结果目录: `results/dct_v3.3_score_first_blca_uni_rep/`
> 交叉核对已确认：**5 折全部为 2026-08-01 新跑**（10:01 → 15:47 逐折），
> 与旧 diagnostics 每折仅重叠 13–17 人，非目录复用、非跳过。

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | 0.7154 | 28 |
| 1 | 0.6918 | 23 |
| 2 | 0.6735 | 18 |
| 3 | **0.7340** | 40 |
| 4 | 0.7222 | 28 |

**两个口径必须分开引用，禁止互换：**

| 口径 | 折 | 均值 | 用途 |
|---|:---:|:---:|---|
| 五折 | 0,1,2,3,4 | **0.7074** | v3.3 自身的完整成绩 |
| 三折 | 1,2,4 | **0.6958** | 与 Priority Queue 新方法对比（新方法只跑了这三折） |

> 此前文档把 **五折均值 0.7074 当作三折均值报告**，并且新方法汇总表里
> v3.3 的「折 1/2/4」填的是旧 diagnostics 的折 0/1/3 数值（错位），
> 两处均已修正。

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
>
> 下列已完成数字来自旧 `highscore` 训练与旧 split，仅用于追溯；统一
> `robust` 协议会在重新分层后的同一套 split 上重跑四个癌种。

### BLCA (380 样本) — 旧协议: 0.7274 ± 0.0351

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 0 | **0.7591** | 7 |
| 1 | 0.7343 | 5 |
| 2 | 0.7199 | 5 |
| 3 | 0.7528 | 6 |
| 4 | 0.6710 | 11 |

> 5折完成！fold0=0.7591 为单折最高，fold3=0.7528 @epoch6 冲到很高后震荡

### BRCA (1045 样本) — 旧 split: 0.6750 ± 0.0573（仅保留为历史记录）

> 该结果的 fold4 验证集只有 10 个事件，而 fold0 有 28 个，不能作为
> 新统一协议的正式 5-fold 结果。需要在均衡 split 上用 `robust`
> 协议重跑全部 5 折，不能只替换 fold4。

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

## DCT v3.8 20ep 快速筛选 (新 split, highscore)

> 2026-07-30 完成。20 epoch 快速筛选版本，用于评估 v3.8 三个创新损失
> (direction/dose/reconfiguration) 的有效性，非正式论文结果。
> full = 全部三损失启用 | base = 三损失全部关闭

### BRCA (1045 样本) — full: 0.6852 | base: 0.7185

| Variant | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean ± SD** |
|:-------:|:------:|:------:|:------:|:------:|:------:|:-------------:|
| full | 0.7128(2) | 0.6667(13) | 0.6857(17) | 0.7054(5) | 0.6552(4) | **0.6852 ± 0.023** |
| base | 0.7736(2) | 0.7488(17) | 0.6419(11) | 0.7099(7) | 0.7185(14) | **0.7185 ± 0.051** |

### BLCA (380 样本) — full: 0.7077 | base: 0.6917 (est.)

| Variant | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean ± SD** |
|:-------:|:------:|:------:|:------:|:------:|:------:|:-------------:|
| full | 0.7349(12) | 0.6403(10) | 0.6857(5) | 0.7384(9) | 0.7393(4) | **0.7077 ± 0.044** |
| base | 0.6534(14) | 0.6858(4) | 0.6352(15) | 0.7498(14) | ~0.7342@2 | **~0.692** (4/5 folds) |

### full vs base 对比

| 癌种 | full | base | Δ | 三损失效果 |
|:----:|:----:|:----:|:-:|:----------:|
| BRCA | 0.6852 | 0.7185 | -0.033 | 有害 |
| BLCA | 0.7077 | ~0.692 | +0.016 | 有益 |

> **结论**：两个癌种结论相反 — BRCA 上三损失拖后腿（-0.033），BLCA 上略有提升（+0.016）。
> 但双方结论所用 split 是从旧 `highscore` 转换而来，并非完全独立于损失函数设计；
> ULIT 认为在 `robust` 协议下结论可能反转，建议暂不下结论，待 robust 20ep 重跑后再评估。
> 
> base 在 BRCA 上达到 0.7185，超过旧 50ep 的 0.6750，提示新 split 均衡化本身可能贡献了约 +0.04；
> full=0.6852 ≈ 旧 0.6750，去掉 split 增益后三损失在 BRCA 上实际净效果约 -0.03。

---

## DCT v3.8 Robust 消融 — BLCA fold0 (2026-07-31)

> 使用 `robust` 协议（事件分层批次、确定性槽、结构损失预热/ramp）
> 在 BLCA fold0 上跑 6 种中间变体，逐一检测 direction/dose/reconfiguration 的单体/组合效果。
> 参考：highscore base=0.6534, highscore full=0.7349

### 旧 highscore 协议 (2026-07-31)

| 变体 | C-Index | Epoch | vs base | vs full |
|------|:------:|:-----:|:-------:|:-------:|
| **direction** | **0.6763** | 15 | **+0.023** | -0.059 |
| dose | 0.6661 | 5 | +0.013 | -0.069 |
| reconfiguration | 0.6415 | 5 | -0.012 | -0.093 |
| direction_dose | 0.6262 | 5 | -0.027 | -0.109 |
| direction_reconfiguration | 0.6415 | 5 | -0.012 | -0.093 |
| dose_reconfiguration | 0.6398 | 7 | -0.014 | -0.095 |

### 新版 robust 20ep 协议 — BLCA fold0 全部 8 变体 (2026-08-01)

> 使用修复后的 `robust` 协议（事件分层批次、确定性槽、结构损失预热/ramp、训练折分箱）
> 在 BLCA fold0 上跑全部 8 变体，max_epochs=20。参考：base=0.7031

| 变体 | C-Index | Epoch | vs base |
|------|:------:|:-----:|:-------:|
| **direction** | **0.7356** | 15 | **+0.0325** |
| full | 0.7171 | 10 | +0.014 |
| direction_dose | 0.7111 | 12 | +0.008 |
| direction_reconfiguration | 0.7036 | 18 | +0.001 |
| base | 0.7031 | 8 | — |
| reconfiguration | 0.7028 | 10 | -0.000 |
| dose | 0.6950 | 10 | -0.008 |
| dose_reconfiguration | 0.6869 | 2 | -0.016 |

> **结论（两轮一致）**：仅 direction 单独有效（+0.0325 vs base），dose 微负，reconfiguration 负效。
> 任意两两组合均比单独 direction 更差。**保留 direction 作为唯一结构损失**。
> robust 协议下 direction 达到 **0.7356**，超过旧 highscore full=0.7349。

### BRCA direction fold0 (robust 20ep) ⚠️ 已终止

> 因 UNI2-h 特征缺 271 人（见下方特征覆盖率分析），BRCA split 含大量全零 WSI 患者，结果无效。已终止并 kill 进程。
> 最佳 val C-Index 曾达到 0.7892 @epoch4，但迅速过拟合至 ~0.60。

### LUSC 8变体筛选 (robust 20ep) — 已完成 (2026-08-02)

> 在 LUSC (454 样本) 上用 robust 协议跑全部 8 变体，max_epochs=20，UNI2-h。
> 结论：direction/dose/reconfiguration 三损失在 LUSC 上不仅无效，反而有害。

| Variant | C-Index | Epoch | vs base |
|---------|:------:|:-----:|:-------:|
| **base** | **0.7475** | 6 | — |
| full | 0.7332 | 15 | -0.014 |
| direction_dose | 0.7134 | 15 | -0.034 |
| dose | 0.6981 | 7 | -0.049 |
| dose_reconfiguration | 0.6918 | 14 | -0.056 |
| direction | 0.6595 | 4 | -0.088 |
| reconfiguration | 0.6595 | 4 | -0.088 |
| direction_reconfiguration | 0.6595 | 4 | -0.088 |

> **与 BLCA 完全相反**：BLCA 上 direction 有益 (+0.033)，LUSC 上 direction 有害 (-0.088)。
> base=0.7475 为 LUSC 最优，三损失全部负效，推测 reconfiguration 未使用时退化到 direction 相同路径。

---
### WSI 编码器特征覆盖率分析 (2026-08-01)

> 检查各癌种 clinical 患者与 UNI v1 / UNI2-h 特征文件的患者 ID 交集。

#### UNI v1 (`/data/CPathPatchFeature/<cancer>/uni/pt_files/`)

| 癌种 | Clinical | Features | 交集 | 覆盖率 |
|:----:|:--------:|:--------:|:----:|:------:|
| BRCA | 1046 | 1062 | 1046 | **100.0%** |
| COADREAD | 573 | 573 | 573 | 100.0% |
| STAD | 366 | 366 | 366 | 100.0% |
| LUAD | 467 | 955 | 466 | 99.8% |
| 其余 6 癌种 | — | — | — | 100.0% |
| **合计** | **5110** | **6114** | **5109** | **99.98%** |

> UNI v1 近乎完美覆盖，v3.3 复现不受影响。

#### UNI2-h (`/data1/TCGA-UNI2-h-features/<cancer>/uni2-h/pt_files/`)

| 癌种 | Clinical | Features | 交集 | 缺特征 | 覆盖率 |
|:----:|:--------:|:--------:|:----:|:------:|:------:|
| BLCA | 381 | 386 | 381 | 0 | 100.0% |
| HNSC | 438 | 450 | 438 | 0 | 100.0% |
| KIRC | 488 | 513 | 488 | 0 | 100.0% |
| LUSC | 460 | 478 | 460 | 0 | 100.0% |
| SKCM | 403 | 433 | 403 | 0 | 100.0% |
| UCEC | 488 | 505 | 488 | 0 | 100.0% |
| LUAD | 467 | 468 | 458 | 9 | 98.1% |
| COADREAD | 573 | 591 | 556 | 17 | 97.0% |
| STAD | 366 | 375 | 353 | 13 | 96.4% |
| **BRCA** | **1046** | **786** | **775** | **271** | **74.1%** |

> **BRCA 缺 271 人 (25.9%)** 为唯一严重问题；COADREAD/LUAD/STAD 各有 1-3% 轻微缺口。
> 当前 split 生成器未按编码器特征过滤，导致缺特征患者被填入全零 WSI 向量。
> **修复方向**：split 生成器需支持编码器专属患者白名单 `intersect(clinical, feature_patients)`。
> 所有 v3.7/v3.8 UNI2-h 实验（BRCA/LUAD/LUSC）需在过滤后的 split 上重跑。

---

## DCT v4.0 — IST-Surv ✅ BLCA完成

> survot_method: intervention_stable_survival_transport
> 参数: UNI2-h (1536d), max_epochs=30, clean/full variant, ist_eps=0.05, ist_num_interventions=3, ist_deletion_penalty=8.0
> 结果目录: `results/ist_surv_v4.0_30ep/clean/full/blca/`

### BLCA (380 样本) — Mean: 0.7078 ± 0.0490 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | 0.6707 | 0 |
| 2 | 0.6767 | 10 |
| 4 | **0.7761** | 12 |

> fold4=0.7761 为 BLCA 所有方法中单折最高，但 fold1/fold2 偏低，方差较大。

---

## DCT v3.8.2 — MGPTR (Prognostic Transport Reconstruction) ✅ BLCA完成

> survot_method: dct_v382_prognostic_transport_reconstruction
> UNI2-h (1536d), max_epochs=30, robust/adaptive_full variant
> 结果目录: `results/dct_v3.8.2_30ep/robust/adaptive_full/blca/`

### BLCA (380 样本) — Mean: 0.7159 ± 0.0376 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | 0.7073 | 20 |
| 2 | 0.6745 | 4 |
| 4 | **0.7658** | 27 |

> BLCA 所有 Priority Queue 方法中**排名第一** (0.7159)，超过 v3.3 基线 (+0.008)。
> fold4=0.7658 为第二高单折（仅次于 v4.0 IST-Surv fold4=0.7761）。
> 三折方差 0.0376 可控，无明显弱折。

---

## DCT v3.8.3 — Centered Intervention Consistency ❌ 已中止

> survot_method: dct_v383_intervention_consistency_centered
> UNI2-h (1536d), max_epochs=30
> 结果目录: `results/dct_v3.8.3_intervention_consistency_centered_30ep/blca/`

### BLCA (380 样本) — 仅 fold1 完成，已中止

| Fold | C-Index | Epoch | 状态 |
|:----:|:------:|:-----:|:----:|
| 1 | 0.5931 | 7 | ✅ 完成 |
| 2 | 0.6193 | 23 (best) | ❌ 未完成 (中断于 e25) |
| 4 | — | — | ❌ 未跑 |

> fold1=0.5931 远低于基线，fold2 best=0.6193 同样偏低。Centered 约束在 BLCA 上无效，已中止。

---
 
## DCT v4.1 — Survival Evidence Ledger ✅ BLCA完成

> survot_method: dct_v41_survival_evidence_ledger
> 配置: `dct_v41_survival_evidence_ledger_blca.yaml`
> UNI v1 (1024d), max_epochs=30, batch_size=8

### BLCA (380 样本) — Mean: 0.7039 ± 0.0490 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | **0.7300** | 11 |
| 2 | 0.6354 | 3 |
| 4 | **0.7462** | 7 |

> fold1=0.7300, fold4=0.7462 均不错，但 fold2=0.6354 拉低均值至 0.7039，略低于 v3.3 (0.7074) 和 v4.0 (0.7078)。fold2 连续三次重跑均低分 (~0.635)，该 split 对 v4.1 不友好。

---

## DCT v3.9 — Risk-Simplex Transport ✅ BLCA完成

> survot_method: dct_v39_risk_simplex_transport
> UNI2-h (1536d), max_epochs=30

### BLCA (380 样本) — Mean: 0.6394 ± 0.0293 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | 0.6320 | 6 |
| 2 | 0.6776 | 13 |
| 4 | 0.6085 | 14 |

> 明显不如 baseline，Risk-Simplex 约束在 BLCA 上效果不佳。

---

## ArcSurv — Archetypal Risk Composition ✅ BLCA完成

> survot_method: archetypal_risk_composition
> UNI v1 (1024d), max_epochs=30

### BLCA (380 样本) — Mean: 0.6757 ± 0.0534 ✅

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | 0.6702 | 29 |
| 2 | 0.6134 | 12 |
| 4 | **0.7436** | 18 |

> fold4=0.7436 不错但 fold2=0.6134 拉低均值。fold2 split 对多种方法都偏难。

---

## BLCA 新方法汇总 (2026-08-02 Priority Queue)

> 全部为新划分 (≥ 2026-07-30)，folds 1/2/4，30ep。
> **不能合成单一排名**：两组使用不同编码器与不同 split 目录，必须分组看。

#### A 组：UNI v1 (1024d) + `5fold` — 同编码器同划分，可比 ✅

| 方法 | Fold1 | Fold2 | Fold4 | **Mean (3f)** | Δ vs v3.3 | 判定 |
|------|:-----:|:-----:|:-----:|:-------------:|:---------:|:----:|
| v4.1 Evidence Ledger | 0.7300 | 0.6354 | 0.7462 | **0.7039** | +0.0081 | 未确立 |
| **v3.3 Score-First** | 0.6918 | 0.6735 | 0.7222 | **0.6958** | baseline | — |
| ArcSurv | 0.6702 | 0.6134 | 0.7436 | **0.6757** | −0.0201 | 未确立 |

> v4.1 三折 std 0.0490 → 均值标准误 0.0283，远大于 +0.0081。
> ArcSurv std 0.0534 → 标准误 0.0308，也大于 −0.0201。
> **A 组内没有任何方法与基线产生可确立的差异。**

#### B 组：UNI2-h (1536d) + `5fold_uni2h` — 基线缺失，暂不可判定 ⚠️

| 方法 | Fold1 | Fold2 | Fold4 | **Mean (3f)** | 对照基线 |
|------|:-----:|:-----:|:-----:|:-------------:|:--------:|
| v3.8.2 MGPTR | 0.7073 | 0.6745 | 0.7658 | **0.7159** | **缺失，待 v3.8 base 折 1/2/4** |
| v4.0 IST-Surv | 0.6707 | 0.6767 | 0.7761 | **0.7078** | **缺失，待 v3.8 base 折 1/2/4** |
| v3.9 Risk-Simplex | 0.6320 | 0.6776 | 0.6085 | **0.6394** | **缺失，待 v3.8 base 折 1/2/4** |
| v3.8.3 Centered | 0.5931 | 0.6193 | — | 中止 | — |

> B 组的正确基线是 **v3.8 `base` 变体**（launcher 定义：v3.7-matched UNI2-h control
> through the v3.8 class，即 v3.3 的 NLL + IPCW 目标），同编码器、同协议、同代码类。
> 但 v3.8 只跑过 BLCA fold0 与 LUSC fold0，**没有 BLCA 折 1/2/4**，
> 因此 B 组三个方法的增减目前全部无法判断。

#### ❌ 已撤回的结论：v3.8.2 「+0.0201 领先基线」

> 该数值用 **B 组成绩** 减 **A 组基线** 得出，跨编码器 + 跨 split 目录，不成立。
> 即使同组比较，v3.8.2 三折 std = 0.0376 → **均值标准误 0.0217 > 0.0201**，
> 仍在噪声内。**当前状态：未确立。**

#### 可确立的结论（幅度远超噪声，不依赖基线口径）

| 方法 | 值 | 相对 A 组基线 | 判定 |
|---|:---:|:---:|:---:|
| v3.9 Risk-Simplex | 0.6394 | −0.0564 | **确立为负** |
| v3.8.3 Centered | 0.5931 (fold1) | ≈ −0.10 | **确立为负** |

### 跨方法 fold 分析（新划分内）

| Fold | v3.8.2 | v4.0 | v3.9 | v3.8.3 | v3.3 | v4.1 | ArcSurv |
|:----:|:------:|:----:|:----:|:------:|:----:|:----:|:-------:|
| | *UNI2-h* | *UNI2-h* | *UNI2-h* | *UNI2-h* | *UNI v1* | *UNI v1* | *UNI v1* |
| 1 | 0.7073 | 0.6707 | 0.6320 | 0.5931 | 0.6918 | 0.7300 | 0.6702 |
| 2 | 0.6745 | 0.6767 | 0.6776 | 0.6193 | 0.6735 | 0.6354 | 0.6134 |
| 4 | 0.7658 | 0.7761 | 0.6085 | — | 0.7222 | 0.7462 | 0.7436 |

> fold4 普遍最高、fold2 普遍最低，但有两个例外：**v4.0**（fold2 0.6767 > fold1 0.6707）
> 与 **v3.9**（fold2 0.6776 是其最好折，fold4 0.6085 是其最差折，模式完全反转）。
> v3.9 的 fold 难度排序与其余方法相反，说明它的输出与数据本身的难度结构脱钩，
> 属于机制未跑通，而非调参不足。

> **最佳 epoch 分布**：fold2 上各方法普遍早熟（4/10/13/23/3/12），
> fold4 上明显更晚（27/12/14/7/18）。即 fold2 早期见顶随后退化，
> 取最大值容易取到早期噪声；fold4 才存在持续学习。
> 另注 v4.0 fold1 最佳出现在 **epoch 0**，即全程未超过初始化。

> 均值均由本表逐折值直接计算。pkl 级重算与 csv best 存在 ≤0.0005 的舍入/并列处理差异
> （例：v4.1 0.7039 vs 0.7034），不影响任何判定。

---

## 📈 版本对比

> ⚠️ **本表横跨划分分界，不可按行直接比较。**
> 左侧 `v3.3 / v3.6 / v3.7 / v3.8 50ep / v3.8 20ep` 为**旧划分**（< 2026-07-30）；
> 右侧 `v3.8.2 / v3.8.3 / v3.9 / v4.0 / v4.1 / ArcSurv` 为**新划分**（≥ 2026-07-30）。
> 实测仅换划分即可移动均值 0.0237、单折 0.0474，因此跨这条界线的差值没有意义。
> BLCA 的有效对比见上方「BLCA 新方法汇总」的 A/B 分组表。

| 癌种 | v3.3 (UNIv1) | v3.6 最佳 (UNIv1) | v3.7 (UNI2-h) | v3.8 50ep | v3.8 20ep full | v3.8 20ep base | v3.8.2 MGPTR | v3.8.3 Centered | v3.9 | v4.0 | v4.1 | ArcSurv |
|:----:|:------------:|:-----------------:|:-------------:|:----------:|:--------------:|:--------------:|:-------------:|:---------------:|:----:|:----:|:----:|:------:|
| UCEC | **0.7964** | — | ⏳ | — | — | — | — | — | — | — |
| KIRC | 0.7958 | — | **0.8149** | — | — | — | — | — | — | — |
| COADREAD | 0.6774 | — | **0.7384** | — | — | — | — | — | — | — |
| BLCA | 0.7311 | 0.7024 (ETAR) | 0.7249 | 0.7274 | 0.7077 | ~0.692 | 0.7159 | 0.5931 | 0.6394 | 0.7078 | 0.7039 | 0.6757 |
| HNSC | ⏳ | — | 0.6406 | — | — | — | — | — | — | — |
| LUAD | ⏳ | 0.7647 (NLL) | 0.6662 | ⏳ | ⏳ | ⏳ | — | — | — | — |
| LUSC | ⏳ | 0.6141 (NLL) | 🔄 fold0=0.4575 | ⏳ | ⏳ | ⏳ | — | — | — | — |
| SKCM | 0.6770 | — | ⏳ | — | — | — | — | — | — | — |
| STAD | 0.6596 | — | ⏳ | — | — | — | — | — | — | — |
| BRCA | ⏳ 待重跑 | ⏳ 待重跑 | ⏳ 待重跑 | 0.6750 ✅ | 0.6852 | **0.7185** | — | — | — | — |

---

## 未结事项总表（唯一台账，新问题一律登记到此处）

| # | 事项 | 状态 | 成本 | 备注 |
|:-:|------|:----:|:----:|------|
| 1 | 修正 v3.3 基线，统一三张表 | ✅ **已完成** | 0 | 根因=`bee66a2` 重划 + 表格错位；本次已改 5 处 |
| 2 | 用同一编码器重排，禁止跨 UNI v1/UNI2-h | ✅ **已完成** | 0 | 已拆为 A/B 两组；副产物=发现 B 组无基线（→ #8） |
| 3 | 拆掉 `dct_lambda_ipcw_rank=0.10` | ⬜ **未开始** | 3 次训练 | 三癌种方向一致为负 (BLCA −0.022 / LUAD −0.022 / LUSC −0.029)，且与 `LOSS_BLACKLIST.md` 已拉黑的 `lambda_rankevent_rank` 同机制（batch 8 下成对可比样本过少）。当前 v3.8/v3.8.2/v3.8.3/v3.9/v4.1 全部继承 λ=0.10 |
| 4 | 验证 v3.8.2 优势来自自适应权重而非 MGPTR | ⬜ **未开始** | 3 次训练 | `fixed_full` vs `adaptive_full` 单变量。假设：它排名靠前是因为自适应控制器自动关掉了有害损失 |
| 5 | 补齐 fold0/fold3 到五折 | ⬜ **未开始** | 每方法 2 次 | 三折标准误 0.022~0.031，撑不住任何结论 |
| 6 | v3.8 warmup/ramp 与 epoch 预算对齐 | 🟡 **分析完成，未动手** | 待定 | 已定位污染：`direction`/`reconfiguration`/`direction_reconfiguration` 三者 C-index 与最佳 epoch **完全相同 (0.6595 @ e4)**，而 `warmup_epochs=5`，即峰值出现在损失开启之前。故「三损失有害 −0.088」不成立，需重测或判定证据不足 |
| 7 | v3.9 / v3.8.3 写成负结果并停止调参 | 🟡 **部分完成** | 0 | 本表已标「确立为负」并给出机制：①塌缩(cos 0.9987)不是病因——修好塌缩后 v3.8.3 反降至 0.5931；②硬结构约束损失预测自由度，v3.9 fold 难度排序反转 |
| 8 | v3.8 `base` BLCA 折 1/2/4 (UNI2-h, 30ep) | ⬜ **未开始** | 3 次训练 | B 组唯一缺失的基线，补齐后 v3.8.2/v4.0/v3.9 才首次可判定 |
| 9 | 验证 `5fold` 与 `5fold_uni2h` 在 BLCA 是否同一划分 | ⬜ **未开始** | 几秒（只读文件） | 比对两组 pkl 的验证患者 key 集合。BLCA UNI2-h 覆盖率 100%，合格患者集应相同，同 seed=42 下 `StratifiedKFold` 理论应给出同一划分，但**未验证**。决定 A/B 组能否合并 |

**结算：9 项中已完成 2、部分完成 2、未开始 5。**
其中 #9 零成本，#3/#4/#8 各 3 次训练，#5 视范围而定。

> ⚠️ 撤回记录：文档原写「BLCA 上 direction 有益 (+0.033)，LUSC 上有害 (−0.088)，
> 两癌种完全相反」。LUSC 侧证据已被 warmup 污染（见 #6），
> 该「癌种间相反」结论同样撤回。

---

## 记录修正日志 (2026-08-03)

1. 新增划分分界说明：`bee66a2` (2026-07-30) 重划 BLCA `5fold`，量化了换划分的影响 (均值 −0.0237)。
2. v3.3 BLCA 拆为旧/新两套划分；新划分五折 = 0.7074、三折 (1/2/4) = 0.6958，明确两口径不可互换。
3. 修正新方法汇总表：此前 v3.3 的「折 1/2/4」误填旧 diagnostics 的折 0/1/3 数值（错位），且把五折均值 0.7074 当三折均值报告。
4. 排名表按编码器/split 拆为 A 组（UNI v1 + `5fold`，有基线）与 B 组（UNI2-h + `5fold_uni2h`，基线缺失）。
5. 撤回 v3.8.2「+0.0201 领先」：跨组比较无效，且同组标准误 0.0217 > 0.0201，改为「未确立」。
6. 版本对比表加跨划分警告。

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

## 🔄 当前运行状态 (2026-08-02)

| 方法 | 癌种 | Fold | 进度 | 最佳 C-Index | 状态 |
|:----:|:----:|:----:|------|:----------:|:----:|
| — | — | — | 全部完成 | — | ✅ |

> Priority Queue 全部 8 个 stage 已完成。GPU 0 空闲。

### 🔴 阻塞项

| 问题 | 详情 | 状态 |
|------|------|:----:|
| UNI2-h BRCA 特征缺口 | 1045 临床患者中仅 775 人有特征 (74.1%)，split 未过滤导致全零 WSI | 🔴 待修复 split 生成器 |
| UNI2-h STAD/COADREAD/LUAD 轻微缺口 | 各缺 1-3% | 🟡 |
| v3.7 LUSC fold0 异常 | 0.4575，需排查 | 🟡 |

### 待办

| 优先级 | 任务 |
|:------:|:-----|
| 🔴 | **修复 split 生成器** — 支持编码器专属患者白名单，`intersect(clinical, feature_patients)` |
| 🔴 | **BRCA UNI2-h split 重建** — 对 775 人子集重新分层生成 5-fold |
| 🟡 | COADREAD/LUAD/STAD UNI2-h split 过滤重建 |
| 🟡 | v3.7 LUSC fold0=0.4575 排查 |
| ⬜ | v3.8 robust LUSC fold0 — direction 变体 (LUSC 100% 覆盖，可直接跑) |
| ⬜ | v3.8 robust 全癌种 5-fold — split 修复后 |
| ⬜ | BRCA v3.3/v3.6/v3.7 重跑 — UNI v1 覆盖 100%，split 按 1046 人重建后可直接跑 |

### 已知问题

| 日期 | 问题 | 详情 | 状态 |
|:----:|------|------|:----:|
| 08-01 | UNI2-h feature cohort mismatch | BRCA robust used a 1045-case clinical split while the UNI2-h directory contains features for only 786 case IDs before exact intersection. Missing slides were silently replaced by zero bags. Robust now requires 100% exact clinical-slide feature coverage, blocks incomplete cancers, fails on patients with no extracted WSI feature, and writes new results under `robust_uni2h`. BRCA/LUAD/COADREAD/STAD remain blocked until features are completed. | ✅ Guard active; run only complete cancers |
| 07-30 | v3.8 三损失效果不一致 | 20ep highscore: BRCA full<base (-0.033), BLCA full>base (+0.016)。split非独立于损失设计，ULIT建议robust协议重跑后再判 | 🟡 待robust验证 |
| 07-30 | 10癌种 split 事件不均衡 | 旧文件虽覆盖患者，但不是当前分层生成器产物；BRCA fold4 仅10个事件。已统一重建50个split，并新增覆盖、重复、有效标签及联合分层审计 | ✅ 数据与代码已修复，待统一重跑 |
| 07-30 | 稀疏事件训练信号不足 | v3.8 原协议用batch=8普通随机批次，BRCA约半数训练批次无事件；三个结构损失还在epoch1后直接全权重开启。新增癌种无关`robust`协议：患者完整事件分散批次、64例折内排序记忆、训练折分箱、确定性验证槽、5轮结构预热与10轮线性增权 | 🟡 待5-fold实测 |
| 07-30 | v3.8 调度器重复fork | 原脚本只有“结果是否完成”检查，没有跨进程互斥；现已增加每GPU调度器锁、每fold写入锁和同机死进程锁回收。旧的并发LUAD fold0结果不得使用，需停掉遗留进程后重跑 | ✅ 代码已修复 |
| 07-29 | v3.7 LUSC fold0=0.4575 | 配置未发现串癌种或维度错误；`highscore`验证阶段使用随机高斯槽，会放大fold方差。新增`stable`协议，仅切换为确定性验证槽并保持global_qcut不变 | 🟡 待单fold复跑 |
| 07-29 | v3.8 BRCA fold0 首次崩溃 | 初始运行 epoch0 即崩溃 (0.6078)，重跑后正常完成 | ✅ 已修复 |

---

## 🏥 多癌种数据集总览 (10 个)

| 癌种 | clinical | Split | v3.3 | v3.6 最佳 | v3.7 | v3.8 | v3.8.2 | v3.8.3 | v3.9 | v4.0 | v4.1 | ArcSurv |
|:----:|:--------:|:-----:|:-----:|:---------:|:----:|:----:|:------:|:------:|:----:|:----:|:----:|:------:|
| BLCA | 381 | 380 | 0.7311 ✅ | 0.7024 ETAR | 0.7249 ✅ | 0.7274 ✅ | 0.7159 | 0.5931 | 0.6394 | 0.7078 | 0.7039 | 0.6757 |
| BRCA | 1046 | 1045 | ⏳ 待重跑 | ⏳ 待重跑 | ⏳ 待重跑 | 0.6750 ✅ |
| COADREAD | 573 | 570 | 0.6774 ✅ | — | 0.7384 ✅ | — |
| HNSC | 438 | 437 | ⏳ | — | 0.6406 ✅ | — |
| KIRC | 488 | 488 | 0.7958 ✅ | — | 0.8149 ✅ | — |
| LUAD | 467 | 458 | ⏳ | 0.7647 NLL | 0.6662 ✅ | ⏳ |
| LUSC | 460 | 454 | ⏳ | 0.6141 NLL | 🔄 fold0=0.4575 | ⏳ |
| SKCM | 403 | 403 | 0.6770 ✅ | — | ⏳ | — |
| STAD | 366 | 362 | 0.6596 ✅ | — | ⏳ | — |
| UCEC | 488 | 487 | 0.7964 ✅ | — | ⏳ | — |
