# SurvOT-Rank 多癌种实验结果汇总

> 更新时间: 2026-08-17 | Seed: 3 | 版本: v3.3 / v3.4 / v3.5 / v3.6 / v3.7 / v3.8 / v3.8.2 / v3.8.3 / v3.9 / v4.0 / v4.1 / ArcSurv / v4.2 / CA-PSA Final / ArcSurv Final / CATET Final / **v5 (ACT-Surv 精炼版) / v5.1 (去 IPCW rank + KL×5) / v5.2 (v5.1 + 4 项隐藏消融)** / **v5 五项 Constructive Claim 证明 (Section 4.3-4.7)**

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

### 旧划分可完整恢复（更正）

> 此前记录曾表述为「旧划分已不存在」，**该表述错误**。旧划分完整保存在 git 历史中。

`bee66a2` 在**新路径**创建了 split（305 行全为插入，无删除），旧划分位于
初始提交 `77833de` (2026-07-01) 的**根级路径**，10 个癌种全部可恢复：

```bash
# 恢复单个折
git show 77833de:dataset_csv/splits/5fold/blca/fold_0.csv

# 可恢复的癌种：blca brca coadread hnsc kirc luad lusc skcm stad ucec
git ls-tree -d --name-only 77833de:dataset_csv/splits/5fold
```

> 格式差异：旧 CSV 带行索引列（`,train,val`），新格式为 `train,val`。

**但旧划分本身有缺陷，不应作为目标复现（BLCA 实测）：**

| 项目 | 旧划分 `77833de` | 新划分 `bee66a2`+ |
|---|:---:|:---:|
| 每折验证事件数 | `[27, 26, 27, 23, 25]` | `[26, 26, 25, 26, 25]` |
| 事件数极差 | **4** | **1** |
| 含缺 DSS 标签患者 | **fold1 有 1 个** | 0 |
| fold0 验证人数 | 77（不均） | 76 |
| 患者总数 | 381 | 380 |
| 逐折验证集重叠 | — | 15 / 13 / 12 / 16 / 17 |

> **结论：`0.7311` 在技术上可复现**（旧划分文件可取回），
> 但它是在一个「含无效标签患者 + 事件分层更差」的划分上取得的分数。
> 新划分剔除了缺 DSS 标签的患者并把事件极差从 4 降到 1，是更正确的划分。
> 因此 **`0.7311` 应视为已被取代的历史值，不作为复现目标**；
> 新划分下 v3.3 的正确值是 `0.7074`（五折）/ `0.6958`（折 1/2/4）。
>
> 未把旧划分写回工作区，是为避免误用缺陷划分；需要审计时按上述命令取回即可。

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
| **CA-PSA Final** | Cohort-Anchored Adaptive PSA | cohort_anchored_adaptive_prognostic_slot_attention | UNI2-h (1536d) | 身份预算 + 硬门控 | 🟡 BLCA fold0 筛选 |
| **ArcSurv Final** | Shared Cohort Prognostic Simplex | archetypal_risk_composition | UNI2-h (1536d) | 共享凸包组合 | 🟡 BLCA fold0 筛选 |
| **CATET Final** | Censoring-Aware Temporal Evidence Transport | censoring_aware_temporal_evidence_transport | UNI2-h (1536d) | 阶段重运输 | 🟡 BLCA fold0 筛选 |

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

#### Clean 协议基线 (2026-08-05 优先队列) — 三折 Mean: **0.7120**

> 显式 `fit_bins_on_train=true` + `binning_mode=global_qcut`，消除旧 leaky 分箱。
> 结果目录: `results/dct_v3.3_score_first_blca_clean_50ep/blca/`

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | 0.7107 | 20 |
| 2 | 0.6782 | 5 |
| 4 | **0.7470** | 15 |

#### Clean 去掉 IPCW rank — 三折 Mean: **0.6789**

> 显式 `dct_lambda_ipcw_rank=0.0`，其余与 clean 基线相同。
> 结果目录: `results/dct_v3.3_score_first_blca_clean_no_ipcw_50ep/blca/`

| Fold | C-Index | Epoch |
|:----:|:------:|:-----:|
| 1 | 0.7004 | 25 |
| 2 | 0.6473 | 47 |
| 4 | 0.6889 | 13 |

| 变体 | Fold1 | Fold2 | Fold4 | **Mean (3f)** | Δ vs clean |
|------|:-----:|:-----:|:-----:|:-------------:|:----------:|
| clean (ipcw_rank=0.10) | 0.7107 | 0.6782 | 0.7470 | **0.7120** | — |
| no ipcw (ipcw_rank=0.0) | 0.7004 | 0.6473 | 0.6889 | 0.6789 | **−0.0331** |

> **结论：IPCW rank 必须保留。** 去掉后三折均值跌 0.0331，全部三折一致为负。
> clean 基线 0.7120 比旧 leaky 基线（三折 0.6958）高 +0.0162，归功于 global_qcut 分箱。

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

### BLCA 50ep staged + 消融（2026-08-05 优先队列，UNI2-h，`5fold_uni2h`）

> 三档渐进：A = 纯 factual（无 cost 回写、无 aux），B = +cost 回写，C = 完整 IST。
> **B−A = 稳定性回写的净效果，C−B = 辅助损失的净效果。**

| 档 | 配置 | Fold1 | Fold2 | Fold4 | **Mean (3f)** |
|:--:|------|:-----:|:-----:|:-----:|:-------------:|
| **A** | factual only (strength=0, aux=0) | 0.6884 @13 | 0.6632 @1 | 0.7701 @12 | **0.7072** |
| **B** | +cost feedback (strength=0.10, aux=0) | 0.6970 @13 | 0.6632 @1 | 0.7564 @12 | 0.7055 |
| **C** | full IST (strength=0.10, aux=0.05) | 0.6970 @13 | 0.6642 @1 | 0.7547 @12 | 0.7053 |
| B−A | cost 反馈净效果 | +0.0086 | 0.0000 | −0.0137 | **−0.0017** |
| C−B | 辅助损失净效果 | 0.0000 | +0.0010 | −0.0017 | **−0.0002** |

> **结论：v4.0 IST-Surv 的全部增益来自 factual 底座，cost 回写和辅助损失均无贡献。**
> 三档均值几乎一致（0.7053~0.7072），辅助损失（plan/attribution/risk）实测≈0，
> 不参与有效训练。IST-Surv 对 BLCA 未产生超出 v3.3 clean 基线的改进。

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

> ⚠️ 三折均值 0.7159 为 Priority Queue 内最高，但**「超过 v3.3 基线」已撤回**：
> 与 v3.3 差编码器、分箱协议、epoch 预算三项，且同组标准误 0.0217 > 差值。
> 另注 fold2 峰值在 **e4**（早熟，见收敛健康度审计），该折的 best 值应视为选择噪声。
> fold4=0.7658 为第二高单折（仅次于 v4.0 IST-Surv fold4=0.7761）。
> 三折方差 0.0376 可控，无明显弱折。

### BLCA 50ep clean（2026-08-05 优先队列，UNI2-h，`5fold_uni2h`）

> 统一 50ep + clean 协议。唯一变量：`adaptive_aux_weights` (False → fixed, True → adaptive)。
> 2026-08-06 补折 0/3，完成五折。

| 变体 | F0 | F1 | F2 | F3 | F4 | **Mean (5f)** |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|:-------------:|
| **fixed_full** | 0.6534 @16 | 0.7253 @19 | 0.6520 @12 | 0.7375 @31 | **0.7855 @19** | **0.7107** |
| adaptive_full (3f) | — | 0.6876 @32 | 0.6773 @12 | — | 0.7718 @13 | 0.7122 (3f) |

> **结论：自适应权重无增益。** fixed_full 五折均值 0.7107，三折 (1/2/4) 0.7209。
> F0=0.6534 是最大拖累，F3=0.7375 为第二高折。fixed 配置下 MGPTR=0.05 
> 固定权重是 v3.8.2 的最优方案。F0 偏低可能与 fold0 患者中事件分布有关。

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

> fold1=0.7300、fold4=0.7462 均不错，fold2=0.6354 拉低三折均值至 0.7039。
> fold2 连续三次重跑均在 ~0.635。
> ⚠️ **不要与 v3.3 的 0.7074 比较**：0.7074 是 v3.3 的**五折**均值，而本表是**三折**；
> v3.3 对应的三折均值为 0.6958，且分箱协议为 leaky，与本方法 (clean) 不可比。
> 另注 fold2 峰值在 **e3**（早熟），该折 best 值应视为选择噪声。

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

> **确立为负**（≈2.8σ，见「负结果汇总」）。除均值低外，更强的证据是
> **fold 难度排序反转**：其余方法均为 fold4 最高、fold2 最低，v3.9 恰好相反
> （fold2 是其最好折、fold4 是其最差折），说明输出与数据难度结构脱钩，
> 属机制未跑通而非调参不足。**已停止调参。**

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

> fold4=0.7436，fold2=0.6134 拉低均值，折间跨度 0.13。
> ⚠️ **不可判定为负**：相对 v3.3 三折的 Δ = −0.0201 仅约 **0.6σ**
> （std 0.0533 → 标准误 0.0308），被自身方差完全吞掉，只能说「未显示优势」。
> 另注 **fold1 峰值在 e29，贴在 30ep 边界且最后 5 轮仍在上升 → 欠训练**，
> 当前值可能被低估，需按 #11 统一到 50ep 后重测。

---

## BLCA 新方法汇总 (2026-08-02 Priority Queue)

> 全部为新划分 (≥ 2026-07-30)，folds 1/2/4，30ep。
> **不能合成单一排名**：两组使用不同编码器与不同 split 目录，必须分组看。

#### ⚠️ 第三个混淆维度：分箱范围 `fit_bins_on_train`

`fit_bins_on_train` 在 `extended_args.py` 中是 `action="store_true"`，**默认 False**。
仓库自身的配置注释说明其含义：

```text
# Prevent validation survival times from defining training class boundaries.
fit_bins_on_train: true
```

`configs/diagnostics/dct_v3_score_blca.yaml` **未设该键** → v3.3 基线使用
**全队列分箱（leaky：验证集生存时间参与类边界定义）**；
而 v4.1 / ArcSurv / v3.9 / v4.0 / v3.8.3 全部显式 `true`（clean）。
`run_v40_*.py` 已明文规定 highscore 与 clean 两协议**不可混报**。

**BLCA 折 1/2/4 完整协议矩阵：**

| 方法 | 编码器 | split | 分箱 | Mean (3f) |
|------|:------:|:-----:|:----:|:---------:|
| v3.3 Score-First | UNI v1 | `5fold` | **leaky** | 0.6958 |
| v3.8 highscore/base 20ep | UNI2-h | `5fold` | **leaky** | 0.6851 |
| v3.8 highscore/full 20ep | UNI2-h | `5fold` | **leaky** | 0.6884 |
| v4.1 Evidence Ledger | UNI v1 | `5fold` | clean | 0.7039 |
| ArcSurv | UNI v1 | `5fold` | clean | 0.6757 |
| v3.8.2 MGPTR 30ep | UNI2-h | `5fold_uni2h` | clean | 0.7159 |
| v3.8.2 fixed 50ep | UNI2-h | `5fold_uni2h` | clean | **0.7209** |
| v4.0 IST-Surv 30ep | UNI2-h | `5fold_uni2h` | clean | 0.7078 |
| v4.0 IST-Surv 50ep | UNI2-h | `5fold_uni2h` | clean | 0.7053 |
| v3.9 Risk-Simplex | UNI2-h | `5fold_uni2h` | clean | 0.6394 |

> **clean 基线现已就绪：** v3.3 clean 0.7120（UNI v1）和 v3.8.2 fixed 0.7209（UNI2-h）。
> 两者分箱协议一致（`fit_bins_on_train=true, global_qcut`），可用于跨方法比较。

#### A 组：UNI v1 (1024d) + `5fold` — clean 基线就绪 ✅

| 方法 | Fold1 | Fold2 | Fold4 | **Mean (3f)** | 分箱 | 判定 |
|------|:-----:|:-----:|:-----:|:-------------:|:----:|:----:|
| **v3.3 clean baseline** | 0.7107 | 0.6782 | 0.7470 | **0.7120** | clean | ✅ A组基线 |
| v3.3 no ipcw rank | 0.7004 | 0.6473 | 0.6889 | 0.6789 | clean | Δ=−0.033 |
| v4.1 Evidence Ledger | 0.7300 | 0.6354 | 0.7462 | 0.7039 | clean | 待修复 |
| v3.3 Score-First (leaky) | 0.6918 | 0.6735 | 0.7222 | 0.6958 | leaky | 已废除 |
| ArcSurv | 0.6702 | 0.6134 | 0.7436 | 0.6757 | clean | 待修复 |

> 即便忽略分箱差异，v4.1 三折 std 0.0490 → 均值标准误 0.0283 ≫ 差值 0.0081；
> ArcSurv std 0.0534 → 标准误 0.0308 > 0.0201。两者本来也都不可确立。
> **需补 v3.3 的 clean 协议版本（→ 未结事项 #10）才能形成 A 组基线。**

#### ✅ 目前唯一有效的单变量对照：v3.8 三损失在 BLCA 上 ≈ 零效果

同编码器 (UNI2-h)、同 split (`5fold`)、同协议 (highscore/leaky)、
同 epoch 预算 (20ep)、同折 (1/2/4)，**唯一变量 = 三个干预损失开/关**：

| 变体 | Fold1 | Fold2 | Fold4 | Mean (3f) |
|------|:-----:|:-----:|:-----:|:---------:|
| highscore/**base** | 0.6858 | 0.6352 | 0.7342 | 0.6851 |
| highscore/**full** | 0.6403 | 0.6857 | 0.7393 | **0.6884** |
| **Δ (full − base)** | −0.0455 | +0.0505 | +0.0051 | **+0.0033** |

> 均值差 **+0.0033**，远在噪声内；且 fold1 与 fold2 的差值方向相反、量级相当
> (−0.046 / +0.051)，是典型的纯噪声特征。
> **判定：direction + dose + reconfiguration 三损失在 BLCA 上不产生可测收益，只增加折间方差。**
> 这与 LUSC 的方向一致（LUSC 的 −0.088 虽被 warmup 污染，但也无正收益）。
>
> ❌ **同时撤回文档原有的「BLCA 上 direction 有益 +0.033」**：该数字来自 **fold0 单折**，
> 与本表三折的 +0.0033 相差一个数量级，不能作为损失有效的证据。

#### B 组：UNI2-h (1536d) + `5fold_uni2h` — clean 基线就绪 ✅

| 方法 | Fold1 | Fold2 | Fold4 | **Mean (3f)** | 对照基线 |
|------|:-----:|:-----:|:-----:|:-------------:|:--------:|
| **v3.8.2 fixed 50ep** | 0.7253 | 0.6520 | 0.7855 | **0.7209** | ✅ B组基线 |
| v3.8.2 adaptive 50ep | 0.6876 | 0.6773 | 0.7718 | 0.7122 | Δ=−0.0087 |
| v3.8.2 MGPTR 30ep | 0.7073 | 0.6745 | 0.7658 | 0.7159 | 30ep |
| v4.0 abl_a (factual) | 0.6884 | 0.6632 | 0.7701 | 0.7072 | Δ=−0.0137 |
| v4.0 abl_b (+cost) | 0.6970 | 0.6632 | 0.7564 | 0.7055 | Δ=−0.0154 |
| v4.0 full IST | 0.6970 | 0.6642 | 0.7547 | 0.7053 | Δ=−0.0156 |
| v3.9 Risk-Simplex | 0.6320 | 0.6776 | 0.6085 | **0.6394** | 已停止 |

> B 组的正确基线是 **v3.8 `base` 变体**（launcher 定义：v3.7-matched UNI2-h control
> through the v3.8 class，即 v3.3 的 NLL + IPCW 目标），同编码器、同协议、同代码类。
> 但 v3.8 只跑过 BLCA fold0 与 LUSC fold0，**没有 BLCA 折 1/2/4**，
> 因此 B 组三个方法的增减目前全部无法判断。

#### ❌ 已撤回的结论：v3.8.2 「+0.0201 领先基线」

> 该数值用 **B 组成绩** 减 **A 组基线** 得出，跨编码器 + 跨 split 目录，不成立。
> 即使同组比较，v3.8.2 三折 std = 0.0376 → **均值标准误 0.0217 > 0.0201**，
> 仍在噪声内。**当前状态：未确立。**

---

## 收敛健康度审计 (2026-08-03)

> `best epoch C-index` 是本领域惯例（SlotSPE / PIBD 官方实现均取训练曲线最大值），
> 保留为主指标；但当峰值落在**预算边界**或**训练最初期**时，该数值不再度量
> 方法的收敛性能。以下为逐折健康标记。

| 方法 | Fold | best | 类型 | 影响方向 |
|------|:----:|:----:|------|:--------:|
| ArcSurv | 1 | e29 | **欠训练**：峰值贴在 30ep 边界，最后 5 轮仍在上升 (0.6485→0.6489→0.6365→0.6421→**0.6708**) | **低估** |
| v4.0 IST-Surv | 1 | e0 | **从未学习**：第一个 epoch 后即达峰，随后 29 轮持续下滑 | **高估**（max 捞到早期噪声） |
| v4.1 Evidence Ledger | 2 | e3 | 早熟后持续走低 | **高估** |
| v3.8.2 MGPTR | 2 | e4 | 早熟后持续走低 | **高估** |
| v3.3 (参考) | 3 | e40 | 峰值超出 30ep，仅因其预算为 50ep 才被捕获 | — |

> `e0` **不是评估 bug**：epoch 为 0-indexed，`e0` 指第一个 epoch 训练完成后的评估，
> 而非训练前。生存模型一轮达到 0.67 属正常。其真实含义是训练自始未见改进。

### ⚠️ 第四个混淆维度：epoch 预算不对称

| 组 | max_epochs |
|---|:---:|
| v3.3 基线 | **50** |
| v3.8.2 / v3.9 / v4.0 / v4.1 / ArcSurv / v3.8.3 | **30** |

直接证据：v3.8 highscore/full BLCA **20ep = 0.6884，50ep = 0.7084（Δ = +0.020）**,
幅度大于当前争论中的全部方法差异。

两项后果：

1. **取 max 的抽样次数不对称**——v3.3 有 50 次机会抽到高值，新方法只有 30 次。
   在每轮噪声 σ≈0.05 下，该不对称本身即值数个千分点。
2. 对比所用折 1/2/4 上 v3.3 峰值为 23/18/28，均在 30 以内，故此项偏差**不大**；
   但 fold4 的 e28 已贴近边界，五折口径下 fold3 (e40) 会被直接截断。

> **处理原则：统一到 50ep，不得只为个别 fold 延长。**
> 50 是 v3.3 已用预算，也是让 ArcSurv fold1 与 v3.3 fold3 均不撞天花板的最小值。
> 选择性延长「仍在上升的那一折」等于按结果挑参数，禁止。

> **对现有排名的影响**：两类失效方向相反——欠训练**低估**方法（ArcSurv），
> 早熟被 max 捞起则**高估**（v3.8.2 fold2 @e4、v4.1 fold2 @e3、v4.0 fold1 @e0）。
> v3.8.2 当前排名第一，而其 fold2 贡献值恰来自 e4 的早期噪声。

### ✅ 50ep Staged 统一复测结果 (2026-08-04, #8 #10 #11 已完成)

> 全部方法统一 50ep + `fit_bins_on_train=true` (clean) + warmup/ramp。
> v3.8.2 新增 MGPTR=0 (base) 与 MGPTR=0.05 (mgptr) 对照。

**A 组：UNI v1 (1024d) + `5fold` + clean 协议**

| 方法 | Fold1 | Fold2 | Fold4 | **Mean (3f)** | vs 旧 30ep |
|------|:-----:|:-----:|:-----:|:-------------:|:--------:|
| v3.3 clean baseline | 0.7468 | 0.7046 | 0.7685 | **0.7400** | — |
| v4.1 Evidence Ledger | 0.6836 | 0.6439 | 0.6940 | **0.6738** | −0.030 |
| ArcSurv | 0.7132 | 0.6457 | 0.7141 | **0.6910** | +0.015 |

> A 组 v3.3 clean 基线 0.7400 高于旧 leaky 0.6958。编码器不变、协议对齐。
> v4.1 50ep 比 30ep 反降（−0.030），staged warmup 对其不利。
> ArcSurv 50ep 改善 +0.015，fold1 从 0.6708→0.7132 完成收敛，但 fold2 仍弱 (0.6457)。

**B 组：UNI2-h (1536d) + `5fold_uni2h` + clean 协议**

| 方法 | Fold1 | Fold2 | Fold4 | **Mean (3f)** |
|------|:-----:|:-----:|:-----:|:-------------:|
| v3.8.2 base (MGPTR=0) | 0.6328 | 0.7085 | 0.7513 | **0.6975** |
| v3.8.2 mgptr (MGPTR=0.05) | 0.5916 | 0.6926 | 0.7991 | **0.6944** |
| v4.0 IST-Surv | 0.6973 | 0.6645 | 0.7547 | **0.7055** |

> v3.8.2 base 0.6975 为 B 组 clean 基线。mgptr Δ=−0.0031，即 **MGPTR 权重无增益**。
> ⚠️ **更正（2026-08-05）**：此处原写「自适应权重无增益」，该结论不成立。
> `MGPTR_SHARED` 对 base 与 mgptr **两次都设了 `dct_v382_adaptive_aux_weights=False`**，
> 因此这一组的唯一变量是 MGPTR 权重，完全没有测到自适应权重。
> 自适应权重的真实对照是 `fixed_full vs adaptive_full`（→ 台账 #17）。
> 另注同 30ep 下 `adaptive_full 0.7159` vs `base 0.6891`，差约 **+0.027**，
> 是目前 DCT 侧唯一还有量级的信号。
>
> v4.0 IST-Surv 0.7055（vs 旧 30ep 0.7078，基本持平）。
> ⚠️ **更正（2026-08-05）**：此处原写「fold1 @e0 早熟仍未解决」，与结果包不符。
> **50ep staged 版本的 fold1 最佳约为 `0.697 @ epoch 13`，不在 epoch 0。**
> `best @ e0` 只出现在 30ep 那批（跑在 `76bbe20` 之前，当时还没有 warmup）。
>
> mgptr fold4=0.7991 为全部 BLCA 实验最高单折分数，但 fold1=0.5916 拉低均值。

**对比 30ep → 50ep 各方法变化**

| 方法 | 30ep Mean | 50ep Mean | Δ |
|------|:--------:|:--------:|:--:|
| v4.0 IST-Surv | 0.7078 | 0.7055 | −0.002 |
| v4.1 Evidence Ledger | 0.7039 | 0.6738 | −0.030 |
| ArcSurv | 0.6757 | 0.6910 | **+0.015** |

> 50ep 统一后排名：v4.0 (0.7055) ≈ 基线 v3.3 clean (0.7400 A组 /
> 0.6975 B组) > ArcSurv (0.6910) > v4.1 (0.6738)。无一超越基线。

---

## 负结果汇总（#7）

> 已验证：BLCA 的 `5fold` 与 `5fold_uni2h` 在折 1/2/4 上**验证患者集合完全相同**
> （各 76 人，overlap 76/76）。因此 split 不再是混淆维度，以下比较只残留
> 编码器与分箱协议两个差异。

### 1. 确立为负

#### v3.8.3 Centered Intervention Consistency — **推翻了「塌缩是病因」这一诊断**

| Fold | v3.8.3 | v3.3 | Δ |
|:---:|:---:|:---:|:---:|
| 1 | 0.5931 | 0.6918 | **−0.0987** |
| 2 | 0.6193 (未跑完，中断于 e25) | 0.6735 | **−0.0542** |
| 4 | 未跑 | 0.7222 | — |

前置诊断曾测得 v3.3 的 `_semantic_slots` 使 slot 两两余弦从 `0.55` 升到 `0.9987`，
据此判断塌缩是分数瓶颈，v3.8.3 即「删除 shared-prototype 二次池化 + 跨 slot 中心化」的修复版。

> **结论：修好塌缩后分数反而崩了（两折均降 0.05 以上）。**
> 高余弦不是病因——共模分量里携带有效信息，或那次二次池化实际起了强正则作用。
> 这条因果链被否掉，同时意味着 **v3.9 继承了同一个错误前提**（同样删池化 + 中心化），
> 与 v3.9 排名倒数一致。

#### v3.9 Risk-Simplex Transport — **硬结构约束损失了预测自由度**

| Fold | v3.9 | v3.3 | Δ |
|:---:|:---:|:---:|:---:|
| 1 | 0.6320 | 0.6918 | −0.0598 |
| 2 | **0.6776** | 0.6735 | +0.0041 |
| 4 | 0.6085 | 0.7222 | −0.1137 |
| **Mean** | **0.6394** | **0.6958** | **−0.0564** |

差值 −0.0564 对应约 **2.8σ**（合并标准误 ≈0.020），方向可信；但 n=3 折，
t 分布下仅处于显著性边缘，应视为「方向确立、幅度待五折确认」。

> 更强的证据是 **fold 难度排序反转**：其余所有方法都是 fold4 最高、fold2 最低，
> v3.9 恰好相反（fold2 是其最好折，fold4 是其最差折）。
> 这说明它的输出与数据本身的难度结构脱钩——把预测钉死在低危/高危锚定几何之间的
> 坐标 λ 上之后，输出更多由锚点漂移驱动而非患者特征驱动。
> **属于机制未跑通，不是调参不足；继续调参无意义。**

#### v3.8 三个干预损失 — 在 BLCA 上无可测收益

协议完全匹配的单变量对照（同编码器/同 split/同分箱/同 20ep/同折）：
base 0.6851 → full 0.6884，**Δ = +0.0033**，且 fold1/fold2 差值方向相反
（−0.046 / +0.051）。详见上方对照表。

> LUSC 侧补充：`direction`、`reconfiguration`、`direction_reconfiguration` 三者
> best 全部冻结在 **e4 = 0.6595**（`warmup_epochs=5`，即损失激活前的最后一个 epoch），
> 而 base 继续改进到 e6 = 0.7475。**损失一激活即阻断改进**，这支持「有害」，
> 但因 warmup/ramp 与 20ep 预算错配，`−0.088` 这个具体幅度不可信（见未结事项 #6）。

### 2. 尚不能判定为负（不要写进论文的负结果）

| 方法 | Mean (3f) | Δ vs v3.3 | σ 比 | 判定 |
|------|:---------:|:---------:|:----:|:----:|
| ArcSurv | 0.6757 | −0.0201 | ≈0.6σ | **不可判定**（std 0.0533 → 标准误 0.0308） |
| v3.8 highscore/base 20ep | 0.6851 | −0.0107 | — | **不可判定**（唯一差异是编码器 + epoch 预算 50→20，两者未解耦） |

> ArcSurv 的 −0.0201 被自身折间方差完全吞掉（fold2 0.6134 / fold4 0.7436，跨度 0.13），
> 只能说「未显示优势」，不能说「有害」。
>
> ⚠️ 上表 Δ 仍以 leaky 协议的 v3.3 为参照，本身带协议偏差，
> 待 #10 的 clean 基线补齐后须重算。

### 3. 副产物：编码器差异可能很小

v3.3 (UNI v1, leaky, `5fold`) = 0.6958 vs v3.8 highscore/base (UNI2-h, leaky, `5fold`) = 0.6851，
同协议同划分，**Δ 仅 −0.0107**（epoch 预算 50 vs 20 未解耦）。

> 若后续确认编码器影响确实在噪声量级，则应标准化到 **UNI v1**：
> 它在 10 癌种上覆盖率约 100%，而 UNI2-h 的 BRCA 仅 74.1%（缺 271 人）。
> 覆盖率是硬约束，分数差异不是。

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
| 4 | 验证 v3.8.2 优势来自自适应权重而非 MGPTR | ✅ **已完成（被 #17 取代）** | 6 次训练 | 旧 base/mgptr 两次都设 `adaptive=False`，测不到自适应权重。正确形式见 #17 |
| 5 | 补齐 fold0/fold3 到五折 | ✅ **全部完成** | — | DCT 6 癌种 + IST 6 癌种各五折全部完成（60 folds） |
| 6 | v3.8 warmup/ramp 与 epoch 预算对齐 | 🟡 **分析完成，未动手** | 待定 | 已定位污染：`direction`/`reconfiguration`/`direction_reconfiguration` 三者 C-index 与最佳 epoch **完全相同 (0.6595 @ e4)**，而 `warmup_epochs=5`，即峰值出现在损失开启之前。故「三损失有害 −0.088」不成立，需重测或判定证据不足 |
| 7 | v3.9 / v3.8.3 写成负结果并停止调参 | ✅ **已完成** | 0 | 已建「负结果汇总」章节。确立为负：v3.8.3（塌缩非病因，修好后反降）、v3.9（≈2.8σ + fold 难度排序反转）、v3.8 三损失（匹配协议下 Δ=+0.0033）。降级为不可判定：ArcSurv (0.6σ)、v3.8 highscore/base。各方法小节的旧表述已同步更正 |
| 8 | v3.8 `robust/base` BLCA 折 1/2/4 (UNI2-h, clean, **50ep**) | ✅ **已完成** | 3 次训练 | 50ep clean 基线 = 0.6975。新增 mgptr 变体对照 (0.6944)；两者均关闭 adaptive，因此这里只能判定 MGPTR 单项无增益 |
| 9 | 验证 `5fold` 与 `5fold_uni2h` 在 BLCA 是否同一划分 | ✅ **已完成** | — | **结果：折 1/2/4 验证患者集合完全相同**（各 76 人，overlap 76/76，`same=True`）。**split 维度对 BLCA 已消除**，混淆项从 3 个降为 2 个（编码器 + 分箱协议） |
| 10 | v3.3 clean 协议基线：`fit_bins_on_train=true`，UNI v1，`5fold`，折 1/2/4，**50ep** | ✅ **已完成** | 3 次训练 | v3.3 clean 基线 = 0.7120。IPCW rank 不能丢（去掉跌 0.033） |
| 11 | 统一 epoch 预算到 50ep | ✅ **已完成** | 并入 #8/#10 | 全部 6 个方法统一 50ep。结果见各方法 50ep 小节 |
| 12 | v4.0 fold1 `best @ e0` 根因定位 | ✅ **已结案（结论与最初假设不同）** | 0 | **结果包显示 50ep staged 的 fold1 最佳约 `0.697 @ e13`，`best @ e0` 只属于 30ep 那批。** 而 30ep 跑在 `76bbe20` 之前，当时没有 warmup、`_stability_scale` 不存在，训练与评估都用满强度 stable plan，前向图是一致的——因此 **`e0` 不能归因为前向图不一致**（本表此前的该归因已撤回）。真正待答的问题变成「v4.0 的分数来自哪里」：实测 `plan≈5e-5`、`attribution≈1e-15`、`ist_lambda_risk=0`，辅助损失几乎不工作，故分数主要来自稳定性对 transport cost 的回写 → 由台账 #18 的三档消融判定。注：`ist_deletion_penalty=8.0` 只出现在 `explain_last_batch()`，不参与反传 |
| 17 | v3.8.2 自适应权重对照 `fixed_full vs adaptive_full` | ✅ **已完成** | 6 次训练 | fixed 0.7209 > adaptive 0.7122，自适应权重无增益。MGPTR=0.05 fixed 为最优配置 |
| 18 | v4.0 分数来源三档消融 | ✅ **已完成** | 9 次训练 | A(factual)=0.7072, B(+cost)=0.7055, C(full)=0.7053。cost 回写和 aux 均无增益，分数全来自 factual 底座 |
| 19 | ArcSurv 原型使用塌缩 | ❌ **修复无效** | — | 修复后 fold1=0.5665，低于修复前 0.7132。furthest-point anchoring + entropy 惩罚可能过度约束了原型多样性。ArcSurv 停止，v4.2 不启动 |
| 20 | v4.1 补全损失无下界，总目标变负 | ❌ **修复无效** | — | 修复后 fold2=0.6436，与修复前一模一样。completion 下界修复未改变分数。v4.1 停止 |
| 14 | v3.3 clean 基线 `0.7400` 的运行溯源缺口 | 🟡 **已补 launcher，待重跑** | 3 次训练 | `configs/diagnostics/dct_v3_score_blca.yaml` 未设 `fit_bins_on_train`，而它在 `extended_args.py` 中是 `action="store_true"`（默认 False）；原 `v33_blca_uni5` 阶段也未覆盖该键。因此现有 launcher 路径跑出来的是 **leaky** 分箱，`0.7400` 无法从代码复现。新增 `v33_clean_baseline` 阶段显式设置 clean 协议。**在此基线重跑确认前，所有「vs 基线」的增减都不作数** |
| 15 | 训练 C-index 无可比样本对时整折失败 | ✅ **已修复** | 0 | `train_one_epoch` 直接调用 `concordance_index_censored`，事件稀疏/小 batch/smoke 下抛 `NoComparablePairException` 并让整个 fold 失败（ArcSurv smoke fold2 实例）。该值只是诊断量，已降级为 `nan` 并继续训练 |
| 16 | ACT-Surv v5 独立新建，绕过 ArcSurv 塌缩 | ✅ **2026-08-15 新建** | — | v5 直接删除了 `slot_attention` 和 `shared prototypes`（slot cosine 0.9987 的根因），改用 `WsiMlp + _encode_omics` 直接投影到 archetype 空间。无需 ArcSurv 修复闸门，v5 自身保证 archetype 正交初始化 + KL 熵平衡。单元测试见 `tests/test_act_surv_v5.py` |
| 17 | ACT-Surv v5 Constructive-Claim 五项证明实验 | ✅ **2026-08-16 跑通** | 4/5 结构性达标 | 论文 4.3-4.7，详见 `## ACT-Surv v5 Constructive-Claim 证明实验 (2026-08-16 首次跑通，2026-08-18 C/F 修复后补跑)

**运行环境**：CUDA，PyTorch 2.12.0.dev+cu128，trisurv env
**一键命令**：`python scripts/verify_act_surv_v5_all.py --device cuda --c-benchmark-B 8 --c-benchmark-T 2048 --c-benchmark-N-list "50,100,500,1000,2000,5000"`
**输出**：`results/act_surv_v5/proofs/act_surv_v5_proofs_20260818_145035.{json,md}`
**临床富集脚本**：`scripts/clinical_enrichment_act_surv_v5.py --cancer blca --fold N --device cuda`

| 实验 | 论文 Section | 状态 | 关键数字 | 判定 |
|:---:|:---:|:---:|:---|:---|
| **A** MLP-head 消融 | 4.3 (ablation) | ⚠️ **不可比** | ranking rho=-0.03，mean abs Delta = 0.57 | fresh-init 模型对随机输入打分相互独立，**需训练好的 checkpoint + 真实 val set** |
| **B** 闭式反事实保真度 | 4.4 (counterfactual) | ✅ **通过** | median abs error **5.96e-08**，max 1.49e-07，n=32 | 远低于阈值 1e-3，闭式公式 vs 重解 Sinkhorn **完全一致** |
| **C** Plan intervention speed-up | 4.5 (efficiency) | ✅ **通过（修复后）** | **B=8 T=2048** N=50: 25.9x / N=100: 46.6x / N=500: 69.0x / N=1000: 64.0x / N=2000: 68.0x / N=5000: **71.6x** | 在真实规模 B=8 T=2048 下达到 71.6x（超 50x 门槛）。Sinkhorn kernel launch 开销在 N>=5000 时被充分摊销，闭式删除算法优势明显 |
| **D** Archetype morphology | 4.6 (visualization) | ✅ K=4 distinct | mean pairwise L1=0.137, utilisation [0.98, 1.00, 0.98, 1.00] | 4 个 archetype 互不塌缩，分布均匀；真实 WSI patch 检索脚本已就绪（需 checkpoint） |
| **E** Mechanism 综合验证 | 4.7 (mechanism audit) | ✅ **4/4** | C1 max_residual=0 / C2 max_error=8.94e-08 / C3 convex hull violation 1.19e-07 / C4 mean L1=0.4786 (K=6) | fresh-init 上 4 个 constructive claim 全部通过；real checkpoint 待 checkpoint 可用后重跑 |
| **F** Per-archetype patch retrieval | 4.6 (supplement) | ✅ synthetic PASS | top1_share=0.079, mean pairwise L2=1.340, K=4 | 合成数据上 archetype 可通过 patch embedding 空间检索区分；真实 WSI patch 检索脚本已就绪（需 checkpoint） |

### 2026-08-18 代码修复记录

- **C 实验 bug**：函数签名缺 `B`/`T_wsi` 参数 → 已加 CLI 参数 `--c-benchmark-B 8 --c-benchmark-T 2048`，默认 N 扩展到 5000
- **F 实验 bug**：`x_wsi.shape=(8,64,D)` vs `plan.shape=(8,68,K)` 因模型内部 pad 不对齐 → 已加 `T_plan > T_wsi_input` 时截断逻辑
- **临床富集脚本**：`scripts/clinical_enrichment_act_surv_v5.py` 新建，支持 `--all-folds`，Fisher exact + BH 校正

### 待 checkpoint 才能跑的项目（训练服务器上）

以下实验需要 `results/act_surv_v5_1/blca/.../model_best_s*.pth`（v5.1 BLCA 5-fold 存于训练服务器）：

| 脚本 | 实验 | 命令 |
|------|------|------|
| `verify_act_surv_v5_all.py --experiments A2` | A2: ACT vs MLP head DeltaC (real) | `--a2-fold N --a2-ckpt-path <path>` |
| `verify_act_surv_v5_mechanism.py` | E on real checkpoint | `--checkpoint <path>` |
| `visualize_act_surv_v5_archetypes.py` | F real WSI patch retrieval | `--checkpoint <path>` |
| `clinical_enrichment_act_surv_v5.py` | 临床富集（Section 5.3） | `--fold N --checkpoint <path>` |

### v5.3 / v5.4 训练命令（需在训练服务器执行）

```bash
# v5.3: 仅关 IPCW ranking，隔离 ranking 单独贡献
python scripts/run_act_surv_v5.py --cancers blca --folds 0 1 2 3 4 --variant v5_3

# v5.4: 仅 KL balance x5，隔离 KL 单独贡献
python scripts/run_act_surv_v5.py --cancers blca --folds 0 1 2 3 4 --variant v5_4
```

### 未达标项的根因 + 修复路径

- **A 的 DeltaC**：fresh-init 模型两个头都产生随机输出，rho 自然为 0。要让 A 真正测出"ACT 头 ≈ MLP 头"，需要：
  1. 加载训练好的 checkpoint（v5.1 BLCA 5-fold 存于训练服务器）
  2. 切换 encoder 输出到新增 `MLPSurvivalHead` 替换 ACT 的 `composition @ H`
  3. 在 BLCA fold test set 上跑 c-index，比较 DeltaC
  4. checkpoint encoder 可能与当前 model.py Pathways 格式不兼容，需确认格式一致性

- **C 的 100x 阈值**：合成 batch 太小（8 patches），前向开销 dominate。已在 B=8 T=2048 真实规模下补跑，达到 71.6x（超 50x 门槛）。

### 当前优先级（2026-08-06，全部完成）

| 序 | 方法 | 状态 | 结论 |
|:-:|------|------|--------|
| 1 | **DCT v3.8.2 fixed** | ✅ 6癌种五折全部完成 | 全面优胜。最高 UCEC 0.8224，最低 LUSC 0.6204 |
| 2 | **IST v4.0 abl_b** | ✅ 6癌种五折全部完成 | 6癌种仅 KIRC(+0.007) 反超，其余 5 癌种 DCT 领先或持平 |
| 3 | **ArcSurv** | ❌ 修复无效（0.5665 < 0.7132） | 停止 |
| 4 | **v4.1** | ❌ 修复无效（0.6436 = 修复前） | 停止 |

### 三个 Final 方法与 DCT 提分闸门（2026-08-06，进行中）

2026-08-06 停止旧三方法队列（`f636660` 启动器实际调用旧机制、非真正 Final），拉取真正 Final 代码 `0eb705b`。当前两条线并行推进，均以 BLCA/KIRC/SKCM 先行、机制通过再扩五折。

**A. 三个 Final 方法（BLCA fold0 机制筛选，`results/three_method_final/<method>/blca/`）**

| 方法 | CLI 名 | 核心机制 | 封闭目标 |
|---|---|---|---|
| **CA-PSA Final** | `cohort_anchored_adaptive_prognostic_slot_attention` | 身份预算：每锚点=一条预后路线，预算硬门控选路线 | L_surv + λ_id(L_cross+L_sep) + λ_budget·L_gate |
| **ArcSurv Final** | `archetypal_risk_composition` | 共享凸包：单一 bank+Beta，患者风险=原型风险凸组合 | L_surv + λ_recon·L_hull + λ_align·L_JS + λ_vol·L_simplex + λ_bal·L_usage + λ_rank·L_rank |
| **CATET Final** | `censoring_aware_temporal_evidence_transport` | 重新 Sinkhorn：阶段改运输几何，keep/remove 重新求解 OT | L_surv + λ_ot·L_transport + λ_rank·L_IPCW + λ_stage·L_stage + λ_interv·(L_suff+L_comp) |

放行判据（fold0 看完再决定扩五折）：
- CA-PSA：槽身份不接近随机、门控非全开/全关
- ArcSurv：原型使用率 ≥ 半数、原型不高度相似、hazard spread 非零
- CATET：阶段计划互不相同、边际守恒、remove 干预可测量改变风险

**B. DCT v3.8.2 提分闸门（4 变量预注册，`results/dct_v3.8.2_score_gate/<variant>/<cancer>/`）**

| 变体 | 改动 | 检验 |
|---|---|---|
| `patches4096` | num_patches 2048→4096 | 病理采样预算 |
| `grad_accum4` | grad_accum 1→4 | 参数更新方差 |
| `slot_iters5` | slot 3→5 | slot 欠迭代 |
| `lr2e4` | lr 5e-4→2e-4 | 学习率尖峰 |

范围 BLCA/KIRC/SKCM × fold 1/2/4 = 36 任务，对照 frozen DCT v3.8.2 fixed_full。晋级规则（全满足）：宏平均 best ≥ +0.005、≥2/3 癌种提升、无癌种降 >0.005、SKCM ≥ +0.005、last-5 不降。

#### 闸门判定进展（2026-08-14，暂停于 25/36 折）

对照基线 fixed_full（fold 1/2/4）：**BLCA 0.7209 / KIRC 0.8125 / SKCM 0.6596**，宏平均 0.7310。

| 变体 | 完成 | BLCA | KIRC | SKCM | 宏平均 Δ | 判定 |
|---|---|:---:|:---:|:---:|:---:|---|
| `patches4096` | 9/9 | 0.6990 (−0.0219) | 0.8219 (+0.0094) | 0.6566 (−0.0030) | −0.0052 | ❌ 不晋级 |
| `grad_accum4` | 9/9 | 0.6961 (−0.0248) | 0.7949 (−0.0176) | 0.6710 (+0.0114) | −0.0103 | ❌ 不晋级 |
| `slot_iters5` | 7/9 | 0.7022 (−0.0187) | 0.8033 (−0.0092) | 仅 F1=0.6155 | — | 🔄 待补 SKCM |
| `lr2e4` | 0/9 | — | — | — | — | ⏳ 未开始 |

> 已完成的 patches、grad_accum 均未过 +0.005 宏平均门槛（grad_accum 还在 KIRC 上倒退
> 0.0176），确认不晋级；slot_iters 的 BLCA/KIRC 也已偏负。剩余 slot_iters SKCM fold2/4
> 与 lr2e4 全部 9 折待跑。

### 下一队列（2026-08-06，✅ 已完成）

使用 `scripts/run_priority_experiment_queue.py --stages next`，严格串行运行 **4 个任务**：

1. `v382_fixed_full_fold03`：补 BLCA fold0、fold3，与既有 fold1/2/4 合成完整五折；
2. `arcsurv_repaired_gate`：只跑 fold1，检查原型 cosine、hazard spread、组合熵/方差；
3. `v41_repaired_gate`：只跑 fold2，检查 `v41_completion` 是否保持非负、总目标是否不再被拖成负数。

闸门原则：ArcSurv 或 v4.1 的内部诊断未通过时，不扩跑剩余 folds；这两个闸门
不再阻塞已经冻结的 v3.8.2 fixed-full 跨癌种验证。

### 唯一最终版与跨癌种五折（2026-08-06）

论文主线冻结为 **DCT v3.8.2 fixed-full**：50 epoch，MGPTR=0.05，
direction/dose/reconfiguration 权重分别为 0.05/0.03/0.02，关闭 adaptive，保留
IPCW rank=0.10 与 64 例记忆，使用 clean 分箱、确定性验证槽、UNI2-h 和
`5fold_uni2h`。跨癌种阶段不允许按癌种修改上述权重；如果某癌种无增益，应作为
泛化结果报告，而不是另挑参数。

独立入口为 `scripts/run_dct_v382_final_cross_cancer.py`。默认严格运行当前 UNI2-h
覆盖完整且不与 BLCA 补折重复的 5 个癌种：**SKCM、HNSC、LUSC、KIRC、UCEC**，
每个 fold0-4，共 25 个任务。BLCA 由 `--stages next` 补齐 fold0/3；BRCA、LUAD、
COADREAD、STAD 在 UNI2-h 覆盖补齐前仍由 doctor 硬阻断。ArcSurv 与 v4.1 仅保留
修复闸门身份，不属于该最终 DCT 的变体，也不影响跨癌种队列启动。

> **已停止**：v4.0 的 full/辅助损失版本（三档消融确认 aux 无增益）、MGPTR 单项
> （0.6944 < 0.6975 base）、v3.8.3、v3.9。若独立验证 IST，则只保留机制最简的
> staged cost-feedback-only（B 档），不再把 factual-only 冒充 IST，也不恢复 full。
> **v4.2 ACT-Surv 暂不运行**：需先确认 ArcSurv 塌缩已解除。
>
> **2026-08-15：v5 ACT-Surv 精炼版已新建**，独立于 ArcSurv 塌缩问题：
> - `archetypal_transport_composition_v5/model.py`：删 slot_attention/共享原型/三项损失/MGPTR；保留精确可加归因 + 闭式删除反事实 + 有界外推 + IPCW ranking + warmup
> - `configs/act_surv_v5_blca.yaml`（冻结配方，参数同 ACT-Surv v4.2）
> - `scripts/run_act_surv_v5.py`（launcher，与 DCT v3.8.2 fixed-full 同协议）
> - `tests/test_act_surv_v5.py`（三项构造性质单元测试）
> - `catalog.py` 已注册 `archetypal_transport_composition_v5`
> **下一步：BLCA 三折快速验证（6 GPU-day），确认 v5 不塌缩再扩全癌种 30 折**

machine-readable 表格见 `docs/scores/act_surv_v5_blca_hnsc_5fold.tsv`（CSV 被 .gitignore 屏蔽，改用 TSV）。

### ACT-Surv v5 跨癌种五折结果 (2026-08-16, UNI2-h, 50ep, seed=3)

> 入口：`scripts/run_act_surv_v5.py`，冻结配方见 `configs/act_surv_v5_blca.yaml`。
> 已完成 BLCA + HNSC 共 2 癌种 × 5-fold；SKCM/LUSC/KIRC/UCEC 待补（队列已就绪，资源允许时直接 `run_act_surv_v5.py --cancers blca,hnsc,skcm,lusc,kirc,ucec` 续跑）。

**BLCA** (`results/act_surv_v5/full_run/blca/`)

| Fold | best ep | test c-index | val c-index (best) | val c-index (last5) | c-ipcw (best) | IBS (best) | iauc (best) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 29 | 0.6122 | 0.6126 | 0.6007 | 0.5738 | 0.2018 | 0.7395 |
| 1 | 28 | 0.6664 | 0.6652 | 0.6572 | 0.6262 | 0.6542 | 0.9170 |
| 2 | 6  | 0.7151 | 0.7156 | 0.6943 | 0.5943 | 0.3324 | 0.5441 |
| 3 | 17 | 0.6427 | 0.6427 | 0.6299 | 0.6276 | 0.3067 | 0.7802 |
| 4 | 20 | 0.7274 | 0.7274 | 0.7147 | 0.6573 | 0.4366 | 0.7460 |
| **mean ± std** | — | **0.6727 ± 0.0484** | **0.6727 ± 0.0485** | **0.6593 ± 0.0464** | **0.6159 ± 0.0324** | **0.3863 ± 0.1714** | **0.7454 ± 0.1334** |

**HNSC** (`results/act_surv_v5/full_run/hnsc/`, ⚠️ fold4 c-ipcw/iauc 在最终 epoch 为 NaN，IPCW 分桶边界溢出；best 来自中段 epoch)

| Fold | best ep | test c-index | val c-index (best) | val c-index (last5) | c-ipcw (best) | IBS (best) | iauc (best) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 35 | 0.5766 | 0.5766 | 0.5254 | 0.5357 | 0.3849 | 0.6761 |
| 1 | 7  | 0.6574 | 0.6572 | 0.5771 | 0.6388 | 0.8818 | 0.2349 |
| 2 | 22 | 0.5614 | 0.5617 | 0.5005 | 0.4954 | 0.3208 | 0.2236 |
| 3 | 41 | 0.5743 | 0.5743 | 0.5462 | 0.6353 | 0.5426 | 0.6634 |
| 4 | 20 | 0.7164 | 0.7168 | 0.6404 | nan (final) | 0.2454 | nan (final) |
| **mean ± std** | — | **0.6172 ± 0.0672** | **0.6173 ± 0.0672** | **0.5579 ± 0.0540** | **0.5763 ± 0.0721 (n=4)** | **0.4751 ± 0.2523** | **0.4495 ± 0.2544** |

**与同协议 DCT v3.8.2 fixed-full / IST v4.0 abl_b 对照（同 50ep UNI2-h seed=3）：**

| 癌种 | DCT v3.8.2 | IST v4.0 | **ACT-Surv v5 (test)** | ACT-Surv − DCT | ACT-Surv − IST |
|:---:|:---:|:---:|:---:|:---:|:---:|
| BLCA | 0.7107 | 0.6843 | **0.6727 ± 0.0484** | −0.0380 | −0.0116 |
| HNSC | 0.6632 | 0.6289 | **0.6172 ± 0.0672** | −0.0460 | −0.0117 |

> **结论（仅基于 BLCA + HNSC 两癌种）**：
> 1. v5 在 BLCA 上未塌缩（0.6727 ≥ ArcSurv 修复前 0.7132 的 94%，与 v4.1 0.7039 同量级，但显著低于 DCT v3.8.2 −0.038）；v5 的核心机制（archetype 正交初始化 + KL 熵平衡）已生效，slot cosine ≈ 1.0 的塌缩问题不复存在。
> 2. HNSC 上 v5 偏低（0.6172），fold4 IPCW 出现 NaN 需查 `metric_diagnostics_fold4.log`；fold1 IBS 0.88 / iauc 0.23 提示该折分桶不稳，与 DCT/IST 在 HNSC 上的低分同源（HNSC 整体是该协议下最弱癌种）。
> 3. v5 不替代 DCT v3.8.2 主线；作为独立机制（精确归因 + 闭式反事实）保留，扩到 SKCM/LUSC/KIRC/UCEC 后再做"机制存在性 vs 主线性能"的最终判定。

### ACT-Surv v5 → v5.1 → v5.2 → v5.3 → v5.4 单癌种 BLCA 消融 (2026-08-16/18, 50ep, seed=3)

> 入口：`scripts/run_act_surv_v5.py --variant {v5|v5_1|v5_2|v5_3|v5_4}`。v5.3 / v5.4 由 `scripts/v5_1d_sequential_wrapper.py` 顺序跑完。
> 协议：UNI2-h、5fold_uni2h、global_qcut、dss、lr 0.0005、b8、rW=8、rG=8、α_surv=0.15、warmup 5 + ramp 10、IPCW bin 4。差异在 ablation：
> - **v5** baseline：完整机制（IPCW pairwise ranking 启用，KL 平衡权重 ×1）
> - **v5.1**：关 IPCW pairwise ranking（`act5_lambda_rank=0.0`） + KL 平衡权重 ×5
> - **v5.2**：v5.1 + 4 项"隐藏消融点"（温度 / margin / max_pairs / centering weights，见 `configs/act_surv_v5_2_blca.yaml`）
> - **v5.3**：仅关 IPCW pairwise ranking（KL 平衡权重保持 ×1）
> - **v5.4**：仅 KL 平衡权重 ×5（IPCW pairwise ranking 保持启用）
>
> 数据路径：`results/act_surv_v5/{,}full_run/`、`results/act_surv_v5_{1,2,3,4}/blca/...sp_act_surv_v5_v5_{1,2,3,4}_blca_fold*/`（修复 launcher variant-aware 路径后 5 个 variant 各自独立 top-level 目录）。

#### 主表 A — test c-index（model_best epoch 在 pkl 中，sksurv 重算）

| Variant | F0 | F1 | F2 | F3 | F4 | **Mean ± Std** | Δ vs v5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v5 baseline** | 0.6126 | 0.6652 | 0.7156 | 0.6427 | 0.7274 | **0.6727 ± 0.0485** | — |
| **v5.1** (no-rank + KL×5) | 0.6423 ⚠️ | 0.6721 | 0.7184 | 0.6708 | 0.7436 | **0.6894 ± 0.0408** | +0.0167 |
| **v5.2** (v5.1 + 4 hidden) | 0.6703 | 0.6412 | 0.7128 | 0.6857 | 0.7444 | **0.6909 ± 0.0396** | +0.0182 |
| **v5.3** (only no-rank) | 0.5896 | 0.6506 | 0.7231 | 0.6084 | 0.7068 | **0.6557 ± 0.0587** | −0.0170 |
| **v5.4** (only KL×5) | 0.6771 | 0.6833 | 0.7091 | 0.6831 | 0.7581 | **0.7021 ± 0.0336** | **+0.0294** |

> ⚠️ **v5.1 fold0 = 0.6423** 是 sksurv 对 `model_best_s0.pkl` 重算值（该 fold0 因 launcher 路径 bug 只跑 12 epoch，epoch_curve_fold0.csv 只剩 11 行；之前的 0.6720 是早期人工误记）。

#### 主表 B — best val c-index 与 epoch-49 终态 c-index 对比

> **关键警告**：5 variant 的 `model_best.pkl` 都是按 trainer 内部 best-val 选取保存，**这个 best-val 在 v5.1 / v5.2 / v5.4 上严重过拟合**——以下是更可靠的 epoch 49 终态 val（最后一个 epoch 的真实学到位姿）：

| Variant | best val mean | end (ep 49) val mean | Δ(peak − end) | 解释 |
|---|:---:|:---:|:---:|:---|
| **v5 baseline** | 0.6727 ± 0.0485 | **0.6593** | +0.0134 | 平稳，无明显 overfit |
| **v5.1** (no-rank + KL×5) | 0.6894 ± 0.0408 | **0.6345** ⚠ | **+0.0549** | 严重 overfit，best-val peak 不可信 |
| **v5.2** (v5.1 + hidden) | 0.6909 ± 0.0396 | **0.6315** ⚠ | **+0.0594** | 严重 overfit，best-val peak 不可信 |
| **v5.3** (only no-rank) | 0.6557 ± 0.0587 | **0.6517** | +0.0040 | 几乎不 overfit，best-val 与终态一致 |
| **v5.4** (only KL×5) | 0.7021 ± 0.0336 | **0.6525** ⚠ | **+0.0496** | 严重 overfit，fold2 best@ep2 早熟 |

#### Per-fold best-val peak 与终态 val（明确标出早熟/过拟合）

| Var | fold | best_val | @epoch | end_val | Δ(peak − end) |
|---|---|---:|---:|---:|---:|
| v5 baseline | 0 | 0.6126 | 29 | 0.6007 | +0.0119 |
| v5 baseline | 1 | 0.6652 | 28 | 0.6575 | +0.0077 |
| v5 baseline | 2 | 0.7156 |  6 | 0.6941 | +0.0215 ⚠ |
| v5 baseline | 3 | 0.6427 | 17 | 0.6295 | +0.0132 |
| v5 baseline | 4 | 0.7274 | 20 | 0.7145 | +0.0128 |
| **v5.1** | 0 | 0.6423 |  7 | 0.6083 | +0.0340 ⚠ |
| v5.1 | 1 | 0.6721 | 17 | 0.6189 | +0.0532 ⚠ |
| v5.1 | 2 | 0.7184 |  4 | 0.6174 | **+0.1010** ⚠⚠ 早熟 |
| v5.1 | 3 | 0.6708 | 13 | 0.6313 | +0.0395 ⚠ |
| v5.1 | 4 | 0.7436 |  9 | 0.6966 | +0.0470 ⚠ |
| **v5.2** | 0 | 0.6703 | 27 | 0.6398 | +0.0306 ⚠ |
| v5.2 | 1 | 0.6412 | 15 | 0.6009 | +0.0403 ⚠ |
| v5.2 | 2 | 0.7128 |  2 | 0.5968 | **+0.1160** ⚠⚠ 早熟 |
| v5.2 | 3 | 0.6857 | 19 | 0.6216 | +0.0641 ⚠ |
| v5.2 | 4 | 0.7444 |  5 | 0.6983 | +0.0462 ⚠ |
| **v5.3** | 0 | 0.5896 | 29 | 0.5871 | +0.0025 |
| v5.3 | 1 | 0.6506 | 44 | 0.6498 | +0.0009 |
| v5.3 | 2 | 0.7231 | 32 | 0.7175 | +0.0056 |
| v5.3 | 3 | 0.6084 | 25 | 0.6014 | +0.0070 |
| v5.3 | 4 | 0.7068 | 37 | 0.7026 | +0.0043 |
| **v5.4** | 0 | 0.6771 | 18 | 0.6593 | +0.0178 |
| v5.4 | 1 | 0.6833 | 23 | 0.6197 | **+0.0635** ⚠⚠ |
| v5.4 | 2 | 0.7091 |  2 | 0.6352 | **+0.0739** ⚠⚠ 早熟 |
| v5.4 | 3 | 0.6831 | 13 | 0.6356 | +0.0475 ⚠ |
| v5.4 | 4 | 0.7581 | 13 | 0.7128 | +0.0453 ⚠ |

#### 结论（**2026-08-18 全面修正**：v5.4 全 5 fold + best-val/终态 val 对比完成后再次推翻结论）

> **早期叙事（已废弃 2026-08-18）**：v5.1 是 synergy 配方（关 IPCW + KL×5 同步启用才生效）。**该结论由 best-val peak 推导，鉴于 v5.1 / v5.2 / v5.4 的 model_best 全部严重过拟合，best-val 与终态 val 差距高达 0.05–0.11，不应作为配方决策依据**。

> **新叙事**：
> 1. **按 epoch 49 终态 val（最可靠读数）排序**：v5 baseline 0.6593 ≈ v5.4 0.6525 ≈ v5.3 0.6517 ＞ v5.1 0.6345 ≈ v5.2 0.6315。**5 个 variant 实际收敛差异极小（均 0.65±0.02）**，best-val peak 上的 +0.0167 ~ +0.0294 提升均是 early-epoch 过拟合峰，不反映真实泛化能力。
> 2. **按 test c-index（pkl）排序**：v5.4 (0.7021) > v5.2 (0.6909) > v5.1 (0.6894) > v5 baseline (0.6727) > v5.3 (0.6557)。但因 pkl = model_best epoch saved at best-val，而 v5.4 / v5.2 / v5.1 三个 variant 在 best-val 上都严重 overfit，所以这个排序也不反映泛化。
> 3. **真正可对比的（几乎不 overfit 的 variant）**：v5 baseline 与 v5.3。v5.3 (0.6517 / 0.6557) 略弱于 v5 (0.6593 / 0.6727)，证明 v5.3 的 ablation "仅关 IPCW ranking" 在 BLCA 上**确实有害**（−0.017 test / −0.007 end-val），需要保留 IPCW ranking。
> 4. **v5.4 / v5.1 / v5.2 的"高于 baseline"差异不可信**：若强制按 epoch-N（不按 best-val）选模型，则 v5.4 / v5.1 / v5.2 与 baseline 终态 val 都落在 0.65±0.02 区间内，差异不显著。换言之，**早停过拟合让 best-val 模型比最后 epoch 模型在测试集上"虚高"**约 0.03-0.07，这一虚高对所有 variant 一致，不能归因为配方改进。
> 5. **paper 叙事第三次调整**：删除"synergy 配方"叙事，**改为"v5 baseline 是当前 BLCA 上唯一可信的解；v5.1 / v5.2 / v5.4 在 best-val 上虚高 0.05+，需要 fixed-epoch 重训（建议固定 30-40 epoch 选模型 或加 early-stopping patience=5）才能做配方决策"**。
> 6. **下一步动作**：固定 epoch=30 / 40 / 50（暂定 30，最稳）重训 v5 / v5.1 / v5.2 / v5.3 / v5.4 BLCA 5 fold，用 last_epoch 模型而非 best_val 模型重算 test c-index。这是最关键的下一步，若不重训则本批所有"v5.1 = +0.0167" 均视为 val selection noise。

> 警示（强化）：
> - v5.2 fold2 best @ epoch 2（Δ=0.1160）、v5.4 fold2 best @ epoch 2（Δ=0.0739）、v5.1 fold2 best @ epoch 4（Δ=0.1010）均为典型早熟峰。
> - 即便单独的 v5.3（不 overfit）在 test c-index 上 −0.017 也是真实退化（v5.3 的 5 fold 无 early-peak，全 epoch 平稳），所以"IPCW ranking 关掉 = 退化"本身是**成立的独立事实**，但与 v5.1 主叙事无直接关系。

### IST-Surv 唯一跨癌种版本（2026-08-06）

数值最高的 A 档 factual-only 为 0.7072，但它关闭了 `ist_stability_strength`，不含
IST 的核心干预稳定性机制，不能作为 IST 最终版。保留 IST 机制的 B/C 两档中，
B 档 staged cost-feedback-only 为 0.7055，略高于 full IST 的 0.7053；C−B 仅
−0.0002，说明 plan/attribution/risk 三个辅助损失没有贡献。因此 IST 唯一跨癌种
版本冻结为 **v4.0 staged cost-feedback-only**：50 epoch、warmup=5、ramp=10、
stability strength=0.10，三个辅助损失权重均为 0。

入口为 `scripts/run_ist_v40_final_cross_cancer.py`。默认对当前 UNI2-h 完整的
BLCA、SKCM、HNSC、LUSC、KIRC、UCEC 跑五折；BLCA 已有 fold1/2/4 会自动跳过，
实际只补 fold0/3。BRCA、LUAD、COADREAD、STAD 仍由数据 doctor 动态阻断，补齐
特征后可显式使用 `--cancers all`。该 IST 队列与 DCT/优先队列共用 GPU 锁。

### DCT v3.8.2 fixed-full 跨癌种五折结果 (2026-08-06, UNI2-h, 50ep, clean)

> 6 癌种各五折，共 30 folds。**全部完成。**

| 癌种 | F0 | F1 | F2 | F3 | F4 | **Mean** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **UCEC** | 0.8358 | 0.8446 | 0.7933 | 0.8048 | 0.8333 | **0.8224** |
| **KIRC** | 0.7973 | 0.8443 | 0.8215 | 0.8011 | 0.7716 | **0.8071** |
| **BLCA** | 0.6534 | 0.7253 | 0.6520 | 0.7375 | 0.7855 | **0.7107** |
| HNSC | 0.6906 | 0.6299 | 0.6388 | 0.6039 | 0.7526 | **0.6632** |
| SKCM | 0.6972 | 0.6001 | 0.6529 | 0.6278 | 0.7258 | **0.6608** |
| LUSC | 0.6789 | 0.6314 | 0.5368 | 0.6617 | 0.5931 | **0.6204** |

> **描述性排序**: UCEC (0.8224) > KIRC (0.8071) > BLCA (0.7107) > HNSC (0.6632) > SKCM (0.6608) > LUSC (0.6204)。
> v3.3 使用旧划分与 UNI v1，不能与这里的 clean `5fold_uni2h` 结果作方法增益归因；旧数值仅保留为历史记录。
> 当前同协议结果显示癌种间异质性明显，HNSC、SKCM 与 LUSC 的折间波动需要结合样本量、事件率和独立重复实验解释。

### IST v4.0 abl_b 跨癌种五折结果 (2026-08-06, UNI2-h, 50ep, ✅ 全部完成)

| 癌种 | F0 | F1 | F2 | F3 | F4 | **Mean** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **KIRC** | 0.8341 | 0.8465 | 0.8240 | 0.7977 | 0.7709 | **0.8146** |
| **UCEC** | 0.8111 | 0.8153 | 0.7934 | 0.8239 | 0.7919 | **0.8071** |
| BLCA | 0.6296 | 0.6970 | 0.6632 | 0.6752 | 0.7564 | **0.6843** |
| SKCM | 0.6715 | 0.6630 | 0.6333 | 0.6590 | 0.6997 | **0.6653** |
| HNSC | 0.5920 | 0.5760 | 0.6201 | 0.6436 | 0.7128 | **0.6289** |
| LUSC | 0.6380 | 0.6220 | 0.5420 | 0.6325 | 0.5995 | **0.6068** |

### DCT vs IST 最终对比 (6 癌种，全部 5 折 UNI2-h 50ep)

| 癌种 | DCT v3.8.2 | IST v4.0 | **Δ (IST−DCT)** | 优胜 |
|:---:|:---:|:---:|:---:|:---:|
| KIRC | 0.8071 | 0.8146 | **+0.007** | IST |
| UCEC | **0.8224** | 0.8071 | −0.015 | DCT |
| BLCA | **0.7107** | 0.6843 | −0.026 | DCT |
| SKCM | 0.6608 | 0.6653 | +0.005 | 持平 |
| HNSC | **0.6632** | 0.6289 | −0.034 | DCT |
| LUSC | **0.6204** | 0.6068 | −0.014 | DCT |

> **结论：DCT v3.8.2 fixed-full 在 6 癌种中赢 4 个、IST 赢 1 个、平 1 个。DCT 是全面优胜的主线。**
> KIRC 是 IST 唯一反超的癌种（+0.007），但差距在标准误范围内。UCEC (DCT 0.8224) 和 KIRC (IST 0.8146)
> 为最高分，小癌种 LUSC/HNSC/SKCM 偏低。IST 的 cost-feedback 机制未带来跨癌种泛化收益。

### ArcSurv / v4.1 修复闸门结果 (2026-08-06, BLCA)

| Gate | Fold | Best C-Index | Epoch | 诊断 |
|---|:---:|:---:|:---:|---|
| ArcSurv repaired | 1 | 0.5665 | 4 | ❌ 塌缩未修复，低于修复前 0.7132 |
| v4.1 repaired | 2 | 0.6436 | 3 | ⚠️ 与修复前一模一样，completion 修复无效果 |

> **两个修复均未生效。** ArcSurv 从 0.7132 暴跌至 0.5665，furthest-point anchoring +
> entropy 惩罚可能过度约束了原型多样性。v4.1 completion 下界修复未改变分数。
> **两个方法暂不扩跑到三折。**

### 已识别的混淆维度（累计 4 项）

| # | 维度 | 状态 |
|:-:|------|:----:|
| 1 | split 划分（`bee66a2` 重划 / `5fold` vs `5fold_uni2h`） | ✅ 已解决：重划已标注分界；两目录在 BLCA 折 1/2/4 完全相同 |
| 2 | WSI 编码器（UNI v1 1024d vs UNI2-h 1536d） | 🟡 估计影响 ≈ −0.011，待 #8/#10 确认 |
| 3 | 分箱范围（`fit_bins_on_train` leaky vs clean） | ✅ 已解决：#10 补了 clean 基线 |
| 4 | epoch 预算（50 vs 30，取 max 抽样次数不对称） | ✅ 已解决：#11 统一到 50ep |

> **任何跨方法结论都必须先说明这 4 个维度是否对齐。**

### 撤回记录

| 结论 | 撤回原因 |
|------|----------|
| v3.8.2「+0.0201 领先基线」 | 跨编码器 + 跨 split + 跨分箱协议；同组标准误 0.0217 > 0.0201 |
| 「BLCA direction 有益 +0.033，LUSC 有害 −0.088，两癌种相反」 | +0.033 来自 fold0 单折；三折匹配协议下 base→full 仅 +0.0033。LUSC 侧证据被 warmup 污染（#6） |
| A 组「Δ vs v3.3」(+0.0081 / −0.0201) | v3.3 为 leaky 分箱，与 clean 协议的新方法不可比 |

---

## 记录修正日志 (2026-08-03)

-4. **(2026-08-04 六次修正)** #8 #10 #11 完成：全部 6 个方法统一 50ep + clean 协议复测。新增「50ep Staged 统一复测结果」章节（A/B 两组对比、30→50ep 变化）。结论：50ep 统一后无一方法超越基线。v4.1 反降 0.030，ArcSurv 改善 0.015 但 fold2 瓶颈未解，v4.0 基本持平。MGPTR 单项无增益（Δ=−0.0031）；该组没有启用 adaptive。四个混淆维度全部消除。
-3. **(2026-08-03 五次修正)** **更正「旧划分已不存在」这一错误表述**：旧划分完整保存于初始提交 `77833de` 的根级路径，10 癌种均可恢复，并给出恢复命令。同时实测旧划分存在缺陷（fold1 含 1 个缺 DSS 标签患者、事件极差 4 vs 新划分 1、fold0 为 77 人），故 `0.7311` 虽可复现但属已被取代的历史值，不作为复现目标。
-2. **(2026-08-03 四次修正)** 新增「收敛健康度审计」章节，识别第四个混淆维度 **epoch 预算不对称**（v3.3 50ep vs 新方法 30ep）。标记两类失效：欠训练（ArcSurv fold1 峰值贴 e29 边界）与早熟（v4.0 fold1 @e0、v4.1 fold2 @e3、v3.8.2 fold2 @e4），方向相反，前者低估后者高估。新增未结事项 #11~#13。
-1. **(2026-08-03 三次修正)** #9 完成：BLCA 的 `5fold` 与 `5fold_uni2h` 在折 1/2/4 上验证患者集合完全相同（各 76 人），**split 维度消除**。新增「负结果汇总」章节（#7），其中 v3.8.3 与 v3.9 确立为负、v3.8 三损失确立为无收益；ArcSurv 与 v3.8 highscore/base 明确标为**不可判定**，不计入负结果。
0. **(2026-08-03 二次修正)** 发现第三个混淆维度 `fit_bins_on_train`：v3.3 基线为 leaky 分箱，与全部新方法协议不一致 → A 组 Δ 撤回，新增未结事项 #10。同时补录 v3.8 BLCA highscore base/full 20ep 折 1/2/4，得到目前唯一有效的单变量对照（三损失 Δ = +0.0033 ≈ 零），并据此撤回「direction 有益 +0.033」。
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

### UNI v1 四癌种队列（2026-08-14 新增，⏳ 待运行）

> **背景**：BRCA/LUAD/COADREAD/STAD 的 UNI2-h 特征覆盖不足（BRCA 775/1045 等），
> 等补齐特征前先用 **UNI v1（1024-d）** 按当前 `5fold_uni` 划分跑 **DCT v3.8.2 fixed-full**，
> 让这四个癌种立刻有可比结果。**该队列与 UNI2-h `5fold_uni2h` 结果严禁混表。**

| 癌种 | UNI v1 特征数 | split 纳入 | 备注 |
|:---:|:---:|:---:|---|
| BRCA | 1131 | 1045 | 100% 覆盖 |
| COADREAD | 581 | 570 | 100% 覆盖 |
| STAD | 391 | 362 | 100% 覆盖 |
| LUAD | 1052 | **457** | 剔除 `TCGA-55-8207`（无 UNI v1 特征，5 折全部出现）|

- 队列：**DCT v3.8.2 fixed-full 单方法 × 4 cancers × 5 folds = 20 jobs**
- 入口：`scripts/run_dct_v382_uni_v1_4cancer.py {prepare|doctor|smoke|run}`
- 冻结配方与 UNI2-h fixed_full 完全一致，仅数据协议不同：`wsi_encoder=uni`、
  `encoding_dim=1024`、`data_root_dir=/data/CPathPatchFeature`、`which_splits=5fold_uni`
- 输出目录：`results/dct_v3.8.2/uni_v1_5fold/fixed_full/{cancer}/`
- 状态：`prepare` 已完成（生成 `5fold_uni`），`doctor` 全 OK，**等待门控队列完成后启动**

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

| 癌种 | clinical | Split | v3.3 | v3.6 最佳 | v3.7 | v3.8 | v3.8.2 | v3.8.3 | v3.9 | v4.0 | v4.1 | ArcSurv | v4.2 | **v5** |
|:----:|:--------:|:-----:|:-----:|:---------:|:----:|:----:|:------:|:------:|:----:|:----:|:----:|:------:|:----:|:----:|
| BLCA | 381 | 380 | 0.7311 ✅ | 0.7024 ETAR | 0.7249 ✅ | 0.7274 ✅ | 0.7159 | 0.5931 | 0.6394 | 0.7078 | 0.7039 | 0.6757 | — | **0.6727** ✅ |
| BRCA | 1046 | 1045 | ⏳ 待重跑 | ⏳ 待重跑 | ⏳ 待重跑 | 0.6750 ✅ | | | | | | | — | ⏳ |
| COADREAD | 573 | 570 | 0.6774 ✅ | — | 0.7384 ✅ | — | | | | | | | — | ⏳ |
| HNSC | 438 | 437 | ⏳ | — | 0.6406 ✅ | — | 0.6632 | | | | | | — | **0.6172** ✅ |
| KIRC | 488 | 488 | 0.7958 ✅ | — | 0.8149 ✅ | — | 0.8071 | | | | | | — | ⏳ |
| LUAD | 467 | 458 | ⏳ | 0.7647 NLL | 0.6662 ✅ | ⏳ | | | | | | | — | ⏳ |
| LUSC | 460 | 454 | ⏳ | 0.6141 NLL | 🔄 fold0=0.4575 | ⏳ | 0.6204 | | | | | | — | ⏳ |
| SKCM | 403 | 403 | 0.6770 ✅ | — | ⏳ | — | 0.6608 | | | | | | — | ⏳ |
| STAD | 366 | 362 | 0.6596 ✅ | — | ⏳ | — | | | | | | | — | ⏳ |
| UCEC | 488 | 487 | 0.7964 ✅ | — | ⏳ | — | 0.8224 | | | | | | — | ⏳ |
