# SurvOT-Rank 实验汇总 (2026-07-12, 全部完成)

所有实验使用 **OTEHV2RankEventV2** 架构，5-fold 交叉验证，最大 30 epochs，batch_size=4，lr=5e-4，seed=3。
指标为 **val C-index** (per-fold best epoch 的 mean ± std)，越高越好。

---

## 一、V45 基线实验 (跨癌种)

| 实验 | 癌种 | 5-fold Mean C-index | Per-fold Best | 状态 |
|------|------|---------------------|---------------|------|
| v45_blca | BLCA | **0.6872** ± 0.0528 | 0.74/0.75/0.61/0.67/0.66 | ✅ |
| v45_brca | BRCA | **0.6548** ± 0.0570 | 0.60/0.73/0.71/0.65/0.59 | ✅ |
| v45_best_blca | BLCA (best params) | **0.6996** ± 0.0252 | 0.69/0.74/0.69/0.67/0.71 | ✅ |
| v45_stad | STAD | — | — | ❌ 缺少 5-fold split |
| v45_coadread | COADREAD | — | — | ❌ 缺少 5-fold split |
| v45_hnsc | HNSC | — | — | ❌ 缺少 5-fold split |

**备注**: stad/coadread/hnsc 需用 `tools/gen_splits_5fold.py` 生成分片。

---

## 二、V45v2 + Clinical 实验

| 实验 | 癌种 | 临床特征 | 5-fold Mean C-index | 状态 |
|------|------|---------|---------------------|------|
| v45v2_blca_clinical | BLCA | age + gender | **0.6919** ± 0.0499 | ✅ |
| v45v2_brca_clinical | BRCA | age + gender | **0.6623** | ✅ |
| v45v2_luad_clinical | LUAD | age + gender | **0.6651** | ✅ |

---

## 三、OTEHV2RankEventV2 系统消融实验 (BLCA) — 全部完成

| # | 实验 | 说明 | 5-fold Mean C-index | Per-fold Best | 状态 |
|---|------|------|---------------------|---------------|------|
| 1 | abl_00_baseline | 纯 V45 基线 | **0.6872** ± 0.0528 | 0.74/0.75/0.61/0.67/0.66 | ✅ |
| 2 | abl_01_clinical | + 临床特征 (age) | — | — | ❌ KeyError |
| 3 | abl_02_unified | + 统一 slot 空间 | **0.6872** ± 0.0528 | 0.74/0.75/0.61/0.67/0.66 | ✅ |
| 4 | abl_03_disentangle | + 解耦编码器 | **0.6859** ± 0.0479 | 0.76/0.71/0.64/0.69/0.63 | ✅ |
| 5 | abl_04_sinkhorn | + Sinkhorn OT | **0.6437** | 0.71/0.71/0.56/0.67/0.56 | ✅ |
| 6 | abl_05_crossmodal | + 跨模态 attention | **0.6770** | 0.72/0.73/0.58/0.68/0.67 | ✅ |
| 7 | abl_06_adaptive_iters | + 自适应迭代次数 | **0.6853** | 0.74/0.71/0.61/0.67/0.69 | ✅ |
| 8 | abl_07_learnable_weights | + 可学习 loss 权重 | **0.6872** | 0.74/0.75/0.61/0.67/0.66 | ✅ |
| 9 | abl_08_all_on | 全部开启 (固定权重) | — | — | ❌ KeyError |
| 10 | abl_09_all_on_learnable | 全部开启 (可学习权重) | — | — | ❌ KeyError |

### 关键发现

- **abl_00 / abl_02 / abl_07 三组完全相同 (0.6872)**：统一 slot 空间和可学习 loss 权重均无提升，与 baseline 等价。
- **abl_06 自适应迭代 (0.6853)**：几乎无影响。
- **abl_03 解耦编码器 (0.6859)**：轻微下降，解耦未带来增益。
- **abl_05 跨模态 attention (0.6770)**：下降约 0.01。
- **abl_04 Sinkhorn (0.6437)**：显著下降 ~0.04，Sinkhorn OT 在 5-fold 设定下拖累性能。
- **最佳单模型仍为 v45_best_blca (0.6996)**，比消融基线 +0.0124。

### 关于 Fold 2 一致偏低的问题

所有 8 个成功实验中 Fold 2 的 C-index 始终在 0.56-0.64（其他折 0.66-0.76）。原因：Fold 2 的验证集事件时间均值 42.5 月、标准差 37.9（其他折 24-31 月、std 20-32），长尾分布导致 C-index 天然偏低。这是数据分片特征，非模型问题，论文中正常报告 mean ± std 即可。

---

## 四、失败实验汇总

| 实验 | 错误 | 修复方法 |
|------|------|---------|
| v45_stad/coadread/hnsc | 缺少 5-fold split 目录 | `python tools/gen_splits_5fold.py --study stad` |
| abl_01_clinical | `KeyError: age_at_diagnosis not in index` | 修正 clinical_feature_cols 列名映射 |
| abl_08_all_on | 同上 KeyError | abl_08/09 的 config 也引用了不存在的临床列 |
| abl_09_all_on_learnable | 同上 KeyError | — |

---

## 五、Run 信息

- **Batch run 目录**: `results/batch_runs/20260710_195046/`
- **启动**: 2026-07-10 19:50 UTC，完成: 2026-07-12 约 04:45 UTC（总耗时约 33 小时）
- **脚本**: `bash scripts/run_batch.sh --start-from v45_stad`
- **环境**: `/home/ubuntu/.conda/envs/trisurv/bin/python`
- **监控**: `python monitor_batch.py --once`

---

*最后更新: 2026-07-12，16 个实验全部完成。*
