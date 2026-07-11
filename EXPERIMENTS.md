# SurvOT-Rank 实验汇总 (2026-07-11)

所有实验使用 **OTEHV2RankEventV2** 架构，5-fold 交叉验证，最大 30 epochs，batch_size=4，lr=5e-4，seed=3。
指标为 **val C-index** (per-fold best epoch 的 mean ± std)，越高越好。

---

## 一、V45 基线实验 (BLCA / BRCA / STAD / COADREAD / HNSC)

| 实验 | 癌种 | 5-fold Mean C-index | 状态 |
|------|------|---------------------|------|
| v45_blca | BLCA | **0.6872** ± 0.0528 | ✅ |
| v45_brca | BRCA | **0.6548** ± 0.0570 | ✅ |
| v45_best_blca | BLCA (best params) | **0.6996** ± 0.0252 | ✅ |
| v45_stad | STAD | — | ❌ 缺少 5-fold split |
| v45_coadread | COADREAD | — | ❌ 缺少 5-fold split |
| v45_hnsc | HNSC | — | ❌ 缺少 5-fold split |

**备注**: stad/coadread/hnsc 需要先用 `tools/gen_splits_5fold.py` 生成分片后才能重跑。

---

## 二、V45v2 + Clinical 实验 (OTEHV2RankEventV2 + 临床特征)

| 实验 | 癌种 | 临床特征 | 5-fold Mean C-index | 状态 |
|------|------|---------|---------------------|------|
| v45v2_blca_clinical | BLCA | age + gender | **0.6919** ± 0.0499 | ✅ |
| v45v2_brca_clinical | BRCA | age + gender | **0.6623** | ✅ |
| v45v2_luad_clinical | LUAD | age + gender | **0.6651** | ✅ |

---

## 三、OTEHV2RankEventV2 系统消融实验 (BLCA)

16 个消融实验依次叠加新能力，测试各组件的贡献。batch run 进行中 (abl_05 之后将持续运行至 abl_09)。

| # | 实验 | 说明 | 5-fold Mean C-index | Per-fold Best | 状态 |
|---|------|------|---------------------|---------------|------|
| 1 | abl_00_baseline | 纯 V45 基线 | **0.6872** ± 0.0528 | 0.74/0.75/0.61/0.67/0.66 | ✅ |
| 2 | abl_01_clinical | + 临床特征 (age) | — | — | ❌ KeyError (列名不匹配) |
| 3 | abl_02_unified | + 统一 slot 空间 | **0.6872** ± 0.0528 | 0.74/0.75/0.61/0.67/0.66 | ✅ |
| 4 | abl_03_disentangle | + 解耦编码器 | **0.6859** ± 0.0479 | 0.76/0.71/0.64/0.69/0.63 | ✅ |
| 5 | abl_04_sinkhorn | + Sinkhorn OT | **0.6437** | 0.71/0.71/0.56/0.67/0.56 | ✅ |
| 6 | abl_05_crossmodal | + 跨模态 attention | 🔄 运行中 | — | 🔄 |
| 7 | abl_06_adaptive_iters | + 自适应迭代次数 | ⏳ | — | ⏳ |
| 8 | abl_07_learnable_weights | + 可学习 loss 权重 | ⏳ | — | ⏳ |
| 9 | abl_08_all_on | 全部开启 (固定权重) | ⏳ | — | ⏳ |
| 10 | abl_09_all_on_learnable | 全部开启 (可学习权重) | ⏳ | — | ⏳ |

**关键发现**:
- **abl_00_baseline** (纯 V45) 与 **abl_02_unified** (统一 slot 空间) 结果完全相同 (0.6872)，统一 slot 空间未带来提升。
- **abl_03_disentangle** 略降 (0.6859)，解耦编码器有轻微负面影响。
- **abl_04_sinkhorn** 明显下降 (0.6437)，Sinkhorn 模块拖累了性能。
- **v45_best_blca** 目前是最佳单模型 (0.6996)，比基线 +0.0124。

---

## 四、失败实验及修复方案

| 实验 | 错误 | 修复方法 |
|------|------|---------|
| v45_stad/coadread/hnsc | `assert os.path.isdir(args.split_dir)` — 缺少分片目录 | 运行 `python tools/gen_splits_5fold.py --study stad` 等生成 CSV |
| abl_01_clinical | `KeyError: age_at_diagnosis, pathologic_stage, histological_grade not in index` | 修改配置中的 `clinical_feature_cols` 或数据列名映射 |

---

## 五、Run 信息

- **Batch run 目录**: `results/batch_runs/20260710_195046/`
- **启动时间**: 2026-07-10 19:50 UTC
- **脚本**: `bash scripts/run_batch.sh --start-from v45_stad`
- **Python 环境**: `/home/ubuntu/.conda/envs/trisurv/bin/python`
- **监控脚本**: `python monitor_batch.py`

---

*自动生成于 2026-07-11. 更新于 batch run 进行中。*
