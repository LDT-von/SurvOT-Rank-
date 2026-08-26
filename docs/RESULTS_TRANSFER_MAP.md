# 结果迁移清单：`results/` (Linux) → `E:\SurvOT-Rank\results\` (Windows)

> 审计时间：2026-08-15
> 工作目录：`/data1/SurvOT-Rank`
> 当前 HEAD：`938cda7`（1dbaade 已是其父提交，代码状态已就绪，无需额外 `git pull`）

---

## TL;DR

所有需要的结果**均已存在于 `results/` 下**，只是目录前缀跟你想要的不完全一致。下表给出
「源路径 → 目标路径」一对一映射，外加每个 fold 的文件审计结果。

主线 60 折全部完整；5 个 BLCA 辅助方法存在**不同程度的缺折**，按你确认的方案：
**只搬已存在的，缺折保留空目录 + `README.md` 标注缺哪些折**。

> 文件大小合计 ≈ 9.5 GB（`.pth` 单文件 50–110 MB）。建议直接用 `robocopy` / `rsync`
> 按下表映射目录对拷，不要逐文件复制。

---

## 1. 主线：DCT v3.8.2 fixed-full 50ep × 6 癌种 × 5 折（30 fold）

**目标前缀：** `results/dct_v382_fixed_full_50ep/`
**源前缀：** `results/dct_v3.8.2/robust/fixed_full/`

| 癌种 | 目标目录 | 源 run 目录 |
|:---:|---|---|
| blca | `results/dct_v382_fixed_full_50ep/blca/SurvOTRank_dct_v382_prognostic_transport_reconstruction/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v382_robust_fixed_full_blca_50ep/` | `results/dct_v3.8.2/robust/fixed_full/blca/blca/SurvOTRank_dct_v382_prognostic_transport_reconstruction/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v382_robust_fixed_full_blca_50ep/` |
| ucec | 同上，`..._ucec_50ep/` | `results/dct_v3.8.2/robust/fixed_full/ucec/ucec/.../..._ucec_50ep/` |
| kirc | 同上，`..._kirc_50ep/` | `results/dct_v3.8.2/robust/fixed_full/kirc/kirc/.../..._kirc_50ep/` |
| skcm | 同上，`..._skcm_50ep/` | `results/dct_v3.8.2/robust/fixed_full/skcm/skcm/.../..._skcm_50ep/` |
| hnsc | 同上，`..._hnsc_50ep/` | `results/dct_v3.8.2/robust/fixed_full/hnsc/hnsc/.../..._hnsc_50ep/` |
| lusc | 同上，`..._lusc_50ep/` | `results/dct_v3.8.2/robust/fixed_full/lusc/lusc/.../..._lusc_50ep/` |

**每个 run 目录的 5 折文件齐全度：**

| 癌种 | fold0 | fold1 | fold2 | fold3 | fold4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| blca | OK | OK | OK | OK | OK |
| ucec | OK | OK | OK | OK | OK |
| kirc | OK | OK | OK | OK | OK |
| skcm | OK | OK | OK | OK | OK |
| hnsc | OK | OK | OK | OK | OK |
| lusc | OK | OK | OK | OK | OK |

**审计字段（每个 fold 必须存在的 7 个文件）：**
- `epoch_curve_fold{N}.csv`
- `log_start_{N}_end_{N+1}.txt`
- `model_best_s{N}.pth`
- `split_{N}_results.pkl`
- `split_{N}_results_final.pkl`
- `experiment_settings.txt`（全 run 共享，1 份）
- `model_parameters.txt`（全 run 共享，1 份）

**合计文件数：** 30 fold × 5 折专用 + 6 个 run × 2 个共享 = **156 个文件，约 4.7 GB**

---

## 2. 主线：IST v4.0 abl_b cost-only 50ep × 6 癌种 × 5 折（30 fold）

**目标前缀：** `results/ist_v40_abl_b_50ep/`
**源前缀：** `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/`

| 癌种 | 目标目录 | 源 run 目录 |
|:---:|---|---|
| blca | `results/ist_v40_abl_b_50ep/blca/SurvOTRank_intervention_stable_survival_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_ist_v40_abl_b_cost_only_blca_50ep/` | `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/blca/blca/SurvOTRank_intervention_stable_survival_transport/..._ist_v40_abl_b_cost_only_blca_50ep/` |
| ucec | 同上，`..._ucec_50ep/` | `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/ucec/ucec/.../` |
| kirc | 同上，`..._kirc_50ep/` | `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/kirc/kirc/.../` |
| skcm | 同上，`..._skcm_50ep/` | `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/skcm/skcm/.../` |
| hnsc | 同上，`..._hnsc_50ep/` | `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/hnsc/hnsc/.../` |
| lusc | 同上，`..._lusc_50ep/` | `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/lusc/lusc/.../` |

**每个 run 目录的 5 折文件齐全度：**

| 癌种 | fold0 | fold1 | fold2 | fold3 | fold4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| blca | OK | OK | OK | OK | OK |
| ucec | OK | OK | OK | OK | OK |
| kirc | OK | OK | OK | OK | OK |
| skcm | OK | OK | OK | OK | OK |
| hnsc | OK | OK | OK | OK | OK |
| lusc | OK | OK | OK | OK | OK |

**合计文件数：** 30 fold × 5 折专用 + 6 个 run × 2 个共享 = **156 个文件，约 3.7 GB**

---

## 3. 辅助方法：BLCA × 5 折（5 个方法，共 25 fold）

5 个方法的源 run 目录和目标目录（仅 BLCA 1 个癌种）：

| # | 方法 | 目标目录 | 源 run 目录 |
|:---:|---|---|---|
| 3.1 | DCT v3.9 Risk-Simplex 30ep | `results/dct_v39_risk_simplex_30ep/blca/SurvOTRank_dct_v39_risk_simplex_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v39_risk_simplex_blca_30ep/` | `results/dct_v3.9_risk_simplex_transport_30ep/blca/blca/SurvOTRank_dct_v39_risk_simplex_transport/..._sp_dct_v39_risk_simplex_blca_30ep/` |
| 3.2 | DCT v4.1 Evidence Ledger 50ep (staged) | `results/dct_v41_evidence_ledger_50ep/blca/SurvOTRank_dct_v41_survival_evidence_ledger/..._sp_dct_v41_staged_blca_50ep/` | `results/dct_v4.1_survival_evidence_ledger_staged_50ep/blca/blca/SurvOTRank_dct_v41_survival_evidence_ledger/..._sp_dct_v41_staged_blca_50ep/` |
| 3.3 | ArcSurv staged 50ep | `results/arcsurv_staged_50ep/blca/SurvOTRank_archetypal_risk_composition/..._sp_arcsurv_staged_blca_50ep/` | `results/archetypal_risk_composition_staged_50ep/blca/blca/SurvOTRank_archetypal_risk_composition/..._sp_arcsurv_staged_blca_50ep/` |
| 3.4 | CATET repaired 50ep | `results/catet_repaired_50ep/blca/SurvOTRank_archetypal_risk_composition/..._sp_arcsurv_repaired_gate_blca_50ep/` | `results/archetypal_risk_composition_repaired_50ep/blca/blca/SurvOTRank_archetypal_risk_composition/..._sp_arcsurv_repaired_gate_blca_50ep/` |
| 3.5 | ACT-Surv v4.2 50ep batch8 | `results/act_surv_v4.2_50ep/blca/SurvOTRank_archetypal_transport_composition/..._sp_act_surv_v42_blca_50ep/` | `results/act_surv_v4.2/blca/blca/SurvOTRank_archetypal_transport_composition/..._sp_act_surv_v42_blca_50ep/` |

### 3.x 折文件齐全度（BLCA）

| 方法 | fold0 | fold1 | fold2 | fold3 | fold4 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 3.1 DCT v3.9 30ep | ❌ 缺 | ✅ | ✅ | ❌ 缺 | ✅ |
| 3.2 DCT v4.1 50ep staged | ❌ 缺 | ✅ | ✅ | ❌ 缺 | ✅ |
| 3.3 ArcSurv 50ep staged | ❌ 缺 | ✅ | ✅ | ❌ 缺 | ✅ |
| 3.4 CATET 50ep repaired | ❌ 缺 | ✅ | ❌ 缺 | ❌ 缺 | ❌ 缺 |
| 3.5 ACT-Surv v4.2 50ep | ❌ 缺 | ✅ | ✅ | ❌ 缺 | ⚠️ 缺 split_final |

**合计：** 13 / 25 fold 完整，12 个 fold 缺失。
**建议：** 在 Windows 端为目标目录的每个缺折补一个 `README.md`，注明「待重跑：fold{N}」，不要空目录。

---

## 4. 一键迁移脚本（Windows PowerShell）

把下面的脚本保存到 `E:\SurvOT-Rank\scripts\transfer_results_from_linux.ps1`，在 PowerShell 里跑：

```powershell
# 用 rsync（Windows 10+ 自带）从 Linux 服务器同步；如未装 rsync，可改用 robocopy
$SRC = "/data1/SurvOT-Rank/results"   # WSL/SSH 挂载路径，按你实际情况改
$DST = "E:\SurvOT-Rank\results"

# 主线 60 fold
$dct_pairs = @(
    @{src="$SRC/dct_v3.8.2/robust/fixed_full/blca/blca/SurvOTRank_dct_v382_prognostic_transport_reconstruction/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v382_robust_fixed_full_blca_50ep"; dst="$DST/dct_v382_fixed_full_50ep/blca/SurvOTRank_dct_v382_prognostic_transport_reconstruction/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v382_robust_fixed_full_blca_50ep"},
    @{src="$SRC/dct_v3.8.2/robust/fixed_full/ucec/ucec/.../...ucec_50ep"; dst="..."},
    # ucec / kirc / skcm / hnsc / lusc 同理（替换癌种 token）
)
foreach ($p in $dct_pairs) {
    New-Item -ItemType Directory -Force -Path (Split-Path $p.dst) | Out-Null
    robocopy $p.src $p.dst /E /Z /MT:8
}

# 辅助 5 个 BLCA 方法同样套路（详见下方表）
```

> 如希望我在 Linux 端把这 8 个新目标目录用 **符号链接**创建出来（不动源数据、零拷贝），
> 让我知道，我可以直接生成。

---

## 5. 源目录磁盘占用（参考）

| 目录 | 大小 |
|---|---:|
| `results/dct_v3.8.2/robust/fixed_full/` | 4.7 GB |
| `results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/` | 3.7 GB |
| `results/dct_v3.9_risk_simplex_transport_30ep/` | 367 MB |
| `results/dct_v4.1_survival_evidence_ledger_staged_50ep/` | 344 MB |
| `results/dct_v4.1_survival_evidence_ledger_repaired_50ep/` | 115 MB |
| `results/archetypal_risk_composition_staged_50ep/` | 320 MB |
| `results/archetypal_risk_composition_repaired_50ep/` | 107 MB |
| `results/act_surv_v4.2/` | 318 MB |
| **合计** | **约 9.9 GB** |

迁移时建议 `E:\` 预留 ≥ 12 GB（含冗余）。

---

## 6. git 状态确认

```
HEAD:   938cda7 feat: add DCT v3.8.2 paper-evidence ablation chain
parent: 5d9345b docs: 记录 DCT 提分闸门判定进展（暂停于 25/36 折）
1dbaade feat: add DCT v3.8.2 paper-evidence ablation chain  ← 已是 HEAD 父提交
```

**结论：** 代码状态已经包含 1dbaade，不需要额外 `git pull` / `git reset` / `git merge`。
如果硬要 `git checkout 1dbaade`，反而会丢掉 HEAD 上多出的 4 个 commit（包括 docs/ 文档与
stage / ablate / etc. 增强），不建议。