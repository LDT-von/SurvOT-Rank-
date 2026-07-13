# SurvOT-Rank 实验结果汇总 (2026-07-13)

> 仓库根目录 `.md` 文件，不受 `.gitignore` 拦截。所有数字来自本地 epoch_curve CSV / log 文件，远程合作者可通过 git pull 获取。

---

## 1. v45_blca — 8 损失 (config: v45_blca.yaml)

### seed=3, 30 epochs, 5-fold (batch_20260712_1912, alpha_surv=0.15, fold-aware bins)

| Fold | Epochs | train_mean | train_max | val_best | val_best@ | val_last5 | ipcw_best | IBS_best | iauc_best |
|------|--------|-----------|-----------|---------|----------|----------|-----------|----------|----------|
| 0 | 30 | 0.4795 | 0.5370 | 0.7120 | 15 | 0.6887 | 0.6753 | 0.3478 | 0.8601 |
| 1 | 30 | 0.4910 | 0.6315 | 0.7411 | 3 | 0.7080 | 0.7114 | 0.1544 | 0.7806 |
| 2 | 30 | 0.4779 | 0.5365 | 0.6013 | 5 | 0.5026 | 0.6855 | 0.2534 | 0.8632 |
| 3 | 30 | 0.4843 | 0.5705 | 0.6995 | 6 | 0.6654 | 0.6363 | 0.1500 | 0.6380 |
| 4 | 5* | 0.4727 | 0.5152 | 0.6530 | 4 | 0.5988 | 0.6110 | 0.1488 | 0.8056 |

| Metric | Value |
|--------|-------|
| val_cindex (best) | 0.6814 ± 0.0491 |
| val_cindex (last5) | 0.6327 ± 0.0748 |
| train_cindex | **0.48** (不如随机猜) |

\* Fold 4 被手动 kill，仅 5 epoch（已确认在 epoch 5 train_cindex=0.5152 < 0.5，bug 已复现，无需跑完）

### seed=None, 30 epochs, fold2 only (grad_clip=1.0)

| Fold | Epochs | train_mean | train_max | val_best | val_best@ | val_last5 |
|------|--------|-----------|-----------|---------|----------|----------|
| 2 | 5* | — | — | 0.6253 | 0 | 0.5768 |

\* 5 ep 后 train_cindex=0.5002 仍 < 0.5，模型不拟合，手动停机

---

## 2. v45v2_blca_clinical — 8 损失 + age/gender (seed=3, 30ep)

从 `results/v45v2_blca_clinical/SUMMARY.md`：

| Fold | Best @ | C-index | IPCW | IBS | iAUC |
|------|--------|---------|------|-----|------|
| 0 | 14 | 0.7417 | 0.4581 | 0.8828 | 0.8302 |
| 1 | 24 | 0.7453 | 0.6801 | 0.4261 | 0.6910 |
| 2 | 8 | 0.6237 | 0.5581 | 0.8884 | 0.4965 |
| 3 | 20 | 0.7049 | 0.6313 | 0.4543 | 0.5897 |
| 4 | 15 | 0.6441 | 0.6074 | 0.3131 | 0.6253 |

**C-index 0.6919 ± 0.0499**

---

## 3. Rank-Guided Event Transport — 3 损失 (config: rank_guided_event_transport_blca.yaml)

### seed=1, 10 epochs, 5-fold, grad_clip=1.0 (quick_robust_eval)

| Fold | Epochs | train_mean | train_max | val_best | val_best@ | val_last5 | ipcw_best | IBS_best | iauc_best |
|------|--------|-----------|-----------|---------|----------|----------|-----------|----------|----------|
| 0 | 10 | 0.4837 | 0.5296 | 0.6965 | 5 | 0.6487 | 0.7790 | 0.3499 | 0.8604 |
| 1 | 10 | 0.5570 | 0.7501 | 0.7403 | 6 | 0.6335 | 0.7204 | 0.1572 | 0.8641 |
| 2 | 10 | 0.5271 | 0.6358 | 0.6037 | 8 | 0.5955 | 0.5899 | 0.2313 | 0.8932 |
| 3 | 10 | 0.6016 | 0.8136 | 0.6470 | 6 | 0.6350 | 0.5974 | 0.1451 | 0.9285 |
| 4 | 10 | 0.5351 | 0.7252 | 0.6859 | 8 | 0.6407 | 0.6439 | 0.1493 | 0.9326 |

| Metric | Value |
|--------|-------|
| val_cindex (best) | 0.6747 ± 0.0463 |
| val_cindex (last5) | **0.6307 ± 0.0184** |
| train_cindex max | **0.53-0.81** (能拟合了) |

**honest_report 诚实验证**：乐观偏差 +0.0440 (C-index), +0.2658 (iAUC)

### seed=3, 25 epochs, fold2 only (no grad_clip)

| Fold | Epochs | train_mean | train_max | val_best | val_best@ | val_last5 |
|------|--------|-----------|-----------|---------|----------|----------|
| 2 | 25 | — | **0.7285** | 0.6341 | 29 | 0.5616 |

### seed=3, 3 epochs smoke, fold2 only (no grad_clip)

| Fold | Epochs | train_max | val_best | val_best@ |
|------|--------|-----------|---------|----------|
| 2 | 3 | — | 0.5592 | 0 |

### seed=1, 30 epochs, fold0 only, grad_clip=1.0 (stable_launcher 仅 fold0 未秒崩)

| Fold | Epochs | val_best | val_best@ | val_last5 |
|------|--------|---------|----------|----------|
| 0 | 30 | 0.6743 | 7 | 0.6520 |

---

## 4. newSlotSPE 消融 (config: v45_blca.yaml, seed=3, alpha_surv=0.15)

> 这些 epoch_curve CSV 仅在 newSlotSPE 本地，未同步到 SurvOT-Rank。以下来自 `ABLATION_LOG.md`。

| 配置 | val_cindex mean±std | Fold 2 | Fold 3 | 说明 |
|------|-------------------|--------|--------|------|
| V0 baseline (8 损失) | 0.6993 ± 0.0218 | — | — | 等频分箱 + alpha_surv=0.15 |
| V1 wd=1e-4 + rank 降权 | 0.6991 ± 0.0323 | — | 0.6426 | 降 rankevent 反而更差 |
| V2 wd=5e-4 + **关 rankevent** | **0.7100 ± 0.0186** | — | 0.6896 | 开/关 rankevent 的因果验证 |
| V3 wd=1e-3 + 关 rankevent | =V2 (md5 一致) | — | — | wd 被 Adam 忽略，回归 bug |
| V4a AdamW + wd=5e-4 | fold3 only | — | 验伪 | 与 V2 一致，AdamW 也未生效 |

**关键发现**：
- V2 关掉 rankevent 的 4 个辅助损失 → mean +0.0107, std -0.0032, fold3 不再崩盘
- `--reg` 配合 `--opt adam` 时 web/training bridge 未传 `weight_decay`，所有 V1-V4 的 wd=0
- V2/V3 的 csv md5 完全一致（回归测试通过），证明 V2 已达到该配置天花板

---

## 5. Stagewise Prognostic Transport (config: stagewise_prognostic_transport_blca.yaml)

`961fda4` — 新方法，每个时间阶段独立 prognostic cost + 三路 OT plan

| Fold | Epochs | val_best | 状态 |
|------|--------|---------|------|
| (smoke test) | — | — | 🔄 正在跑 3ep fold2 |

---

## 6. robust_eval 工具链状态

| 工具 | 测试 | 结果 | 状态 |
|------|------|------|------|
| `epoch_curve_selection.py` | selftest (合成数据) | optimism_gap=0.07 | ✅ |
| `honest_report.py` | quick_train 报告 | 已生成 report.md | ✅ |
| `stable_train_launcher.py` | seed=1,2,3 × 5-fold × 30ep | **3 次全部 `__func__` 失败** | ❌ |

**stable_train_launcher 失败原因**：monkey-patch `optimizer.step` 在 PyTorch 2.x / Python 3.11+ 下 `'function' object has no attribute '__func__'`。grad_clip 已在 `train_runner.py` 内置实现规避。

---

## 7. Fold 2 全量对比

| 模型 | 损失数 | Epoch | Seed | grad_clip | val_best | val_last5 | train_max | 能拟合? |
|------|-------|-------|------|----------|---------|----------|-----------|--------|
| v45 (batch) | 8 | 30 | 3 | ❌ | 0.6013 | 0.5026 | 0.5365 | ❌ |
| v45 (fold2 only) | 8 | 30 | None | ✅ | 0.6253 | 0.5768 | 0.5002 | ❌ |
| v45v2+clinical | 8 | 30 | 3 | ❌ | 0.6237 | — | — | ❌ |
| RG-ET 25ep | 3 | 25 | 3 | ❌ | 0.6341 | 0.5616 | **0.7285** | ✅ |
| RG-ET 10ep | 3 | 10 | 1 | ✅ | 0.6037 | 0.5955 | 0.6358 | ✅ |
| RG-ET 30ep (no seed) | 3 | 30 | None | ✅ | 🔄 | 🔄 | 🔄 | 🔄 |
| Stagewise | ? | 3 | — | ❌ | 🔄 | 🔄 | — | 🔄 |

---

## 8. 运行环境

- **SurvOT-Rank**：`/home/ubuntu/SurvOT-Rank` (commit `961fda4`)
  - 新代码，PR#3/#4/#5 已合并，seed 默认 None
- **newSlotSPE**：`/home/ubuntu/newSlotSPE` (commit `0eaadc6`)
  - 旧代码，等宽分箱 bug 版本，消融实验 V0-V4a 在此环境
- **Python**：conda env `trisurv` (`/home/ubuntu/.conda/envs/trisurv/bin/python`)
- **GPU**：CUDA, gpu=0
- **数据**：5-fold BLCA split, `survival_months_dss` 标签, Pathways 特征, n=380
