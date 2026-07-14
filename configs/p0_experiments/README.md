# P0 实验配置文件说明

> 本目录包含 P0 优先级实验的配置文件（必须完成，才能把结论立住）

---

## 实验列表

| 序号 | 实验名 | 目的 | 对比基准 |
|:---:|---|---|---|
| P0-1 | `v45_baseline_globalbin_blca` | v45 全 8 损失 + 分箱 B 对照 | v45_norank(分箱B)=0.6406 |
| P0-2a | `v45_norank_seed3_blca` | 固定 seed=3 复核 v45 稳定性 | v45_norank(seed=None)=0.6406 |
| P0-2b | `v45_norank_seed5_blca` | 固定 seed=5 复核 v45 稳定性 | v45_norank(seed=None)=0.6406 |
| P0-3a | `v50_norank_seed3_blca` | 固定 seed=3 复核 v50 稳定性 | v50_norank(seed=22646)=0.6572 |
| P0-3b | `v50_norank_seed5_blca` | 固定 seed=5 复核 v50 稳定性 | v50_norank(seed=22646)=0.6572 |
| P0-4a | `v50_ablation_only_ot_eventsurv` | stripped：仅 OT + EventSurv | — |
| P0-4b | `v50_ablation_spec_only` | +Spec（时间特化）| P0-4a |
| P0-4c | `v50_ablation_spec_cover` | +Cover（时间覆盖）| P0-4b |
| P0-4d | `v50_ablation_full` | +Compete（全开）| P0-4c |

---

## 运行方式

### Linux 服务器（推荐）
```bash
cd /home/ubuntu/SurvOT-Rank
bash scripts/run_p0_experiments.sh
```

### Windows 本地（调试用）
```powershell
cd E:\SurvOT-Rank
.\scripts\run_p0_experiments.ps1
```

---

## 注意事项

1. **只跑 fold 0 和 fold 2**：节省时间，两个折足够判断趋势
2. **30 epoch**：与历史实验保持一致
3. **分箱 B**：全部使用全局分箱（已在代码层面修复）
4. **seed 固定**：P0-2 用 seed=3 和 seed=5，P0-1/3 用 seed=3

---

## 预期结果解读

### P0-1（v45 全开 vs 关 rankevent）
- 如果 `v45 全开(分箱B)` ≈ `v45_norank(分箱B)=0.6406`：说明关闭 rankevent 没有贡献，分箱修复是主要因素
- 如果 `v45 全开(分箱B)` < 0.6406：说明关闭 rankevent 确实有帮助

### P0-2（v45 种子稳定性）
- 如果 seed=3 和 seed=5 的 last5 都 ≈ 0.6406（±0.02）：说明 v45_norank 稳定
- 如果方差很大（>0.05）：说明 v45 对种子敏感

### P0-3（v50 种子稳定性）
- 如果 seed=3 和 seed=5 的 last5 都 ≈ 0.6572（±0.02）：说明 v50_norank 稳定
- 如果方差很大（>0.05）：说明 v50 对种子敏感，结论不可靠

### P0-4（v50 损失消融）
- `stripped` < `+spec` < `+cover` < `full`：说明时间局部机制逐层有效
- 如果某一档没有增益，可以精简 V50 的设计

---

## 结果汇总位置

实验结果自动写入 `results/p0_experiments/<实验名>/` 目录。

手动汇总到 `EXPERIMENT_SUMMARY.md` 的 §1 表格中。
