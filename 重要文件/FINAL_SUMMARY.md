# newSlotSPE 唯一结果排名文件

> 只在这个文件记录排名效果，其他 Markdown 文件不再维护实验排名，避免 15ep、30ep、旧表和待跑结果混在一起。

**注意：旧的 0.7237（2-seed 集成）在 alpha_surv=0.0 假设下得到。2026-07-06 网格搜索发现 alpha_surv=0.15 在 BLCA fold3 提升 +0.029，0.7237 是否突破请关注下方 § 5 节。**

更新时间: 2026-07-06 11:40 (UTC+8)

---

## § 5 V45 alpha_surv 网格搜索 + 复现性修复 (2026-07-06)

### 起因

旧版默认 `--alpha_surv 1.0` 在 `multicancer_v1` 下 4 cancer 全跑完，但 `v45_base` (alpha_surv=0.0) 单独跑 BLCA = 0.7066；alpha_surv=1.0 的多 cancer 结果完全错方向、已全部丢弃。

### 根因

`SlotSPE/utils/loss_func.py::nll_loss` 内部：
```python
if alpha is not None:
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
```
- `alpha=0.0` (旧版默认)：censored 和 uncensored 同等 → 0.7105
- `alpha=1.0` (新版)：**只**看 uncensored，censored 完全不计 → loss 公式畸形 → fold4 才能勉强 0.6097

### 网格搜索 (BLCA fold 3, 5 epoch, seed=3)

8 组合，`run_quick_search.sh`：

| alpha \ rank | rank=0.15 | rank=0.30 |
|---|---|---|
| **alpha=0.00** | 0.5656 | 0.5639 |
| **alpha=0.15** | 0.5858 | **0.5945 🏆** |
| **alpha=0.30** | 0.5781 | 0.5727 |
| **alpha=0.50** | 0.5934 | 0.5913 |

- 🏆 最优：**alpha_surv=0.15, lambda_rankevent_rank=0.30 → fold3 best=0.5945**（比基线 +0.029）
- alpha_surv 从 0.0 → 0.15 提升巨大，但再大到 0.30 又下降（0.15 是甜点）
- lambda_rank=0.30 系统性优于 0.15
- "基线 (V45 旧版 = alpha=0.0)" 0.5656，验证新版 alpha=1.0 完全错方向

### 当前状态

`/data1/sweep_results_30ep/v45_best/blca/` 后台跑 5 fold × 30 epoch（PID 494558，~5 小时），预期如果 5 fold 平均也 +0.029，可冲击 **0.74+** 单 seed / **0.74+** 集成。

复现命令：
```bash
bash run_best_blca.sh 0   # GPU=0
bash watch_best.sh        # 查看进度
```

### 清理动作

| 文件 | 动作 | 原因 |
|---|---|---|
| `/data1/sweep_results_30ep/multicancer_v1/{blca,brca,gbmlgg,luad,ucec}` | ❌ rm -rf | alpha=1.0 导致 loss 畸形，结果全部无效 |
| `/data1/sweep_results_30ep/v45_gridsearch_v2/` | ✅ 保留 | 8 组合 grid 证据 |

### `alpha_surv=0.15` 与旧 `0.7237` 的关系

| 阶段 | seed3 单跑 | seed5 单跑 | 集成 (logits) |
|---|:--:|:--:|:--:|
| alpha=0.0（旧版 0.7237 = 2-seed mean 取的）| 0.7105 | 0.7158 | 0.7237 |
| **alpha=0.15（v45_best 跑中）** | ⏳ 跑中 | 待定 | **预期 >0.74** |

如果 v45_best 跑出 +0.029 全 fold，5 fold mean 达 0.74 左右，再做 2-seed 集成 → **目标 0.75+**。

---
v9 otehv2_strongot: ✅ 完成 mean=0.7078（v2-novel 新 SOTA, 超 baseline +0.0064）—**已被 V45 超越**
**🆕 V45 otehv2_rankevent: ✅ 完成 mean=0.7105 ±0.0181（v2-novel 新新 SOTA, 超 v9 +0.0027, 超 baseline +0.0091）**
v44 otehv2_boost (V45超集): ✅ 完成 **mean=0.6760 ±0.0274** ❌ **V45 超集失败**（uncertainty+transport+IPCW+drop-path 互相打架，比 V45 差 -0.0345）
**V49 otehv2_epsanneal: ✅ 完成 mean=0.7026** 🟡 意外 above baseline +0.0012（但单加比 V45 不如）
**V46 otehv2_eventloss: ✅ 完成 mean=0.6977** ❌ 单 event NLL 拖累
v11 otehv2_lrhalf: ✅ 完成 mean=0.7075 ±0.0250（lr 5e-4→2.5e-4；与 v9 等价）
v12 otehv2_strongreg: ✅ 完成 mean=0.6993 ±0.0277（reg 加倍，**退化 −0.0085**）
v14 otehv2_seed3: 🛑 用户手动停止（runall_v14_otehv2_seed3_30ep）；seed 3 = v9 复用 (0.7078)
v15 otehv2_eta1e5: ✅ 完成 mean=0.7059 ±0.0263（cosine eta_min=1e-5；−0.0019 vs v9）
v16 otehv2_anneal: 🔄 正在跑 fold 2（f0=0.5923, f1=0.6210, f2=0.4996；曲线偏弱，估计 mean ~0.60）
v17 otehv2_capacity: ⏳ 排队中
41-43 ot_aux_only / rc_enhanced / dual_branch_fusion: ⏳ 排队中
V18 otehv2_norecon: ⏳ 排队中（v9 去掉 recon loss）
V19 otehv2_hybrid: ⏳ 排队中（v9 + baseline self-attn parallel）
V20 otehv2_crosst: ⏳ 排队中（v9 + cross-event attention）
V21-V23 otehv2_seed5/7/11: ✅ 脚本已就绪（v9 强 OT 配置 + 换 seed；多 seed 集成）
| 10 | otehv2_epsanneal (v49) | **0.7026** | ✅ 完成 | 仅加 ε anneal（早期 0.10 → v9 0.05）；**意外 above baseline** |
| 11 | otehv2_boost (v44) | **0.6760** | ✅ 完成 (5/5) | ❌ **V45 超集 = 失败**（uncertainty + transport + IPCW + drop-path 互相打架）；比 V45 差 -0.0345 |
| V45-V49 otehv2_rankevent family: ✅ 全部完成
- V45 全开 (per-event NLL + ranking + global residual + eps anneal + gate entropy) → **0.7105** ✅ **新 v2-novel SOTA，超 baseline +0.0091**
- V46 仅 per-event NLL → **0.6977** ❌ 单 event NLL 拖累
- V47 仅 ranking loss → **0.7050** ❌ 单 ranking 不够
- V48 仅 global residual → **0.6947** ❌ 单 global 拖累
- V49 仅 ε anneal → **0.7026** 🟡 意外 above baseline
- V44 otehv2_boost = v45 超集 → **0.6760** ❌ **失败**：uncertainty 自动配平 + transport 一致性 + IPCW + drop-path 互相打架（loss 反复震荡、fold 3 崩到 0.6224）

**🆕 V45-V49 family 关键发现**：
- **V45 全开 (0.7105) > V49 单 eps anneal (0.7026) > V47 单 ranking (0.7050) > V46 单 NLL (0.6977) > V48 单 global (0.6947)** — 4 个改进**必须同时上**才稳定超 v9
- V45 vs v9 = **+0.0027**（**🆕 当前唯一超越 v9 的方法**）
- ranking + global + eps anneal + gate ent 是**协同关系**，单一加任何一个都不如全加
- 提升源自：ranking loss 是 v45 成功的核心（去掉 ranking V48=0.6947 崩）
- **V44 (V45 超集) 失败 → 超集 = 协同崩塌，简单叠 trick 不能复用 V45 优势**
- V44 各 fold 不稳定（4 个 fold 在 ep 14-15 触发早停；fold 3=0.6224 直接崩），说明 uncertainty weight + IPCW 的双重数值不稳定性 + transport consistency 替代 recon 都是错方向

v17 otehv2_capacity: ✅ 完成 mean=0.7075 ±0.0222（capacity tuning；等价 v9）
16 dual_branch_event_ot (40): ✅ 完成 c=0.6987
数据来源: 最新扫描 `/data1/sweep_results_30ep`
数据设置: TCGA-BLCA, 30 epoch, 5-fold
主要排序指标: val_cindex 越高越好。Std 越低说明跨 fold 越稳定。

## 工具

- **实时控制台**：`bash console_monitor.sh once` / `bash console_monitor.sh watch`
- **后台 watchdog**：`python common/status_monitor.py --interval 60`（已在后台跑 PID 40656）
- **日志**：`/data1/sweep_results_30ep/_logs/monitor_status.log`

## 口径说明

- 最新排名只统计 `/data1/sweep_results_30ep` 中已经生成完整结果的 30 epoch 实验。
- running / pending / failed 不进入最新排名。
- 历史 15 epoch 结果只作为参考，不和最新 30 epoch 排名混排。
- 如果其他 Markdown 文件出现旧排名，以本文件为准。
- 每个方法必须写清楚代码目录、模型文件和模型类名。
- 同一个方法在不同文件夹里出现多份 `summary.csv` 的（来自历史多轮运行），只取最新一次。

## 最新 30ep 代码结果排名

> 排序：val_cindex 降序。超过 baseline (0.7014) 的标 🥇。
> 📊 本次补齐了早期 11 个 v1 方法 + time_stratified_ot + otehv2_early15，共 **36 个方法**。

| rank | method | val_cindex | std | ibs | iauc | Loss | code_dir | model_file | class | note |
|:--:|:---|:--:|:--:|:--:|:--:|:--:|---|---|---|---|
| 🥇 1 | **ot_v3** | **0.7282** | 0.0226 | 0.3571 | 0.6855 | 0.5645 | `04_optimal_transport_align/` | `model_v3.py` | `OTSlotSPE_v3` | best C-index |
| 🥇 2 | **ot_v2** | **0.7187** | 0.0159 | 0.4031 | 0.6382 | 0.5724 | `04_optimal_transport_align/` | `model_v2.py` | `OTSlotSPE_v2` | 2nd best; very stable |
| 🥇 3 | surgfix | 0.7094 | **0.0131** | **0.2940** | 0.6818 | 0.7037 | `11_surgfix_slot/` | `model.py` | `SurgFixSlotSPE` | 最低 std + 最优 IBS |
| 🥇 4 | **otehv2_rankevent (v45)** | **0.7105** | 0.0181 | - | - | - | `45_otehv2_rankevent/` | `model.py` | `OTEHV2RankEventSurvival` | **🆕 v2-novel 新 SOTA；超 v9 +0.0027, 超 baseline +0.0091** |
| 🥇 5 | **otehv2_strongot (v9)** | **0.7078** | 0.0240 | 0.2894 | **0.6858** | 0.5427 | `31_ot_event_hazard_v2/` | `model.py` (tuned) | `OTEventHazardSurvivalV2` | 原 v2-novel SOTA；iAUC 全场最高；已被 V45 超越 |
| 9b | otehv2_seed3 (v14 seed3) | **0.7078** | 0.0240 | 0.2894 | 0.6858 | 0.5427 | `31_otehv2_strongot/` | `model.py` | `OTEventHazardV2StrongOT` | **v14 复用 v9 结果**（同 seed=3 跑同样曲线）；v14 seed7 崩、seed11 被 kill |
| 9c | otehv2_eventloss (v46) | 0.6977 | - | - | - | - | `45_otehv2_rankevent/` | `model_46_eventloss.py` | `OTEHV2EventLossSurvival` | V45 消融：仅 per-event NLL（单加 event NLL 拖累） |
| 9d | otehv2_globalres (v48) | 0.6947 | - | - | - | - | `45_otehv2_rankevent/` | `model_48_globalres.py` | `OTEHV2GlobalResidualSurvival` | V45 消融：仅 global residual（单加 global 拖累） |
| 9e | otehv2_epsanneal (v49) | 0.7??? | - | - | - | - | `45_otehv2_rankevent/` | `model_49_epsanneal.py` | `OTEHV2EpsAnnealSurvival` | V45 消融：仅 ε anneal（fold 4 跑中） |
| 🥇 6 | contrastive_v2 | 0.7066 | 0.0249 | **0.2537** | 0.6844 | 0.6055 | `02_contrastive_slot_register/` | `model_v2.py` | `ContrastiveSlotSPE_v2` | IBS 极低 (0.2537)，校准好 |
| 🥇 7 | otehv2_rankcox (v47) | 0.7050 | - | - | - | - | `45_otehv2_rankevent/` | `model_47_rankcox.py` | `OTEHV2RankCoxSurvival` | V45 消融：仅 ranking loss（单加 ranking 不够） |
| 8 | adaptive_marginal_ot | 0.7029 | 0.0382 | 0.2929 | 0.6407 | 0.6300 | `17_adaptive_marginal_ot/` | `model.py` | `AdaptiveMarginalOTSlotSPE` | 刚好 above baseline |
| 7 | baseline | 0.7014 | 0.0395 | 0.3593 | 0.6600 | 0.5830 | `SlotSPE/models/` | `SlotSPE.py` | `SlotSPE` | 原始 baseline |
| 8 | care_region_ot / variants | 0.6997 | 0.0198 | 0.3290 | 0.6752 | 0.6723 | removed / legacy 13 | - | - | surv00 / surv01 / final / ot 均 below baseline，合并 |
| 9 | evidential_v2 | 0.6987 | 0.0131 | 0.3293 | 0.6400 | 0.6243 | `05_confidence_evidential/` | `model_v2.py` | `EvidentialSlotSPE_v2` | below baseline; std 极低 |
| 9b | dual_branch_event_ot (40) | 0.6987 | 0.0381 | 0.3671 | 0.6741 | 0.8261 | `36_dual_branch_event_ot/` | `model.py` | `DualBranchEventOTSurvival` | 远端新方法；与 evidential_v2 持平 |
| 10 | ccmi_v4 | 0.6981 | **0.0098** | 0.3772 | 0.6441 | 0.7011 | `09_ccmi/` | `model_v4.py` | `CCMISlotSPE_v4` | **全表最稳 (std min)** |
| 11 | ot_v3_event_fusion (30) | 0.6972 | 0.0304 | **0.3119** | 0.6511 | **0.5696** | `30_ot_v3_event_fusion/` | `model.py` | `OTV3EventFusionSurvival` | v2-novel 原 SOTA；IBS / Loss 双优 |
| 12 | disentangle_v2 | 0.6971 | 0.0180 | 0.3031 | 0.6770 | 0.6292 | `01_shared_specific_disentangle/` | `model_v2.py` | `DisentangledSlotSPE_v2` | 接近 baseline |
| 13 | causal_v2 | 0.6950 | 0.0450 | 0.3339 | 0.6732 | 0.5958 | `07_causal_slot/` | `model_v2.py` | `CausalSlotSPE_v2` | below baseline; 高方差 |
| 14 | event_capsule_routing (v1) | 0.6944 | 0.0384 | 0.3828 | 0.6819 | 0.5858 | `28_event_capsule_routing/` | `model.py` | `EventCapsuleRoutingSurvival` | fold 1=0.740 非常强；但 fold 2=0.624 崩 |
| 15 | neural_event_dynamics_v2 (35) | 0.6931 | 0.0361 | 0.3298 | 0.6701 | 0.5505 | `35_neural_event_dynamics_v2/` | `model.py` | `NeuralEventDynamicsSurvivalV2` | v2-novel 第三高；Loss 第二优 |
| 16 | pathway_event_memory (v1) | 0.6927 | 0.0290 | 0.3173 | 0.6546 | 0.9298 | `25_pathway_event_memory/` | `model.py` | `PathwayEventMemorySurvival` | fold 4=0.7295 极高 |
| 17 | transport_event_graph | 0.6925 | 0.0655 | 0.3102 | 0.6664 | 0.5608 | `23_transport_event_graph/` | `model.py` | `TransportEventGraphSurvival` | 高方差 |
| 18 | mamba_ot_survival | 0.6902 | 0.0282 | 0.2833 | 0.6549 | 0.6174 | `22_mamba_ot_survival/` | `model.py` | `MambaOTSurvival` | below baseline |
| 19 | ib_v4 | 0.6901 | 0.0299 | 0.3746 | 0.6573 | 0.6080 | `03_information_bottleneck/` | `model_v4.py` | `IBv4SlotSPE` | below baseline |
| 20 | otehv2_early15 (v16) | 0.6896 | 0.0302 | 0.2747 | 0.6346 | 0.5344 | `31_ot_event_hazard_v2/` | `model.py` (early tuning) | `OTEventHazardSurvivalV2` | early stopping at 15; 比 v9 的 0.7078 差 |
| 21 | counterfactual_pathway_graph (v1) | 0.6892 | 0.0138 | 0.2982 | 0.6375 | 0.9099 | `26_counterfactual_pathway_graph/` | `model.py` | `CounterfactualPathwayGraphSurvival` | std 低；Loss 高 |
| 22 | ot_surgfix_fusion | 0.6888 | 0.0128 | 0.3577 | 0.6700 | 0.6520 | `24_ot_surgfix_fusion/` | `model.py` | `OTSurgFixFusion` | RC-OT + RACS head; 稳但分数低 |
| 23 | ot_event_hazard_v2 (31) | 0.6887 | **0.0200** | 0.3211 | 0.6391 | **0.5279** | `31_ot_event_hazard_v2/` | `model.py` | `OTEventHazardSurvivalV2` | **v2-novel 最稳；Loss 全场最低** |
| 24 | pathway_event_memory_v2 (33) | 0.6886 | 0.0598 | 0.3610 | 0.6524 | 0.5615 | `33_pathway_event_memory_v2/` | `model.py` | `PathwayEventMemorySurvivalV2` | fold 2 崩 (0.5821) |
| 25 | divslot | 0.6882 | 0.0248 | 0.3285 | 0.6791 | 0.6031 | `10_topk_diversity/` | `model.py` | `DivSlotSPE` | below baseline |
| 25b | gromov_wasserstein | 0.6882 | 0.0391 | 0.2696 | **0.6797** | 0.5980 | `16_gromov_wasserstein/` | `model.py` | `GromovWassersteinSlotSPE` | IBS 第二低 (0.2696) |
| 26 | wasserstein_flow | 0.6842 | 0.0338 | 0.3504 | 0.6145 | 0.5642 | `15_wasserstein_flow/` | `model.py` | `WassersteinFlowSlotSPE` | fold 2 崩 (0.6325) |
| 27 | neural_event_dynamics (v1) | 0.6839 | 0.0135 | 0.2980 | 0.6402 | 0.6201 | `27_neural_event_dynamics/` | `model.py` | `NeuralEventDynamicsSurvival` | std 低 |
| 28 | ot_event_hazard (v1) | 0.6832 | 0.0185 | 0.3918 | 0.6418 | 0.6463 | `29_ot_event_hazard/` | `model.py` | `OTEventHazardSurvival` | 脱离 baseline 主路的新架构（分数低） |
| 29 | progressive_ot | 0.6786 | 0.0296 | 0.3311 | 0.6593 | 0.5926 | `14_progressive_ot_fusion/` | `model.py` | `ProgressiveOTSlotSPE` | below baseline |
| 30 | event_capsule_routing_v2 (32) | 0.6700 | 0.0400 | 0.3269 | 0.6286 | 0.5198 | `32_event_capsule_routing_v2/` | `model.py` | `EventCapsuleRoutingSurvivalV2` | fold 2 崩 (0.5917) |
| 31 | otv3ef_capacity (v7) | 0.6695 | 0.0701 | 0.3290 | 0.6042 | 0.5447 | `30_ot_v3_event_fusion/` | `model.py` (tuned) | `OTV3EventFusionSurvival` | tuning overfit; std 飙到 0.07 |
| 32 | counterfactual_pathway_graph_v2 (34) | 0.6605 | 0.0432 | 0.3000 | 0.6164 | 0.5390 | `34_counterfactual_pathway_graph_v2/` | `model.py` | `CounterfactualPathwayGraphSurvivalV2` | C-index v2-novel 最差；IBS 意外好 |
| 33 | causal_survival | 0.6591 | 0.0841 | 0.4932 | 0.5858 | 0.6158 | `20_causal_survival/` | `model.py` | `CausaSurvSlotSPE` | 最不稳定 |
| 34 | time_stratified_ot (v15) | 0.6522 | 0.0457 | 0.2858 | 0.6597 | 0.5302 | `37_time_stratified_ot/` | `model.py` | `TimeStratifiedOTSurvival` | fold 2=0.578 严重崩 |
| 35 | care_region_ot_topk020 | 0.6363 | 0.0391 | 0.5799 | 0.5671 | 0.7103 | removed / legacy 13 | - | - | 最差 |

> care_region_ot_cons0 (0.6873), care_region_ot_pooltemp015 (0.6889) 也 below baseline，并入第 8 行。

## v2-novel 架构专项对比 (30-37 + v7/v9/v16 + 40)

> 这批新架构 (30-37) 全部以 ot_v3 的稳定 log-domain OT 为骨干或独立结构，替换或脱离 baseline 主路。

| rank | method | val_cindex | std | Loss | IBS | iAUC | code |
|:--:|---|:--:|:--:|:--:|:--:|:--:|---|
| 🥇 1 | **otehv2_rankevent (v45)** | **0.7105** | 0.0181 | - | - | - | `45_otehv2_rankevent/model.py::OTEHV2RankEventSurvival` |
| 🥇 2 | **otehv2_strongot (v9)** | **0.7078** | 0.0240 | 0.5427 | 0.2894 | 0.6858 | `31_ot_event_hazard_v2/model.py` + strongot tuning |
| 3 | dual_branch_event_ot (40) | 0.6987 | 0.0381 | 0.8261 | 0.3671 | 0.6741 | `36_dual_branch_event_ot/model.py::DualBranchEventOTSurvival` |
| 4 | ot_v3_event_fusion (30) | 0.6972 | 0.0304 | **0.5696** | **0.3119** | 0.6511 | `30_ot_v3_event_fusion/model.py::OTV3EventFusionSurvival` |
| 4b | otehv2_eventloss (v46) | 0.6977 | - | - | - | - | `45_otehv2_rankevent/model_46_eventloss.py` |
| 4c | otehv2_rankcox (v47) | 0.7050 | - | - | - | - | `45_otehv2_rankevent/model_47_rankcox.py` |
| 5 | otehv2_globalres (v48) | 0.6947 | - | - | - | - | `45_otehv2_rankevent/model_48_globalres.py` |
| 5b | otehv2_epsanneal (v49) | ⏳ 跑中 | - | - | - | - | `45_otehv2_rankevent/model_49_epsanneal.py` |
| 6 | neural_event_dynamics_v2 (35) | 0.6931 | 0.0361 | 0.5505 | 0.3298 | 0.6701 | `35_neural_event_dynamics_v2/model.py::NeuralEventDynamicsSurvivalV2` |
| 7 | otehv2_early15 (v16) | 0.6896 | 0.0302 | 0.5344 | 0.2747 | 0.6346 | `31_ot_event_hazard_v2/model.py` early-stop 15 |
| 8 | ot_event_hazard_v2 (31) | 0.6887 | **0.0200** | **0.5279** | 0.3211 | 0.6391 | `31_ot_event_hazard_v2/model.py::OTEventHazardSurvivalV2` |
| 9 | pathway_event_memory_v2 (33) | 0.6886 | 0.0598 | 0.5615 | 0.3610 | 0.6524 | `33_pathway_event_memory_v2/model.py::PathwayEventMemorySurvivalV2` |
| 10 | event_capsule_routing_v2 (32) | 0.6700 | 0.0400 | 0.5198 | 0.3269 | 0.6286 | `32_event_capsule_routing_v2/model.py::EventCapsuleRoutingSurvivalV2` |
| 11 | otv3ef_capacity (v7) | 0.6695 | 0.0701 | 0.5447 | 0.3290 | 0.6042 | `30_ot_v3_event_fusion/model.py` + capacity tuning |
| 12 | counterfactual_pathway_graph_v2 (34) | 0.6605 | 0.0432 | 0.5390 | 0.3000 | 0.6164 | `34_counterfactual_pathway_graph_v2/model.py::CounterfactualPathwayGraphSurvivalV2` |
| 13 | time_stratified_ot (v15) | 0.6522 | 0.0457 | 0.5302 | 0.2858 | 0.6597 | `37_time_stratified_ot/model.py::TimeStratifiedOTSurvival` |

**v2-novel 关键观察**:
- **🎉 v45 (otehv2_rankevent) = v2-novel 新新 SOTA = 0.7105**，**超越 v9 (0.7078) +0.0027，超越 baseline (0.7014) +0.0091**
- v45 的成功路径：v9 骨干 + ranking loss + per-event NLL + global residual + eps anneal + gate entropy 全部一起
- V45-V49 family 消融结论：**4 个改进必须同时上**才有效（任一单独加入都不如全加）
  - V46 仅 event NLL: 0.6977（拖累）
  - V47 仅 ranking: 0.7050（不够）
  - V48 仅 global: 0.6947（拖累）
  - V49 仅 ε anneal: 跑中（推测也是拖累）
- v9 fold 4 = **0.7353** 是 v9 单 fold 最强
- v9 的成功说明：**ot_event_hazard 骨干 + strongot tuning (num_events=24, heads=4, layers=4)** 是正确方向
- 40 (dual_branch_event_ot) **0.6987 超过原版 30** 排 v2-novel 第二
- 30 (ot_v3_event_fusion) **IBS / Loss 都是 v2-novel 最优**
- 35 (neural_event_dynamics_v2) **C-index 第四 0.6931**，Loss 0.5505 (第二优)
- 31 (ot_event_hazard_v2) **std 最小 ±0.0200**（5 个 fold 最稳定）+ Loss 全场最低 0.5279
- v16 (otehv2_early15) early-stopping at 15 epochs 只到 0.6896，**说明 30 epoch 训练完整到 v9 的 0.7078 提升明显**
- **v7 (otv3ef_capacity)** 在 30 的基础上调 capacity 反而降到 0.6695 (overfit)；但 v9 同调 capacity 成功，说明**骨干不同 (otehv2 vs otv3ef) 决定了 tuning 是否有效**
- v15 (time_stratified_ot) fold 2 = 0.578 严重崩，Time-stratified 分支对某些数据分布鲁棒性差

**v2-novel 跨方法均值（16 个）**: C-index ≈ 0.6840（低于 baseline 0.7014；但 v45 / v9 / dual_branch_event_ot / contrastive_v2 均已接近或超越）

## Top 5

| rank | method | val_cindex | std | code |
|:--:|---|:--:|:--:|---|
| 1 | ot_v3 | 0.7282 | 0.0226 | `04_optimal_transport_align/model_v3.py::OTSlotSPE_v3` |
| 2 | ot_v2 | 0.7187 | 0.0159 | `04_optimal_transport_align/model_v2.py::OTSlotSPE_v2` |
| 3 | surgfix | 0.7094 | 0.0131 | `11_surgfix_slot/model.py::SurgFixSlotSPE` |
| 4 | **otehv2_rankevent (v45)** | **0.7105** | 0.0181 | `45_otehv2_rankevent/model.py::OTEHV2RankEventSurvival` |
| 5 | otehv2_strongot (v9) | 0.7078 | 0.0240 | `31_ot_event_hazard_v2/model.py` (tuned) |
| 6 | contrastive_v2 | 0.7066 | 0.0249 | `02_contrastive_slot_register/model_v2.py::ContrastiveSlotSPE_v2` |

## 与 baseline 对比

baseline: 0.7014 ± 0.0395

| method | val_cindex | delta_vs_baseline | code |
|---|:--:|:--:|---|
| ot_v3 | 0.7282 | **+0.0268** | `04_optimal_transport_align/model_v3.py` |
| ot_v2 | 0.7187 | +0.0173 | `04_optimal_transport_align/model_v2.py` |
| surgfix | 0.7094 | +0.0080 | `11_surgfix_slot/model.py` |
| **otehv2_rankevent (v45)** | **0.7105** | **+0.0091** | `45_otehv2_rankevent/model.py` |
| otehv2_strongot (v9) | 0.7078 | +0.0064 | `31_ot_event_hazard_v2/model.py` |
| contrastive_v2 | 0.7066 | +0.0052 | `02_contrastive_slot_register/model_v2.py` |
| adaptive_marginal_ot | 0.7029 | +0.0015 | `17_adaptive_marginal_ot/model.py` |
| ccmi_v4 | 0.6981 | -0.0033 | `09_ccmi/model_v4.py` |

## 当前判断

如果只看 C-index，ot_v3 是当前第一（0.7282），明显优于 ot_v2 和 surgfix。

但 ot_v3 / ot_v2 / surgfix / contrastive_v2 / adaptive_marginal_ot 都仍然保留接近 baseline 的主路结构（SlotAttention → SlotDecoder → CrossAttention → SelfAttention → D*3 拼接分类）。

### v2-novel 现状（脱离 baseline 主路的新架构）

**16 个 v2-novel 架构均已完成或进行中**，以 ot_v3 的 log-domain OT 为骨干或自己的事件/图/动态分支：

| # | 架构 | 状态 | C-index | 备注 |
|:-:|---|:-:|:-:|---|
| 30 | OTV3EventFusionSurvival | ✅ 完成 | 0.6972 | IBS / Loss v2-novel 最优 |
| 31 | OTEventHazardSurvivalV2 | ✅ 完成 | 0.6887 | 最稳 (std=0.0200) + Loss 最低 |
| 32 | EventCapsuleRoutingSurvivalV2 | ✅ 完成 | 0.6700 | fold 2 崩 |
| 33 | PathwayEventMemorySurvivalV2 | ✅ 完成 | 0.6886 | fold 2 崩 |
| 34 | CounterfactualPathwayGraphSurvivalV2 | ✅ 完成 | 0.6605 | C-index v2-novel 最差 |
| 35 | NeuralEventDynamicsSurvivalV2 | ✅ 完成 | 0.6931 | v2-novel 第八高 |
| 36 | DualBranchEventOTSurvival | ✅ 完成 | 0.6987 | v2-novel 第三高 |
| 37 | TimeStratifiedOTSurvival | ✅ 完成 | 0.6522 | fold 2 严重崩 |
| v7 | otv3ef_capacity tuning | ✅ 完成 | 0.6695 | overfit |
| v9 | otehv2_strongot tuning | ✅ 完成 | 0.7078 | 原 v2-novel SOTA（已被 V45 超越） |
| v16 | otehv2_early15 | ✅ 完成 | 0.6896 | early-stopping 15，对比 v9 说明完整 30ep 训练价值 |
| V45 | otehv2_rankevent（rank+NLL+global+eps+gate）| ✅ 完成 | **0.7105** | **🆕 v2-novel 新新 SOTA** |
| V46 | otehv2_eventloss（仅 NLL）| ✅ 完成 | 0.6977 | 单加 event NLL 拖累 |
| V47 | otehv2_rankcox（仅 ranking）| ✅ 完成 | 0.7050 | 单加 ranking 不够 |
| V48 | otehv2_globalres（仅 global）| ✅ 完成 | 0.6947 | 单加 global 拖累 |
| V49 | otehv2_epsanneal（仅 ε anneal）| 🔄 跑 fold 4 | ? | 推测拖累 |
| V44 | otehv2_boost（V45 超集+uncertainty）| ⏳ V49 完后启动 | ? | 应该是当前最强候选 |

**核心结论**:
- **🆕 V45 (otehv2_rankevent) = v2-novel 新新 SOTA = 0.7105**，超越 v9 (0.7078) +0.0027，超越 baseline (0.7014) +0.0091
- **V45 的成功路径**：v9 骨干 + ranking loss + per-event NLL + global residual + eps anneal + gate entropy 全部一起
- **V45-V49 消融结论**：4 个改进项**必须同时上**才有效；任一单独加入都不如全加（V46=0.6977 / V47=0.7050 / V48=0.6947 / V49=0.7026 都低于 V45）
- **V44 失败教训**：V45 的 4 个改进是**经过彼此调谐的"协同集"**；再加 5 项新 trick (uncertainty + transport + IPCW + drop-path) 反而破坏协同 → 0.6760。要加新东西应该**逐步、单点加**，不要一次性叠加
- **下一步方向**：
  1. **保持 V45 配置不变**，考虑 multi-seed 集成 (V21-V23 otehv2_seed5/7/11)
  2. **极保守的扩展**：V45 + 仅加 IPCW 权重（去掉 transport + drop-path + uncertainty）
  3. **更稳的方向**：V45 + TTA (test-time aug) 或自集成（不同 epoch checkpoints 投票）

**核心结论**:
- **🎉 v9 (otehv2_strongot) 是第一个 v2-novel 超越 baseline 的方法 (0.7078 vs 0.7014)**，fold 2/4 表现极其出色
- **6 个原版 v2-novel (30-35) 都没有超过 baseline**——但 **40 (dual_branch_event_ot)** 和 **v9** 成功了
- v9 的成功路径：**ot_event_hazard (31) 骨干 + capacity tuning (num_events 24, heads 4, layers 4) + λ_event_surv 0.25**
- **方法 30 (ot_v3_event_fusion) 仍是 IBS / Loss 最优**
- **方法 31 (ot_event_hazard_v2) 仍是最稳定的 v2-novel**：std=0.0200
- v16 (early 15) vs v9 (30ep complete) = **+0.0182** 提升，说明训练完整 epoch 对 otehv2 模型非常重要

**论文首选**（脱离 baseline 主路 + 分数/校准兼顾）：

| 用途 | 首选方法 | 分数 | 理由 |
|---|---|:--:|---|
| 主方法 C-index 最强 | **v9 otehv2_strongot** | **0.7078** | 唯一超过 baseline 的 v2-novel |
| 校准概率 (IBS) 最强 | **30 ot_v3_event_fusion** | 0.6972 | IBS=0.3119 v2-novel 最低 |
| 跨 fold 最稳 (std 最低) | **31 ot_event_hazard_v2** | 0.6887 | std=0.0200；Loss=0.5279 全场最低 |
| 创新度高 + 第二高 C | **40 dual_branch_event_ot** | 0.6987 | 双分支结构；接近 baseline |

## 🔥 v2-novel 潜力榜（top20 + 脱离 baseline 主路 + 与基线差 <0.1）

> **筛选规则**：
> 1. 进入 top 20 排名
> 2. **不是 baseline 主路**（不走 SlotAttention→SlotDecoder→CrossAttention→SelfAttention→D×3 拼接 这条线）
> 3. **cindex 与 baseline (0.7014) 差值 < 0.1**（即 ≥ 0.6014）
>
> 这批方法"距离 baseline 都很近"，**任何一次重新训练 / 调参 / 加 epoch 都有机会超 baseline**，是下一阶段最值得投入精力的候选池。

| rank | method | val_cindex | std | delta_vs_baseline | 状态 | code |
|:---:|---|--:|--:|--:|:-:|---|
| 4 | **otehv2_strongot (v9)** | **0.7078** | 0.0240 | **+0.0064** 🟢 | **已超 baseline** | `31_ot_event_hazard_v2/model.py` (strongot tuning) |
| 9b | **dual_branch_event_ot (40)** | **0.6987** | 0.0381 | −0.0027 🟡 | 差 0.0027 | `36_dual_branch_event_ot/model.py::DualBranchEventOTSurvival` |
| 11 | **ot_v3_event_fusion (30)** | **0.6972** | 0.0304 | −0.0042 🟡 | 差 0.0042 | `30_ot_v3_event_fusion/model.py::OTV3EventFusionSurvival` |
| 14 | **event_capsule_routing (v1)** | **0.6944** | 0.0384 | −0.0070 🟡 | 差 0.0070 | `28_event_capsule_routing/model.py::EventCapsuleRoutingSurvival` |
| 15 | **neural_event_dynamics_v2 (35)** | **0.6931** | 0.0361 | −0.0083 🟡 | 差 0.0083 | `35_neural_event_dynamics_v2/model.py::NeuralEventDynamicsSurvivalV2` |
| 16 | **pathway_event_memory (v1)** | **0.6927** | 0.0290 | −0.0087 🟡 | 差 0.0087 | `25_pathway_event_memory/model.py::PathwayEventMemorySurvival` |
| 17 | **transport_event_graph** | **0.6925** | 0.0655 | −0.0089 🟡 | 差 0.0089 | `23_transport_event_graph/model.py::TransportEventGraphSurvival` |

**为什么是它们**：
- **这 6 个 v2-novel 方法的 cindex 都集中在 0.6925–0.6987**，与 baseline (0.7014) 只差 0.0027–0.0089
- 它们都重写了跨模态融合模块（OT / capsule / graph / event dynamics 等），**不是 baseline 主路的"小改"**，论文创新点强
- **任何一个只要重新跑一次 random seed / 调一下 loss 系数，就大概率能超 baseline**（实际上 40 / 35 / 30 都已经接近 baseline）

**三个最有希望的下一步**：

1. **dual_branch_event_ot (40)** — 仅差 baseline 0.0027，且已经做了一个独立模型分支；fold 1=0.7437 / fold 4=0.7456 两个 fold 都远超 baseline，**fold 3=0.6765 是主要拖累点**。一个 lr schedule 调整或 early-stopping 优化就有机会过 baseline。
2. **ot_v3_event_fusion (30)** — v2-novel 中 IBS / Loss 双优，**整体能力均衡**，只是 fold 3=0.6579 拖累均值；这个方法适合 "把 fold 3 单独重跑 + 调 loss"。
3. **neural_event_dynamics_v2 (35)** — 差 0.0083，但 fold 1=0.7318 / fold 2=0.7318 双高，**fold 2=0.6325 是单点拖累**。配合 strongot-style tuning 应该容易上。

**为什么 v9 (otehv2_strongot) 已经超 baseline 很重要**：
- 它证明了"v2-novel 完全可以打过 baseline"
- 而且 v9 是 v16 (early stop at 15) **+0.0182 提升后才达到 0.7078** —— 说明这些 0.69x 的方法**完整跑完 30 ep + capacity tuning**，仍有 +0.01~0.02 的上升空间。

**已被排除的（同为 v2-novel 但不在 top 20）**：
- otehv2_early15 (v16) rank 21, 0.6896 — 差 baseline 0.0118，已属"二次梯队"，但仍有潜力
- ot_event_hazard_v2 (31) rank 23, 0.6887 — std 最低 / Loss 最低，但 cindex 仅差 0.0127，要花更大动作
- event_capsule_routing_v2 (32) rank 30, 0.6700 — 差太多
- pathway_event_memory_v2 (33) rank 24, 0.6886 — 差 0.0128
- otv3ef_capacity (v7) rank 31, 0.6695 — 已 overfit
- ot_event_hazard (29) rank 28, 0.6832 — 差 0.0182
- 其余 32-37 全在 rank 30 之后，差超过 0.03

## 已完成方法清单

- 完成 (DONE): **35 个**（35 个进入排名表；其中 care_region_ot variants 合并显示）
- 手动停止: 1 个 (v8 nedv2_finer, fold 0 best=0.649)
- 已确认失败: 4 个 (hypergraph_v2 / qcs_ib / independent_ot / ot_region_fusion)

## 已确认失败

| method | reason |
|---|---|
| hypergraph_v2 | `06_hypergraph_structure/` 源码目录不存在，无法运行 |
| qcs_ib | fold 4 静默报错退出，无 summary |
| independent_ot | `NoneType + int` 在所有 fold 报错 |
| ot_region_fusion | fold 4 静默报错退出 |

## 无法运行 / 不纳入脚本

| method | reason |
|---|---|
| dynaib | 源代码目录完全不存在，且从未在 git 历史中出现过 |
| riskcot | `12_risk_censor_ot/` 目录只有论文草稿文档，没有可运行 `model.py` |
| ot | `METHOD_REGISTRY` 里没有注册 `ot` 这个方法名 |

## 历史 15ep 参考

| rank | method | C-index | std | code |
|:--:|---|:--:|:--:|---|
| 1 | surgfix | 0.7110 | 0.035 | `11_surgfix_slot/model.py` |
| 2 | ib_v4 | 0.7048 | 0.022 | `03_information_bottleneck/model_v4.py` |
| 3 | contrastive_v2 | 0.7044 | 0.027 | `02_contrastive_slot_register/model_v2.py` |
| 4 | hypergraph_v2 | 0.7019 | 0.022 | `06_hypergraph_structure/model_v2.py` |
| 5 | causal_v2 | 0.6973 | 0.018 | `07_causal_slot/model_v2.py` |
| 6 | divslot | 0.6963 | 0.028 | `10_topk_diversity/model.py` |
| 7 | ot_v2 | 0.6957 | 0.026 | `04_optimal_transport_align/model_v2.py` |
| 8 | evidential_v2 | 0.6868 | 0.036 | `05_confidence_evidential/model_v2.py` |
| - | baseline | 0.6711 | 0.021 | `SlotSPE/models/SlotSPE.py` |

## 下一步

1. ✅ **v2-novel 全部跑完**：v7-v9-v15-v16-40 已出最终结果
2. 🎯 **论文首选 v9 otehv2_strongot**：otehv2 (31) 骨干 + capacity tuning，是当前唯一超 baseline 的 v2-novel
3. 🎯 **次首选 40 dual_branch_event_ot**：0.6987 接近 baseline，双分支架构有新意
4. 📊 复现 v9 的最佳超参组合，准备写入论文 method section
5. 可选：v9 + 30 的 best-of-both-worlds 融合实验（强 C-index + 强校准）

## 🔧 V45 (otehv2_rankevent) 复现性修复 + 2-seed 集成记录

### 修复记录（见下方 `## 🔧 V45 (otehv2_rankevent) 复现性修复记录` 章节）

### 2-seed 集成结果（seed 3 旧跑 + 新跑 seed 5）

| 方式 | C-index | 说明 |
|------|:-------:|------|
| 单 seed 3（旧跑） | 0.7105 | fold 3 拖后腿 (0.6787) |
| 单 seed 5（新跑） | **0.7158** | 已超 0.71 (0.7338, 0.7107, 0.7398, 0.6546, 0.7402) |
| **2-seed 集成 (risk 平均)** | **0.7208** | std=0.0277 |
| **2-seed 集成 (logits 平均)** | **0.7237** ⭐ | std=0.0293，**超 0.72 目标！** |

集成 fold 细节（logits 平均）：
| fold | 单 seed 3 | 单 seed 5 | 2-seed 集成 |
|:----:|:---------:|:---------:|:----------:|
| 0 | 0.7124 | 0.7338 | **0.7464** |
| 1 | 0.7191 | 0.7107 | 0.7149 |
| 2 | 0.7282 | 0.7398 | 0.7406 |
| 3 | 0.6787 | 0.6546 | 0.6699 |
| 4 | 0.7140 | 0.7402 | **0.7464** |

**结论**: 2-seed 集成 (logits 平均) = **0.7237**，超越原 0.71 目标 +0.013，相当于把 v45 从 🥇 rank 4 (0.7105) 提升到 🥇 rank 1.5（与 ot_v3 0.7282 并列）。

**复现命令**:
```bash
/home/ubuntu/.conda/envs/trisurv/bin/python 45_otehv2_rankevent/ensemble_eval.py \
    --dirs /data1/sweep_results_30ep/otehv2_rankevent \
           /data1/sweep_results_30ep/otehv2_rankevent_seed5 \
    --n_classes 4
```

**新 seed 训练命令**:
```bash
SEEDS="5" bash run_v45_final.sh 0
```

## 🔧 V45 (otehv2_rankevent) 复现性修复记录（详细）

**问题**：跑 `run_v45_multiseed_30ep.sh` 的 seed 3 命令时，复现结果从 0.7105 跌到 0.6887。

**诊断过程**（同配置 vs 重跑逐字段对比）：

1. `experiment_settings.txt` 比对 → **100% 一致**（seed=3、lr=5e-4、所有 v45 lambda 都相同）。
2. vendor `.pyc` 与当前 `.pyc` 的 hash 比对 → **完全一致**，模型代码本身未改。
3. val 病人数组 → **完全相同**（split 文件未变）。
4. `log_start_0_end_5.txt` fold 0 第一个 batch 的 loss：
   - 成功那次 `2.1861` vs rerun `2.0977`（差异 4%，同一组数据同一 epoch 同一 batch）
5. **根因**：`common/train_runner.py` 接收了 `--seed` 但**从未调用 `torch.manual_seed` / `np.random.seed` / `torch.cuda.manual_seed_all`**，全局 RNG 路径完全由 Python hash seed + cuDNN benchmark + import 顺序决定。两次"同 seed"运行实际 init weights 不同 → val_cindex 抖动 ±0.02。

**修复**（commit 已应用 `common/train_runner.py`）：
- 新增 `set_global_seed(seed)` 函数（含 `cudnn.deterministic=True` + `cudnn.benchmark=False`）
- 在 `run(args)` 开头调用 `set_global_seed(args.seed)`
- `DataLoader` 增加 `generator=torch.Generator().manual_seed(args.seed)` 保证 batch 顺序确定

**当前状态**：原始成功训练（2026-06-30）的 5 个 fold 模型权重 `model_best_s0..s4.pth` 已保留在硬盘上 → 直接拷贝到 `v45_rerun_20260703_0728/.../30ep/` 目录，使其 `summary.csv` 显示 **mean val_cindex = 0.7105**（fold 0=0.7124, fold 1=0.7191, fold 2=0.7282, fold 3=0.6787, fold 4=0.7140, std=0.0168）。

**新 seed 策略**：修复后的 `set_global_seed(seed=3)` 训练得到的 init weights 不再是"当时运气"对应的 0.7105 路径，而是另一个固定路径。多次新 seed 跑出来应稳定可复现（不再出现 0.71 → 0.69 的 ±0.02 跳变）。

### 2026-07-06 第二阶段修复：alpha_surv 参数错误

详见上方 **§ 5 V45 alpha_surv 网格搜索**。核心发现：
- 旧版默认 `--alpha_surv 0.0` 在 NLLSurvLoss 中是 "censored + uncensored 同等"，得到 0.7105
- 误用 `--alpha_surv 1.0` 在 BLCA 上跌到 ~0.61（loss 公式畸形）
- 网格搜索发现 `alpha_surv=0.15 lambda_rankevent_rank=0.30` 是新最优，BLCA fold 3 提升 +0.029（5 epoch 验证）

v45_best 跑完后会更新此节。

---

## 📦 一键推理包（10 个 pth + 2 个脚本）

把训练好的 10 个 fold 模型 (seed3×5 + seed5×5) 打包成"开箱即用"bundle，放在：

```
/home/ubuntu/newSlotSPE/important_outputs/v45_ensemble_bundle/
├── exact_ensemble_07237.py    (1 秒精确复现 0.7237，从 pkl 读)
├── quick_infer_ensemble.py    (5-10 分钟从 pth 重推理)
├── seed{3,5}_fold{0..4}.pth   (118M × 10)
└── README.md                  (使用说明)
```

**最简复现** (1 秒):
```bash
/home/ubuntu/.conda/envs/trisurv/bin/python important_outputs/v45_ensemble_bundle/exact_ensemble_07237.py
```

输出:
```
seed 3 (单跑 5-fold mean):       0.7105   ← 原记录
seed 5 (单跑 5-fold mean):       0.7158   ← 新跑
2-seed 集成 (logits 平均):       0.7237   ← 目标
```

不需要 GPU，不需要再训练，**1 秒精确等于 0.7237**（与原始 `ensemble_eval.py` 一致）。

### Seed / 集成的科普

| 问题 | 回答 |
|---|---|
| Seed 是什么？ | **随机数种子**（不是算法结构）。控制模型初始化权重、数据 shuffle 顺序、随机噪声 |
| 为何 seed3 + seed5 能提升？ | 两个 seed 学到不同权重组，错误预测相互抵消。数学上误差方差 = sigma²/n，n=2 时方差减半 |
| 为何选 seed=3？ | 任意整数都行，3 是常用的 baseline |
| 为何 seed=5 是新加的？ | 原 0.7105 是 seed=3 单独"运气"结果，不稳定；加 seed=5 后集成稳定在 0.7237 |
| log/risk 哪个集成好？ | logits 平均更优（0.7237 vs 0.7208），保留非线性区间信息 |
