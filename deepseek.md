# DeepSeek 分析记录

## 2026-07-12：5-Fold 折间崩溃根因分析

### 测试环境

- **SurvOT-Rank 代码版本**：`5b66668` (Rank-Guided Event Transport) 合并 PR#3/#4/#5
  - `b6425d6`: PR#3 — `_disc_label` 等频分箱修复
  - `7496619`: PR#4 — V50 方法合并
  - `75c9959`: PR#5 — V2 死开关 + alpha_surv 修复
  - PR 系列生效确认：`alpha_surv=0.15`、等频分箱、fold-aware binning 均已应用
- **配置文件**：`configs/v45_blca.yaml`
- **运行命令**：`python -m survot_rank.cli train --config configs/v45_blca.yaml --set "gpu=0"`
- **结果文件**：`results/batch_runs/20260712_1912/v45_blca.log`
- **对比项目**：`/home/ubuntu/newSlotSPE`（代码版本 `0eaadc6`），等宽分箱 bug 版本

---

### 核心发现：所有折的 train_cindex 全部低于 0.5

从 `results/batch_runs/20260712_1912/v45_blca.log` 提取：

| Fold | Train C-index (min) | Train C-index (max) | Train C-index (mean) | Val C-index (best) |
|------|--------------------|--------------------|--------------------|--------------------|
| 0 | 0.4154 | 0.5370 | **0.4795** | 0.7120 |
| 1 | 0.4064 | 0.6315 | **0.4910** | 0.7411 |
| 2 | 0.4257 | 0.5365 | **0.4779** | 0.6013 |
| 3 | 0.3761 | 0.5705 | **0.4843** | 0.6995 |
| 4 | 0.4276 | 0.5152 | **0.4727** | (未完成) |

关键反常点：
- **train_cindex < 0.5 = 模型在自己的训练集上不如随机猜测**
- **val_cindex (0.60-0.74) 远大于 train_cindex (0.47-0.49)**，严重反直觉
- Fold 2 不是特例——所有折都无法拟合训练集，Fold 2 只是验证集长尾分布（mean=34月, std=33.9）最先暴露

---

### 根因：v45 有 8 个损失函数，梯度互相冲突

v45 (`SurvOTRank_otehv2_rankevent`) 的损失组成：

| # | 损失名称 | 权重 | 来源 |
|---|---------|------|------|
| 1 | OT 传输距离 (`lambda_ot`) | 0.06 | 父类 SlotSPE |
| 2 | Token 多样性 (`lambda_div`) | 0.01 | 父类 SlotSPE |
| 3 | 跨模态重建 (`lambda_recon`) | 0.20 | 父类 SlotSPE |
| 4 | 门控熵惩罚 (`lambda_rankevent_gate_ent`) | 0.005 | RankEvent |
| 5 | NLL 平均生存 (`lambda_surv`) | 0.25 | 父类 |
| 6 | 逐事件 NLL (`lambda_rankevent_per_event`) | 0.15 | RankEvent |
| 7 | Cox 排序 (`lambda_rankevent_rank`) | 0.15 | RankEvent |
| 8 | 全局一致性 (`lambda_rankevent_global_cons`) | 0.02 | RankEvent |

模型同时在 8 个目标的加权和上优化，8 个梯度方向互相拉扯。高方差 fold 上（如 Fold 2），梯度冲突被放大到让优化器在 Pareto 前沿上反复震荡，导致训练集都无法拟合。

---

### 证据：newSlotSPE 的 ABLATION_LOG 独立验证了同一结论

从 `/home/ubuntu/newSlotSPE/ABLATION_LOG.md`（代码版本 `0eaadc6`，配置文件 `v45_blca.yaml`，种子 seed=3）：

| 配置 | 5-fold mean | 5-fold std | 说明 |
|------|-----------|-----------|------|
| **V0 baseline** (8 损失全开) | 0.6993 | 0.0218 | 等同于 SurvOT-Rank v45 |
| V1 (降 rankevent 权重) | 0.6991 | **0.0323** (更差) | fold 3 崩到 0.6426 |
| **V2 (彻底关 rankevent)** | **0.7100** | **0.0186** (最稳) | fold 3 涨到 0.6896 |

newSlotSPE 的消融实验独立证明：**关掉 rankevent 的 4 个辅助损失后，mean 涨、std 降、fold 3 不再崩盘**。但 V2 仍然是 7 个损失（OT+Div+Recon+NLL 父类 4 项还在）。

---

### 修复历程（均对 Fold 2 无效的尝试）

| 修复 | 代码版本 | 效果 |
|------|---------|------|
| `_disc_label` 等频分箱（PR#3） | `b6425d6` | Fold 2 仍 0.60 |
| `alpha_surv` 1.0→0.15（PR#5） | `75c9959` | Fold 2 仍 0.60 |
| fold-aware binning | `e4808e4` | Fold 2 仍低分 |

说明问题不在分箱方式，在损失函数数量。

---

### 为什么 newSlotSPE 看起来"折间均衡"

newSlotSPE 的 `_disc_label` 使用等宽分箱（bug），82% 样本归入同一 bin。任务退化二分类，所有折的难度被"抹平"。这不是真正的鲁棒性——等频分箱修复后暴露了真实问题。

---

### 解决方向：Rank-Guided Event Transport（3 损失）

`5b66668` (`configs/rank_guided_event_transport_blca.yaml`)：
- 砍掉 5 个辅助损失，只保留 OT + Ranking + Stage order 三项
- 引入 `prognostic_pair_cost` 将预后代价直接加入 OT cost
- 待测试 Fold 2

---

### 配置错误记录

| 错误 | 影响 | 修复 |
|------|------|------|
| `alpha_surv=1.0` | 完全丢弃 66% 删失患者 | 改为 0.15 |
| `_disc_label` 等宽分箱 | 82% 样本同一 bin | PR#3 修复为等频 |
| V2 `unified_objective` 死开关 | abl_02==abl_00 | PR#5 修复 |
| `--reg` 不传 `--opt adamW` 无效 | wd 实验白跑 | ABLATION_LOG 已记录 |
| seed 默认 3 | 不需要的固定种子 | 改为默认 None |
