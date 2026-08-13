# SurvOT-Rank 文档地图

> 这是文档的唯一总入口。旧文档继续保留用于复现，但不再各自维护“当前主线”判断。

## 当前事实入口

| 问题 | 入口 |
|---|---|
| 方法注册名、别名、当前角色 | [METHODS.md](METHODS.md) |
| 方法身份、论文优先级、负结果和缺口 | [ACADEMIC_METHOD_EVALUATION.md](ACADEMIC_METHOD_EVALUATION.md) |
| 当前正式结果、划分边界、可比性 | [FINAL_SUMMARY.md](../FINAL_SUMMARY.md) |
| 2026-07-16 之前的历史实验 | [EXPERIMENT_SUMMARY.md](../EXPERIMENT_SUMMARY.md) |
| 包结构、CLI 与兼容层 | [FRAMEWORK.md](FRAMEWORK.md) |
| 旧路径迁移 | [MIGRATION.md](MIGRATION.md) |
| 配置/脚本入口 | [configs/INDEX.md](../configs/INDEX.md) / [scripts/README.md](../scripts/README.md) |

## 方法机制

- [methods/README.md](methods/README.md)：按家族整理的方法机制说明；
- `DCT_V*.md`：DCT 各阶段的协议、诊断和版本记录；
- [V40_INTERVENTION_STABLE_TRANSPORT.md](V40_INTERVENTION_STABLE_TRANSPORT.md)：IST-Surv v4.0；
- [LOSS_BLACKLIST.md](LOSS_BLACKLIST.md)：已否定或禁止重复引入的损失设计。

机制文档解释“方法如何工作”，不单独决定它是不是当前主线，也不保存最新分数。

## 路线图

- [roadmap/00_README.md](roadmap/00_README.md)：历史路线图入口；
- [roadmap/THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md](roadmap/THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md)：CA-PSA / CATET / ArcSurv 正式跨癌种计划；
- 尚未提交的孵化稿属于本地工作资产，验证并纳入版本控制前不进入稳定索引。

路线图中的潜力排序是时间快照，不能覆盖 [METHODS.md](METHODS.md) 的当前状态。

## 状态词

- **Primary**：当前论文主线；
- **Candidate**：正式评估候选；
- **Repair**：旧结论后正在通过修复闸门重新验证；
- **Research**：消融、机制比较或孵化分支；
- **Reference/Historical**：基线和复现档案。

新增文档时，先选择上述职责之一；不要再创建新的“总览”“最终总结”或“唯一主线”文件。
