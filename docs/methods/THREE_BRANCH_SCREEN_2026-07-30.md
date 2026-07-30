# ArcSurv / IST-Surv / Evidence Ledger 筛选记录（2026-07-30）

这份记录用于保存三条研究分支的工程状态与本地筛选证据。所有数值
均来自同一台 RTX 3060 Laptop GPU；本地只有 BLCA 的 UNI
（1024 维）特征，因此它们不是 UNI2-h 正式五折结果，也不能与论文
表格中的最终结果直接比较。

## ArcSurv

本轮首先修复了一个会令方法退化为常数预测的对称初始化问题：
`Beta` 全零会使所有 cohort archetype 完全重合，患者组成恒为
均匀分布，而体积正则在这个完全对称点的一阶梯度仍为零。当前使用
可复现的非对称凸权重初始化，并增加组成熵、组成方差与记忆库覆盖
诊断。

真实 BLCA 对照：

- 旧的全零初始化连续两轮验证 C-index 均为 `0.5000`；
- 修复后首轮验证 C-index 为 `0.5564`；
- 更强的初始化（2.0）为 `0.5120`；
- 更低的组合温度（0.10）为 `0.4388`；
- `alpha_surv=0.15` 的首轮对照为 `0.4922`，低于本次探索性
  事件聚焦配置（1.0）。后者会忽略删失似然，只能用于 idea
  筛选，不能直接作为论文正式协议；仓库主配置仍保留 0.15。

最终 20 epoch、fold-0 筛选使用
`beta_init_scale=1.5, temperature=0.25, alpha_surv=1.0`：

- 最佳验证 C-index：`0.6685`（epoch 6）；
- 最终 epoch：`0.5618`；
- 20 轮中共有 4 轮超过 `0.60`；
- 结论：存在可重复出现的预测信号，值得进入正式 5-fold，但曲线
  波动较大，而且患者组成熵接近最大值；下一步应修复组成辨识度，
  不能把这个探索性单折峰值作为论文结果。

## v4.0 IST-Surv

瘦身版保留两个核心干预视图：删除 WSI patch 和删除 pathway；
默认关闭重复的 risk-logit stability，只保留 transport-plan 与
signed-attribution stability。事实视图和两个干预视图现在合并成
一次批量 Sinkhorn，稳定运输再调用一次，因此每个 forward 只有
两个 Sinkhorn 批次。

在 `[B,R,C]=[4,1024,128]`、30 次 Sinkhorn 迭代下，本机计时：

- 逐视图求解：`0.1465 s`；
- 批量求解：`0.0607 s`；
- Sinkhorn 子过程加速：`2.415x`。

BLCA/UNI、512 patches、2 个真实训练 batch 的 smoke test 通过：
完整性误差与边际误差均为 0（日志精度下），
`ist_sinkhorn_batches=2`，无非有限值。

按用户要求，本地长跑在完成 3 个 epoch 后停止，以避免继续占用
笔记本显卡；中断前验证 C-index 为 `0.5036, 0.4161, 0.6307`，
最佳为 epoch 3 的 `0.6307`。这只是容量和方向性证据，不能代替
远端 UNI2-h 的完整 20 epoch 筛选。

## v4.1 Evidence Ledger

核心机制已从“补全完整模态”改为“只补全可恢复的低秩共享证据，
显式保留私有证据的不确定性”。缺失模态不会再伪造一份确定的完整
ledger；置信度同时受共享证据可恢复性和 private uncertainty
约束。

BLCA/UNI smoke test（1 epoch、2 个训练 batch）通过：

- completion loss：`0.2057`；
- private uncertainty calibration：`0.0594`；
- ledger conservation：`0.0044`；
- survival consistency：`0.0516`；
- auxiliary objective：`0.0141`；
- 全流程无非有限值。

按本轮计划，v4.1 只做核心机制升级与 smoke test，不进行 20 epoch
排名。

## 当前验证

- 三条方法定向测试：`21 passed`；
- 仓库完整测试（核心升级并兼容并行 v3.8 更新后）：
  `286 passed, 56 warnings`；
- 改动 Python 文件均通过单文件编译检查；
- 仓库 doctor 全部通过。

训练期间主分支并行加入了 v3.8 子进程 CUDA 校验，但意外覆盖了此前
防重复 fork 的 scheduler/fold 双层锁。当前代码已经合并两者：
既在训练子进程中验证 CUDA，也保留双层运行锁、split 审计与 robust
协议；相关 12 项回归测试通过。

## 后续远端执行顺序

1. 先运行 doctor，确认 split、RNA、clinical 与目标 WSI encoder
   路径完整；
2. ArcSurv 使用论文配置（保留删失似然）先做 BLCA fold-0 的
   20 epoch 复核；只有最佳点能在相邻 epoch 或第二个 fold
   复现，才扩到 5-fold；
3. v4.0 使用 UNI2-h、2048 patches、30 次 Sinkhorn、两个干预
   视图完成 BLCA fold-0 的 20 epoch 筛选；通过后再扩展 5-fold；
4. v4.1 维持当前 smoke-only 状态，先做 shared/private 机制消融，
   不与前两条分支争抢完整训练资源；
5. 所有论文判断使用 5-fold mean ± std、固定的验证集 checkpoint
   选择与 train-side 生存参考量，不使用单个峰值作结论。
