# DCT v3.8.2 提分任务与证据计划

## 目标边界

目标是在同一 UNI2-h、`5fold_uni2h`、train-fold-only 分箱和 50 epoch 协议下，
提高 DCT 的跨癌种表现并公平对照 SlotSPE。禁止为每个癌种选择不同方法权重，也不把
论文 SlotSPE 的 UNI/30ep/按癌种搜索 slot 数结果冒充同协议基线。

当前第一闸门选择 BLCA、KIRC、SKCM 的 fold 1/2/4：BLCA 用于防止已有优势退化，
KIRC 距 SlotSPE 很近，SKCM 是共同六癌种中的最大差距。第一阶段共 4×3×3=36 个任务。

## 版本、用途与可支持结论

| 版本 | 唯一变化 | 用途 | 通过后可以支持 | 不能支持 |
|---|---|---|---|---|
| `patches4096` | patch 2048→4096 | 检查病理输入是否受采样预算限制 | 更大病理采样预算有益 | DCT机制更有效 |
| `grad_accum4` | 累积步数1→4 | 检查参数更新方差 | 优化稳定性改善 | micro-batch内IPCW pair增加 |
| `slot_iters5` | slot迭代3→5 | 检查共享语义坐标是否欠迭代 | 更深slot细化有益 | transport损失有效 |
| `lr2e4` | 学习率5e-4→2e-4 | 检查尖峰与后期退化 | 较低学习率更稳定 | 新方法贡献 |
| `predictive_core` | 成组关闭4个未证实辅助项 | 检查辅助束对预测的净作用 | 辅助束整体不必要/有害 | 归因到某一个loss |
| `capacity_stable` | 合并第一阶段四项 | 晋级前联合确认 | 联合训练配方有效 | 单项贡献 |

`predictive_core`与`capacity_stable`属于第二阶段，只有第一阶段汇总后才能运行。

## 预注册晋级规则

候选版本必须在匹配的9个任务上全部产生有限曲线，并同时满足：

1. 相对 frozen `fixed_full` 的宏平均 best C-index 提升至少 0.005；
2. 三个癌种至少两个提高；
3. 任一癌种下降不超过 0.005；
4. SKCM 至少提高 0.005；
5. 宏平均 last-5 C-index 不下降。

未通过的版本停止，不用其他fold或癌种补救。通过的第一阶段因素进入联合确认；联合版本
通过后才扩展至当前六个特征完整癌种的5折（30任务）。BRCA、COADREAD、LUAD、STAD
必须在UNI2-h覆盖与clean split审计通过后才能加入最终十癌种队列。

## 运行命令

```bash
# 查看36个第一阶段任务及每个版本的证据用途
python scripts/run_dct_v382_score_gate.py plan

# 数据、split与配置检查
python scripts/run_dct_v382_score_gate.py doctor

# 每个版本只跑第一个fold的2-epoch烟雾测试
python scripts/run_dct_v382_score_gate.py smoke

# 正式第一阶段
python scripts/run_dct_v382_score_gate.py run

# 汇总并生成预注册晋级判定
python scripts/summarize_dct_v382_score_gate.py

# 只有第一阶段通过后运行第二阶段
python scripts/run_dct_v382_score_gate.py plan --variants phase2
python scripts/run_dct_v382_score_gate.py run --variants phase2
```

## SlotSPE公平对照阻塞项

现有外部 newSlotSPE 入口只接受 `uni/gigap/r50/chief`，且没有DCT当前的
train-fold-only分箱与缺失WSI硬错误开关。因此当前不能生成一个可声称“同协议”的SlotSPE
任务。完成以下兼容后，才登记10癌种×5折SlotSPE控制队列：支持`uni2-h`与1536维输入、
使用完全相同的`5fold_uni2h`、train-fold-only `global_qcut`、缺失特征立即失败、50 epoch，
并保存逐epoch曲线。论文原表只作为外部参照，不作为这个闸门的直接控制组。
