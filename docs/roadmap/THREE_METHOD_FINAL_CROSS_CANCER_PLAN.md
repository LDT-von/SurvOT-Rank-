# CA-PSA / CATET / ArcSurv 最终跨癌种五折运行计划

> 目的：每个方法只保留一个已筛选版本，在同一数据、特征、划分和训练预算下完成正式跨癌种五折实验。

## 1. 唯一运行版本

| 方法 | 冻结版本 | 选择依据 | 明确排除 |
|---|---|---|---|
| CA-PSA | `CA-PSA full 50ep` | BLCA 旧协议五折 `0.7217 ± 0.0383`，三者中现有证据和分数最好 | 不运行结构消融或额外门控变体 |
| CATET | `CATET repaired 50ep` | 当前可信结果为 BLCA fold0/2：`0.6458 / 0.6837`；使用单调 epsilon、物理 batch 16 和完整 rank/intervention 配方 | 不运行已作废的旧 `catet_fix` 五折 |
| ArcSurv | `ArcSurv staged 50ep` | BLCA fold1/2/4：`0.7132 / 0.6457 / 0.7141`，三折均值 `0.6910` | 不运行 hard-repaired 版本；其 fold1 仅 `0.5665` |

### ArcSurv 版本复现约束

当前 HEAD 已包含 furthest-point anchor 与 patient-composition sharpness 等 hard repair。直接运行当前默认行为并不等于已筛选的 `ArcSurv staged 50ep`。

正式 launcher 必须通过显式配置复现 staged 配方：

- `arc_warmup_epochs=5`
- `arc_ramp_epochs=10`
- `arc_bank_update_epochs=1`，只在首轮建立原型库
- `arc_lambda_sharpness=0.0`
- 关闭 memory 冻结后的 furthest-point anchor seeding
- `batch_size=8`

若关闭 anchor seeding 的配置开关尚不存在，必须先实现并测试；在此之前不得启动 ArcSurv 正式队列。

## 2. 统一数据协议

三种方法全部使用完全一致的正式协议：

- WSI 特征：UNI2-h，`encoding_dim=1536`
- 数据根目录：`/data1/TCGA-UNI2-h-features`
- split：`5fold_uni2h`
- 生存终点：DSS
- folds：`0,1,2,3,4`
- seed：`3`
- max epochs：`50`
- `fit_bins_on_train=true`
- `binning_mode=global_qcut`
- `event_stratified_batches=true`
- 患者级划分
- 不允许 UNI 回退，不允许缺失 WSI 静默零填充
- 不允许按癌种修改学习率、损失权重、slot数或其他方法参数

数据协议从 UNI v1 改为 UNI2-h 只属于统一评估协议，不产生新的方法版本。

## 3. 当前允许运行的完整癌种

根据 2026-08-06 的 UNI2-h 覆盖审计，当前正式队列包含：

1. BLCA
2. UCEC
3. KIRC
4. SKCM
5. HNSC
6. LUSC

BRCA、LUAD、COADREAD、STAD 暂不进入队列。只有在 UNI2-h 患者覆盖达到 100%，并建立对应 `5fold_uni2h` 后才能补跑；不得临时混用 UNI v1。

## 4. 任务数量与运行顺序

每个方法：

`6 cancers × 5 folds = 30 jobs`

三个方法总计：

`3 methods × 6 cancers × 5 folds = 90 jobs`

正式串行顺序：

1. CA-PSA：BLCA → UCEC → KIRC → SKCM → HNSC → LUSC，每个癌种 fold0 → fold4
2. ArcSurv staged：同上，共30个任务
3. CATET repaired：同上，共30个任务

该顺序只影响结果出现时间，不改变任何训练参数。每个已完成fold都必须支持安全跳过和断点续跑。

## 5. 方法固定参数

### 5.1 CA-PSA full

- `capsa_max_slots=16`
- `capsa_slot_iters=3`
- `capsa_heads=4`
- `capsa_dropout=0.15`
- `capsa_gate_temperature=0.6666667`
- `capsa_gate_gamma=-0.1`
- `capsa_gate_zeta=1.1`
- `capsa_gate_threshold=0.5`
- `capsa_gate_prior_start=-1.0`
- `capsa_gate_prior_end=-2.2`
- `capsa_lambda_sparse=0.01`
- `capsa_lambda_align=0.02`
- `batch_size=8`

### 5.2 ArcSurv staged

- `arc_num_archetypes=6`
- `arc_bank_size=256`
- `arc_temperature=0.25`
- `arc_beta_init_scale=1.5`
- `arc_lambda_recon=0.05`
- `arc_lambda_align=0.05`
- `arc_lambda_balance=0.01`
- `arc_lambda_volume=0.01`
- `arc_lambda_rank=0.10`
- staged复现参数以第1节为准

### 5.3 CATET repaired

- `catet_num_stages=4`
- `catet_prog_cost=0.20`
- `catet_lambda_ot=0.04`
- `catet_lambda_rank=0.08`
- `catet_lambda_intervention=0.05`
- `catet_keep_ratio=0.25`
- `catet_intervention_margin=0.05`
- `catet_rank_margin=0.0`
- `catet_rank_max_pairs=4096`
- `batch_size=16`

## 6. 启动前硬检查

每个癌种、每个fold启动前必须通过：

1. 精确子进程 Python/CUDA 分配检查；
2. UNI2-h 文件存在且最后一维为1536；
3. split患者全部具有WSI特征和组学数据；
4. train/validation患者无交集；
5. 训练折事件数、验证折事件数均大于0；
6. 训练折可比较生存pair不为0；
7. 磁盘空间、结果目录和运行锁正常；
8. 记录commit、配置快照、split哈希和特征覆盖摘要。

任何覆盖缺失都必须阻断当前癌种，不能用零向量继续。

## 7. 必须保存的结果

所有方法通用：

- `epoch_curve_fold*.csv`
- `split_*_results_final.pkl`
- 每折训练日志
- `model_parameters.txt`
- best checkpoint与best epoch
- best C-index、last5 C-index、IPCW C-index、IBS、integrated AUC
- train/validation患者数与事件数
- 配置、seed、commit和split哈希

方法诊断：

- CA-PSA：active slot count、gate概率、slot使用率、跨fold身份稳定性
- ArcSurv：archetype cosine、hazard spread、active archetype fraction、composition entropy/variance、bank count
- CATET：各stage plan差异、rank pairs、keep/remove风险变化、intervention margin、OT有限性

## 8. 结果汇总规则

每个癌种只在五折全部完成后生成正式均值：

- 报告五折 `mean ± std`
- 同时报告逐折分数和best epoch
- 标记best epoch是否早熟（epoch ≤ 5）或撞训练边界（epoch ≥ 47）
- 不将两折或三折中间均值写成最终结果
- 不混入旧UNI v1结果
- 不因为某癌种分数低而修改该癌种专属参数

最终形成三张 `6 cancers × 5 folds` 表，以及一张三方法同协议总对比表。

## 9. 实施步骤

1. 增加统一跨癌种launcher，支持 `plan / doctor / smoke / run / summarize`；
2. 增加 ArcSurv anchor seeding 显式开关，验证关闭后精确对应 staged版本；
3. 为三种方法建立 UNI2-h 1536d 配置覆盖；
4. 每种方法先做一个BLCA fold0、1 epoch smoke，只验证数据、前向、反向与诊断字段；
5. 通过后按照第4节严格串行执行90个正式任务；
6. 自动跳过已有完整结果，失败任务保留日志并从当前fold恢复；
7. 全部完成后更新 `FINAL_SUMMARY.md`，但旧结果保留并明确标记为历史协议。
