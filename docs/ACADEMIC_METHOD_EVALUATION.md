# SurvOT-Rank 方法学术潜力与撞题审计

> 最后更新：2026-07-21
> 文献检索截止：2026-07-21
> 仓库快照：`311cd88`
> 评价对象：`survot_rank/training/model_factory.py` 中注册的全部方法，以及有实验记录的历史版本  
> 结果主来源：`EXPERIMENT_SUMMARY.md`；未进入仓库的服务器汇总只标为“暂定”  
> 用途：以后每次要求“评价所有方法”时更新本文件，不用重新建立另一套口径

## 1. 先给结论

当前注册表共有 **12 个方法标识**；其中 V45、V45v2、V50 共用同一演进实现目录，因此磁盘上对应 **10 个正式方法目录**。本文按方法标识分别评价，避免共用目录掩盖版本差异。

目前不应该把所有实现都包装成独立论文。它们多数是同一研究路线的连续演化，拆开投稿会有明显的“切香肠”风险。

论文优先级应固定为：

1. **DCT v3.3：当前第一主线，但不是已经安全的高创新论文。** 仓库内分数最高，目标函数也已简化到两个核心项；论文只能把贡献限定为“删失风险集驱动的 transport 分布敏感性”，不能把 prototype、OT、IPCW ranking 或反事实分别当创新，更不能声称因果反事实。
2. **CA-PSA：第二主线，但缝合风险高于 DCT。** SlotSPE、AdaSlot、BO-QSA、Dual-State Slot Attention、SurvQ 已分别覆盖其主要部件。只有“队列共享身份 + 患者状态 + 生存监督下自适应激活”被证明是不可拆分、且带来新的身份稳定性能力时，才不是模块拼接。
3. **CATET：高创新、低性能的储备线。** 问题定义清楚，但现有分数不足以支撑主论文；只有恢复性能并证明干预解释可靠后才值得继续。
4. **V50：强经验基线，不是最佳主论文。** 分数和稳定性尚可，但损失较多，而且已有拆解结果没有证明 time-local/coverage/competing 三项带来收益。
5. **其余方法：作为基线、消融或历史演进保留。** 不建议分别投稿。

## 2. 评价口径

### 2.1 分数不是录用概率

- “当前可投档次”按**现在已经落盘的证据**判断。
- “补齐后合理上限”是假设缺失实验全部完成且结论稳定，不是承诺。
- 期刊分区会随年份、学科和中科院/JCR 口径变化；本文的 Q1–Q4 是研究质量档次判断，不对应某一本期刊的永久分区。
- CCF A/B/C 只用于计算机会议档次判断。医学期刊与生物信息学期刊不能机械换算为 CCF。
- 单一 BLCA、单次 5-fold、每折挑选 best epoch，只足以做开发期比较；对高档投稿还需要独立重复、置信区间、显著性检验和外部/多癌种验证。

### 2.2 严格评分维度

每条方法按 10 分制观察五项：

- **创新性**：是否提出了新的、不可被一个现成模块替换的问题与机制。
- **证据强度**：结果完整性、稳定性、消融、统计检验和跨癌种验证。
- **叙事清晰度**：核心机制能否用一条因果链解释，损失是否过多。
- **撞题风险**：越高表示越接近已有论文，不等同于抄袭。
- **过度宣称风险**：是否容易把相关性、注意力或模型敏感性写成因果/生物机制。

### 2.3 “抄袭”与“撞题”不是一回事

当前代码和文档审计**没有发现可以据此认定的文字或代码抄袭证据**。但没有做全仓库逐文件许可证/代码指纹取证，所以不能给出法律意义的“绝对无抄袭”保证。

学术上的真实风险主要是：

- 使用 OT、Slot Attention、IPCW ranking、Hard-Concrete 或稀疏模块本身都不是新贡献；
- 若论文只把已有模块串联，会被审稿人判断为增量组合；
- 名称中出现 `counterfactual`、`faithful`、`evidence` 时尤其容易过度宣称；
- 同一主干的多个版本若分别投稿，可能构成重复发表或切香肠式投稿风险。

## 3. 总表：当前证据与投稿判断

| 优先级 | 方法 | 当前最好证据 | 活跃损失项* | 创新性 | 撞题风险 | 当前可投档次 | 补齐后合理上限 | 定位 |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | **DCT v3.3 score-first** | BLCA `0.7311±0.0293`；BRCA `0.7062±0.0420`；LUAD `0.7100±0.0348`；LUSC `0.6254±0.0364` | 2 | **6.5/10** | **中高** | Q3；证据补强后可试 Q2 | Q1/Q2；CCF B 有条件，CCF A 不现实 | 第一主线；必须证明特有机制有效 |
| 2 | **CA-PSA** | 服务器暂定 5-fold best mean `0.7217`，尚未纳入主结果档案 | 3 | **6.0/10** | **高** | Q3；当前不宜声称完成论文证据 | Q1/Q2；CCF B/C 有条件 | 第二主线；缝合风险高于 DCT |
| 3 | **V50 Time-local Competing** | 5-fold best `0.7148±0.028`；last5 `0.6572±0.012` | 8 | 5.0/10 | 高 | Q3/Q2 应用型 | Q2；CCF C | 强基线 |
| 4 | **V45v2** | 5-fold best `0.7063±0.035`；last5 `0.6394±0.028` | 配方可变、分支多 | 4.0/10 | 高 | Q3/Q4 | Q2/Q3 应用型 | 历史基线 |
| 5 | **V45 RankEvent** | 5-fold best `0.6848±0.041`；last5 `0.6406±0.028` | 历史完整配方最多 9 | 3.5/10 | 高 | 不建议独立投稿 | Q3（仅应用型） | 历史演进 |
| 6 | **V60 OT Event Rank** | 5-fold best `0.6790±0.054`；last5 `0.6063±0.035` | 约 4 | 4.5/10 | 高 | Q4/Workshop | Q3；CCF C 边缘 | 基线；停止主推 |
| 7 | **V70 PSPC** | 服务器暂定 5-fold best mean `0.6786`，尚未入主档案 | 3 | 5.5/10 | 高 | Q4/Workshop | Q2/Q3；CCF C | 暂停，除非性能跃升 |
| 8 | **Stagewise Prognostic Transport** | 仅 fold2 best `0.6741`，非完整 5-fold | 约 4 | 5.5/10 | 中高 | 不可投稿 | Q2/Q3 | 假设/消融线 |
| 9 | **CATET** | 5-fold best `0.6534±0.079`；last5 `0.5474±0.032` | 4 | 7.0/10 | 中 | 暂不可投；最多 Q4 概念稿 | Q2；CCF C，若性能和解释性同时成立 | 高创新储备 |
| 10 | **Faithful Evidence Transport** | 5-fold best `0.6519±0.080`；last5 `0.5892±0.060` | 约 6 | 6.0/10 | 中 | Q4/不建议 | Q2/Q3 专门方向 | 解释性消融线 |
| 11 | **Rank-Guided Event Transport** | 5-fold best `0.6495±0.076`；last5 `0.5923±0.087` | 4 | 5.5/10 | 高 | Q4/不建议 | Q3 | 机制基线 |
| 12 | **OT Event Hazard V2** | 没有独立、完整、可比的新协议结果 | 多项辅助损失 | 3.0/10 | 高 | 不可独立投稿 | Q4/Q3 应用型 | 架构起点 |

\* “活跃损失项”按当前代表性配置/设计目标统计，不把内部实现中仅用于计算、但权重为 0 的项算进去；配置变化时必须重新审计。

### 结果口径提醒

- DCT 的正式 BLCA 结论以最新复现归档为准：best `0.7311±0.0293`、last5 `0.6453±0.0706`；fold2 使用 NaN 修复后的重跑结果，逐折日志、配置、环境和数据哈希已进入 `reproducibility_archives/`。旧记录中的 `0.7328` 不再作为当前主结果。
- DCT v3.3 多癌种结果已更新：BRCA `0.7062±0.0420`、LUAD `0.7100±0.0348`、LUSC `0.6254±0.0364`。其中 v3.3 未启用 train-only bins，不能与严格协议下的新结果直接等同。
- DCT v3.4 BRCA event-aware 配方已在 fold0 epoch21 暂停：best `0.6189`。其失败来自有放回事件采样、`alpha_surv=2/3`、rank memory 与随机验证 slots 的叠加，不是“损失项数量过多”。
- DCT v3.5 R/Q/G/L 已按单变量原则进入 fold0/2 筛选；完整协议见 [`docs/DCT_V35_SCREENING.md`](DCT_V35_SCREENING.md)。
- DCT v3.5 R/Q/G/L 是**诊断变体，不是四个论文方法**：R 修复验证随机性与有放回采样偏差；Q 检验 learned queries 是否改善 slot 绑定；G 检验 evidence-conditioned marginal 是否有独立价值；L 检验 30M 级容量是否是过拟合主因。代码问题已处理，但在 fold0/2 结果出来前不能声称科学问题已经解决。
- DCT fold1 有过多进程污染记录。论文前必须用锁定代码与环境重跑或至少提供可核验的干净重复。
- CA-PSA 与 V70 的数字目前来自服务器汇总，原始逐 epoch 曲线、配置快照、checkpoint 元数据尚未正式进入本仓库，因此只能标为暂定。
- `dct_fix`、DCT v3、v3.2 属于历史演进/消融，不应被包装成三条独立方法。
- 仓库同时存在 `v60_ot_event_rank` 和被服务器简称为“V60”的 CA-PSA。以后禁止只写“V60”，必须写完整方法名，避免结果串线。

## 4. 逐项严格评价

### 4.1 DCT v3.3 — Distributional Counterfactual Transport

**核心想法。** 全局 WSI/pathway prototypes 提供跨患者可比坐标；训练折内估计时间阶段与删失分布；使用 IPCW 生存排序；原始 DCT 机制还包含 evidence-conditioned marginals、风险 anchor 干预和重新求解 Sinkhorn。v3.3 为追求稳健分数，将训练目标简化为：

`NLL + 0.10 × IPCW pairwise ranking`

OT、anchor、stage-risk、coordinate 等辅助项在 score-first 配方中关闭。

**优点。** 当前分数最高；目标函数从复杂多损失收敛到两个主项；“全局原型坐标 + 删失感知排序 + post-hoc transport sensitivity”可以形成相对完整的论文故事。

**致命审稿问题。** 最高分主要来自 score-first 目标，而不是原始反事实 transport 训练项。审稿人会问：如果所有 DCT 特有损失均关闭，提升究竟来自 DCT 表征、IPCW ranking，还是普通容量/训练技巧？这必须靠结构消融回答。若 post-hoc intervention 既不参与训练，也没有方向一致性、剂量响应和随机 anchor 阴性对照，它会被评价为“标准高分预测器后附加一个解释模块”，这是当前最强的“缝合”证据。

**v3.5 到底解决了什么。** `311cd88` 已完成工程与实验设计层面的四个单变量修复，但尚未由结果证明：

- **R** 直接修复同一 checkpoint 重复验证会因随机 slot 初始化而改变排序的问题，并去掉有放回采样导致的患者覆盖偏差；这是正确性基线，必须保留。
- **Q** 用 learned per-slot queries 检验跨样本绑定稳定性；learned query 本身已有 BO-QSA/SurvQ 近邻，不能作为新贡献。
- **G** 只检验 evidence gate 改变 OT marginals 是否有价值；它不自动证明 evidence 具有生物意义或解释忠实性。
- **L** 只检验缩小维度/层数能否缓解过拟合；它是容量对照，不是创新模块。

因此，“那几个方法”目前是**代码已解决、实验结论未解决**。只有 R 先通过确定性复验，且 Q/G/L 在预先规定的 fold0/2 筛选中形成稳定差异，才能决定最终 DCT 配方。

**BRCA 低分的定位（2026-07-20）。** 最新归档显示，BRCA 的问题主要不是 BLCA 代码无法迁移，而是数据分布与训练协议不匹配：

- BRCA DSS 只有 98/1046 个观测事件（9.4%），BLCA 为 129/381（33.9%）；每个 BRCA 验证折只有 10–28 个事件，C-index 方差天然更大。
- 训练集每折约 835 人，但仍使用约 30.4M 参数、batch 8、50 epochs、固定 `lr=5e-4` 和 `weight_decay=5e-4`；相当于把 BLCA 的训练时长和容量直接迁移到一个重删失癌种。
- BRCA 五折 best epoch 为 42、26、16、7、5；对应 last5 mean 为 0.4207、0.5859、0.4508、0.5202、0.5231。五折都出现 best-to-last5 明显下降，平均下降约 0.19，说明后期在拟合验证集偶然排序/训练集表示，而不是稳定学习风险关系。
- DCT 的 IPCW ranking 是按 batch 内可比 pair 计算。BRCA 事件率低时，大多数 batch 的有效事件排序信号很少，且 censoring KM 的尾部权重更敏感；这会使梯度更噪、更容易被高容量 WSI 表征放大。
- 配置中没有启用 early stopping；best epoch 只用于汇报，训练仍固定跑满 50 epochs。因此 `best=0.6886` 是开发期峰值，`last5=0.5001` 才暴露了当前配方的不稳定性。

目前没有证据证明 BRCA 是标签反转或 split 泄漏：代码明确按 `c=0` 观测事件、`c=1` 删失处理，且 train-fold reference 拟合路径正确。仍需额外排查 WSI 缺失/多 slide 聚合和 BRCA patch 质量，不能把所有损失归因于模型。

**修复优先级。** 先做不改模型的诊断：按 fold 输出事件数、IPCW pair 数、censoring KM 曲线、WSI 缺失率和事件时间分布；然后比较 early stopping、有效 batch 增大（梯度累积）、降低学习率/训练轮数、冻结或缩小 WSI encoder。只有这些协议修复后 BRCA 仍然低，才考虑 BRCA 专门的 loss 或 stage 设计。不能直接用 BRCA 的 best epoch 重新调参后再声称跨癌种泛化，必须把协议预先固定并在所有癌种一致执行。

**最接近工作与边界。** [MOTCat](https://arxiv.org/abs/2306.08330) 已把 OT 用于 WSI–genomics 生存对齐；[MMP](https://arxiv.org/abs/2407.00224) 已使用形态/通路 prototypes 与 OT cross-alignment；2026 年的 [ProtoPathway](https://arxiv.org/abs/2605.21454) 又进一步覆盖了可学习形态 prototypes、Reactome pathway 表征、稳定跨模态对应和内生解释；删失生存的 learning-to-rank 也不是新问题（例如 [Learning to Rank for Censored Survival Data](https://arxiv.org/abs/1806.01984)）。[CURE](https://arxiv.org/abs/2602.19987) 已直接使用“multimodal counterfactual time-to-event”表述，而 AISTATS 2025 Oral 的 [DISCOUNT](https://proceedings.mlr.press/v258/you25a.html) 已明确提出“distributional counterfactual explanations with optimal transport”。因此 DCT 不能把“prototype”“pathway fusion”“OT”“ranking”“distributional counterfactual”或“counterfactual survival”单独当创新点，只能主张这些组件在**训练折删失风险集驱动的 transport 干预与重新耦合敏感性机制**上的统一设计。

**风险。** 撞题中高；过度宣称风险高；当前缝合观感约 **7/10**。定向检索尚未发现完整同构的“训练折阶段/删失风险集参考 → cost-space intervention → evidence-conditioned re-Sinkhorn → 生存输出变化”链条，所以不是“整条方法已经被撞”。但 `counterfactual` 必须定义为 model-based transport intervention/sensitivity，不能写成治疗因果效应，也不能暗示可识别的个体反事实生存时间。更安全的名称是 **Censoring-Aware Distributional Sensitivity Transport**。

**投稿判断。** 现在直接投稿只能按 Q3 或 Q2 边缘看待。BLCA 结果已经完成可复现归档，但 BRCA 的 `0.6886` best 与 `0.5001` last5 暴露出跨癌种泛化和训练稳定性问题。完成多癌种、干净重复、结构/目标拆解、校准和敏感性真实性验证后，可形成 Q1/Q2 生物信息/医学 AI 稿件；若强调算法并提供大规模严谨验证，可尝试 CCF B。当前证据不支持 CCF A。

**必须补的实验。** 

1. 冻结同一训练协议，至少 3 seeds × 5 folds，并报告 bootstrap 95% CI 与配对显著性。
2. 继续对标 SlotSPE 的癌种；BRCA 已显示当前配方不能直接宣称跨癌种泛化，必须分析并修复其 28.7% 的 best-to-last5 下降。
3. 结构消融：local slots / global prototypes / DCT backbone / IPCW rank 分开；尤其比较“普通 backbone + 同一 IPCW rank”。
4. 机制消融：no-anchor、no-stage、no-evidence-marginal、no-re-Sinkhorn、随机 anchor、训练折 KM 与错误全数据 KM。
5. 分数之外报告 time-dependent AUC、IBS、校准曲线、风险分层 log-rank；评估参考量必须只在训练折拟合。
6. 保存每折 best checkpoint、epoch、配置、seed、commit、数据 split 哈希与依赖版本，重建一键复现实验清单。

### 4.2 CA-PSA — Cohort-Anchored Adaptive Prognostic Slot Attention

**核心想法。** 每个 slot 拆成队列共享 anchor 和患者特异 state，同索引跨模态融合；Hard-Concrete 门控决定每名患者激活的 slot 数量；目标为 `NLL + sparse + align`。

**优点。** 用一个统一机制同时回应三个真实缺陷：跨患者 slot 身份不稳定、跨模态需要事后猜配对、固定 slot 数量缺乏患者适应性。三个损失比 V45/V50 清晰。

**撞题情况。** 风险高，不是空白区：[SlotSPE](https://arxiv.org/abs/2512.01116) 已将 slot-based prognostic event、选择性激活和跨模态重建用于多模态生存；[AdaSlot](https://arxiv.org/abs/2406.09196) 已提出动态 slot 数量；[BO-QSA](https://openreview.net/forum?id=_-FN9mJsgg) 已用可学习 query 改善 slot 初始化与绑定稳定性；[Dual-State Slot Attention](https://arxiv.org/abs/2606.12601) 已明确拆分稳定 identity 与局部 state；[SurvQ](https://openreview.net/forum?id=4oA5xPOTmy) 已把可学习 queries 用于多模态癌症生存；ProtoPathway/FeatProto 等 2025–2026 工作又使“稳定 prototype 身份 + WSI/omics 解释”更加拥挤。实时检索未发现把三部分以 CA-PSA 的完整形式同时用于 WSI+omics 生存的论文，但这只能写成“据检索未发现完全相同机制”，不能写“全球首创”。

**致命审稿问题。** CA-PSA 是否只是 SlotSPE + AdaSlot + Dual-State 的组合？共享 anchor 是否真的形成稳定、可复现的预后身份，还是仅仅同位置参数共享？Hard-Concrete 是否只是稀疏正则而非患者自适应发现？

**投稿判断。** 暂定 BLCA best mean `0.7217` 有竞争力，但尚不足以独立成文。当前创新性按 **6.0/10**、缝合风险按 **8/10** 看待。补齐十癌种、身份稳定性和动态 slot 的必要性验证，且证明三部分形成不可替代的统一机制后，合理目标才是 Q1/Q2 或 CCF B/C；若只给 BLCA C-index 和常规消融，最多 Q3/Q2 边缘。

**必须补的实验。** 

1. SlotSPE 同癌种、同特征、同 split、同 best-epoch 规则的公平复现。
2. 固定 8 slots、动态 slots；随机初始化、独立模态 anchors、共享 anchors；无 state 分解；无 align；无 sparse 的全因子消融。
3. 跨 seed/跨 fold 的 slot identity 一致性、同一 anchor 的 pathway/形态富集一致性、门控数量分布。
4. 用相同参数量的普通 learnable queries、Perceiver/Set Transformer 作为容量对照。
5. 缺失模态、低样本量、噪声 pathway、WSI patch subsampling 的鲁棒性。

### 4.3 V50 — Time-local Competing Prognostic Events

**核心想法。** 在事件 transport 主干上加入时间特异性、事件覆盖和竞争正则。

**优点。** `0.7148±0.028` 且 last5 `0.6572±0.012`，是当前最稳定的强基线之一。

**问题。** 代表性 no-rank 配方仍有约 8 个活跃目标：外层 NLL、OT、diversity、event survival、reconstruction、time-specificity、coverage、competing。故事容易被认为是 regularizer stacking。更关键的是，已有拆解没有证明后三个 V50 专属项稳定增益，因此当前高分不能自动归因于 V50 的核心创新。

**投稿判断。** 作为主论文，当前大致 Q3/Q2 应用型；做完整多癌种可到 Q2 或 CCF C，但不建议与 DCT/CA-PSA 竞争主线。最有价值的角色是强基线和“复杂多损失未必优于简洁目标”的对照。

### 4.4 V45v2 — Clinical/Three-way RankEvent v2

**优点。** `0.7063±0.035`，具备可用性能。

**问题。** 临床编码、三路融合、解耦、自适应路由和可学习权重形成太多可选分支，难以界定哪一项是论文贡献。若临床变量并非所有癌种一致可得，还会破坏公平对比。

**投稿判断。** 当前 Q3/Q4；若只做应用整合可争取 Q2/Q3，但不应作为算法主线。保留作“更多模块不等于更好”的历史对照。

### 4.5 V45 — OTEHV2 RankEvent

历史完整版本最多包含 9 个损失/正则项，属于典型的目标函数堆叠。`0.6848±0.041` 没有显示出足以抵消复杂度的收益。与 SlotSPE、通用事件建模、ranking survival 都有邻近。

**投稿判断。** 不独立投稿；只作为 V50、DCT 的架构演化与损失简化消融。

### 4.6 V60 OT Event Rank

**核心想法。** log-domain Sinkhorn 形成事件表示，结合事件级生存监督与删失感知排序。

**问题。** OT 生存融合已被 MOTCat 占据，ranking survival 也很成熟；`0.6790±0.054` 既不领先也不稳定。方法的独特性不足以单独支撑论文。

**投稿判断。** 当前 Q4/Workshop；完整强化后最多 Q3/CCF C 边缘。建议停止主推，保留为紧凑 OT-event 基线。

### 4.7 V70 — Patient-Specific Prognostic Circuits

**核心想法。** 不使用 OT/Slot，而是学习患者条件化的稀疏可复用模块图；目标为 `NLL + node sparse + edge sparse`。

**撞题情况。** [Neural Attentive Circuits](https://arxiv.org/abs/2210.08031) 已联合学习稀疏模块与连接结构。把该思想迁移到生存任务具有应用价值，但若没有生存特有的可识别机制与显著性能提升，很容易被视为领域迁移。

**投稿判断。** 暂定 `0.6786` 不足以继续扩大投入。当前 Q4/Workshop；只有在多癌种明显超过 DCT/CA-PSA、并证明 circuits 的稳定临床含义后，才有 Q2/Q3 或 CCF C 可能。

### 4.8 Stagewise Prognostic Transport

**核心想法。** 不同生存阶段使用不同的 cost/transport plan。

**问题。** 只有单折 `0.6741`，证据不完整；阶段边界、删失处理和多次比较都可能造成不稳定。阶段特异 OT 是合理假设，但目前只是 DCT/CATET 的中间机制，不足以独立成文。

**投稿判断。** 当前不可投稿；补齐后最多 Q2/Q3，前提是多个癌种都显示阶段 plan 可解释且优于共享 plan。

### 4.9 CATET — Censoring-Aware Temporal Evidence Transport

**核心想法。** 时间边缘风险改变 OT cost，evidence gate 改变 transport，风险集监督处理删失，并做干预敏感性分析。

**优点。** 问题定义与机制链比 V45/V50 清晰，创新性位列前列。它针对“注意力/transport 权重是否真实影响预测”的质疑；已有研究已指出注意力解释可能不忠实，例如 [On the Relationship between Explanation and Prediction](https://arxiv.org/abs/2201.12114)，因此问题重要。

**问题。** `0.6534±0.079` 且 last5 `0.5474±0.032`，性能和稳定性都不足。高创新不能替代基本预测有效性；如果 intervention 只测模型内部变化，也不能宣称临床或生物因果解释。

**投稿判断。** 目前暂不可投，概念稿最多 Q4。若恢复到 DCT 级别性能，并用 deletion/insertion、随机化、反事实一致性和专家富集证明解释质量，可达 Q2 或 CCF C；Q1 需要跨癌种持续成立。

### 4.10 Faithful Evidence Transport

**核心想法。** evidence gate 实际改变 OT plan，并用 keep/remove 干预、稀疏和 faithfulness 目标约束解释。

**问题。** 方向合理，但“干预后预测变化”是解释性领域的常见评估思路，不足以单独构成高创新；约 6 项损失、`0.6519±0.080` 和较大方差进一步削弱论文性。

**投稿判断。** 当前 Q4/不建议；若转为专门的医学 XAI 论文，建立严格 faithfulness benchmark 和病理/通路专家验证，可能达到 Q2/Q3。

### 4.11 Rank-Guided Event Transport

**核心想法。** feature cost 与 prognostic pair cost 联合构建 transport，并加入连续风险排序和 stage order。

**问题。** MOTCat 已覆盖 OT 多模态生存，删失 ranking 已是成熟路线。当前 `0.6495±0.076` 没有证明二者结合的必要性，且方差大。

**投稿判断。** 不建议独立投稿；保留为“直接把 ranking 注入 OT”的负面/机制基线。

### 4.12 OT Event Hazard V2

这是整个 event-transport 家族的架构起点：WSI/omics slots、多个 OT cost、event tokens、Transformer 与 hazard 输出。它对工程演化重要，但和 MOTCat、SlotSPE 及通用多模态生存融合高度邻近，也缺少独立新协议结果。

**投稿判断。** 不独立投稿，只作祖先基线和结构图中的版本起点。

## 5. 撞题地图：哪些表述已经不能直接当创新点

| 我们可能使用的表述 | 已有近邻 | 严格判断 | 安全写法 |
|---|---|---|---|
| “首次用 OT 做 WSI+omics 生存” | [MOTCat](https://arxiv.org/abs/2306.08330) | 已撞，不能写 | 强调新的 cost、删失估计、阶段干预或敏感性机制 |
| “首次用 prototypes 对齐形态与通路” | [MMP](https://openreview.net/forum?id=3MfvxH3Gia) | 已撞，不能写 | 强调全局坐标如何服务于跨患者可比干预 |
| “稳定形态 prototype 与 pathway 对应天然可解释” | [ProtoPathway](https://arxiv.org/abs/2605.21454) | 2026 年近邻已覆盖 | DCT 不把稳定 prototype 本身当贡献，只把它作为风险集 transport 干预的坐标系 |
| “首次用 slots 建模多模态预后事件” | [SlotSPE](https://arxiv.org/abs/2512.01116) | 已撞，不能写 | 强调共享身份/患者状态的明确分解与可验证稳定性 |
| “每个患者动态选择 slot 数量” | [AdaSlot](https://arxiv.org/abs/2406.09196) | 通用机制已撞 | 强调生存监督下跨模态同身份激活，而非动态数量本身 |
| “可学习 query 保持 slot 身份” | [BO-QSA](https://openreview.net/forum?id=_-FN9mJsgg) | 邻近 | 证明队列级预后身份，不只改善初始化 |
| “identity 与 patient state 分开” | [Dual-State Slot Attention](https://arxiv.org/abs/2606.12601) | 概念高度邻近 | 明确跨模态生存场景的新约束、监督与验证指标 |
| “queries 用于多模态癌症生存” | [SurvQ](https://openreview.net/forum?id=4oA5xPOTmy) | 已撞 | 不能把 query 本身作为贡献 |
| “稀疏可复用模块连接” | [Neural Attentive Circuits](https://arxiv.org/abs/2210.08031) | 通用架构已撞 | 需要生存特有机制与显著临床证据 |
| “counterfactual multimodal survival” | [CURE](https://arxiv.org/abs/2602.19987) | 术语与方向已被使用 | DCT 明确写 model-based transport sensitivity，避免治疗因果声称 |
| “distributional counterfactual + optimal transport” | [DISCOUNT](https://proceedings.mlr.press/v258/you25a.html) | 方法名称和大方向已被占用 | 避免把 DCT 名称本身当创新，突出 censoring-aware risk-set intervention 与 re-coupling |
| “可解释/解耦 WSI+transcriptomics” | [PIBD](https://openreview.net/forum?id=otHZ8JAIgh) 等 | 宽泛表述已拥挤 | 给出可证伪的解释忠实性和身份稳定性定义 |

## 6. 论文组合建议

### 论文 A：DCT 主论文

建议标题口径：**Censoring-Aware Distributional Sensitivity Transport for Multimodal Cancer Survival**。

贡献只保留三点：

1. 用跨患者共享 prototype 坐标定义可比较的多模态预后表示；
2. 用训练折删失分布支持 score-aligned 生存学习；
3. 用重求 transport 的 post-hoc intervention 衡量模型分布敏感性。

不要把 V45/V50/CATET 的所有历史损失重新塞回来。论文真正的危险不是“损失太少”，而是**最高分配方与论文声称的 DCT 特有机制脱节**。结构消融必须证明 DCT 表征仍然必要。

### 论文 B：CA-PSA 主论文

建议标题口径：**Cohort-Anchored Patient-Adaptive Prognostic Slots for Multimodal Survival**。

核心贡献必须写成一个统一机制，而不是三个模块：

> 队列 anchor 定义稳定的预后身份，患者 state 表达个体异质性，生存监督门控决定该身份在每名患者中的实际激活。

CA-PSA 与 DCT 可以是两篇不同论文，但必须有不同的问题定义、核心图和主要实验；不能只是替换 backbone 后复用全部主张。

### CATET/V50/其余线

- CATET：作为未来解释性专项，先解决性能；暂不投稿。
- V50：进入所有主论文的强内部基线；不要再增加辅助损失。
- V45/V45v2/V60/RG-ET/Stagewise/FET：组成演进、消融和负结果附录。
- V70：暂停；只有相对 DCT/CA-PSA 出现明确跨癌种优势再恢复。

## 7. 现在离“可以写论文”还缺什么

可以开始写方法和相关工作，但现在还不能把结果部分视为完成。最低发表包应包含：

1. **数据与协议对标**：按 SlotSPE 的癌种、特征与 split 逐项对标；若无法完全相同，明确列出差异。
2. **强基线**：SlotSPE、MOTCat、MMP、PIBD、SurvQ，以及本仓库 V50/DCT/CA-PSA 的同协议比较。
3. **统计严谨性**：至少 3 seeds；每折固定验证选择规则；报告 mean±std、95% CI、配对检验和效应量。
4. **完整指标**：C-index 之外加入 time-dependent AUC、IBS、校准、风险分层。
5. **机制消融**：每个论文贡献都有唯一、直接、参数量匹配的对照。
6. **鲁棒性**：缺失模态、低样本、噪声、不同 slot/prototype 数、不同 censoring 比例。
7. **可解释性真实性**：不能只展示好看的热图；要有删除/插入、随机化、跨 seed 稳定性和通路/病理专家验证。
8. **复现包**：锁定 commit、环境、split 哈希、配置、seed、逐 epoch 日志、best checkpoint 与汇总脚本；禁止手工复制最优分数。
9. **数据泄漏审计**：所有 KM/IPCW、时间边界、归一化、cutoff 和参考风险都只能由训练折拟合。
10. **命名与结果治理**：清除“V60”歧义；结果目录必须带完整 method id、版本、癌种、seed 与 fold。

## 8. 下一批实验的停止/继续规则

### 立即做

1. 基于已提交的 DCT BLCA/BRCA 归档，先分析 BRCA 过拟合来源，再决定统一正则化、早停和癌种特定超参是否允许进入正式协议。
2. 将 CA-PSA 的逐折曲线、配置和 checkpoints 汇总进主档案，确认 `0.7217±0.0383` 可复算。
3. 在完全相同协议下跑 DCT、CA-PSA、V50、SlotSPE 四条主比较线。
4. 先在 BLCA 做小规模结构消融，确认贡献成立后再扩到全部癌种。

### 有条件做

- CATET：只有简化后 BLCA best mean 达到至少 `0.70` 且方差明显下降，才扩癌种。
- V70：只有 BLCA 稳定超过 `0.71`，或出现 DCT/CA-PSA 不具备的强临床解释证据，才继续。

### 不再做

- 不再给 V45/V50 叠加新的辅助损失；
- 不为每个历史 commit 单独跑完整论文实验；
- 不把同一 BLCA 5-fold 的小幅 best-epoch 波动当成新方法成功；
- 不用 test fold 挑 epoch 或调超参数。

## 9. 本文件的更新规则

以后更新本文件必须同时完成以下项目：

1. 更新顶部日期、检索截止日期和当前 git commit；
2. 从 `METHOD_REGISTRY` 重新盘点注册方法，新增/删除方法都要解释；
3. 只从可追溯的逐折结果生成总表，暂定服务器数字不得静默转正；
4. 每个结果记录：癌种、fold、seed、best epoch、best metric、last-k、配置路径、checkpoint、commit；
5. 新论文只加入“撞题地图”一次，并说明它改变了哪条方法的创新边界；
6. 不覆盖历史结论：若分数或评级改变，在更新日志说明原因；
7. 期刊/会议评级始终同时给出“当前证据”和“补齐后上限”；
8. 若出现代码来源复用，补充上游仓库、许可证、具体文件和修改范围。

## 10. 更新日志

- **2026-07-19**：创建首版。覆盖全部注册方法、DCT/CA-PSA/V50 等真实或暂定结果；完成 SlotSPE、MOTCat、MMP、AdaSlot、BO-QSA、Dual-State Slot Attention、SurvQ、NAC、CURE 等近邻工作检索；确定 DCT 与 CA-PSA 为两条主论文线。
- **2026-07-20**：同步 `21da4cf`。DCT BLCA 正式结果校正为 `0.7311±0.0293`，新增 BRCA `0.6886±0.0382`；确认复现归档已提交，并将 DCT 的主要风险从“结果可复现”更新为“重删失癌种上的训练协议不适配、跨癌种泛化与后期过拟合”。
- **2026-07-20（本地待提交修复）**：加入训练折分箱、fold 事件/离散 bin 日志、可选跨 batch IPCW 风险记忆和真正早停；新增 BRCA stable 与 no-rank control 配置。两者必须成对运行，不能只报告 stable 版本的峰值。
- **2026-07-21**：同步 `311cd88` 的 DCT v3.5 R/Q/G/L 严格筛选设计；明确四者只是诊断变体。新增 ProtoPathway 与 DISCOUNT 撞题审计，将 DCT 创新性由 `7.5` 下调为 `6.5`、撞题风险由“中”上调为“中高”，将 CA-PSA 创新性由 `7.0` 下调为 `6.0`、撞题风险上调为“高”。评分下调的原因是 2026 近邻工作进一步占用了 prototype/pathway/identity/counterfactual-OT 部件，而当前最高分配方尚未证明 DCT 特有干预机制带来可验证价值。
