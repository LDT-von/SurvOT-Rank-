# Requirements Document

## Introduction

SurvOT-Rank 当前的最优方法 OTEHV2RankEvent（代号 V45，位于
`survot_rank/research/methods/prognostic_event_transport/model.py`，继承自
`OTEventHazardV2Survival`，位于
`survot_rank/research/methods/ot_event_hazard_v2/model_v2.py`）相对 SlotSPE
baseline 的 C-index 提升幅度较小（+0.0091，且仅在 BLCA 单一癌种上验证），同时存在以下可发表性风险：

1. 核心组件 `MultiHeadSlotAttention`（`survot_rank/research/components/slot_attention.py`）几乎是
   Locatello et al. (2020) Slot Attention 的原样复制，缺乏架构新颖性；
2. 训练目标由 per-event NLL 生存损失、Cox 式成对排序损失等多个独立加权项简单相加而成
   （`OTEHV2RankEvent.forward()`），是"拼装式"目标而非有原则的统一似然；
3. 模型只使用 WSI（病理全切片图像）和 Omics（转录组通路）两种模态，未纳入临床变量；
4. 每个 slot 是单一的整体向量，没有"身份（角色）与状态（内容）"的解耦表示；
5. 验证范围只覆盖 BLCA 一种癌症类型，尽管仓库中已具备 blca/brca/coadread/hnsc/stad 五种癌症类型的
   临床、RNA 通路和 5-fold 数据切分。

本特性将上述五个增强方向整合为一个统一的研究工程特性，目标是把 SurvOT-Rank 从一个 SlotSPE 的
渐进式改进，转变为具备可辩护原创性的贡献：三模态融合（新增临床编码器）、多癌种验证与横向对比、
统一的 NLL+排序目标、slot 身份/状态解耦表示、以及 Slot Attention 内部路由机制的重新设计
（Sinkhorn/OT 路由、跨模态条件化更新、自适应迭代次数任选一种或组合）。

所有变更必须保持对现有 V45 baseline 的向后兼容：现有 YAML 配置（如
`configs/v45_blca.yaml`）在不修改的情况下必须仍可运行并复现相同的模型结构与训练行为；新增能力
必须通过新增的可选配置字段开启，默认关闭时模型行为与当前 V45 完全一致。

本仓库是 PyTorch 研究代码库（非 Web/产品应用），因此本文档中的验收标准聚焦于机器学习代码的正确性
（张量形状/dtype 不变量、无 NaN/Inf、梯度可达性、退化输入下的损失行为、配置/检查点的向后兼容性），
而非业务逻辑测试。Windows 开发环境没有 WSI 的 `.pt` 特征文件，因此单元测试与属性测试必须使用合成的
小张量（synthetic tensors），完整训练验证只能在远程 Linux GPU 环境上进行。

## Glossary

- **SurvOT-Rank**: 本仓库的多模态（WSI + Omics [+ Clinical]）生存分析研究框架。
- **V45_Model / OTEHV2RankEvent**: 当前最优方法类，位于
  `survot_rank/research/methods/prognostic_event_transport/model.py`，训练配置别名为
  `otehv2_rankevent`。
- **Parent_Model / OTEventHazardV2Survival**: V45_Model 的父类，位于
  `survot_rank/research/methods/ot_event_hazard_v2/model_v2.py`。
- **Slot_Attention_Module**: `MultiHeadSlotAttention` 类，位于
  `survot_rank/research/components/slot_attention.py`，用于将 WSI patch 特征或 Omics 通路特征
  聚合为固定数量的 slot 表示。
- **WSI_Modality**: 全切片病理图像（Whole Slide Image）patch 特征模态，来自 `.pt` 特征文件。
- **Omics_Modality**: 基因表达/通路（Pathways）特征模态。
- **Clinical_Modality**: 患者临床变量特征模态（如年龄、分期、分级等结构化临床数据），当前作为
  第三模态被引入。
- **Clinical_Encoder**: 将 Clinical_Modality 原始特征编码为与 WSI/Omics 同维度的表示的轻量网络
  模块。
- **Event_Token**: OT 融合（`MultiScaleOTFusion`）输出的、代表隐含"事件类型"的融合表示，形状为
  `[batch, num_events, dim]`。
- **OT_Plan**: `log_sinkhorn_plan` 函数计算的最优传输方案矩阵，用于在 WSI slot 与 Omics slot（或
  纳入 Clinical slot 后）之间建立软对应关系。
- **Slot_Identity**: 每个 slot 固定或可学习的、跨前向传播不变的角色/身份嵌入（disentanglement
  方向新增概念），形状为 `[num_slots, dim]`，在同一 batch 内所有样本间共享。
- **Slot_State**: 每个 slot 在单次前向传播中由输入内容决定的、可变的状态表示（与 Slot_Identity
  相对），形状为 `[batch, num_slots, dim]`。
- **Unified_Objective**: 将当前"NLL + per-event NLL + ranking + global consistency + gate
  entropy"多项独立加权和，重新设计为原则性统一目标（如边际校准似然）后的训练目标函数。
- **Training_Runner**: `survot_rank/training/train_runner.py`，训练主循环入口。
- **Model_Factory**: `survot_rank/training/model_factory.py`，方法名到模型类的注册与加载器。
- **Method_Registry**: `Model_Factory` 中的 `METHOD_REGISTRY` 字典，记录方法名到
  `(模块路径, 类名)` 的映射。
- **Cancer_Study**: 一种 TCGA 癌症类型代码，本特性覆盖 `blca`、`brca`、`coadread`、`hnsc`、
  `stad` 五种。
- **Cross_Cancer_Report**: 汇总多个 Cancer_Study 训练结果（C-index/IBS/iAUC 等指标）的统一对比
  报告/脚本产出，包含 CSV 与 Markdown 两种格式。
- **Sinkhorn_Router**: 使用 log-domain Sinkhorn 迭代（复用 `log_sinkhorn_plan` 中的实现）替代
  Slot_Attention_Module 内部 softmax 竞争的路由机制。
- **Convergence_Criterion**: 用于判断 Slot_Attention_Module 迭代是否已收敛、从而提前停止迭代的
  度量，定义为相邻迭代 slot 张量之差的 L2 范数 `‖slot_t − slot_(t-1)‖₂`。

## Requirements

### Requirement 1: 临床模态编码器与三模态融合架构

**User Story:** 作为研究者，我希望在现有 WSI+Omics 双模态融合流程中新增一个临床变量编码器和第三模态
融合路径，以便模型能够利用患者临床信息进一步提升预后预测能力并增强论文贡献的完整性。

*背景假设*：本需求描述的是架构能力，前提假设是训练数据管线（`SurvivalDatasetFactory` /
`SurvivalDataset`，位于
`survot_rank/research/legacy/slotspe_runtime/dataset/dataset_survival.py`）会被扩展为能够
提供每个样本的 Clinical_Modality 原始特征张量（例如年龄、分期、分级等经过数值化/独热编码后的
向量）。具体临床字段来源、缺失值处理策略与数据补全方式留待设计阶段确定，不在本需求的验收范围内。

#### Acceptance Criteria

1. THE Clinical_Encoder SHALL 将形状为 `[batch, clinical_feature_dim]` 的 Clinical_Modality
   输入张量映射为形状为 `[batch, dim]` 的表示，其中 `dim` 等于 `wsi_projection_dim`（与 WSI/Omics
   投影维度一致）。
2. WHERE 配置中启用 Clinical_Modality（即存在显式的三模态开关配置项且为真），THE V45_Model SHALL
   将 Clinical_Encoder 的输出通过 Slot_Attention_Module 聚合为形状为 `[batch, num_slots_clinical,
   dim]` 的 Clinical_Modality slot 表示（其中 `num_slots_clinical` 为配置中指定的 Clinical 模态
   slot 数量，命名风格与现有 `slot_num_wsi`、`slot_num_omics` 保持一致），并参与 OT_Plan 计算与
   Event_Token 融合。
3. IF 配置中三模态开关已启用，但 Clinical_Modality 输入张量缺失（即调用方未提供该张量）或其张量
   形状与配置声明的 `[batch, clinical_feature_dim]` 不一致，THEN THE V45_Model SHALL 抛出明确指出
   "Clinical 输入缺失或维度不匹配"原因的异常，且不产生静默的错误计算结果或占位输出。
4. WHERE 配置中未启用 Clinical_Modality（即三模态开关为假或未设置），THE V45_Model SHALL 仅使用
   WSI_Modality 与 Omics_Modality 计算 OT_Plan 与 Event_Token，且该计算路径 SHALL 不实例化、不
   调用 Clinical_Encoder 以及任何 Clinical 相关的 Slot_Attention_Module 聚合逻辑。
5. IF Clinical_Modality 输入张量在批次内包含缺失值占位符（如约定的 NaN 或哨兵值），THEN THE
   Clinical_Encoder SHALL 在不产生 NaN 或 Inf 输出的前提下完成前向传播，且其输出张量形状 SHALL
   与批次内不含缺失值占位符时的输出张量形状一致（均为 `[batch, dim]`）。
6. FOR ALL batch size ≥ 1、clinical_feature_dim ≥ 1 与 dim ≥ 1 的取值组合，Clinical_Encoder 前向
   传播后反向传播 SHALL 为 Clinical_Encoder 的所有可训练参数产生非 NaN、非 Inf 的梯度。
7. WHEN 三模态融合被启用且 batch 中样本数大于等于 2，THE 三模态融合后的 Event_Token 张量形状 SHALL
   与仅使用二模态时的 Event_Token 张量形状在 `[batch, num_events, dim]` 的 `num_events` 和 `dim`
   维度上保持一致（即引入第三模态不改变下游事件 token 的形状契约）。

### Requirement 2: 多癌种验证配置与跨癌种结果汇总

**User Story:** 作为一名研究人员，我希望能够为多个癌种分别生成验证配置文件并自动汇总各癌种的验证
结果，以便高效地比较模型在不同癌种数据集上的表现。

#### Acceptance Criteria

1. THE 系统 SHALL 为每个癌种生成一份独立的配置文件，命名规则为 `v45_{Cancer_Study}.yaml`，其中
   `{Cancer_Study}` 为对应癌种的研究标识符（`blca`、`brca`、`coadread`、`hnsc`、`stad`）。
2. THE 系统 SHALL 在所有癌种专属配置文件中，仅使 `data.study` 与 `train.results_dir` 两个字段
   随癌种取值变化，其余字段（包括但不限于 `data.data_root_dir` 与 `data.data_path`）在所有配置
   文件中保持完全一致。
3. WHEN 某一癌种的验证运行完成，THE 系统 SHALL 将该癌种的验证结果记录到汇总数据中，汇总数据的
   列名须包含 `val_cindex`、`val_cindex_ipcw`、`val_IBS`、`val_iauc` 四项指标。
4. WHEN 生成跨癌种汇总统计，THE 系统 SHALL 通过读取汇总结果 DataFrame 中索引标记为 `mean` 与
   `std` 的行，获取并输出 `val_cindex`、`val_cindex_ipcw`、`val_IBS`、`val_iauc` 四项指标各自的
   均值与标准差。
5. WHEN 跨癌种汇总统计计算完成，THE 系统 SHALL 分别生成 CSV 格式与 Markdown 格式的汇总结果文件
   （即 Cross_Cancer_Report），两种格式均须包含各癌种的原始指标值以及 `mean`、`std` 汇总行。
6. IF 汇总所依赖的 `summary.csv` 文件存在但缺少 `mean` 索引行，THEN THE 系统 SHALL 判定该文件
   格式无效，跳过该文件的均值与标准差计算，并在 CSV 与 Markdown 两种汇总输出中标注该癌种结果
   缺失或无效。
7. THE 系统 SHALL 按字母序（`blca`、`brca`、`coadread`、`hnsc`、`stad`）排列各癌种在汇总结果
   （CSV 与 Markdown 两种格式）中出现的行顺序，确保多次运行产生的汇总输出行序一致。

### Requirement 3: 统一的 NLL 与排序目标

**User Story:** 作为研究者，我希望将当前独立加权求和的 per-event NLL 损失与 Cox 式排序损失，重构为一
个有原则的统一生存目标（如边际校准的似然，将排序约束直接嵌入似然计算），以便训练目标具备理论一致性
而非"拼装式"多项加权和。

#### Acceptance Criteria

1. THE Unified_Objective SHALL 接受与当前 `OTEHV2RankEvent._per_event_surv_loss` 和
   `OTEHV2RankEvent._ranking_loss` 相同的输入（事件 logits 与/或风险 logits，其中至少提供二者之一、
   离散时间标签 `y`、删失指示 `c`），并返回一个 0 维（标量）的浮点张量作为损失，该张量的 dtype SHALL
   与输入 logits 张量的 dtype 一致。
2. WHEN 一个 batch 中所有样本均被删失（`c` 全为 1），THE Unified_Objective SHALL 返回一个有限
   （非 NaN、非 Inf）的标量值，且该值的反向传播 SHALL 不产生 NaN 或 Inf 梯度。
3. WHEN 一个 batch 中不存在任意一对满足 Cox 可比条件（即不存在 `t_i < t_j` 且事件 `i` 未删失）的样本
   对，THE Unified_Objective 中与排序相关的项 SHALL 退化为零贡献而不引发运行时错误。
4. FOR ALL 满足下界约束（batch size ≥ 1、事件数 ≥ 1、类别数 ≥ 2）的合法 batch size、事件数、类别数
   取值组合，Unified_Objective 的反向传播 SHALL 同时为上游事件 logits（对应
   `_per_event_surv_loss` 所作用的输入）和上游风险 logits（对应 `_ranking_loss` 所作用的输入）产生
   非 NaN、非 Inf 的梯度。
5. WHEN 输入的成对风险排序与真实事件时间顺序完全一致（即模型预测的风险排序与观测生存时间的降序完全
   吻合，不一致可比对数量为 0）时，THE Unified_Objective 中排序相关的贡献 SHALL 小于或等于任意其他
   不一致可比对数量大于 0 的排序输入所对应的贡献；即当不一致可比对数量单调增加时，该排序相关贡献
   SHALL 呈单调非减关系（避免有原则设计使排序不可判别的退化情况）。
6. WHERE 配置显式选择沿用旧版损失（例如通过一个兼容开关将损失函数切回原独立加权和形式），THE
   V45_Model 训练循环 SHALL 使用与当前 `OTEHV2RankEvent.forward()` 完全一致的损失计算路径。

### Requirement 4: Slot 身份与状态解耦表示

**User Story:** 作为研究者，我希望将 Slot_Attention_Module 中每个 slot 的单一整体向量表示，重新设计为
"身份（角色，如所属事件类型）与状态（当前前向传播内容）"分离的表示，以便获得可解释的 disentanglement
故事并支撑后续研究方向。

#### Acceptance Criteria

1. WHERE 配置启用身份/状态解耦（兼容开关为真），THE Slot_Attention_Module SHALL 为每个 slot 维护一个
   形状为 `[num_slots, dim]` 且在同一 batch 内所有样本间共享的 Slot_Identity 嵌入，其取值在同一次模型
   前向传播的所有迭代步骤中保持不变。
2. WHERE 配置启用身份/状态解耦（兼容开关为真），THE Slot_Attention_Module SHALL 为每个 slot 维护一个
   形状为 `[batch, num_slots, dim]` 的 Slot_State 表示，其取值在迭代过程中根据输入内容与 GRU（或替代
   的状态更新机制）持续更新。
3. WHERE 配置启用身份/状态解耦（兼容开关为真），WHEN Slot_Attention_Module 完成前向传播，THE 输出
   slot 表示 SHALL 由 Slot_Identity 与 Slot_State 通过显式的组合方式（如拼接后投影，或加性调制）得到，
   且输出张量形状 SHALL 与当前 `MultiHeadSlotAttention.forward` 的输出形状 `[batch, num_slots, dim]`
   一致。
4. WHERE 配置启用身份/状态解耦（兼容开关为真），IF 在固定随机种子条件下，两次前向传播使用相同的
   Slot_Identity 参数但不同的输入内容，THEN 两次输出的 Slot_State 分量 SHALL 满足 L2 范数差异大于
   1e-6（判定为不相同，用以验证 Slot_State 确实随输入变化），且两次的 Slot_Identity 参数 SHALL 逐元素
   误差为 0（用以验证 Slot_Identity 保持不变）。
5. WHERE 配置启用身份/状态解耦（兼容开关为真），FOR ALL 合法的 batch size、num_slots、dim 取值组合，
   解耦后的 Slot_Attention_Module 前向传播的反向传播 SHALL 为 Slot_Identity 参数与 Slot_State 相关
   参数均产生非 NaN、非 Inf 的梯度。
6. WHERE 配置显式关闭身份/状态解耦（兼容开关为假），THE Slot_Attention_Module SHALL 使用与当前单一
   整体向量表示一致的前向传播路径，使得在相同随机种子与相同输入下，输出数值与当前实现的绝对误差在
   1e-6 容差范围内一致。

### Requirement 5: Slot Attention 内部路由机制重设计

**User Story:** 作为研究者，我希望重新设计 Slot_Attention_Module 内部的竞争/路由机制（当前是标准
Slot Attention 的 softmax 竞争 + GRU 更新），使其在理论上与论文的 OT 主题统一，并/或引入跨模态条件化
更新和自适应迭代次数，以构成本方法区别于原始 Slot Attention 的实质性架构贡献。

#### Acceptance Criteria

1. WHERE 配置选择 Sinkhorn_Router 作为路由机制，THE Slot_Attention_Module SHALL 使用 log-domain
   Sinkhorn 迭代（与 `log_sinkhorn_plan` 一致的数值稳定实现）计算 slot 与输入 token 之间的分配矩阵，
   替代原有的 softmax 竞争归一化，且实际执行的 Sinkhorn 迭代次数 SHALL 不超过配置中声明的
   Sinkhorn_Max_Iterations 上限（取值为 1 到 1000 之间的正整数）。
2. WHEN Sinkhorn_Router 计算的分配矩阵用于更新 slot 表示，THE 更新后的 slot 张量形状 SHALL 与当前
   `MultiHeadSlotAttention.forward` 的输出形状 `[batch, num_slots, dim]` 一致。
3. FOR ALL 合法的 batch size、输入 token 数（记为 `N`）、num_slots（记为 `K`）取值组合，
   Sinkhorn_Router 计算出的分配矩阵的每一行之和 SHALL 与目标行边际值 `1/K` 的绝对误差不超过
   `1e-3`，每一列之和 SHALL 与目标列边际值 `1/N` 的绝对误差不超过 `1e-3`（即在数值容差范围内
   满足最优传输的边际约束；若配置显式提供自定义边际向量，则以该自定义边际向量替代 `1/K`、`1/N`
   作为目标值）。
4. WHERE 配置启用跨模态条件化更新，THE WSI 分支的 Slot_Attention_Module SHALL 在 slot 状态更新步骤
   中接受一个形状为 `[batch, num_slots, dim]`（其 `batch` 与 `num_slots` 维度与自身 slot 张量一致）
   的、来自 Omics 分支当前 slot 状态的张量作为额外输入，反之 Omics 分支 SHALL 同样接受一个形状约束
   相同的、来自 WSI 分支的张量作为额外输入。
5. WHERE 配置启用自适应迭代次数，THE Slot_Attention_Module SHALL 在每次迭代后计算
   Convergence_Criterion（定义为相邻迭代 slot 张量之差的 L2 范数 `‖slot_t − slot_(t-1)‖₂`），并在
   该指标低于配置的 Convergence_Threshold（取值 ≥ 0）时提前停止迭代，且实际执行的迭代次数 SHALL 不
   超过配置的最大迭代次数上限。
6. IF 自适应迭代次数在第一次迭代（`t=1`，即计算出 `slot_1` 后与初始值 `slot_0` 比较）后即满足收敛
   阈值，THEN THE Slot_Attention_Module SHALL 仍至少执行一次迭代后才允许停止（避免零迭代导致 slot
   表示退化为初始随机采样值）。
7. FOR ALL 合法的 batch size、num_slots、dim、输入 token 数取值组合，启用任意组合的路由机制重设计
   选项后的 Slot_Attention_Module 前向传播 SHALL 不产生 NaN 或 Inf 输出，且反向传播 SHALL 为所有
   相关可训练参数产生非 NaN、非 Inf 的梯度。
8. WHERE 配置未启用任何路由机制重设计选项（Sinkhorn_Router、跨模态条件化、自适应迭代次数均关闭或
   未设置），THE Slot_Attention_Module SHALL 使用与当前 `MultiHeadSlotAttention.forward` 完全一致
   的 softmax 竞争 + 固定迭代次数 + GRU 更新路径。
9. WHERE 配置启用跨模态条件化更新，IF 将传入的跨模态额外输入张量替换为不同取值的另一张量（其余
   输入保持不变），THEN THE Slot_Attention_Module 输出的 slot 表示 SHALL 产生数值差异（即证明该
   额外输入被实际用于计算，而非被接受但忽略的退化实现）。

### Requirement 6: 跨能力集成与端到端向后兼容性

**User Story:** 作为研究者，我希望临床模态融合、统一目标、slot 解耦与路由重设计这几项增强能够组合
生效并流经同一条前向/损失计算路径，同时确保在全部新增能力关闭时，模型行为与 V45 baseline 完全一致，
以便已发表或已复现的 V45 结果不会因本特性的引入而失效。

#### Acceptance Criteria

1. WHILE Clinical_Modality 融合、Slot 身份/状态解耦、路由机制重设计中的任意子集处于启用状态，THE
   V45_Model 前向传播 SHALL 成功产出最终风险 logits 张量，形状 SHALL 为
   `[batch, n_classes]`（其中 batch ≥ 1，n_classes ≥ 2），且不产生 NaN 或 Inf。
2. WHEN Clinical_Modality 融合被启用，THE Event_Token 计算路径 SHALL 经过（而非绕过）
   Slot_Attention_Module 的路由机制重设计逻辑与 Unified_Objective 的损失计算逻辑（即三者共享同一条
   数据流，而非并行独立分支）。WHEN 对 Unified_Objective 输出的标量损失执行反向传播，THE
   Clinical_Modality 融合模块输入张量所接收梯度的 L2 范数 SHALL 大于 0（以此验证该梯度确实经由
   Slot_Attention_Module 的路由逻辑回传至 Clinical_Modality 输入，从而证明三者共享同一条数据流而非
   并行独立分支）。
3. GIVEN 一个未修改的现有配置文件（如 `configs/v45_blca.yaml`），WHEN 该配置文件通过
   `survot_rank.cli train --config` 加载并构建模型，THE Model_Factory 与 V45_Model SHALL 产出与
   本特性引入之前完全相同的模块结构（层数、参数形状）与前向计算路径，不因新增可选功能的代码存在而
   改变默认行为。
4. IF 一个现有的 V45 模型 checkpoint（`state_dict`）在所有新增能力保持关闭的配置下被加载，THEN THE
   V45_Model SHALL 成功加载该 checkpoint 且不出现形状不匹配或缺失/多余键错误。
5. THE Method_Registry SHALL 保留现有方法别名（`31`、`45`、`pet`、`prognostic_event_transport`
   等）到现有模型类的映射不变。THE Method_Registry SHALL NOT 因本特性新增能力（Clinical_Modality
   融合、Unified_Objective、Slot 解耦、路由机制重设计）的引入而新增或替换任何现有条目；上述新增
   能力 SHALL 仅通过新增可选配置字段的方式实现。
6. FOR ALL batch size 属于 `{1, 2, 4}`、slot 数与事件数均属于闭区间 `[1, 16]` 的合法取值组合，
   启用全部新增能力后的完整前向 + 损失计算 + 反向传播流程 SHALL 使用合成的小张量输入，在不依赖真实
   WSI `.pt` 特征文件或远程 GPU 环境的前提下，于单次 CPU 执行不超过 10 秒内完成，并验证张量形状、
   dtype 与数值有效性契约。
7. GIVEN 全部新增能力保持关闭的配置与相同的参数初始值，WHEN 使用相同的输入张量分别执行本特性引入
   前与引入后的 V45_Model 前向传播，THE 两次输出的风险 logits 张量 SHALL 逐元素绝对误差不超过
   `1e-5`。
8. IF 配置中同时启用的新增能力组合之间存在已声明为互斥的冲突（例如两个能力被设计为不可同时开启），
   THEN THE Model_Factory SHALL 拒绝构建模型、不产出任何可用的模型实例，并返回明确指明发生冲突的
   能力名称的错误信息，且调用前的程序状态 SHALL 保持不变。

### Requirement 7: 配置驱动的可复现性

**User Story:** 作为研究者，我希望每一项新增能力都通过 YAML 配置中的显式字段控制，并有合理的默认值，
以便实验可以通过配置差异而非代码分支来复现和对比。

#### Acceptance Criteria

1. FOR ALL 本特性新增的配置开关（三模态融合、Unified_Objective、Slot 解耦、Sinkhorn_Router、
   跨模态条件化、自适应迭代次数），THE 对应的模型类 SHALL 在配置字段缺失时使用与当前 V45 baseline
   行为一致的默认值，即：在相同随机种子与相同输入下，模型前向输出与 V45 baseline 逐元素数值相等
   （绝对误差不超过 1e-6），且模型参数名称列表与各参数形状（shape）与 V45 baseline 完全一致。
2. WHEN 一个 YAML 配置文件通过 `survot_rank.cli train --config` 与 `--set key=value` 组合覆盖新增
   配置字段时，THE 系统 SHALL 使覆盖后的取值在实际构建的模型结构或损失计算路径中生效，并体现为
   以下至少一项可观察差异随配置取值改变而改变：模型模块组成、网络层数、任一参数张量的形状，或
   训练日志中记录的损失数值；WHEN 同一配置字段在 YAML 配置文件与 `--set` 参数中被同时赋值时，
   THE 系统 SHALL 以 `--set` 参数给出的取值作为最终生效值，忽略 YAML 配置文件中的同名字段取值。
3. IF 传入的配置文件或 `--set` 参数中包含未在系统中定义的字段名，或某字段的取值无法解析为其声明的
   数据类型，THEN THE 系统 SHALL 在训练流程开始前终止执行，输出指明该字段名及具体问题原因的错误
   提示，并且不产生任何权重文件或结果目录的写入。
4. THE 新增配置字段命名 SHALL 仅由小写英文字母、数字与下划线组成（不允许出现大写字母或连字符），
   并以对应方法或模块名称作为前缀（如 `otehv2_*`、`lambda_rankevent_*`），从而与现有
   `configs/v45_blca.yaml` 中字段的命名风格保持一致。
