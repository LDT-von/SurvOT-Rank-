# Implementation Plan: SurvOT-Rank Enhancements

## Overview

本实现计划严格按照 design.md 的依赖顺序拆分：Step A（Slot Attention V2，需求4+5）→
Step B（统一目标函数，需求3）→ Step D 骨架（OTEHV2RankEventV2 子类 + 注册 + 配置校验，
需求6+7）→ Step C（临床编码器与三模态融合，需求1，依赖 Step A 的 slot attention 接口
与 Step D 骨架提供的子类）→ Step D 收尾（把 A/B/C 全部接入 OTEHV2RankEventV2，需求6+7）
→ Step E（多癌种配置与汇总工具，需求2，独立于 A-D，可随时插入执行）。

每个任务都足够小，可以单独通过 `spec-task-execution` 子代理一次执行、一次验收（"一点一点改"）。
所有新代码都是新增类/新增文件/新增配置字段，不修改 `OTEHV2RankEvent`、
`OTEventHazardV2Survival`、`MultiHeadSlotAttention`、`nll_loss`/`NLLSurvLoss` 等现有类的
默认行为，保证任意步骤停下时 V45 baseline 仍然可用。

本次设计文档（design.md）不包含"Correctness Properties"章节，因此本计划只使用合成张量
单元测试（pytest，无 GPU、无真实 WSI 数据），不引入基于属性的测试（PBT）框架。测试子任务
统一标记为可选（`*`），Windows 本地即可运行。

## Tasks

- [x] 1. Slot Attention V2 —— Identity/State 解耦（需求4）
  - 在 `survot_rank/research/components/slot_attention.py` 中新增
    `MultiHeadSlotAttentionV2` 类（不修改现有 `MultiHeadSlotAttention`）。

  - [x] 1.1 实现 `MultiHeadSlotAttentionV2` 骨架与身份/状态解耦路径
    - 新增 `self.slot_identity = nn.Parameter(torch.randn(num_slots, dim) * 0.02)`
      （`[num_slots, dim]`，batch 间共享，跨迭代步骤不变）
    - 保留迭代变量 `slot_state`（即原 `slots`，`[batch, num_slots, dim]`）
    - `use_disentangled_slots=True` 时：
      `output = self.identity_proj(torch.cat([slot_state, identity.expand(b,-1,-1)], dim=-1))`，
      其中 `identity_proj = nn.Linear(dim*2, dim)`
    - `use_disentangled_slots=False` 时：直接 `output = slot_state`，逐位复用
      `MultiHeadSlotAttention.forward` 的现有迭代逻辑（softmax + 固定 iters + GRU），保证
      与旧实现数值一致
    - _Requirements: 4.1, 4.2, 4.3, 4.6_

  - [ ]* 1.2 为 identity/state 解耦编写单元测试（`tests/test_slot_attention_v2.py`）
    - 断言输出形状 `[batch, num_slots, dim]`（需求4 AC3）
    - 固定随机种子，相同 `slot_identity`、不同输入内容 → `slot_state` 分量 L2 差 > 1e-6，
      `slot_identity` 逐元素误差为 0（需求4 AC4）
    - 对 `batch∈{1,2,4}`、`num_slots/dim` 多组取值，`loss.backward()` 后
      `slot_identity` 与 state 相关参数 `.grad` 均非 NaN/Inf（需求4 AC5）
    - `use_disentangled_slots=False` 时与固定种子下的当前 `MultiHeadSlotAttention` 输出
      绝对误差 ≤ 1e-6（需求4 AC6）
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 2. Slot Attention V2 —— Sinkhorn 路由机制（需求5a）
  - [x] 2.1 在 `slot_attention.py` 中新增 `_log_sinkhorn_assign(cost, max_iter, eps=0.05)`
      并接入 `MultiHeadSlotAttentionV2` 的路由开关
    - 复用 `model_v2.py::log_sinkhorn_plan` 的 log-domain 数值稳定写法，行边际
      `1/num_slots`、列边际 `1/num_input_tokens`（支持自定义边际向量覆盖默认值）
    - `router="sinkhorn"` 时用该分配矩阵替代 `dots.softmax(dim=-2)` 那一步；
      `router="softmax"`（默认）时路径不变
    - 实际执行的 Sinkhorn 迭代次数不超过 `sinkhorn_max_iters`（1~1000 的正整数上限）
    - _Requirements: 5.1, 5.2_

  - [ ]* 2.2 为 Sinkhorn 路由编写单元测试（追加到 `tests/test_slot_attention_v2.py`）
    - 对 `batch`、输入 token 数 `N`、`num_slots K` 多组合法取值组合，断言分配矩阵每行和与
      `1/K` 绝对误差 ≤ 1e-3，每列和与 `1/N` 绝对误差 ≤ 1e-3（需求5 AC3）
    - 断言更新后 slot 张量形状为 `[batch, num_slots, dim]`（需求5 AC2）
    - 断言实际迭代次数不超过配置的 `sinkhorn_max_iters`（需求5 AC1）
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 3. Slot Attention V2 —— 跨模态条件化更新（需求5b）
  - [x] 3.1 在 `MultiHeadSlotAttentionV2.forward` 中新增可选参数
      `cross_modal_state: Optional[Tensor] = None`
    - `[batch, num_slots, dim]`，`cross_modal_conditioning=True` 且非 None 时：
      `updates = updates + self.cross_modal_proj(cross_modal_state)`，
      `cross_modal_proj = nn.Linear(dim, dim)`（新增可训练层）
    - `cross_modal_conditioning=False` 或参数为 None 时行为与现有路径一致
    - _Requirements: 5.4_

  - [ ]* 3.2 为跨模态条件化编写单元测试（追加到 `tests/test_slot_attention_v2.py`）
    - 固定其余输入，仅替换 `cross_modal_state` 为不同取值的另一张量 → 输出 slot 表示产生
      数值差异（需求5 AC9，证明该输入被实际使用而非被忽略）
    - 断言 `cross_modal_state` 形状约束（`batch`、`num_slots` 与自身 slot 张量一致）下
      前向不产生 NaN/Inf，反向所有相关参数 `.grad` 非 NaN/Inf（需求5 AC7）
    - _Requirements: 5.4, 5.7, 5.9_

- [x] 4. Slot Attention V2 —— 自适应迭代次数与工厂函数（需求5c + 5.8 + 7.1）
  - [x] 4.1 实现 Convergence_Criterion 与自适应迭代循环，并在文件末尾新增
      `build_slot_attention(dim, num_slots, heads, iters, config)` 工厂函数
    - 每次迭代后计算 `criterion = (slot_state - slot_state_prev).norm(p=2)`
    - 循环：`for t in range(max_iters_cap): ...; if adaptive_iters and t >= 1 and
      criterion < convergence_threshold: break`（保证至少完整跑完第 0、1 两轮才允许停止，
      满足"第一次迭代后即收敛仍要再跑一次"的最低要求）
    - `build_slot_attention`：当 `slot_router/slot_disentangled/slot_cross_modal_cond/
      slot_adaptive_iters` 均为默认值（`softmax`/`False`）时返回原始
      `MultiHeadSlotAttention` 实例；否则返回配置好的 `MultiHeadSlotAttentionV2` 实例
    - _Requirements: 5.5, 5.6, 5.8, 7.1_

  - [ ]* 4.2 为自适应迭代次数与工厂函数编写单元测试（追加到 `tests/test_slot_attention_v2.py`）
    - 构造在第 1 次迭代后即满足收敛阈值的合成输入，断言实际执行迭代次数 ≥ 2（需求5 AC6）
    - 断言实际迭代次数不超过 `max_iters_cap`（需求5 AC5）
    - 断言启用任意路由重设计选项组合后前向不产生 NaN/Inf，反向所有相关参数梯度非
      NaN/Inf（需求5 AC7，覆盖组合场景）
    - 断言 `build_slot_attention` 在所有新增配置字段缺失时返回的实例，与直接实例化
      `MultiHeadSlotAttention` 在固定种子下数值一致（≤1e-6）（需求5 AC8，需求7 AC1）
    - _Requirements: 5.5, 5.6, 5.7, 5.8, 7.1_

- [ ] 5. Checkpoint —— 确保 Step A（Slot Attention V2）所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. 统一生存目标函数 UnifiedSurvivalObjective（需求3）
  - [x] 6.1 在 `survot_rank/research/legacy/slotspe_runtime/utils/loss_func.py` 中新增
      `UnifiedSurvivalObjective(nn.Module)`（不修改现有 `nll_loss`/`NLLSurvLoss` 等）
    - `forward(event_logits=None, risk_logits=None, y=None, c=None) -> Tensor`（0 维，
      dtype 与输入 logits 一致）；`event_logits`、`risk_logits` 至少提供一个
    - `_per_event_nll`：复用 `OTEHV2RankEvent._nll_per_sample` 的逐样本计算逻辑
    - `_pairwise_margin_penalty`：复用 `_ranking_loss` 的可比对筛选逻辑
      （`comparable = (e>0.5) & (ti<tj)`），无可比对时返回 `risk.sum()*0.0`（保持梯度图、值为 0）
    - 惩罚项使用 `softplus(-(diff - margin))` 对可比对求和/均值，天然满足"不一致可比对数量
      单调增加时贡献单调非减"的契约
    - 全部删失（`c` 全 1）时返回有限标量，反向不产生 NaN/Inf 梯度
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [ ]* 6.2 为 UnifiedSurvivalObjective 编写单元测试（`tests/test_unified_objective.py`）
    - 断言返回 0 维张量，dtype 与输入 logits 一致（需求3 AC1）
    - 全删失 batch → 有限值，反向无 NaN/Inf 梯度（需求3 AC2）
    - 无可比对样本对 → 排序项退化为 0 且不抛异常（需求3 AC3）
    - 对合法 `batch/事件数/类别数` 组合，反向同时为 `event_logits` 与 `risk_logits`
      产生非 NaN/Inf 梯度（需求3 AC4）
    - 构造不一致可比对数量分别为 0、1、5 的排序输入，断言排序贡献单调非减
      （需求3 AC5）
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 7. Checkpoint —— 确保 Step B（UnifiedSurvivalObjective）所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. OTEHV2RankEventV2 骨架、方法注册与配置校验（需求6.3-6.5/6.8，需求7.1-7.4）
  - [x] 8.1 在 `survot_rank/research/methods/prognostic_event_transport/model.py` 中新增
      子类 `OTEHV2RankEventV2(OTEHV2RankEvent)`，并在
      `survot_rank/training/extended_args.py` 中新增对应 CLI 参数
    - `__init__` 用 `getattr(args, name, default)` 读取全部新增配置字段（三模态、
      unified objective、slot 解耦/路由/跨模态/自适应迭代），默认值使新开关全部为
      `False`/`"softmax"` 等，与 V45 行为一致；字段缺失时不实例化任何新模块
    - `forward` 暂时直接调用 `super().forward(**kwargs)`（本任务只搭骨架，Step D 收尾
      任务 11 再接入新能力），保证当前默认行为与 `OTEHV2RankEvent` 完全一致
    - `extended_args.py`：新增 `--otehv2v2_use_clinical`、`--otehv2v2_clinical_feature_dim`、
      `--otehv2v2_num_slots_clinical`、`--otehv2v2_use_unified_objective`、
      `--lambda_unified_rank`、`--otehv2v2_slot_disentangled`、`--otehv2v2_slot_router`、
      `--otehv2v2_slot_cross_modal_cond`、`--otehv2v2_slot_adaptive_iters`、
      `--otehv2v2_sinkhorn_max_iters`、`--otehv2v2_convergence_threshold` 等参数（默认值
      与骨架类一致），并把 `"otehv2_rankevent_v2"` 加入 `METHOD_CHOICES`
    - _Requirements: 6.3, 7.1, 7.4_

  - [x] 8.2 在 `survot_rank/training/model_factory.py` 的 `METHOD_REGISTRY`/
      `METHOD_ALIASES` 中新增条目（不修改/删除现有条目）
    - `METHOD_REGISTRY["otehv2_rankevent_v2"] = (prognostic_event_transport 目录, "OTEHV2RankEventV2")`
    - `METHOD_ALIASES["45v2"] = "otehv2_rankevent_v2"`
    - _Requirements: 6.5_

  - [x] 8.3 新增配置字段 schema 校验，并为将来的互斥冲突检测预留接口
    - 在 `model_factory.py` 中新增轻量 `_validate_config(args)`，在 `get_model` 构建模型
      前调用；当前版本只做字段类型校验（未定义字段名或取值无法解析为声明类型时终止，
      给出明确的字段名与问题原因），不产生任何权重文件或结果目录写入；不做组合冲突拒绝
      （当前没有互斥组合，函数结构预留 `ValueError` 分支供后续扩展）
    - _Requirements: 7.2, 7.3, 6.8_

  - [ ]* 8.4 为骨架子类编写单元测试（`tests/test_otehv2_rankevent_v2.py`）
    - 固定随机种子，用合成小张量分别跑 `OTEHV2RankEvent` 与 `OTEHV2RankEventV2`
      （新开关全关闭）前向，断言 `logits` 逐元素 `allclose(atol=1e-5)`（需求6 AC7）
    - 断言两者 `state_dict()` 键集合与各参数形状完全一致，`OTEHV2RankEventV2` 可用
      `strict=True` 加载 `OTEHV2RankEvent` 的 `state_dict`（需求6 AC4）
    - 断言 `METHOD_REGISTRY` 中现有条目（`otehv2_rankevent`、`ot_event_hazard_v2`）未被
      修改，新条目 `otehv2_rankevent_v2`/`45v2` 存在（需求6 AC5）
    - 断言传入未定义字段名或不可解析类型时 `_validate_config`/训练入口在开始前终止且不
      写入任何文件（需求7 AC3）
    - _Requirements: 6.4, 6.5, 6.7, 6.8, 7.1, 7.2, 7.3_

- [x] 9. 临床编码器与三模态融合接入（需求1）
  - [x] 9.1 新建 `survot_rank/research/components/clinical_encoder.py`：`ClinicalEncoder`
    - `[batch, clinical_feature_dim] -> [batch, dim]`；`self.impute` 可学习填充向量替换
      NaN 占位符缺失值，再经 `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
    - _Requirements: 1.1, 1.5_

  - [ ]* 9.2 为 ClinicalEncoder 编写单元测试（`tests/test_clinical_encoder.py`）
    - 断言输出形状 `[batch, dim]`，`dim` 等于传入的 `wsi_projection_dim`（需求1 AC1）
    - 输入包含 NaN 占位符时前向不产生 NaN/Inf，输出形状与不含 NaN 时一致（需求1 AC5）
    - 对 `batch≥1`、`clinical_feature_dim≥1`、`dim≥1` 多组合法取值，反向传播为所有可训练
      参数产生非 NaN/Inf 梯度（需求1 AC6）
    - _Requirements: 1.1, 1.5, 1.6_

  - [x] 9.3 在 `OTEHV2RankEventV2`（`model.py`）中接入 Clinical 分支与三方 OT 融合
    - `__init__` 中仅当 `otehv2v2_use_clinical=True` 时实例化
      `self.clinical_encoder = ClinicalEncoder(...)` 与
      `self.slot_attention_clinical = MultiHeadSlotAttention(dim=dim,
      num_slots=otehv2v2_num_slots_clinical, iters=args.slot_iters, heads=8)`；关闭时
      完全不创建这些模块（`state_dict()` 不包含相关键）
    - `forward` 中新增 Clinical 分支：缺失 `x_clinical` 或形状不匹配时抛出包含
      "Clinical 输入缺失或维度不匹配" 的 `ValueError`；否则编码 → 升维为
      `[B,1,dim]` → 送入 `slot_attention_clinical` 得到 `slots_clinical`
    - 新增 `ThreeWayOTFusion` 组合器（wsi-omic / wsi-clinical / omic-clinical 三对模态
      两两计算 OT 代价与 plan，concat 后复用与 `MultiScaleOTFusion.forward` 同构的
      event-query 注意力聚合逻辑，输入通道数从 3 变为 6），保证输出 `event_tokens`
      的 `num_events`、`dim` 与二模态时一致
    - 关闭 `otehv2v2_use_clinical` 时该分支代码完全不被调用，路径与骨架任务 8.1 一致
    - _Requirements: 1.2, 1.3, 1.4, 1.7_

  - [x] 9.4 数据管线新增可选 Clinical 字段（不改变现有 5 元组默认路径）
    - `survot_rank/research/legacy/slotspe_runtime/dataset/dataset_survival.py`：
      `SurvivalDataset.__getitem__` 在 `dataset_factory.use_clinical_modality` 为真时，
      额外从 `self.label_df` 取出配置声明的临床列、数值化/独热编码后拼成 1-D tensor 作为
      第 6 个返回值；默认情况下仍返回原有 5 元组
    - `survot_rank/research/legacy/slotspe_runtime/utils/core_utils.py`：
      `_unpack_data`/`_process_data_and_forward` 在 `getattr(args, "otehv2v2_use_clinical",
      False)` 为真时，新增 `input_args["x_clinical"] = data[5].to(device)` 分支，不改变
      现有 5 元组解析逻辑
    - _Requirements: 1.2 (数据管线前提假设)_

  - [ ]* 9.5 为 Clinical 分支与三模态融合编写单元测试（追加到
      `tests/test_otehv2_rankevent_v2.py`）
    - 启用 `otehv2v2_use_clinical=True` 但不传 `x_clinical`（或传入错误形状），断言抛出
      指明"Clinical 输入缺失或维度不匹配"的异常，且不产生占位输出（需求1 AC3）
    - `otehv2v2_use_clinical=False` 时断言模型 `state_dict()` 中不包含
      `clinical_encoder`/`slot_attention_clinical` 相关键（需求1 AC4）
    - 启用三模态融合、batch≥2 时，断言 `event_tokens` 的 `num_events`、`dim` 维度与仅
      二模态时一致（需求1 AC7）
    - _Requirements: 1.2, 1.3, 1.4, 1.7_

- [ ] 10. Checkpoint —— 确保 Step C（临床模态融合）所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. 跨能力集成：路由重设计与统一目标接入 OTEHV2RankEventV2（需求5.4/5.9/3.6，
    需求6.1/6.2/6.6/6.8）
  - [x] 11.1 用 `build_slot_attention` 工厂替换 `OTEHV2RankEventV2` 内 wsi/omic/clinical
      三个分支的 slot attention 实例化，并接入双向跨模态条件化
    - `self.slot_attention_wsi`/`omic`/`clinical` 改为调用
      `build_slot_attention(dim, num_slots, heads=8, iters=args.slot_iters, config=args)`
      （字段全部缺失时行为与骨架任务 8.1 完全一致，回退到 `MultiHeadSlotAttention`）
    - `forward` 中当 `otehv2v2_slot_cross_modal_cond=True` 时，wsi 分支与 omic 分支的
      slot attention 各自把对方上一轮的 `slot_state` 通过 `cross_modal_state` 参数传给
      对方，实现双向条件化
    - _Requirements: 5.4, 6.1_

  - [x] 11.2 在 `OTEHV2RankEventV2` 中接入 UnifiedSurvivalObjective 兼容开关
    - `otehv2v2_use_unified_objective=True` 时，损失计算改为调用
      `UnifiedSurvivalObjective(event_logits=..., risk_logits=..., y=y, c=c)`
    - `otehv2v2_use_unified_objective=False`（默认）时，损失计算路径逐行复制
      `OTEHV2RankEvent.forward` 现有的 5 项加权和逻辑（per-event NLL + ranking +
      global consistency + gate entropy + OT/diversity/recon），两条路径完全独立、
      不共享状态
    - _Requirements: 3.6, 6.1_

  - [ ]* 11.3 为跨能力组合编写集成级单元测试（追加到 `tests/test_otehv2_rankevent_v2.py`）
    - 对 `batch∈{1,2,4}`、slot 数与事件数均属于 `[1,16]` 的合法取值组合，启用全部新增
      能力（clinical + unified objective + slot 解耦 + sinkhorn 路由 + 跨模态条件化 +
      自适应迭代）后跑完整前向 + 损失 + 反向传播，断言：最终 `logits` 形状为
      `[batch, n_classes]`、无 NaN/Inf；单次 CPU 执行不超过 10 秒（需求6 AC1, AC6）
    - 对 `UnifiedSurvivalObjective` 输出标量反向传播后，断言 Clinical 输入张量所接收
      梯度的 L2 范数 > 0（验证梯度经由 slot attention 路由与统一目标回传到 Clinical
      输入，三者共享同一条数据流）（需求6 AC2）
    - 断言 `_validate_config` 在当前所有新增能力组合下均不拒绝构建（当前无互斥组合，
      为需求6 AC8 的"预留冲突检测"契约提供回归基线）
    - _Requirements: 6.1, 6.2, 6.6, 6.8_

- [ ] 12. Checkpoint —— 确保 Step D（跨能力集成）所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. 多癌种配置文件生成（需求2.1, 2.2）
  - [x] 13.1 新建 `tools/gen_multicancer_configs.py`，生成 `configs/v45_{study}.yaml`
      （`study ∈ {blca, brca, coadread, hnsc, stad}`）
    - 以 `configs/v45_blca.yaml` 为模板读取，仅替换 `data.study` 与
      `train.results_dir` 两个字段，其余字段（含 `data.data_root_dir`、
      `data.data_path`）逐字段保持一致后写出 5 个文件
    - 按字母序（blca/brca/coadread/hnsc/stad）生成文件
    - _Requirements: 2.1, 2.2_

  - [ ]* 13.2 为配置生成脚本编写单元测试（`tests/test_gen_multicancer_configs.py`）
    - 用临时目录 + 模板 YAML 跑生成脚本，断言生成的 5 个配置文件仅 `data.study` 与
      `train.results_dir` 不同，其余字段（含 `data.data_root_dir`、`data.data_path`）
      与模板完全一致（需求2 AC2）
    - 断言生成的文件名严格匹配 `v45_{study}.yaml` 命名规则（需求2 AC1）
    - _Requirements: 2.1, 2.2_

- [x] 14. 跨癌种结果汇总工具（需求2.3-2.7）
  - [x] 14.1 新建 `tools/aggregate_cross_cancer.py`（改造自现有
      `tools/aggregate_multicancer.py`，不修改原文件）
    - 接受 `--results-root` 与 `--studies`（默认 5 个癌种，字母序）参数
    - 对每个 study 在结果目录下 `rglob("summary.csv")`；目录缺失或找不到文件时标记
      `status=missing`；找到但缺少 `"mean"` 索引行时标记
      `status=invalid, reason=missing_mean_row`，跳过该文件的均值/标准差计算但不中断
      其余 study 的处理
    - 汇总数据列固定为 `val_cindex, val_cindex_ipcw, val_IBS, val_iauc`，通过读取
      `mean`/`std` 索引行获取并输出各指标的均值与标准差
    - 按字母序排列各癌种在输出中的行顺序；同时写出 CSV 与 Markdown 两种格式的汇总文件，
      两种格式均包含各癌种原始指标值以及 `mean`、`std` 汇总行，并标注缺失/无效癌种
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 14.2 为跨癌种汇总工具编写单元测试（`tests/test_aggregate_cross_cancer.py`）
    - 用临时目录构造若干个合成 `summary.csv`（部分包含 `mean`/`std` 行、部分缺失
      `mean` 行、部分癌种目录完全缺失），运行汇总脚本
    - 断言输出 CSV/Markdown 均包含固定四列指标以及各癌种的 `mean`/`std` 汇总行
      （需求2 AC3, AC4, AC5）
    - 断言缺少 `mean` 行的癌种被标注为缺失/无效而不中断整体汇总（需求2 AC6）
    - 断言输出行顺序严格按字母序（需求2 AC7）
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7_

- [ ] 15. 最终检查点 —— 确保全部测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- 标记 `*` 的子任务为可选测试任务，可跳过以加快 MVP 进度，但强烈建议在每个 Step 完成后
  执行，作为增量验收标准。
- 所有测试均使用合成小张量（`torch.randn`/`torch.randint`），不依赖 GPU、不依赖真实
  WSI `.pt` 特征文件，可在 Windows 本地直接用 `pytest` 运行。
- Step E（任务13、14）与 Step A-D 之间无代码依赖，可在任意时间点插入执行，不影响其余
  任务的顺序。
- 每个 Checkpoint 任务（5、7、10、12、15）都是让用户确认阶段性成果的自然停顿点；
  Step D 骨架任务8完成后，现有 `configs/v45_blca.yaml` 应仍可通过
  `survot_rank.cli train --config` 无修改运行，产出与本特性引入前完全相同的模型结构。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "6.1", "8.1", "9.1", "13.1", "14.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "6.2", "8.2", "8.3", "9.2", "13.2", "14.2"] },
    { "id": 2, "tasks": ["2.2", "3.1", "8.4", "9.3"] },
    { "id": 3, "tasks": ["3.2", "4.1", "9.4"] },
    { "id": 4, "tasks": ["4.2", "9.5", "11.1"] },
    { "id": 5, "tasks": ["11.2"] },
    { "id": 6, "tasks": ["11.3"] }
  ]
}
```
