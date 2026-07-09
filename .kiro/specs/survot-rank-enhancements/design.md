# Design Document

## Overview

本设计把 requirements.md 中的 7 个需求，转化为可以在 `survot_rank` 现有代码结构上**增量叠加**的
具体实现方案。核心设计原则只有一条：**新增的每一项能力都是默认关闭的可选项，关闭时代码路径必须
退化为与当前 `OTEHV2RankEvent`（V45）完全一致（数值误差 ≤ 1e-5/1e-6）**。所有新逻辑都通过
"新增类/新增方法 + 配置开关分支"来实现，不修改现有类的默认行为，从而保证增量开发时随时可以停在
任意一步，现有实验结果都不会失效。

实现顺序按依赖关系从底层到上层排列（这也是后续 tasks.md 拆分任务的顺序基础）：

```
Step A: Slot Attention 层改造（需求 4 + 5）—— 独立于其他模态，最底层
Step B: 统一目标函数（需求 3）—— 独立于模态数量，只依赖 logits/y/c
Step C: 临床模态编码器与三模态融合（需求 1）—— 依赖 Step A 的 Slot Attention 接口
Step D: 跨能力集成与配置装配（需求 6 + 7）—— 把 A/B/C 串起来，验证组合与向后兼容
Step E: 多癌种配置与结果汇总（需求 2）—— 纯配置/脚本工作，与 A-D 无代码依赖，可并行
```

## Architecture

### 1. 总体数据流（新增能力全部启用时）

```
x_wsi ──► WSI_Mlp ──► SlotAttentionV2(mode=wsi) ──► slots_wsi ─┐
x_omics ─► sig_networks ──► SlotAttentionV2(mode=omic) ─► slots_omic ─┤
x_clinical ─► ClinicalEncoder ──► SlotAttentionV2(mode=clinical) ─► slots_clinical ─┤
                                                                      │
                                                    (三方 OT cost + log_sinkhorn_plan)
                                                                      ▼
                                                          MultiScaleOTFusion (扩展为 N 模态)
                                                                      ▼
                                                             event_tokens (Transformer)
                                                                      ▼
                                                     event_hazard / event_gate / global_head
                                                                      ▼
                                                              logits [batch, n_classes]
                                                                      │
                                            UnifiedSurvivalObjective(logits, event_logits, y, c)
                                                                      ▼
                                                                  aux_loss
```

所有新增箭头（Clinical 分支、SlotAttentionV2 内部机制、UnifiedSurvivalObjective）均由配置开关
控制；关闭时数据流精确退化为当前 `OTEHV2RankEvent.forward()` 的现状。

### 2. 文件级改动清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `survot_rank/research/components/slot_attention.py` | 新增类 `MultiHeadSlotAttentionV2`，原 `MultiHeadSlotAttention` **不修改** | 向后兼容通过"新类 + 工厂函数选择"实现，而非在旧类里加 if 分支 |
| `survot_rank/research/components/clinical_encoder.py` | 新文件 | `ClinicalEncoder` 模块 |
| `survot_rank/research/legacy/slotspe_runtime/utils/loss_func.py` | 新增函数/类，不修改现有 `nll_loss`/`NLLSurvLoss`/等 | `UnifiedSurvivalObjective` |
| `survot_rank/research/methods/prognostic_event_transport/model.py` | 新增子类 `OTEHV2RankEventV2(OTEHV2RankEvent)`，不修改 `OTEHV2RankEvent` | 承载三模态融合、路由重设计、统一目标的接入 |
| `survot_rank/training/model_factory.py` | 在 `METHOD_REGISTRY`/`METHOD_ALIASES` **新增条目**，不修改/删除现有条目 | 新增 `"otehv2_rankevent_v2"` |
| `survot_rank/research/legacy/slotspe_runtime/dataset/dataset_survival.py` | 新增可选返回字段，默认路径不变 | `SurvivalDataset.__getitem__` 在配置启用 clinical 时多返回一个张量 |
| `survot_rank/research/legacy/slotspe_runtime/utils/core_utils.py` | `_unpack_data`/`_process_data_and_forward` 新增可选分支 | 组装 `x_clinical` kwarg |
| `configs/v45_{study}.yaml`（新建 5 个） | 新文件 | 需求 2 |
| `tools/aggregate_cross_cancer.py` | 新文件 | 需求 2 的汇总工具 |
| `tests/`（新建） | 新文件，仅合成张量单测 | 覆盖需求 1/3/4/5/6 的属性测试 |

**关键决策：新增子类而不是原地修改。** `OTEHV2RankEvent` 是已发表/已复现结果（V45, C-index
0.7105/0.7237 集成）的直接来源。把所有新增能力放进新的子类 `OTEHV2RankEventV2`，`METHOD_REGISTRY`
里新增一个方法名 `otehv2_rankevent_v2`（而不是复用 `otehv2_rankevent`），使旧配置
`configs/v45_blca.yaml`（`survot_method: otehv2_rankevent`）在代码语义上完全走不到任何新代码路径，
这是比"同一个类里加 flag 分支"更强的向后兼容保证方式（需求 6.3/6.7 的"数值一致"验收标准因此自动
满足，不需要专门维护一份"退化路径"测试）。

## Components and Interfaces

### Step A: Slot Attention 层改造（需求 4 + 5）

新文件内新增类，`slot_attention.py` 追加以下内容（不改动现有 `MultiHeadSlotAttention`）：

```python
class MultiHeadSlotAttentionV2(Module):
    """在标准 Slot Attention 基础上支持三个可独立开关的重设计选项。

    Args:
        dim, heads, dim_head, iters, eps, hidden_dim: 与 MultiHeadSlotAttention 相同
        num_slots: 同上
        use_disentangled_slots: bool = False   # 需求4：identity/state 解耦
        router: str = "softmax"                # "softmax" | "sinkhorn"  需求5a
        sinkhorn_max_iters: int = 20            # 需求5 AC1
        cross_modal_conditioning: bool = False  # 需求5b
        adaptive_iters: bool = False            # 需求5c
        convergence_threshold: float = 0.0
        max_iters_cap: int = 10                 # 自适应模式下的迭代上限
    """
```

**identity/state 解耦（需求4）**：
- `self.slot_identity = nn.Parameter(torch.randn(num_slots, dim) * 0.02)`（形状 `[num_slots, dim]`，
  batch 间共享，对应 Glossary `Slot_Identity`）。
- 迭代过程中维护的仍是 `slot_state`（形状 `[batch, num_slots, dim]`，即原来的 `slots` 变量，
  对应 `Slot_State`）。
- 前向传播最后一步（仅当 `use_disentangled_slots=True`）：
  `output = self.identity_proj(torch.cat([slot_state, self.slot_identity.unsqueeze(0).expand(b, -1, -1)], dim=-1))`，
  其中 `identity_proj = nn.Linear(dim * 2, dim)` 做拼接后投影（对应需求4 AC3 的"拼接后投影"选项）。
  当 `use_disentangled_slots=False` 时直接 `output = slot_state`，与现有实现逐位一致。

**Sinkhorn 路由（需求5a）**：
- 复用 `ot_event_hazard_v2/model_v2.py::log_sinkhorn_plan` 的数值稳定实现思路，在
  `slot_attention.py` 内新增一个局部函数 `_log_sinkhorn_assign(cost, max_iter, eps=0.05)`，
  输入 slot↔token 的代价矩阵（由 `q·kᵗ` 取负得到），输出满足行/列边际的分配矩阵，替代原来的
  `dots.softmax(dim=-2)` 那一步。
- 边际约束：行边际 `1/num_slots`，列边际 `1/num_tokens`（对应需求5 AC3），用
  `log_mu = torch.full((b, num_slots), 1.0/num_slots).log()` 等构造，与 `model_v2.py` 中写法
  保持同构，方便审阅者/审稿人一眼看出"路由机制与论文 OT 主题统一"。

**跨模态条件化更新（需求5b）**：
- `forward` 签名新增可选参数 `cross_modal_state: Optional[Tensor] = None`（形状
  `[batch, num_slots, dim]`，需求5 AC4/AC9）。
- 在 GRU 更新前，若 `cross_modal_conditioning=True` 且 `cross_modal_state is not None`：
  `updates = updates + self.cross_modal_proj(cross_modal_state)`，其中
  `cross_modal_proj = nn.Linear(dim, dim)`，新增的可训练层保证需求5 AC9（换一个不同的
  `cross_modal_state` 必须让输出数值变化——因为它直接加进了 GRU 输入）。
- 调用方（`OTEHV2RankEventV2.forward`）在两个模态分支的 slot attention 各迭代 1 步之间互相把
  对方上一轮的 `slot_state` 传进来，实现"双向条件化"。

**自适应迭代次数（需求5c）**：
- 每次迭代后计算 `criterion = (slot_state - slot_state_prev).norm(p=2)`（对应需求5 AC5 的
  `Convergence_Criterion` 定义）。
- 循环条件：`for t in range(max_iters_cap): ...; if adaptive_iters and t >= 1 and criterion < convergence_threshold: break`
  ——`t >= 1` 保证至少执行 2 次原始迭代计数（对应需求5 AC6："第一次迭代后即满足阈值仍要再跑一次"，
  这里用 `t>=1`即至少完整跑完第 0、1 两轮才允许停止，比 AC6 的最低要求更保守，肯定满足）。

工厂函数（放在 `slot_attention.py` 末尾）：

```python
def build_slot_attention(dim, num_slots, heads, iters, config):
    """config 是一个简单的 dict/Namespace，读取新增的可选字段；
    所有字段缺失时返回原始 MultiHeadAttention 实例（需求5 AC8 / 需求7 AC1）。
    """
    if not any([
        getattr(config, "slot_router", "softmax") != "softmax",
        getattr(config, "slot_disentangled", False),
        getattr(config, "slot_cross_modal_cond", False),
        getattr(config, "slot_adaptive_iters", False),
    ]):
        return MultiHeadSlotAttention(dim=dim, num_slots=num_slots, heads=heads, iters=iters)
    return MultiHeadSlotAttentionV2(dim=dim, num_slots=num_slots, heads=heads, iters=iters, ...)
```

### Step B: 统一目标函数（需求 3）

在 `loss_func.py` 追加：

```python
class UnifiedSurvivalObjective(nn.Module):
    """将 per-event NLL 与 Cox 排序约束嵌入同一个边际校准似然。

    forward(event_logits=None, risk_logits=None, y=None, c=None) -> Tensor (0-dim)
    """
```

设计上不是"发明一个全新的似然公式"，而是采用**margin-calibrated likelihood**的构造方式：
在原始 NLL 的基础上，对每一个可比样本对 `(i,j)`（`t_i < t_j` 且 `i` 未删失）施加一个 log-barrier
式的惩罚，直接乘进似然的对数域，而不是在 loss 数值上外部加权：

```python
def forward(self, event_logits=None, risk_logits=None, y=None, c=None):
    assert event_logits is not None or risk_logits is not None
    base_nll = 0.0
    if event_logits is not None:
        base_nll = self._per_event_nll(event_logits, y, c)  # 复用现有 nll_per_sample 逐样本实现
    risk = self._risk_from_logits(risk_logits if risk_logits is not None else event_logits.mean(dim=1))
    concordance_penalty = self._pairwise_margin_penalty(risk, y, c)  # softplus margin, 退化为0当无可比对
    return base_nll + self.rank_weight * concordance_penalty
```

- `_per_event_nll`：直接复用 `OTEHV2RankEvent._nll_per_sample` 现有实现逻辑（数值上一致，只是
  搬到 loss_func.py 成为独立可测试单元）。
- `_pairwise_margin_penalty`：复用 `OTEHV2RankEvent._ranking_loss` 的可比对筛选逻辑
  （`comparable = (e>0.5) & (ti<tj)`），无可比对时返回 `risk.sum()*0.0`（保持梯度图连通、值为 0，
  对应需求3 AC3）。
- 单调性（需求3 AC5）：因为惩罚项是 `softplus(-(diff - margin))` 对每个可比对求和/均值，"不一致
  可比对数量"增加时该求和严格非减，天然满足单调非减契约，不需要额外设计。
- 旧版兼容开关（需求3 AC6）：`OTEHV2RankEventV2` 里用
  `if getattr(args, "use_unified_objective", False): loss = UnifiedSurvivalObjective(...)(...)
  else: loss = <原 OTEHV2RankEvent 的 5 项加权和逐行复制>`。两条路径都保留，不是"新旧接口切换"，
  是两段完全独立的代码，避免共享状态导致的隐蔽 bug。

### Step C: 临床模态编码器与三模态融合（需求 1）

新文件 `survot_rank/research/components/clinical_encoder.py`：

```python
class ClinicalEncoder(nn.Module):
    """[batch, clinical_feature_dim] -> [batch, dim]"""
    def __init__(self, clinical_feature_dim, dim, dropout=0.1):
        super().__init__()
        self.impute = nn.Parameter(torch.zeros(clinical_feature_dim))  # 可学习的缺失值填充
        self.net = nn.Sequential(
            nn.LayerNorm(clinical_feature_dim),
            nn.Linear(clinical_feature_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # 缺失值占位符（约定用 NaN）替换为可学习填充值，而不是 0（避免 0 恰好是合法取值时产生歧义）
        mask = torch.isnan(x)
        x = torch.where(mask, self.impute.expand_as(x), x)
        return self.net(x)
```

`OTEHV2RankEventV2.forward` 中新增（仅当 `use_clinical_modality=True`）：

```python
if self.use_clinical_modality:
    if "x_clinical" not in kwargs or kwargs["x_clinical"] is None:
        raise ValueError("Clinical 输入缺失：三模态开关已启用但未提供 x_clinical")
    if kwargs["x_clinical"].shape[-1] != self.clinical_feature_dim:
        raise ValueError(f"Clinical 输入维度不匹配: 期望 {self.clinical_feature_dim}, 实际 {kwargs['x_clinical'].shape[-1]}")
    x_clinical_proj = self.clinical_encoder(kwargs["x_clinical"])          # [B, dim]
    x_clinical_proj = x_clinical_proj.unsqueeze(1)                         # [B, 1, dim] 视为单 token 序列
    slots_clinical = self.slot_attention_clinical(x_clinical_proj)         # [B, num_slots_clinical, dim]
    cost_cli_wsi = cosine_cost(slots_clinical, slots_wsi)
    ...  # 三方 OT：对 MultiScaleOTFusion 的输入从 2 路 slot 扩展为 3 路
```

**三模态 OT 融合的扩展方式**：不改 `MultiScaleOTFusion` 类本身的接口契约（避免破坏
`ot_event_hazard_v2` 现有二模态调用方），而是在 `OTEHV2RankEventV2` 里新增一个
`ThreeWayOTFusion` 组合器，内部对三对模态两两计算 OT 代价与 plan（wsi-omic, wsi-clinical,
omic-clinical），再把三个 plan concat 后交给和现有 `MultiScaleOTFusion.forward` 相同结构的
event-query 注意力聚合逻辑（复制该聚合子逻辑到新类中，输入通道数从 3 变为 6，因为三对模态各贡献
3 种代价度量）。这保证需求1 AC7："三模态融合后 `Event_Token` 的 `num_events`、`dim` 与二模态时
一致"——聚合层的输出维度契约不变，只是内部融合的路数变多。

关闭时（需求1 AC4）：`OTEHV2RankEventV2` 完全不实例化 `self.clinical_encoder` /
`self.slot_attention_clinical`（在 `__init__` 里用 `if self.use_clinical_modality:` 包裹模块
创建），这样"不实例化"是字面意义上的真——不仅是不调用，参数列表里也不会出现这些层，
`state_dict()` 对比会完全一致。

### Step D: 跨能力集成（需求 6）与配置装配（需求 7）

`OTEHV2RankEventV2.__init__` 读取的新增配置字段全部使用 `getattr(args, name, default)` 模式，
默认值使全部开关为 `False`/`"softmax"` 等于当前行为：

```python
# 三模态
otehv2v2_use_clinical: bool = False
otehv2v2_clinical_feature_dim: int = 0
otehv2v2_num_slots_clinical: int = 8

# 统一目标
otehv2v2_use_unified_objective: bool = False
lambda_unified_rank: float = 0.15

# slot 解耦 + 路由重设计
otehv2v2_slot_disentangled: bool = False
otehv2v2_slot_router: str = "softmax"          # "softmax" | "sinkhorn"
otehv2v2_slot_cross_modal_cond: bool = False
otehv2v2_slot_adaptive_iters: bool = False
otehv2v2_sinkhorn_max_iters: int = 20
otehv2v2_convergence_threshold: float = 0.0
```

命名前缀 `otehv2v2_*` 与 `lambda_unified_*` 遵循需求7 AC4 的字符集约束（小写字母/数字/下划线），
且以模块名为前缀，与现有 `otehv2_*`、`lambda_rankevent_*` 风格一致，同时刻意与 V45 的
`otehv2_*` 前缀区分开（`otehv2v2_*`），避免 `--set otehv2_eps=...` 这种旧配置的字段名不小心
命中新代码路径。

**冲突检测（需求6 AC8）**：目前设计的选项彼此正交（clinical/unified-objective/slot-router/
cross-modal/adaptive-iters 互不冲突），暂无需要拒绝的组合；`model_factory.get_model` 在构建
`OTEHV2RankEventV2` 前调用一个轻量 `_validate_config(args)`，为将来新增互斥选项预留位置——当前
版本里该函数只做字段类型校验（需求7 AC3：未定义字段名/类型不可解析时终止），不做组合冲突拒绝，
因为现阶段没有互斥组合需要拒绝。

**Method Registry 新增（不修改现有条目，需求6 AC5）**：

```python
METHOD_REGISTRY["otehv2_rankevent_v2"] = (
    os.path.join("survot_rank", "research", "methods", "prognostic_event_transport"),
    "OTEHV2RankEventV2",
)
METHOD_ALIASES["45v2"] = "otehv2_rankevent_v2"
```

**数值向后兼容验证（需求6 AC7）**：新增一个 pytest 用例，用固定随机种子构造合成输入，分别用
`OTEHV2RankEvent`（旧类）和 `OTEHV2RankEventV2`（新类，所有新开关关闭）跑前向，断言两者输出
`logits` 逐元素 `allclose(atol=1e-5)`。这是本设计里"向后兼容"从口头承诺变成可执行断言的关键测试。

### Step E: 多癌种配置与结果汇总（需求 2）

`configs/v45_{study}.yaml`（`study ∈ {blca,brca,coadread,hnsc,stad}`）：以 `configs/v45_blca.yaml`
为模板，仅替换 `data.study` 与 `train.results_dir` 两个字段（对应需求2 AC2），其余字段（含
`data.data_root_dir`、`data.data_path`）逐字段保持一致——用一个小型生成脚本
`tools/gen_multicancer_configs.py` 读取模板 YAML、替换字段、写出 5 个文件，保证不会手工复制出
不一致。

`tools/aggregate_cross_cancer.py`：改造自仓库已有的 `tools/aggregate_multicancer.py`（当前只处理
硬编码的 Linux 绝对路径与 2 个癌种），新版本：
- 接受 `--results-root` 与 `--studies` 参数（默认 5 个癌种，字母序，对应需求2 AC7）。
- 对每个 study，在结果目录下 `rglob("summary.csv")`，读出后校验 `"mean" in df.index`
  （需求2 AC6：无效则标记 `status=invalid, reason=missing_mean_row`，不中断其余 study）。
- 目录缺失/无 summary.csv：标记 `status=missing`（需求2 AC4 的判定依据，比原脚本的
  `print`+`continue` 更结构化，输出里能看到具体缺失原因）。
- 输出列固定为 `val_cindex, val_cindex_ipcw, val_IBS, val_iauc`（需求2 AC3），同时写 CSV 与
  Markdown 两个文件（需求2 AC5）。

## Data Models

### 新增张量契约（贯穿 Step A/C/D）

| 名称 | 形状 | 来源 | 备注 |
|---|---|---|---|
| `x_clinical` | `[batch, clinical_feature_dim]` | dataset/collate 新增字段 | dtype float32；缺失值用 NaN 占位 |
| `Slot_Identity` | `[num_slots, dim]` | `nn.Parameter`，batch 间共享 | 需求4 |
| `Slot_State` | `[batch, num_slots, dim]` | 迭代变量 | 需求4，即"原 slots 变量" |
| `slots_clinical` | `[batch, num_slots_clinical, dim]` | `slot_attention_clinical` 输出 | 需求1 |
| `event_tokens` | `[batch, num_events, dim]` | 三模态融合输出 | 与二模态时形状契约一致 |
| Sinkhorn 分配矩阵 | `[batch, num_slots, num_input_tokens]` | `_log_sinkhorn_assign` | 行和≈1/num_slots，列和≈1/num_input_tokens，容差 1e-3 |

### 数据管线改动（`dataset_survival.py` / `core_utils.py`）

`SurvivalDataset.__getitem__` 新增可选返回：当 `dataset_factory.use_clinical_modality` 为真时，
额外从 `self.label_df`（已含 clinical_df 合并结果）里按配置的列名列表取出临床字段、数值化/独热
编码后拼成一个 1-D tensor 作为第 6 个返回值；`_collate_pathways` 和默认 collate 相应增加一路
`torch.stack`。`_unpack_data`/`_process_data_and_forward`（`core_utils.py`）在
`omics_format` 分支之外新增：

```python
if getattr(args, "otehv2v2_use_clinical", False):
    input_args["x_clinical"] = data[5].to(device)
```

不改变现有 5 元组路径的解析逻辑，只在配置开启时多读一路。

## Error Handling

| 场景 | 处理方式 | 对应需求 |
|---|---|---|
| Clinical 开关开启但输入缺失/维度不匹配 | `ValueError`，消息包含"Clinical 输入缺失或维度不匹配" | R1 AC3 |
| Unified_Objective 遇到全删失 batch | 返回有限标量（`_pairwise_margin_penalty` 返回 0，`base_nll` 的 uncensored 项自然为 0） | R3 AC2 |
| Unified_Objective 无可比对样本对 | `_pairwise_margin_penalty` 返回 `risk.sum()*0.0`（有梯度图，值为0） | R3 AC3 |
| Sinkhorn 路由迭代超过配置上限 | 循环 `for _ in range(min(configured_iters, sinkhorn_max_iters))`，静默截断，不报错（上限本身就是配置契约） | R5 AC1 |
| 配置字段名未定义/类型不可解析 | 在 `extended_args.process_args_extended` 里做一次 schema 校验，训练开始前 `sys.exit(1)` 并打印字段名，不创建 `results_dir` 下的任何写入 | R7 AC3 |
| 新增能力组合冲突（预留） | `model_factory._validate_config` 抛 `ValueError`，说明冲突的能力名 | R6 AC8 |
| Checkpoint 加载（关闭态） | 因为关闭态下 `OTEHV2RankEventV2` 的 `state_dict` 键集合与 `OTEHV2RankEvent` 完全一致（未实例化任何新模块），`load_state_dict(strict=True)` 天然成立，不需要额外的键名映射逻辑 | R6 AC4 |

## Testing Strategy

Windows 开发环境没有 WSI `.pt` 特征文件，因此测试策略分两层：

**Tier 1 — 合成张量单元/属性测试（本地可跑，CI 可跑，是本次实现的主要验证手段）**

在新建的 `tests/` 目录下（`tests/test_slot_attention_v2.py`、`tests/test_unified_objective.py`、
`tests/test_clinical_encoder.py`、`tests/test_otehv2_rankevent_v2.py`），每个新增模块单独用
`torch.randn`/`torch.randint` 构造 `batch∈{1,2,4}`、`num_slots/num_events∈[1,16]` 的小张量，
断言：
- 前向输出形状契约（对照 requirements.md 每条 AC 里给出的具体形状）；
- `torch.isnan(out).any()`/`torch.isinf(out).any()` 均为 False；
- `loss.backward()` 后相关参数 `.grad` 非 NaN/Inf；
- 关闭态与旧类/旧函数数值 `allclose`（1e-5~1e-6，按各需求给定容差）；
- Sinkhorn 边际约束、自适应迭代次数下限、跨模态条件化"换输入必换输出"等需求特有的属性。

这一层测试**不依赖 GPU、不依赖真实数据**，可以在当前 Windows 环境直接用 `pytest` 跑，是每个
增量步骤（Step A → B → C → D）完成后的验收标准。

**Tier 2 — 端到端 smoke 训练（仅远程 Linux GPU 环境）**

复用仓库已有的 `configs/smoke_v45_blca.yaml` 模式，新增
`configs/smoke_v45_v2_blca.yaml`（`survot_method: otehv2_rankevent_v2`，开启全部新开关，
`max_epochs: 1`，`max_smoke_batches: 2`），在真实数据齐备的远程环境上跑通一次，验证真实数据
管线（`x_clinical` 的临床列来源、collate、DataLoader）没有在合成测试覆盖不到的地方出问题。
这一层不在本次 Windows 环境的验收范围内，留给用户在服务器上执行。

**回归测试（贯穿全程）**：每完成一个 Step，运行一次"关闭态 vs 旧类"的数值一致性测试
（Tier 1 中的关键一环），确保增量修改没有意外影响 V45 baseline。
