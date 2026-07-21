# DCT：2026 文献边界、瘦身审计与下一版研究路线

> 更新：2026-07-21  
> 代码基线：`5af1c93`（本地评价提交不改变模型代码）  
> 原则：先完成现有 v3.5 R/Q/G/L 单变量筛选；任何新机制均作为诊断变量，不覆盖已归档的 v3.3。

## 1. 结论

DCT 的创新性可以从当前约 `6.5/10` 提高，但不能靠继续堆叠 prototype、cross-attention、MoE 或 pathway expert。2026 年相关工作已经明显占据这些部件。最有希望的提升方式是把 DCT 从“高分预测器 + post-hoc transport”收敛为一个可证伪的科学机制：

> **训练折删失风险集定义干预参考；多几何一致性决定 transport 证据是否可靠；在固定边缘约束下沿风险参考连续干预 cost 并重新求解 OT；用方向性、剂量响应和随机参考阴性对照验证预测敏感性。**

其中只有“可靠性调节的 transport intervention”适合成为新增机制。跨 batch memory、降维和缺失模态处理只能作为稳定训练或扩展实验，不应包装成并列创新点。

## 2. 当前代码瘦身审计

### 2.1 确认没有梯度的继承参数

在包含 factual logits 与当前 IPCW rank auxiliary loss 的完整反向传播中，下列参数没有梯度：

- `prognostic_pair_cost`：来自 Rank-Guided 父类；DCT 自己改用 `stage_pair_cost`。
- `stage_head`：只服务父类 `_stage_order_loss`；DCT 使用自己的 stage-risk 路径。
- `stage_score`：来自 Stagewise 父类；DCT 最终风险由 `event_hazard + event_gate` 解码。

16 维单元配置的实测为：总参数 `91,664`，无梯度参数 `1,270`。正式配置默认 `wsi_projection_dim=256` 时，上述死参数约 `265k`，只占约 `30.37M` 的不足 `1%`。结论是：**应从最终 paper model 删除以清理叙事，但它们不是过拟合和低分的主因。** 为保持旧 checkpoint 可加载，历史 DCT 类应保留；最终配方确认后再建立干净的 paper-only 类。

### 2.2 不能直接删除的内容

- `dct_lambda_ot/rank/anchor/stage_risk/coordinate=0` 的代码是已做/待做消融入口，不应在论文实验结束前删除。
- `fet_lambda_sparse/faith=0` 是继承历史，最终 paper-only 类可移除，当前类保留用于旧结果复现。
- 历史 config、runner 和结果摘要不是模型冗余，而是复现证据；应移动到 `configs/archive/`、`scripts/archive/`，不能直接删除。
- `risk_anchor_costs` 虽不参与当前训练目标，却是 post-hoc intervention 的必要参考，不能删。

### 2.3 真正需要实验决定的“大头”

1. `wsi_projection_dim=256`、双层 event Transformer 与两套 Slot Attention 才是容量主体；先完成 v3.5L（128 维、1 层）筛选。
2. 每阶段同时计算 cosine/euclidean/dot 三种几何：训练 factual 共 `4×3=12` 个 Sinkhorn，评估 factual/low/high 共 `36` 个。需要比较单几何、三几何均值和三几何可靠性调节。
3. local Slot Attention 后再投影到 global prototypes 是双重压缩。必须比较 local-only、prototype-only 和 local→prototype，不能因名字好听默认双重结构必要。
4. 论文最终训练目标应继续保持 `NLL + IPCW rank`；不得重新同时打开五个旧辅助损失。

## 3. 2026 年最相关工作及处理方式

| 工作 | 2026 状态 | 它解决什么 | 对 DCT 的决定 |
|---|---|---|---|
| [FeatProto](https://doi.org/10.1109/JBHI.2026.3710553) | IEEE JBHI 2026 | 全局/局部 WSI 特征、genomics 统一 prototype 空间、EMA prototype 更新、层次匹配 | **不要移植。** 它进一步占据 prototype 解释与稳定对应；DCT 应将 prototype 降为坐标工具而非主创新 |
| [ProtoPathway](https://arxiv.org/abs/2605.21454) | 2026 预印本、审稿中 | learnable morphology prototypes、Reactome 图、prototype×pathway attention、稳定身份解释 | **不要移植。** 加 pathway GNN 或 prototype attention 会直接增强撞题与缝合观感 |
| [MoMKD](https://arxiv.org/abs/2602.21395) | CVPR 2026 | momentum memory 扩大跨 batch 监督上下文，缓解 batch-local alignment 不稳定 | **吸收原则，不搬网络。** DCT 已有 train-epoch 内 IPCW rank memory；将 `memory_size` 作为小 batch/稀事件癌种的优化消融，不列创新点 |
| [EMMS](https://arxiv.org/abs/2606.20757) | 2026 预印本 | 用证据不确定性和模态可靠性处理缺失模态生存融合 | **可吸收可靠性思想。** 不使用 Dempster–Shafer/GRFN；改为 DCT 自身三种 OT 几何的一致性可靠度 |
| [Missing-aware NSCLC survival](https://doi.org/10.1038/s41746-026-02783-3) | npj Digital Medicine 2026 | CT/WSI/clinical 的 missing-aware intermediate fusion | **只做扩展实验。** 当前 TCGA WSI+RNA 完整样本不能借此声称解决缺失模态；未来 modality dropout/缺失模态评估可采用其协议思想 |
| [EAGLE](https://doi.org/10.1038/s41467-026-74918-9) | Nature Communications 2026 | 用少量信息 tile 获得高效、可审计 WSI 表征，并做外部验证 | **用于输入和实验协议，不加进 DCT 核心。** 可比较 EAGLE/UNI2-h 特征或更少 patch，但不能把 patch selector 当 DCT 创新 |
| [CURE](https://arxiv.org/abs/2602.19987) | 2026 预印本 | retrieval-aware multimodal time-to-event counterfactual prediction、cross-attention、MoE | **作为命名与宣称警戒。** 不搬 MoE；DCT 的 counterfactual 必须限定为 model-based transport sensitivity，而非治疗反事实 |

## 4. 唯一建议新增的机制：可靠性调节证据边缘

### 4.1 动机

DCT 当前让 evidence gate 直接改变 OT 行列边缘；若 cosine、euclidean、dot 三种几何对同一患者的匹配意见冲突，gate 仍可把不可靠证据强行放大。借鉴 EMMS 的“可靠信息应获得更大融合权重”原则，但用 DCT 自身的 transport 几何定义可靠性，不引入新的专家、注意力或损失。

对患者 `i`、阶段 `s`、几何 `g` 的 cost，定义边分布：

```text
q_is^g = softmax(-vec(C_is^g) / tau)
qbar_is = mean_g(q_is^g)
u_is = mean_g KL(q_is^g || qbar_is) / log(G)
r_is = clamp(1 - u_is, 0, 1)
```

其中 `r_is` 是三几何一致性可靠度。令原 evidence marginal strength 为 `lambda0`，新强度为：

```text
lambda_is = lambda0 * ((1-gamma) + gamma*r_is)
a'_is = (1-lambda_is)*Uniform + lambda_is*a_evidence
b'_is = (1-lambda_is)*Uniform + lambda_is*b_evidence
```

- `gamma=0` 必须逐元素复现当前 DCT。
- `gamma=1` 时，几何一致才允许 evidence gate 强烈改变质量分配。
- 不增加 trainable parameters，不增加辅助损失，只改变 transport 的可信度控制。
- 论文中应称 **Reliability-Tempered Evidence Marginals (RTEM)**；它是 DCT 内部机制，不是新方法家族。

### 4.2 为什么它比加入 FeatProto/MoE 更合适

- 直接解决 DCT 的现有问题：evidence marginal 可能在低证据或跨几何冲突时过拟合。
- 与 re-Sinkhorn、evidence-conditioned marginals 和风险干预处于同一数学对象上，叙事不分叉。
- 不增加第三个训练损失，也不显著增加参数。
- 能产生可检验预测：可靠度低的患者应具有更小、更不稳定的 intervention response；可靠度高的患者应有更清晰剂量响应。

## 5. 实验顺序

### 阶段 A：不得跳过的现有筛选

先完成 v3.5 R/Q/G/L 的 fold0/2。尤其必须先知道：

- R 是否消除重复验证波动；
- L 是否在 BRCA/UCEC 降低 best-last5 gap 且不损失 BLCA；
- G 是否证明 evidence marginals 不是负贡献；
- Q 是否真的改善绑定，而不是增加可学习初始化。

在这些结果出来前，不应把 RTEM 写成正式 v4 默认。

### 阶段 B：两个单变量诊断

固定 v3.5R 的其余设置，只增加：

1. **U：RTEM**，`gamma=1`；回答跨几何可靠性是否改善稳定性。
2. **M：现有 IPCW memory=64**；回答跨 batch 风险集是否改善 BRCA/UCEC 稀事件排序。M 是 MoMKD 原则的任务内适配，但不是论文创新。

先跑 BLCA、BRCA、LUAD、UCEC 的 fold0/2。U 与 M 不在第一轮组合，避免无法归因。

### 阶段 C：机制真实性

- intervention strength `alpha ∈ {0,.25,.5,.75,1}` 的风险剂量响应；
- low/high anchor 方向一致率与 fold/seed 置信区间；
- 随机 anchor、打乱 stage、固定 coupling、不 re-Sinkhorn 阴性对照；
- reliability 分位数组的 intervention response、校准和 C-index；
- train-fold reference 与错误全数据 reference 的泄漏对照。

### 采用门槛

RTEM 只有满足以下任一条件才进入最终 DCT：

- fold0/2 平均 C-index 至少提高 `0.005`，且没有癌种下降超过 `0.01`；或
- C-index 基本持平（绝对差 `<0.005`），但 best-last5 gap 至少降低 `20%`，并显著提高 intervention 方向一致率/剂量响应。

否则删除 RTEM，保留最简 DCT，不为论文故事强留模块。

## 6. 预期创新上限

- 只加 2026 模块、没有机制实验：创新性会从 `6.5` 降到约 `5.5–6.0`，因为更像 FeatProto/ProtoPathway/EMMS 拼接。
- 完成瘦身、结构归因和风险干预阴性对照：可稳定到约 `7.0`。
- RTEM 在多癌种同时改善稳定性，并证明“可靠性 → transport response → survival sensitivity”的预先规定关系：可到约 `7.0–7.5`。
- 即使如此，也不能把它写成治疗因果反事实；更安全的论文名称应使用 `transport sensitivity/intervention`。
