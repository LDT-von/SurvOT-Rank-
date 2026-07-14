# 现状快照与下一步分析 (2026-07-14)

> 对应 git tag `snapshot-2026-07-14-binningB-v50best`（commit `4fb6935`）。
> 本文档回答三个问题：(1) 现在的方法还能怎么创新；(2) 损失函数够不够、有没有问题；
> (3) 接下来必须做哪些实验才能把结论立住。

---

## 1. 现状快照（标注情况和分数）

### 1.1 代码谱系（都源自 SlotSPE 骨架，未脱离）

```
SlotSPE 原始（slot attention + cross-attn + MoE decoder）
  └─ V31 OTEventHazardV2Survival（cross-attn → OT 融合，唯一换掉的底层机制）
       ├─ V45 OTEHV2RankEvent（+全局残差 +4项rankevent损失）
       │    ├─ V45v2 OTEHV2RankEventV2（+临床模态/统一目标/可学习权重开关）
       │    └─ V50 OTEHTimeLocalCompeting（+risk/protect双通路 +时间竞争hazard）
       └─ RG-ET RankGuidedEventTransport（跳过V45，直接3损失+prognostic cost注入OT）
            └─ SPT StagewisePrognosticTransport（+分阶段OT plan）
                 └─ FET FaithfulEvidenceTransport（+证据门控keep/removed）
                      └─ DCT DistributionalCounterfactualTransport（+风险原型插值反事实）
V60（独立方法，未继承V45系）：stable log-domain OT + compact event ranking
CA-TET（独立方法）：删失感知risk-set + 阶段化OT + 干预验证
```

Slot attention / WSI_Mlp / SNN_Block 编码器全部继承自 SlotSPE 官方代码（已逐字节比对一致），**没有任何方法脱离这套底层特征编码骨架**，创新都发生在"OT融合之后 → hazard输出"这一段。

### 1.2 分箱 B（当前唯一可信口径）5-fold 排名

| # | 方法 | seed | avg_best | avg_last5 | gap |
|---|---|:---:|:---:|:---:|:---:|
| 1 | v50_norank | 22646 | 0.7148 | **0.6572 ± 0.011** | -0.058 |
| 2 | v45v2_norank | 323 | 0.7063 | 0.6394 ± 0.025 | -0.067 |
| 3 | v45_norank | None(未记录) | 0.6848 | 0.6406 ± 0.025 | -0.044 |
| 4 | V51 seed3 | 3 | 0.6786 | 0.6207 ± 0.058 | -0.058 |
| 5 | V60 | 3 | 0.6791 | 0.6063 ± 0.031 | -0.073 |
| 6 | V51 seed5 | 5 | 0.6583 | 0.6088 ± 0.047 | -0.050 |

⚠️ 全部单一随机种子，探索性质，非最终结论。

### 1.3 已排除/已证伪的假设

| 假设 | 结论 | 证据 |
|---|---|---|
| V45 的 8 项辅助损失都有用 | ❌ 证伪 | V2/V4a/V4b/v45_norank 关闭 rankevent 4 项后分数不降反升 |
| fold2 低分是模型问题 | ❌ 部分证伪 | v45v2_norank fold2 从 0.5685(ep2)回弹到 0.6665(ep12) → 是慢收敛不是坏折；但 V51 fold2 两个seed都崩 → 是架构问题，需分方法讨论 |
| clinical 模态（age/gender）有增益 | ❌ 暂不成立 | v45v2(+clinical) vs v45(无) last5 几乎相等(0.6394 vs 0.6406) |
| V60 的 OT-event backbone 能力不行 | ❌ 不成立 | best不低(0.6791)，问题是没用AdamW+wd导致过拟合(-0.073)，不是架构问题 |
| SlimBridge/Modality Dropout 架构可用 | ✅ 证实为问题 | 两个独立seed的fold2都<0.60，排除分箱和运气因素 |
| PCGrad能解决RG-ET的IBS崩塌 | ⚠️ 未验证 | pcgrad.py写好但从未接入训练循环，标注"完成"的结果实际不含PCGrad |

---

## 2. 损失函数审计：够不够，有没有问题

### 2.1 当前各方法损失数一览

| 方法 | 损失数 | 具体项 | 评价 |
|---|:---:|---|---|
| V45/V45v2(未关) | 8 | OT+Div+Recon+GateEnt+EventNLL+PerEvent+Rank+GlobalCons | **过多，已证伪**，互相打架致 train_cidx<0.5 |
| V50(未关) | 11 | 上8项+Spec+Cover+Compete | 同上问题+3项，更重 |
| V45/V45v2/V50(norank) | 4/4/7 | 关rankevent后 | 当前实际在跑的配置 |
| RG-ET/SPT | 3 | OT+Rank+StageOrder | 已经很紧凑 |
| FET | 5 | +sparse+faith | 略多但可解释性驱动，合理 |
| CA-TET | 3 | OT+risk-set rank+intervention | 紧凑，但rank在batch=4下几乎失效（见2.3）|
| DCT | 4 | OT+rank+反事实+原型稀疏 | 反事实项是循环论证（margin loss强制方向）|
| V60 | 未详细审计 | "compact event ranking" | 需要补充审计 |

**结论：数量本身已经在往"够用"的方向收敛（3~7项），核心矛盾已从"损失太多"转移到下面两个新问题。**

### 2.2 问题一：batch_size=4 让排序类损失普遍失效

所有方法的 `rg_lambda_rank` / `spt_lambda_rank` / `catet_lambda_rank` 都依赖 batch 内的成对比较（Cox-style pairwise ranking 或 risk-set）。batch=4 时：
- 每个 batch 平均只有 1-3 个可比对（uncensored + 时间可比）
- 梯度信号接近纯噪声
- **这意味着目前"关rankevent"和"排序损失"两条线的对比，可能都被batch=4这个共同瓶颈污染**——不知道排序损失到底有没有用，因为它从来没在够大的batch下测过。

**这是当前最大的一个未验证变量，比继续设计新损失更优先。**

### 2.3 问题二：intervention/faithfulness 类损失的方向性问题

DCT 的反事实损失（`_counterfactual_loss`）显式要求"low_risk < factual < high_risk"，这是把结论当作训练目标——不可证伪，前面已判定为循环论证。

CA-TET 的 intervention 损失（sufficiency + comprehensiveness）**方向无关**（只要求删除证据后风险要变，不预设方向），这是更干净的写法，但同样从未在够大batch下测过是否真的比事后注意力更忠实。

### 2.4 问题三：没有一次做过"loss ablation × 正确分箱 B"的组合

`loss_group_sweep.py`（curated/pruned/full三档）目前的两次实际运行都在**分箱A\***下（已作废），且被判定"随机组合无意义"提前终止。**分箱B修复后，没有任何方法重新做过系统的损失消融**——现在"v50最优"的结论里，7个损失（OT+Div+Recon+EventSurv+Spec+Cover+Compete）哪个真正贡献、哪个是噪声，完全未知。

### 2.5 损失函数层面的结论

**够用（数量上），但可信度不够**：
1. 排序类损失从未在合理batch下测试过真实效果——**待办第一优先级**。
2. v50的7个损失从未消融过，"time-local机制有效"这个结论目前建立在"整体7项 > 关闭4项的V45"，无法定位到底哪个子机制起作用。
3. DCT的反事实机制方法论有硬伤，不建议再投入。

---

## 3. 还能不能继续创新——三个方向，按可行性排序

### 方向A（最推荐）：把"时间局部竞争"机制单独拎出来，去掉OT依赖做纯净消融

V50当前把"时间局部竞争hazard"和"OT融合"耦合在一起，无法证明前者的贡献独立于后者。
**具体做法**：设计一个V50的消融变体——保留OT融合骨架不变，只对比"标准hazard头" vs "risk/protect双通路+时间竞争hazard头"，固定其余全部超参。这是目前唯一还没做过、且能直接回答"V50真正的创新点是否work"的实验，比设计全新方法性价比高得多。

### 方向B（有一定新意但要小心）：risk-set排序在合理batch下的真实效果

前面提过的CA-TET/RG-ET的排序损失从未在batch≥16下测试。**如果**在合理batch下risk-set排序确实带来增益，这会是一个可以写的发现（"删失感知排序在小batch下被低估"）；**如果**没有增益，至少能排除一个长期悬而未决的疑点。这不是全新idea，是把现有idea验证清楚，但对论文的说服力贡献很大。

### 方向C（探索性，风险较高）：借鉴检索到的MoE-hazard文献做真正的时间路由

上次检索到的 [Dual MoE for Discrete-Time Survival (2510.26014)](https://arxiv.org/html/2510.26014v1) 和 [MoE Heads for Survival Calibration (2511.09567)](https://arxiv.org/html/2511.09567v2) 都强调"专家表达能力"和"聚类可解释性"的权衡。V50现在的时间竞争机制某种程度上是MoE-hazard的一个简化变体（只用softmax路由，没有显式专家聚类目标）。**可以考虑**：给V50加一个显式的"事件-时间段聚类一致性"正则，让每个事件真正对应一段可解释的临床时期（比如"早期复发风险""晚期进展风险"），这样可解释性会强于现在的隐式spec/cover正则。**风险**：这是新增复杂度，与"损失已经够多"的诊断方向相反，只有在方向A证明"时间竞争机制确实有效"之后才值得投入。

**不建议的方向**：继续派生新的独立方法名（如再来一个V70）。当前问题不是idea不够，是**已有7-8个变体没有一个被完整消融和多seed验证过**。再加新方法只会让"损失函数是否够用"这个问题更难回答。

---

## 4. 接下来必须做的实验（按优先级）

### P0（不做这些，任何结论都立不住）

1. **v45全8损失 + 分箱B + 5-fold** 的对照实验（缺失）。当前"关rankevent+分箱B=0.6406"无法拆分是分箱功劳还是关损失功劳。
2. **v50_norank 补固定seed 3和5**（当前只有seed=22646一个点）。确认0.6572是否稳定复现，还是运气。
3. **v50的7损失消融**（对应方向A）：至少跑"仅OT+EventSurv"vs"+Spec+Cover"vs"+Compete"vs"全开"四档，在分箱B、固定seed下。

### P1（决定能不能写论文）

4. **v50_norank 跨癌种验证**（至少BRCA或HNSC）。只在BLCA上测过，审稿人第一个问题就是"是不是只对BLCA过拟合"。
5. **排序损失在batch=16或32下重跑一次**（对应方向B），确认是否被batch=4低估。
6. **PCGrad真正接入train_one_epoch并重跑RG-ET**，回答IBS崩塌问题是否真能被梯度手术解决——这是唯一"已经写好但没验证"的现成工具，性价比最高。

### P2（锦上添花，非必需）

7. V60补AdamW+wd重跑，排除"过拟合而非架构问题"的干扰后再决定是否保留这个方法。
8. 与至少一个外部真实SOTA（MOTCat/SurvPath）在同一分箱口径下直接对比。

---

## 5. 一句话总结

代码没有脱离SlotSPE骨架，创新集中在hazard头这一段，其中V50的"时间竞争+双通路"是目前唯一在两种分箱口径下都排第一的机制级创新。损失函数数量已经够精简，但**排序类损失从未在合理batch下验证过真实效果**、**v50的7个损失从未消融过**，这两个空白比设计新idea更紧迫。下一步不建议再派生新方法，应把P0三项实验做完，才能知道现在手里到底攥着什么。
