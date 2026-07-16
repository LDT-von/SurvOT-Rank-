# DCT 分布反事实运输 — 发表路线图

## 1. 定位
- 一句话：学两套(低危/高危)阶段化证据分布，把病人 factual 的 OT plan 向两端插值，得到"模型忠实的隐式反事实"并报告风险变化 + 运输距离。
- 代码：`survot_rank/research/methods/distributional_counterfactual_transport/model.py` → `DistributionalCounterfactualTransport`（继承 FET）
- 注册名：`distributional_counterfactual_transport`
- config：`configs/distributional_counterfactual_transport_blca.yaml`

## 2. 当前状态
- 旧数据 best=0.7217（全场最高）但 last5=0.5920、gap=-0.130——**best 是 early-peak 噪声，30ep 内未收敛**。修复前数据，**已作废**。
- ⏳ 待修复后重跑；DCT 尤其可能需要 60ep（历史上 best 出现在最后一个 epoch）。

## 3. 方法缺陷 / 已知问题
- **🔴 反事实损失是循环论证**：`_counterfactual_loss` 显式要求 `low_risk < factual < high_risk`（by margin），把结论当训练目标，不可证伪。**必须改成方向无关的形式**（对照 CATE-T 的干预写法）。
- **未收敛**：3 次解码(factual/low/high)共享头 + 反事实约束，收敛慢，30ep 不够。
- 风险原型 `risk_prototypes` 是全局共享参数，可能学成两个平凡端点。

## 4. 缺失实验清单（发论文必需）
- [ ] **先修循环论证损失**，再谈其它——这是前置条件。
- [ ] 延长到 60ep + 补正则，确认真实平台在哪。
- [ ] **与 CURE 正面对比**（见 §7），讲清 OT-plan 反事实 vs 检索式反事实的区别。
- [ ] 反事实忠实性验证：插值产生的风险变化是否单调、可解释，还是循环损失逼出来的假象。
- [ ] 跨癌种 + 多 seed。

## 5. 可解释性怎么做到位
- 核心卖点：transport-plan 层面的反事实——"把证据分布搬向高危/低危端，风险如何变、需要搬多远(transport distance)"。
- 必须证明：反事实风险 delta 和 transport distance 是**模型真实敏感度**的反映，不是 margin 损失强行拉出来的（这正是循环论证要害）。
- 展示 low/high 原型对应的 slot-pair 模式是否对上已知高危/低危生物学。

## 6. 发表门槛 checklist
- [ ] 循环论证损失改为方向无关/可证伪
- [ ] 修复后能收敛（可能需 60ep）
- [ ] 与 CURE 同口径对比并胜出或有明确差异化
- [ ] 反事实忠实性有独立验证（非损失自证）
- [ ] 多癌种 + SOTA + 多 seed

## 7. 目标分区 + 竞品文献（条件性）
- 现实：改损失前**不建议投**；改好后方向其实是空的，**1–2 区**可争。
- 竞品/撞车：
  - [AISTATS 2025 Oral: Distributional Counterfactual Explanation w/ OT](https://arxiv.org/abs/2401.13112) — **仅命名/框架撞车，是通用 XAI 不是生存**，related work 引用+区分即可，非死因。
  - [CURE 2602.19987](https://arxiv.org/abs/2602.19987) — 最近的邻居（反事实+多模态+生存，检索式），**必须对比**。
  - [MoPaDi](https://pubmed.ncbi.nlm.nih.gov/39554184/) — 病理反事实(diffusion, 分类)，威胁较小。
- 关键判断：**"OT 分布反事实 + 多模态生存"这个精确组合，文献里没人做过**——方向是空的，新颖性成立。

## 8. 一句话结论
方向空白是最大优点，但**循环论证损失是硬伤、且没收敛**。路线很清晰：先把反事实损失改成可证伪 → 延长训练 → 对比 CURE。改完这三点，DCT 是"文献空白最大"的候选。
