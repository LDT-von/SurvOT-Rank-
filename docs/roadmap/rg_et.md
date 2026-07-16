# RG-ET 排序引导事件运输 — 发表路线图

## 1. 定位
- 一句话：把一个可学习的 prognostic pair cost 注入 OT 代价（改变运输几何），再用连续时间排序 + 阶段序约束训练，损失紧凑（OT + rank + stage-order）。
- 代码：`survot_rank/research/methods/rank_guided_event_transport/model.py` → `RankGuidedEventTransport`（继承 OTEventHazardV2）
- 注册名：`rank_guided_event_transport`
- config：`configs/rank_guided_event_transport_blca.yaml`

## 2. 当前状态
- 旧数据：能拟合(train 0.86)，但 **epoch 20 后 IBS 从 0.25 崩到 0.75、iAUC 崩到 0.39**——生存分布彻底崩溃。best=0.6496/last5=0.5923，修复前，**已作废**。
- ⏳ eps 单调修复 + 梯度累积对 RG-ET 帮助应最大（它的 rank loss 在 batch=4 下几乎是噪声）。待重跑。

## 3. 方法缺陷 / 已知问题
- **机制改动幅度有限**："预后 cost 注入 OT"是一个不错但不大的改动，独立成篇的问题定义偏弱。
- **IBS 崩溃**：排序损失把 hazard 推向极端 → 生存曲线校准崩坏。历史上试过 PCGrad 但从未真正接入。
- batch=4 下 rank loss 只有 1–3 个可比对，梯度接近噪声（梯度累积后应改善，待验证）。

## 4. 缺失实验清单（发论文必需）
- [ ] **prognostic-cost 消融**：注入 vs 不注入，证明这个核心机制有贡献。
- [ ] 修复后确认 IBS 不再崩（这是 RG-ET 能否成立的生死线）。
- [ ] rank loss 权重敏感性 + 有效 batch(累积)对 rank 的影响。
- [ ] 跨癌种 + 多 seed。

## 5. 可解释性怎么做到位
- 卖点：prognostic pair cost → 哪些 WSI-pathway 对被判为"预后相关"从而降低运输代价被选中。
- 展示注入前后 OT plan 的变化，说明预后信号确实改变了运输几何。

## 6. 发表门槛 checklist
- [ ] IBS/校准全程稳定（修复后验证）
- [ ] prognostic-cost 消融显著有效
- [ ] 预测性能不落后于 v50/SPT
- [ ] 跨癌种 + 多 seed + SOTA 对比

## 7. 目标分区（条件性）
- 现实/天花板：**2 区**。简洁完整，但创新幅度有限，冲 1 区困难。可作为 SPT/FET/DCT 这条继承链的**基线/消融锚点**，价值不小。

## 8. 一句话结论
最简洁，但也最"小"。成败在两点：修复后 **IBS 是否还崩**、prognostic-cost 消融是否显著。它更可能作为整条 OT-event 线的干净基线存在，而非单独主打 1 区。
