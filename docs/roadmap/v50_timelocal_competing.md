# V50 时间局部竞争 — 发表路线图

## 1. 定位
- 一句话：在 OT 事件骨架上，用"时间局部竞争 hazard 头 + 风险/保护双通路"建模事件随时间的竞争。
- 代码：`survot_rank/research/methods/prognostic_event_transport/model.py` → `OTEHTimeLocalCompeting`
- 注册名：`otehv2_timelocal_competing`（别名 50）
- config：`configs/v50_blca.yaml` / P0 消融见 `configs/p0_experiments/v50_ablation_*`

## 2. 当前状态（当前最可信）
- **分箱 B 下实测最强、最稳**：v50_norank 三个 seed（22646/3/5）best≈0.698、last5 0.657–0.678、gap 小（-0.018~-0.058），是唯一 fold2 稳定在 0.64+ 的方法。
- 主线地位已被 P0 三种子复核坐实。
- ⚠️ 仍只在 BLCA 单癌种验证。

## 3. 方法缺陷 / 已知问题（最重要，必须正视）
- **🔴 致命：标榜的"创新正则"没贡献，甚至有害。** P0-3 消融（fold0+2）：
  - stripped(只 OT+EventSurv) best=0.6746 > +Spec 0.6510 > +Cover 0.6353 > full 0.6466。
  - 即 spec(时间特化)/cover(时间覆盖)/compete(竞争稳定)三项**净负增益**。v50 的优势来自"timelocal_competing 骨架 + 关 rankevent"，**不是来自你打算当卖点的三个正则**。审稿人做消融立刻会发现。
- 竞争风险本身是极成熟领域（DeepHit、cause-specific transformer、CRISP-NAM），"竞争风险"四个字毫无新意。
- 保护通路(protect)只有 0.001 的弱 L2 约束，容易发散。
- best 仍偶发早峰（fold4 best@ep0），说明评估口径仍需 last5 为准。

## 4. 缺失实验清单（发论文必需）
- [ ] **机制隔离实验**：固定 OT 融合骨架，只对比"标准 hazard 头" vs "risk/protect 双通路+时间竞争头"，证明这个头本身有独立贡献（当前无法把贡献从 OT 里剥离）。
- [ ] **重塑后的正则消融**：既然 spec/cover/compete 有害，要么删掉、要么找到真正有用的时间正则形式，并重跑消融证明"新卖点"确实 work。
- [ ] **跨癌种**：至少 BRCA + HNSC + STAD，证明不是只对 BLCA 过拟合。
- [ ] **战胜公开 SOTA**：MOTCat / SurvPath / OTSurv 同分箱同划分直比。
- [ ] **完整 5-fold × ≥3 固定 seed**（当前 3 seed 部分是 2 折 P0）。

## 5. 可解释性怎么做到位
- 卖点：每个事件在每个时间 bin 上的"责任权重"(risk_g softmax over events) → 可画出"哪个事件对应早期/中期/晚期风险"。
- 必须证明：这个 event→时间段的分化**临床上可解释**（如早期复发 vs 晚期进展），而不是随机分配。
- risk/protect 双通路 → 展示哪些模态证据是"保护性因素"（负 hazard 贡献），对上已知预后生物学。
- 建议加"事件-时间段聚类一致性"可视化，比现在隐式的 spec/cover 正则更有说服力。

## 6. 发表门槛 checklist
- [ ] 有一个 demonstrably 有贡献的机制（当前正则不达标 → 卖点要重写成"OT 融合 × 时间竞争 hazard 头"的耦合）
- [ ] 多癌种一致增益
- [ ] 同口径战胜 ≥2 个公开 SOTA
- [ ] 多 seed 完整 5-fold，报 mean±std，以 last5/收敛值为准
- [ ] 可解释性有定量+定性证据（不只是画图）

## 7. 目标分区 + 竞品文献（条件性）
- 现实：**2 区**（JBHI/CBM 级）——因为它是实测最强、最稳。
- 天花板：**1 区**（MedIA/TMI），前提是 §4/§6 全部达标且卖点重塑成功。
- 竞品：竞争风险深度学习（DeepHit、[cause-specific hazard transformer](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-024-05799-2)、[CRISP-NAM](https://arxiv.org/html/2505.21360v5)）；OT 多模态生存（MOTCat/SurvPath/OTSurv）。

## 8. 一句话结论
分数最硬，但**创新点被自己的消融证伪了**——现在离发表差的不是分数，是"一个能站住的、消融里确实有贡献的机制卖点"。先做机制隔离实验 + 重塑正则，再上跨癌种和 SOTA 对比。
