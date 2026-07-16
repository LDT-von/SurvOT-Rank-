# Stagewise 分阶段预后运输 (SPT) — 发表路线图

## 1. 定位
- 一句话：RG-ET 只学一套共享的 prognostic pair cost，SPT 给每个事件阶段学**独立的**阶段特异 pair cost，喂对应事件 token。
- 代码：`survot_rank/research/methods/stagewise_prognostic_transport/model.py` → `StagewisePrognosticTransport`（继承 RG-ET）
- 注册名：`stagewise_prognostic_transport`
- config：`configs/stagewise_prognostic_transport_blca.yaml`

## 2. 当前状态
- 旧 A\* 数据：best=0.6741、"能拟合，30ep 后仍平稳"（是崩溃方法里相对稳的）。
- 分箱 B 下未单独完整验证；受 eps bug 影响，**已修**，待 `run_fix_verify_fold02.sh` 重跑。

## 3. 方法缺陷 / 已知问题
- **创新幅度有限**：审稿人极易把"分阶段=多套 OT"说成 RG-ET 的自然扩展。这是天花板的主要压制因素。
- 阶段数(4)与离散时间 bin 数(4)耦合，缺乏独立设计动机。
- 阶段间没有显式约束保证"阶段真的对应不同时间语义"，可能退化成冗余的多套 OT。

## 4. 缺失实验清单（发论文必需）
- [ ] **关键消融**：stage-specific OT vs shared OT(=RG-ET)，必须证明分阶段带来**显著且稳定**的增益，否则方法不成立。
- [ ] 阶段数敏感性（2/4/6 阶段）。
- [ ] 证明各阶段 plan 确实分化（不同阶段关注不同 slot-pair），而非冗余。
- [ ] 跨癌种 + 多 seed。

## 5. 可解释性怎么做到位
- 卖点：每个阶段一套 OT plan → 可视化"早/中/晚阶段分别关注哪些 WSI-pathway 对"。
- 必须证明阶段分化对应可解释的时间语义（配合 stage_order_loss 的单调约束展示）。

## 6. 发表门槛 checklist
- [ ] stage vs shared OT 消融显著胜出（这是生死线）
- [ ] 阶段分化可视化 + 可解释
- [ ] 跨癌种一致 + 多 seed
- [ ] 与 SOTA 对比

## 7. 目标分区（条件性）
- 现实/天花板：**2 区**。"自然扩展"的质疑很难突破到 1 区，除非分阶段带来大幅且机制上不可替代的增益。

## 8. 一句话结论
最容易讲清楚，也最容易被说成"RG-ET 加了几套 OT"。**能不能发，取决于一个消融**：stage-specific 是否显著稳定优于 shared。做不出这个差距，它就只能当 RG-ET 的消融项。
