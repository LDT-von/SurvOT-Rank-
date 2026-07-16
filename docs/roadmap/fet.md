# FET 忠实证据运输 — 发表路线图

## 1. 定位
- 一句话：每个阶段学一个 slot-pair 证据门控，门控**改变该事件用的 OT plan**，让解释耦合到预测路径而非旁路注意力；配 sparse + faithfulness 正则。
- 代码：`survot_rank/research/methods/faithful_evidence_transport/model.py` → `FaithfulEvidenceTransport`（继承 SPT）
- 注册名：`faithful_evidence_transport`
- config：`configs/faithful_evidence_transport_blca.yaml`

## 2. 当前状态
- 旧 A\* 数据：best=0.6837 但 **欠拟合（train 仅 0.62）+ IBS 剧烈震荡（0.21→0.60 反复）+ val 单点尖峰**。修复前，**已作废**。
- ⏳ 待修复后重跑（eps 修复 + 梯度累积应显著改善震荡）。

## 3. 方法缺陷 / 已知问题
- **可解释性范式成熟、增量性强**：sufficiency/comprehensiveness 是 XAI 标准（ERASER 系），搬到生存 OT 是应用创新，不是新范式 → 天花板受限。
- **keep/removed 多次解码共享头**，梯度互相拉扯，可能是 IBS 震荡的来源之一。
- faithfulness 项用 `relu(margin - |risk_full - risk_removed|)`，只要求"删了要变"，方向无关（干净），但 margin 选取敏感。

## 4. 缺失实验清单（发论文必需）
- [ ] **忠实性基准**：证据门控 vs 事后注意力/IG/SHAP 在 sufficiency/comprehensiveness 上的定量对比，必须明显更忠实。
- [ ] keep / removed / random 三档干预对照。
- [ ] 修复后确认 IBS 不再震荡、train 能拟合到 0.7+。
- [ ] 跨癌种 + 多 seed。

## 5. 可解释性怎么做到位（这是本方法立命之本）
- 卖点：解释直接改 OT plan → 预测和解释同一条路径，天然忠实。
- 必须定量证明：删掉高证据边 → 风险显著变（comprehensiveness）；只留高证据边 → 重现 factual（sufficiency）；且优于 detached attention。
- 展示 stage×slot-pair 证据 + WSI/omic slot assignment，对上已知预后通路。

## 6. 发表门槛 checklist
- [ ] 修复后 IBS 稳定、能正常拟合
- [ ] 忠实性定量显著优于事后解释基线（核心卖点）
- [ ] 预测性能不明显落后 v50/CATE-T
- [ ] 跨癌种 + 多 seed

## 7. 目标分区 + 竞品文献（条件性）
- 现实/天花板：**2 区**（可解释性驱动，增量应用难上 1 区）。
- 竞品：faithfulness 评估范式（[sufficiency/comprehensiveness 对比研究](https://ar5iv.labs.arxiv.org/html/2204.05514)）；病理反事实解释 [MoPaDi](https://pubmed.ncbi.nlm.nih.gov/39554184/)；CATE-T（同项目内的干预式解释，注意区分）。

## 8. 一句话结论
是一个"可解释性论文"，成败在**忠实性能否定量打赢事后解释基线**，而不是 c-index。先修复确认不震荡，再把忠实性对比做扎实，稳 2 区。
