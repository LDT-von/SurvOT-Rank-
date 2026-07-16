# CATE-T 删失感知时间证据运输 — 发表路线图

## 1. 定位
- 一句话：把删失感知的 risk-set 监督注入"分阶段 OT 几何"，并在**同一个用于预测的 transport plan** 上做 keep/remove 干预来验证证据忠实性。
- 代码：`survot_rank/research/methods/censoring_aware_temporal_evidence_transport/model.py` → `CensoringAwareTemporalEvidenceTransport`
- 注册名：`censoring_aware_temporal_evidence_transport`
- config：`configs/censoring_aware_temporal_evidence_transport_blca.yaml`

## 2. 当前状态
- **问题定义最好、纸面上限最高**，但分数未被可信验证。
- 旧数据（catet_fix seed27785）best=0.6534 / last5=0.5474，垫底——但那是**修复前**的结果（早停过猛 + eps 间断双重污染），**已作废**。
- ⏳ 待 `run_fix_verify_fold02.sh`（eps 单调 + 梯度累积 + 关早停）重跑后才知道真实水平。

## 3. 方法缺陷 / 已知问题
- **排序监督与预测解耦**：`_risk_set_transport_loss` 排的是辅助标量 `transport_evidence`（gate×edge_risk 的均值，∈[0,1]），**不是预测用的风险分 `logits`**。rank loss 没直接改进预测排序。
- **多次解码共享头**：full / evidence / removed 三条路径共用 `event_encoder`/`event_hazard`，干预约束会把头拽向"抗扰动"而非"准确"，可能拖累 factual 预测。
- batch=4 下 risk-set 排序几乎失效（已用梯度累积缓解，待验证）。

## 4. 缺失实验清单（发论文必需）
- [ ] **消融**（方法文档已列，必须补齐）：shared OT vs stage-specific OT；edge-risk cost 有/无；risk-set 监督有/无；full CATE-T。
- [ ] **干预对照**：keep / remove / random 三种干预，证明选中的证据 sufficiency（重现 factual 风险）且 comprehensiveness（删掉显著改变风险）。
- [ ] **忠实性 vs 事后注意力**：证明"改 OT 几何的证据"比 detached attention map 更忠实。
- [ ] 跨癌种 + 战胜 SOTA + 多 seed 完整 5-fold。
- [ ] 修复后 fold0/2 曲线：train_cindex 应爬到 0.6+，val 峰值后移。

## 5. 可解释性怎么做到位（这是本方法的核心卖点）
- 核心主张：解释进入预测路径——证据门控**改变 OT plan 本身**，不是画个旁路注意力。
- 必须定量证明：removed 证据后风险显著变化（comprehensiveness），selected 证据重现 factual 风险（sufficiency），且删失下这套仍成立。
- 展示 stage×slot-pair 证据热图 + patch/pathway 级稳定性（跨 seed/fold）。
- 干预方向无关（只要求"删了要变"，不预设方向）——比 DCT 的循环论证干净，要突出这一点。

## 6. 发表门槛 checklist
- [ ] 修复后能正常收敛（不再早峰崩溃）
- [ ] 完整消融证明每个组件（stage OT / edge-risk / risk-set / 干预）都有贡献
- [ ] 忠实性定量优于事后注意力基线
- [ ] 多癌种 + SOTA 对比 + 多 seed
- [ ] 把"排序监督解耦预测"这个缺陷修掉或论证清楚

## 7. 目标分区 + 竞品文献（条件性）
- 现实：修复验证前**不够投稿**；数字追上来后 **1–2 区**（问题定义最完整）。
- 竞品：[CURE 2602.19987](https://arxiv.org/abs/2602.19987)（反事实+多模态+生存，但用检索非 OT）；faithfulness/删除干预范式（ERASER 系 sufficiency/comprehensiveness）；OT 多模态生存 SOTA。

## 8. 一句话结论
纸面最强，但**现在是"故事强、数字空"**。第一步就是修复后重跑拿到真实收敛曲线；只要能追到 v50 量级，凭"删失感知 + 模型内干预闭环"这个完整问题定义，是最有 1 区相的一个。
