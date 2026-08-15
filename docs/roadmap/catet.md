# CATE-T 删失感知时间证据运输 — 发表路线图（修复后，2026-08-15）

## 1. 定位
- 一句话：把删失感知的 IPCW risk-set 监督注入"分阶段 OT 几何"，并在**同一个用于预测的 transport plan** 上做 keep/remove 干预来验证证据忠实性。
- 代码：`survot_rank/research/methods/censoring_aware_temporal_evidence_transport/model.py` → `CensoringAwareTemporalEvidenceTransport`
- 注册名：`censoring_aware_temporal_evidence_transport`
- config：`configs/censoring_aware_temporal_evidence_transport_blca.yaml`
- 文档：`docs/methods/catet_censoring_aware_temporal_evidence_transport.md`

## 2. 修复后状态（2026-08-15）
- 代码与 `backup/three_method_final_2026_08_13/catet_final_model.py` 对齐；v2 的 `CohortAnchoredRouter` / `archetype prior` / `_route_consistency_loss` 已删除，连带 `extended_args.py` 中的 `catet_cohort_routes` / `catet_cohort_topk` / `catet_lambda_route` / `catet_use_archetype_prior` 一并清空。
- 5 个机制偏离全部修复：阶段特异 base cost、真 counterfactual re-transport + IPFP、IPCW risk-set ranking 直接监督 `risk_score(logits)`、`gate_budget = (gate.mean - catet_keep_ratio)^2` 正则、IPFP 显著改善 plan 数值精度。
- 单元测试 `tests/test_censoring_aware_temporal_evidence_transport.py` 7/7 通过，覆盖：阶段特异成本、IPFP marginal error < 1e-3、risk score 单调、IPCW 权重单调、censored stage 屏蔽、超参一致性、有限梯度。

## 3. 当前实验数字（不可外推）
- BLCA fold0/2 已拿到 `0.6458 / 0.6837`，均值 `0.6648`。
- 完整 5-fold + 机制签名未跑，**0.6648 ≠ 全 5-fold 真实水平**。
- `Δ`（`catet_intervention_cost`）、`catet_keep_ratio`、`eps` 噪声强度尚未做过 sweep。

## 4. 剩余任务（按"留下结果"原则逐项 gate）
1. `scripts/audit_catet.py`：输出 5 种机制签名（方向一致性率、剂量单调性、plan 守恒、sufficiency/comprehensiveness gap、random-gate 基线）。
2. 6 项机制消融：`shared_stage_cost / no_ipcw / no_censored_stage / masked_plan (v1 行为) / random_gate (负对照) / final_model`，每个 5-fold C-index + 机制签名。
3. 跨癌种统一协议：UNI2-h 特征 + `5fold_uni2h` + 50ep，先跑完 BLCA 全 5-fold，再扩 LUAD / BRCA / KIRC。
4. 分数度量扩展：除 C-index 外，报告 time-dependent AUC、IBS、ECE 校准。
5. 多 seed：≥3 seeds × 5 folds，bootstrap 95% CI + paired test。

## 5. 投稿判断
- 完整 5-fold 全 5 项 gate 同时通过 → 重新评估 Q3（CCF C / 中文学报）或 Q2（CCF B 偏解释性）。
- 任一 gate 未通过 → 暂缓投稿，先把 gap 收口。

## 6. 一句话结论
Idea 兑现度满分，**剩的就是把实验数字拉满**。剩下的 5 项任务就是从"故事强、数字空"走到"故事强、数字到位"的全部距离。
