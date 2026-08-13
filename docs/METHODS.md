# SurvOT-Rank 方法注册与状态索引

> 状态快照：2026-08-11。这个文件只回答“方法叫什么、现在处于什么角色、从哪里进入”，不保存实验分数。

## 先看结论

- **当前论文主线**：DCT v3.8.2 fixed-full，对应注册名 `dct_v382_prognostic_transport_reconstruction`。
- **正式候选**：CA-PSA、CATET repaired、ArcSurv staged。
- **修复闸门**：IST-Surv v4.0 正在进行 repair-gate 检查；旧机制结果不能代表修复版结论。
- **基线/历史对照**：V60 OT Event Rank、V50、V45/PET、V31。
- 其余可运行实现是研究分支，用于消融、追溯或机制比较，不再各自宣称“当前主线”。

“可注册运行”和“论文优先级”是两回事。所有下表方法仍可由模型工厂加载；状态只用于阻止文档和实验入口继续混称主线。

## 唯一入口

```bash
# 人工查看
python -m survot_rank.cli methods

# 只看当前主线
python -m survot_rank.cli methods --status primary

# 机器可读
python -m survot_rank.cli methods --json
```

代码中的唯一注册源是
`survot_rank/research/methods/catalog.py`。训练参数的合法方法名、模型工厂注册表和数字别名均从这里生成，不再分别维护。

## 21 个规范注册名

| 状态 | 方法 | 规范注册名 | 常用别名 | 机制说明 |
|---|---|---|---|---|
| **Primary** | DCT v3.8.2 fixed-full | `dct_v382_prognostic_transport_reconstruction` | `dct_v382`, `dct_v3_8_2` | [DCT v3.8.2](DCT_V382_MGPTR.md) |
| **Candidate** | CA-PSA | `cohort_anchored_adaptive_prognostic_slot_attention` | `ca_psa`, `capsa` | [CA-PSA](methods/cohort_anchored_adaptive_prognostic_slot_attention.md) |
| **Candidate** | CATET | `censoring_aware_temporal_evidence_transport` | — | [CATET](methods/catet_censoring_aware_temporal_evidence_transport.md) |
| **Candidate** | ArcSurv | `archetypal_risk_composition` | `arcsurv`, `arc_surv` | [ArcSurv](methods/arcsurv_archetypal_risk_composition.md) |
| **Repair** | IST-Surv v4.0 | `intervention_stable_survival_transport` | `v40`, `ist_surv` | [IST-Surv](V40_INTERVENTION_STABLE_TRANSPORT.md) |
| **Reference** | V60 OT Event Rank | `v60_ot_event_rank` | `60` | [README 中的参考目标](../README.md#v60-ot-event-rank-reference-objective) |
| **Reference** | V50 Time-Local Competing Risk | `otehv2_timelocal_competing` | `50` | [V50](methods/50_timelocal_competing.md) |
| **Reference** | V45/PET | `otehv2_rankevent` | `45`, `pet`, `prognostic_event_transport` | [V45](methods/45_otehv2_rankevent.md) |
| **Reference** | V45v2 | `otehv2_rankevent_v2` | `45v2` | [V45v2](methods/45_otehv2_rankevent_v2.md) |
| **Reference** | V31 OT Event Hazard | `ot_event_hazard_v2` | `31` | [V31](methods/31_ot_event_hazard_v2.md) |
| Research | DCT Score-First | `distributional_counterfactual_transport` | — | [DCT](methods/distributional_counterfactual_transport.md) |
| Research | DCT v3.6 Listwise | `dct_listwise_transport` | — | [DCT v3.6](DCT_V36_LISTWISE.md) |
| Research | DCT v3.8 TIC | `dct_transport_intervention_consistency` | `dct_v38` | [DCT v3.8](DCT_V38_TRANSPORT_CONSISTENCY.md) |
| Research | DCT v3.8.3 Centered | `dct_v383_intervention_consistency_centered` | `dct_v383`, `dct_v3_8_3` | [DCT v3.8.3](DCT_V383_INTERVENTION_CONSISTENCY_CENTERED.md) |
| Research | DCT v3.9 Risk Simplex | `dct_v39_risk_simplex_transport` | `dct_v39`, `dct_v3_9`, `rst` | [DCT v3.9](DCT_V39_RISK_SIMPLEX_TRANSPORT.md) |
| Research | DCT v4.1 Evidence Ledger | `dct_v41_survival_evidence_ledger` | `dct_v41`, `dct_v4_1` | [DCT v4.1](methods/dct_v41_survival_evidence_ledger.md) |
| Research | Rank-Guided Event Transport | `rank_guided_event_transport` | — | [RG-ET](methods/rank_guided_event_transport.md) |
| Research | Stagewise Prognostic Transport | `stagewise_prognostic_transport` | — | [Stagewise PT](methods/stagewise_prognostic_transport.md) |
| Research | Faithful Evidence Transport | `faithful_evidence_transport` | — | [FET](methods/faithful_evidence_transport.md) |
| Research | ACT-Surv v4.2 | `archetypal_transport_composition` | `act_surv`, `actsurv`, `v42`, `dct_v42` | — |
| Research | V70 PSPC-Surv | `v70_patient_specific_prognostic_circuits` | `70`, `pspc_surv`, `pspc` | [V70](methods/v70_patient_specific_prognostic_circuits.md) |

## 命名约定

- 统一写 **CATET**；`CA-TET`、`CATE-T` 只视为旧文档拼法。
- 统一写 **IST-Surv v4.0**；“DCT v4.0”仅用于追溯旧结果表。
- 统一写 **V60 OT Event Rank**，不要裸写“V60”，以免与服务器上曾使用的 CA-PSA 临时简称混淆。
- **CA-PSA** 与 V60 OT Event Rank 是两个独立方法。
- 配置文件是实验实例，不是新的方法版本；launcher 的动态覆盖也必须记录在结果快照中。

## 信息源分工

| 需要回答的问题 | 唯一入口 |
|---|---|
| 当前方法身份、论文优先级与负结果判断 | [ACADEMIC_METHOD_EVALUATION.md](ACADEMIC_METHOD_EVALUATION.md) |
| 当前正式结果、划分边界与可比性 | [FINAL_SUMMARY.md](../FINAL_SUMMARY.md) |
| 截至 2026-07-16 的历史实验档案 | [EXPERIMENT_SUMMARY.md](../EXPERIMENT_SUMMARY.md) |
| 三方法正式跨癌种计划 | [THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md](roadmap/THREE_METHOD_FINAL_CROSS_CANCER_PLAN.md) |
| 方法机制 | [methods/README.md](methods/README.md) |
| 配置与脚本入口 | [configs/INDEX.md](../configs/INDEX.md) / [scripts/README.md](../scripts/README.md) |

更新方法状态时，应先更新代码目录中的 catalog，再同步本页；不要在 README、路线图和单方法文档中另建一套“当前主线”判断。
