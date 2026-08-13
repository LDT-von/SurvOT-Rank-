# 配置文件索引

> 状态快照：2026-08-11。当前共有 79 份 YAML。配置是“某次实验的参数实例”，不是新的方法注册名。

方法身份统一从以下命令查询：

```bash
python -m survot_rank.cli methods
```

## 目录职责

| 位置 | 职责 | 是否作为新实验起点 |
|---|---|---|
| `configs/*.yaml` | 可复现实验或方法级默认配置 | 是，优先选择与规范注册名同名的文件 |
| `configs/ablation/` | 消融配置 | 仅用于对应消融 |
| `configs/diagnostics/` | 数值诊断与故障定位 | 否 |
| `configs/fix/` | 历史修复验证快照 | 否，除非复现该修复 |
| `configs/p0_experiments/` | 2026-07 的 P0 筛选档案 | 否，作为历史记录 |

## 当前入口

| 状态 | 方法 | 配置/入口 |
|---|---|---|
| Primary | DCT v3.8.2 fixed-full | 由 `scripts/run_dct_v382_final_cross_cancer.py` 在 DCT 基础配置上显式覆盖 |
| Candidate | CA-PSA | `cohort_anchored_adaptive_prognostic_slot_attention_blca.yaml` |
| Candidate | CATET | `censoring_aware_temporal_evidence_transport_blca.yaml` |
| Candidate | ArcSurv | `archetypal_risk_composition_blca.yaml` |
| Repair | IST-Surv v4.0 | `intervention_stable_survival_transport.yaml`；repair gate 以当前工作脚本为准 |
| Reference | V60 OT Event Rank | `v60_ot_event_rank_blca.yaml` |
| Reference | V45/PET | `v45_blca.yaml` |

三类 DCT 注册方法没有独立 YAML：

- `dct_listwise_transport`：由 v3.6 launcher 生成动态覆盖；
- `dct_transport_intervention_consistency`：由 v3.8 launcher 生成动态覆盖；
- `dct_v382_prognostic_transport_reconstruction`：由 v3.8.2 launcher 生成动态覆盖。

这些不是孤立模型，但复现实验时必须同时保存“基础 YAML + launcher 生成的覆盖参数”，不能只复制基础 YAML。

## 已知兼容项

- `fix/v45_baseline_globalbin_blca.yaml` 与
  `p0_experiments/v45_baseline_globalbin_blca.yaml` 内容相同。两份暂时保留，以免破坏旧命令；新引用统一使用 `p0_experiments/` 中的历史快照。
- `ablation/rank_guided_event_transport/*.yaml` 位于二级目录，旧
  `run_ablation.sh` 只扫描 `configs/ablation/*.yaml`，不会自动发现它们。
- `specific_simple` 会进入结果参数编码。已有配置中的历史值不因整理而改名，以免改变复现路径。

## 新配置命名规则

1. 文件名：`<canonical_method>_<cancer>[_<purpose>].yaml`。
2. `name` 必须与文件名主体一致。
3. `data.study`、`train.results_dir` 和癌种名称必须一致。
4. `train.survot_method` 使用 catalog 中的规范注册名，不在新配置里使用数字别名。
5. 动态覆盖必须在 launcher、结果目录配置快照和运行日志中同时保存。

当前方法角色见 [docs/METHODS.md](../docs/METHODS.md)，运行脚本见
[scripts/README.md](../scripts/README.md)。
