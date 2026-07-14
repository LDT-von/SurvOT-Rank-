# SurvOT-Rank 方法版本总览

本目录按可运行模型版本整理。配置文件是实验实例，不等于新的模型版本。

| 版本 | 注册名称 | 代码 | 定位 |
|---|---|---|---|
| 31 | `ot_event_hazard_v2` | `ot_event_hazard_v2/model_v2.py` | OT-event 基线 |
| 45 | `otehv2_rankevent` | `prognostic_event_transport/model.py` | 原始 RankEvent |
| 45v2 | `otehv2_rankevent_v2` | `prognostic_event_transport/model.py` | 临床与统一目标扩展 |
| 50 | `otehv2_timelocal_competing` | `prognostic_event_transport/model.py` | 时间局部竞争风险实验 |
| RG-ET | `rank_guided_event_transport` | `rank_guided_event_transport/model.py` | 当前推荐论文主线 |

## 推荐使用

论文主方法使用 `rank_guided_event_transport`。

`ot_event_hazard_v2` 用作 OT-event 基线，`otehv2_rankevent` 用作旧 V45 对照，
`otehv2_rankevent_v2` 和 `otehv2_timelocal_competing` 用于扩展实验与消融。

## 损失数量

RG-ET 只使用：

```text
最终 Survival NLL（训练器）
+ OT regularization
+ continuous-time ranking loss
+ event stage order loss
```
