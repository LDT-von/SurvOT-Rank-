# 45v2 - OTEHV2 RankEvent V2

## 代码

仍位于：`survot_rank/research/methods/prognostic_event_transport/model.py`

类名：`OTEHV2RankEventV2`

注册名称：`otehv2_rankevent_v2`

## 新增能力

- clinical encoder
- clinical slot attention
- three-way OT fusion
- unified survival objective
- optional slot disentanglement
- optional adaptive slot routing
- optional learnable loss weighting

## 损失特点

该版本根据配置开关动态组合损失。不同配置下实际生效的损失不同，不能只看 YAML 中的参数名判断。

常见组合包括：

```text
diversity
+ gate entropy
+ event NLL
+ per-event NLL
+ ranking
+ global consistency
+ unified objective
```

## 论文用途

用于临床模态、统一目标和结构增强实验。由于分支和开关较多，暂不建议作为唯一主方法。
