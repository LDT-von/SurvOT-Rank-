# 50 - Time-Local Competing Event

## 代码

`survot_rank/research/methods/prognostic_event_transport/model.py`

类名：`OTEHTimeLocalCompeting`

注册名称：`otehv2_timelocal_competing`

## 核心想法

在 RankEvent V2 的 event hazard 上增加时间局部竞争结构：

- 时间特异性约束
- 时间覆盖约束
- competing regularization
- event hazard 与 global residual

## 损失

除基础损失外，还可能启用：

```text
time specificity
+ time coverage
+ competing regularization
+ event NLL
+ per-event NLL
+ ranking
+ global consistency
```

## 论文用途

作为时间建模方向的实验版本。只有在多个癌种和严格消融中稳定优于 RG-ET，才考虑作为独立主线。
