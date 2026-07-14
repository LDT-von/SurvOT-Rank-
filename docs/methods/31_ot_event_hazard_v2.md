# 31 - OT Event Hazard V2

## 代码

`survot_rank/research/methods/ot_event_hazard_v2/model_v2.py`

注册名称：`ot_event_hazard_v2`

## 结构

```text
WSI patches -> WSI MLP -> WSI slots
omics/pathways -> omics encoder -> omics slots
    -> cosine / Euclidean / dot OT
    -> MultiScaleOTFusion
    -> event tokens
    -> Transformer encoder
    -> event hazard head + gate
    -> survival logits
```

## 损失

除训练器中的最终 Survival NLL 外，原始实现可能启用：

- OT distance regularization
- event mean NLL
- reconstruction
- diversity

## 定位

这是 OT-event 结构的基线版本。它没有连续时间 ranking，也没有将预后代价直接加入 OT cost。

## 论文用途

作为 `No Rank-Guidance` 或 OT-event baseline，不建议作为最终主方法。
