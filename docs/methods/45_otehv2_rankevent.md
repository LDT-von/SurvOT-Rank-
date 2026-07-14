# 45 - OTEHV2 RankEvent

## 代码

`survot_rank/research/methods/prognostic_event_transport/model.py`

类名：`OTEHV2RankEvent`

注册名称：`otehv2_rankevent`

## 结构

```text
WSI/omics slots
    -> three-cost OT
    -> event tokens
    -> Transformer encoder
    -> event hazard + event gate
    -> global residual head
    -> survival logits
```

## 默认损失

旧 V45 配置可能同时启用：

```text
最终 NLL
+ OT
+ diversity
+ reconstruction
+ event mean NLL
+ per-event NLL
+ pairwise ranking
+ global consistency
+ gate entropy
```

## 主要问题

- 多个损失共同作用，难以判断性能来源。
- 旧 ranking 使用离散时间箱编号，不是连续 event time。
- event mean NLL、per-event NLL 和 global consistency 存在监督重复。
- reconstruction 直接约束 WSI slot 与 omics slot，可能抹平模态差异。

## 论文用途

作为历史 V45 对照和 loss ablation，不要把所有辅助损失都写成创新点。
