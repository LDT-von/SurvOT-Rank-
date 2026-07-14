# RG-ET - Rank-Guided Event Transport

## 代码

`survot_rank/research/methods/rank_guided_event_transport/model.py`

类名：`RankGuidedEventTransport`

注册名称：`rank_guided_event_transport`

配置：`configs/rank_guided_event_transport_blca.yaml`

## 方法主线

```text
WSI/omics slots
    -> feature cost + learned prognostic pair cost
    -> Sinkhorn transport plans
    -> event token fusion
    -> ordered time-stage event tokens
    -> event hazards and gate
    -> survival logits
```

## 当前创新假设

普通 OT 只根据特征相似性分配 transport mass。RG-ET 增加 learned prognostic pair cost，
使最终生存排序的梯度可以回传到 transport plan：

```text
C_RG = C_feature + lambda_prog * C_prognostic
```

event token 通过 stage embedding 和 stage order loss 形成有序风险阶段，而不是没有语义约束的独立 token。

## 损失

模型内部只保留：

```text
lambda_ot * OT regularization
+ lambda_rank * continuous event-time ranking
+ lambda_stage * stage order loss
```

最终 Survival NLL 仍由训练器计算。

## 论文主张边界

可以主张：

> Rank-guided prognostic event transport for multimodal survival analysis.

暂时不能主张：

- 首次使用 OT 做 WSI-omics 生存预测；
- 首次使用 Slot Attention、Transformer 或 event token；
- 已经证明 event token 对应真实生物学事件；
- 已经证明该方法在所有癌种都优于已有方法。

## 必须完成的验证

- no OT
- single-cost OT
- three-cost OT
- no prognostic pair cost
- no continuous ranking
- no stage order loss
- full RG-ET

主线应以平均 C-index、折间标准差、IBS 和 iAUC 综合评价，不能只根据 Fold 2 决定模块去留。
