# DCT v3.8.3：干预一致性损失 + 去塌缩运输几何

> 状态：已实现、已单测（7 项）、全量 330 测试通过。**尚无真实癌种结果。**
> 目的：把 v3.8 一大堆损失组合子变体收敛成**一个跑就够**的版本。
> 代码：`survot_rank/research/methods/dct_v383_intervention_consistency_centered/model.py`
> 配置：`configs/dct_v383_intervention_consistency_centered_blca.yaml`
> 注册名：`dct_v383_intervention_consistency_centered`（别名 `dct_v383`）

---

## 1. 一句话

v3.8.3 = v3.8 的 direction/dose/reconfiguration 三个损失（full 变体，robust 协议）
**保持不变**，唯一改动是把塌缩的 shared-prototype 运输几何换成中心化几何。

## 2. 为什么它能替代 v3.8 的一堆版本

v3.8 有 8 个损失组合变体（base/direction/dose/reconfiguration/两两/full）再乘
highscore/robust 协议、再加 v3.8.2 的 adaptive，跑不完。但实测证明**它们共享同
一个坏地基**：三个损失作用的"anchor 干预响应"在 v3.3 塌缩表征上是浮点噪声。

| 表征 | plan 偏离均匀 | high_risk_gain | plan_shift |
|---|---|---|---|
| v3.8 现状（塌缩） | 0.076 | ~1e-5（噪声） | 0.09 |
| v3.8.3 中心化 | 0.689 | ~1e-3 | 0.77 |

（复现：`python tools/diagnose_transport_collapse.py`）

在噪声上调损失权重不可能提分，这解释了 v3.8 全部子变体分数不动、BRCA full<base
(-0.033) 的现象。**所以不该继续扫损失组合，而该修地基。** v3.8.3 就是修好地基
的那一个版本，损失组合固定为 full，你只需要跑它。

## 3. 与 v3.8 的唯一区别（干净的单变量对照）

| | v3.8 robust/full | v3.8.3 |
|---|---|---|
| 三个干预损失 | ✓ | ✓（一字不改） |
| margin / warmup / ramp / alpha | robust 默认 | 完全相同 |
| slot 编码 | local + shared-prototype 池化（**塌缩**） | local + learned query + **跨 slot 中心化** |

因此 `v3.8 robust/full  vs  v3.8.3` 本身就是论文里一张有说服力的对照表：它证明
"运输干预一致性"这类损失需要一个非退化的运输几何才可能成立。你也可以用
`dct_v383_center_slots=false` 在 v3.8.3 内部复现塌缩基线做消融。

## 4. 诚实边界

- learned slot query（BO-QSA 已有）与中心化（标准去相关）是**正确性修复**，
  不是创新点。
- v3.8.3 只保证损失从"优化噪声"变成"优化真信号"（**必要条件**）。中心化后训练
  分数是否真涨需 BLCA fold0/2 实测（充分性未验证）。
- 中心化后 plan_shift 约 0.7，远超默认 `reconfiguration_margin=0.02`，该项会接近
  饱和（约束自动满足），预计主要有效项是 direction。若要 reconfiguration 重新
  起作用需调大其 margin。这是修复后的预期行为，不是 bug。

## 5. 运行

```bash
# 单测
python -m pytest tests/test_dct_v383_intervention_consistency_centered.py -q

# 训练（BLCA，UNI2-h，robust，full 三损失，中心化）
python -m survot_rank.cli train \
  --config configs/dct_v383_intervention_consistency_centered_blca.yaml
```

## 6. 建议的验证顺序

1. **BLCA fold0+fold2，同 seed，三条对比**（只回答"修地基是否有用"）：
   - v3.8 robust/full（塌缩基线）
   - v3.8.3（中心化，默认）
   - v3.8.3 `dct_v383_center_slots=false`（v3.8.3 内部的塌缩消融，应≈第一条）
   取 per-fold best（与 SlotSPE/PIBD/本仓库全部基线同口径）。
2. 若 v3.8.3 best 追平或超过 v3.8，再扩 5 折与多癌种；否则说明干预损失即使在
   真信号上也无正收益，转向 v3.9。
3. 训练期监控 `v38_high_plan_shift`：应稳定在 0.5+，若掉到 0.1 级说明中心化未
   生效或表征又塌了。
