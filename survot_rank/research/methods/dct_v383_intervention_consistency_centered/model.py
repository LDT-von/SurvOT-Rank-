"""DCT v3.8.3: 把 v3.8 的干预一致性损失接到一个非退化的运输几何上。

## 为什么需要 v3.8.3

v3.8 的三个损失（direction / dose / reconfiguration）全部作用在"anchor 干预
后 plan / risk 的变化幅度"上。但它们继承了 v3.3 的 `_semantic_slots`
shared-prototype 池化，而该池化实测把 slot 两两余弦推到 0.99+，运输计划
塌缩到接近均匀。后果（本机实测，见 `tools/diagnose_transport_collapse.py`）：

| 表征 | plan 偏离均匀 | high_risk_gain | plan_shift |
|---|---|---|---|
| v3.8 现状（塌缩） | 0.076 | 1e-5（噪声） | 0.09 |
| 中心化后 | 0.689 | 1e-3 | 0.77 |

即 v3.8 的三个损失在塌缩表征上优化的是浮点噪声：`softplus(margin - gain)`
在 gain≈1e-5 时几乎是不随患者变化的常数，梯度不区分任何人。这解释了
v3.8 全部子变体（base / 单项 / 两两 / full / v3.8.2）分数都不动、甚至
BRCA full<base(-0.033) 的现象——三个损失是纯干扰项。

## v3.8.3 改了什么

**唯一改动：把 slot 编码从"local slot + shared-prototype 池化"换成
"local slot + 跨 slot 中心化"，并用 learned slot query 提供索引稳定性。**
其余全部继承 v3.8：三个损失、margin、warmup、alpha、Sinkhorn 迭代数一字不改。

因此 v3.8 与 v3.8.3 的对照是一个干净的单变量实验：唯一变量 = 运输几何是否
退化。这本身就是论文里一张有说服力的表——它证明"运输干预一致性"这一类
损失需要一个非退化的运输几何才可能成立。

## 诚实边界

* learned slot query（BO-QSA 已有）与跨 slot 中心化（标准去相关）在本方法
  中是**正确性修复**，不是创新点。
* 本方法只保证 v3.8 的损失从"优化噪声"变成"优化真信号"（必要条件）；
  中心化后训练分数是否真的提高需要 BLCA fold0/2 实测（充分性未验证）。
* 中心化后 plan_shift 量级约 0.7，远超默认 `reconfiguration_margin=0.02`，
  该损失会接近饱和（约束自动满足），主要有效项预计是 direction。这不是 bug，
  是修复表征后的预期行为；若要让 reconfiguration 重新起作用需调大其 margin。
* `counterfactual` 指基于模型的运输敏感性，不是可识别的因果治疗效应。
"""

from __future__ import annotations

from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)


class DCTV383InterventionConsistencyCentered(DCTTransportInterventionConsistency):
    """v3.8 干预一致性损失 + 去塌缩的中心化运输几何。"""

    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        # learned slot query 提供跨患者的 slot 索引稳定性，替代会塌缩的
        # shared-prototype 池化。必须在 super() 之前设置，因为 slot attention
        # 在祖先类 __init__ 中构建。
        if not getattr(args, "dct_v383_keep_legacy_slot_init", False):
            args.dct_slot_init_mode = "learned"

        super().__init__(args, omic_input_dim, omic_names, pathway_names)

        # 移除塌缩来源。del 而非权重置零：论文口径上"移除模块"与"关掉权重"
        # 不是一回事，单测会断言 state_dict 里不再有这些键。
        del self.shared_wsi_prototypes
        del self.shared_omic_prototypes

        self.v383_center_slots = bool(getattr(args, "dct_v383_center_slots", True))

    @staticmethod
    def _center_slots(slots):
        """跨 slot 中心化：移除所有 slot 共有的共模分量。

        运输代价只取决于 slot 之间的相对关系；共模分量对跨模态匹配是纯噪声，
        却会让 cost 矩阵各行趋同、把 Sinkhorn 解推向均匀 plan。中心化后
        sum_i v_i = 0，两两内积之和严格为 -sum_i ||v_i||^2 < 0（零参数）。
        """
        return slots - slots.mean(dim=1, keepdim=True)

    def _encode_transport_slots(self, x_wsi_proj, x_omics, kwargs):
        """用中心化的 local slot 替代 v3.3 的 shared-prototype 坐标。

        这是 v3.8.3 相对 v3.8 的唯一改动。返回签名保持与 v3.3 一致；后两个
        坐标分配矩阵在中心化路径下不存在，返回占位张量（仅用于可解释性日志，
        不参与任何计算）。
        """
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)
        if self.v383_center_slots:
            slots_wsi = self._center_slots(slots_wsi)
            slots_omic = self._center_slots(slots_omic)
        placeholder = slots_wsi.new_zeros(slots_wsi.size(0), slots_wsi.size(1), 1)
        return slots_wsi, slots_omic, placeholder, placeholder
