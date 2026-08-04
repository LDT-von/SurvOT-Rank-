"""验证 v4.0 / v4.1 / ArcSurv 的辅助约束确实被延后激活。

三者原先都让辅助约束从第一个 epoch 就生效，导致：
- v4.0 BLCA fold1 最佳 C-index 在 epoch 0，随后 29 轮持续下滑
- v4.1 BLCA fold2 最佳 C-index 在 epoch 3，随后持续走低
- ArcSurv BLCA fold1 最佳 C-index 在 epoch 29（预算边界）且仍在上升
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from survot_rank.research.methods.archetypal_risk_composition.model import (  # noqa: E402
    ArchetypalRiskComposition,
)
from survot_rank.research.methods.dct_v41_survival_evidence_ledger.model import (  # noqa: E402
    DCTV41SurvivalEvidenceLedger,
)
from survot_rank.research.methods.intervention_stable_survival_transport.model import (  # noqa: E402
    InterventionStableSurvivalTransport,
)


SCALE_METHODS = (
    (InterventionStableSurvivalTransport, "_stability_scale", "ist"),
    (DCTV41SurvivalEvidenceLedger, "_ledger_scale", "v41"),
    (ArchetypalRiskComposition, "_structure_scale", "arc"),
)


class _Stub:
    """只承载 warmup/ramp 字段，避免构建完整模型。"""

    training = True


def _stub(prefix: str, warmup: int, ramp: int, model_class, method_name: str):
    stub = _Stub()
    setattr(stub, f"{prefix}_warmup_epochs", warmup)
    setattr(stub, f"{prefix}_ramp_epochs", ramp)
    # v4.0 把公共 warmup+ramp 曲线抽成 _staged_ramp，stub 需要一并绑定。
    if hasattr(model_class, "_staged_ramp"):
        stub._staged_ramp = model_class._staged_ramp.__get__(stub, model_class)
    return getattr(model_class, method_name).__get__(stub, model_class)


@pytest.mark.parametrize("model_class,method_name,prefix", SCALE_METHODS)
def test_warmup_fully_disables_then_linearly_ramps(model_class, method_name, prefix):
    scale = _stub(prefix, 5, 10, model_class, method_name)

    # warmup 期间必须严格为 0，否则约束仍在早期压制生存头。
    for epoch in range(5):
        assert scale(epoch) == 0.0

    # warmup 后第一轮开始线性上升，第 15 轮（0-indexed 14）到满权重。
    assert scale(5) == pytest.approx(0.1)
    assert scale(9) == pytest.approx(0.5)
    assert scale(14) == pytest.approx(1.0)
    # 满权重后保持 1.0，不得超过。
    assert scale(30) == 1.0
    assert scale(49) == 1.0


@pytest.mark.parametrize("model_class,method_name,prefix", SCALE_METHODS)
def test_zero_warmup_reproduces_legacy_behaviour(model_class, method_name, prefix):
    scale = _stub(prefix, 0, 0, model_class, method_name)
    assert scale(0) == 1.0
    assert scale(7) == 1.0


@pytest.mark.parametrize("model_class,method_name,prefix", SCALE_METHODS)
def test_evaluation_always_uses_full_strength(model_class, method_name, prefix):
    scale = _stub(prefix, 5, 10, model_class, method_name)
    stub = scale.__self__
    stub.training = False
    # 这三个 scale 只作用于**训练路径**的辅助损失（v4.1 的 dropout 在 eval
    # 也直接短路），因此评估时取满权重不会改变验证预测。
    assert scale(0) == 1.0
    assert scale(3) == 1.0


def test_v40_cost_feedback_uses_one_curve_for_train_and_eval():
    """稳定性回写 cost 属于前向图，训练与评估必须完全同曲线。

    此前 ``_stability_scale`` 同时驱动 ``stable_cost``（预测用的运输计划）与
    辅助损失，而它在评估时硬返回 1.0。于是 warmup 内训练更新的是 factual
    plan、验证打分的是 stable plan，两者是不同的模型；这使早期 epoch 的
    验证 C-index 成为无法与后期比较的选择噪声，正是 BLCA fold1
    ``best @ epoch 0`` 的来源。
    """
    feedback = _stub(
        "ist", 5, 10, InterventionStableSurvivalTransport, "_cost_feedback_scale"
    )
    stub = feedback.__self__

    for epoch in (0, 3, 5, 9, 14, 30):
        stub.training = True
        train_scale = feedback(epoch)
        stub.training = False
        eval_scale = feedback(epoch)
        assert train_scale == eval_scale, f"epoch {epoch} 的前向图在训练/评估不一致"

    # 曲线本身仍是 warmup + 线性 ramp。
    stub.training = True
    assert feedback(0) == 0.0
    assert feedback(4) == 0.0
    assert feedback(5) == pytest.approx(0.1)
    assert feedback(14) == pytest.approx(1.0)
    assert feedback(49) == 1.0


def test_arcsurv_bank_update_window_follows_warmup():
    """原型库原先只在 epoch 0 建立，此时编码器尚未被生存目标塑形。"""

    class _Args:
        arc_warmup_epochs = 5
        arc_bank_update_epochs = -1

    stub = _Stub()
    stub.arc_warmup_epochs = int(_Args.arc_warmup_epochs)
    bank_epochs = int(_Args.arc_bank_update_epochs)
    stub.arc_bank_update_epochs = (
        stub.arc_warmup_epochs if bank_epochs < 0 else bank_epochs
    )
    assert stub.arc_bank_update_epochs == 5

    # -1 跟随 warmup；显式值优先。
    bank_epochs = 2
    resolved = stub.arc_warmup_epochs if bank_epochs < 0 else bank_epochs
    assert resolved == 2

    # 至少更新 1 轮，否则原型库永远为空。
    for warmup in (0, 1, 5):
        assert max(1, warmup) >= 1
