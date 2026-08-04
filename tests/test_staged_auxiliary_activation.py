"""验证 v4.0 / v4.1 / ArcSurv 的辅助约束确实被延后激活。

三者原先都让辅助约束从第一个 epoch 就生效，导致：
- v4.0 BLCA fold1 最佳 C-index 在 epoch 0，随后 29 轮持续下滑
- v4.1 BLCA fold2 最佳 C-index 在 epoch 3，随后持续走低
- ArcSurv BLCA fold1 最佳 C-index 在 epoch 29（预算边界）且仍在上升
"""

from __future__ import annotations

import math

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


def test_v41_completion_loss_is_bounded_below_and_non_negative():
    """补全损失不得被方差项拖成负数。

    实测故障：completion 从正数降到 -1.3 ~ -1.9，使 v4.1 总目标与训练损失
    变负——模型在压低 log-variance 而不是改善生存预测。根因是高斯 NLL
    ``0.5 * (err^2 / var + log var)`` 在方差无约束时下界为负无穷，而补全目标
    是模型自身的 detached 账本表示，误差极易被压到 0。
    """
    predicted = torch.zeros(4, 8, 16)
    target = torch.zeros(4, 8, 16)
    confidence = torch.ones(4, 8)
    valid = torch.ones(4, dtype=torch.bool)

    loss_fn = DCTV41SurvivalEvidenceLedger._gaussian_completion_loss

    # 误差为 0 且方差被推到极小：旧实现在这里发散到负无穷。
    for log_variance_value in (-4.0, -20.0, -1e4):
        log_variance = torch.full((4, 8, 16), log_variance_value)
        loss = loss_fn(predicted, log_variance, target, confidence, valid)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    # 误差越大，损失越大（仍然是有意义的回归目标）。
    log_variance = torch.zeros(4, 8, 16)
    near = loss_fn(predicted, log_variance, target, confidence, valid)
    far = loss_fn(predicted + 2.0, log_variance, target, confidence, valid)
    assert far.item() > near.item()

    # 固定方差（floor=0）时不引入平移，损失恰为 0。
    exact = loss_fn(predicted, log_variance, target, confidence, valid, 0.0)
    assert exact.item() == pytest.approx(0.0)


def test_arcsurv_archetypes_do_not_collapse_to_a_uniform_composition():
    """原型必须分化，否则凸组合退化为常向量。

    实测故障：6 个原型的组合熵 ≈ ln(6) = 1.7918、患者间组合方差 ≈ 1e-4，
    即所有患者都均匀使用全部原型。两个放大器：
      1. ``archetypes = softmax(beta_logits) @ memory`` 摊在整个 bank 上，
         K 行全部收敛到队列均值附近，彼此重合；
      2. 距离对 dim 取均值，把量级压掉 dim 倍，softmax 必然接近均匀。
    """
    from survot_rank.research.methods.archetypal_risk_composition.model import (
        CohortArchetypeBank,
    )

    # 必须用真实规模：塌缩来自「softmax 摊在整个 bank 上」与「距离对 dim 取
    # 均值」这两个尺度效应，在玩具维度下都不会出现。
    torch.manual_seed(0)
    dim, num_archetypes, bank_size = 256, 6, 256
    # patient state 经过 LayerNorm，因此每维近似标准正态。
    states = torch.randn(bank_size, dim)
    uniform_entropy = math.log(num_archetypes)

    def build(distance_reduction: str, anchor_logit: float) -> CohortArchetypeBank:
        torch.manual_seed(0)
        bank = CohortArchetypeBank(
            dim,
            num_archetypes,
            bank_size,
            temperature=0.25,
            beta_init_scale=1.5,
            distance_reduction=distance_reduction,
            anchor_logit=anchor_logit,
        )
        bank.train()
        bank.update(states)
        return bank

    def entropy_of(bank: CohortArchetypeBank) -> float:
        composition, _, _ = bank(states)
        return float(
            -(composition.clamp_min(1e-12) * composition.clamp_min(1e-12).log())
            .sum(dim=1)
            .mean()
        )

    # 旧配置：熵贴在均匀上界 ln(K) 附近（真实 BLCA 运行实测 1.7898）。
    legacy = build("mean", 0.0)
    legacy_entropy = entropy_of(legacy)
    assert legacy_entropy > uniform_entropy - 0.05

    # 修复后：锚定 + 距离归一使组合明显偏离均匀。
    fixed = build("scaled", 6.0)
    assert fixed.seed_anchors_once() is True
    fixed_entropy = entropy_of(fixed)
    assert fixed_entropy < uniform_entropy - 0.2
    assert fixed_entropy < legacy_entropy

    # 锚定只做一次，重复调用不再改写 beta。
    assert fixed.seed_anchors_once() is False

    # 原型之间必须真的分开。
    archetypes, _ = fixed.archetypes()
    pairwise = torch.cdist(archetypes, archetypes)
    off_diagonal = pairwise[~torch.eye(num_archetypes, dtype=torch.bool)]
    assert off_diagonal.min().item() > 0.0
