"""v4.2 ACT-Surv：验证可加归因与删除反事实确实是凸组合的推论。

这些不是「近似成立」的性质——如果任何一条不精确成立，合并就退回缝合了。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from survot_rank.research.methods.archetypal_transport_composition.model import (  # noqa: E402
    ArchetypalTransportComposition,
)


def make_args(**overrides):
    args = dict(
        omic_sizes=[5, 7, 3],
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=32,
        rna_format="Pathways",
        slot_num_wsi=4,
        slot_num_omics=4,
        slot_iters=2,
        dct_slot_init_mode="deterministic",
        dct_slot_eval_seed=11,
        act_num_archetypes=5,
        act_epsilon=0.10,
        act_lambda_balance=0.01,
        act_lambda_rank=0.10,
        act_rank_margin=0.0,
        act_rank_max_pairs=64,
        act_hazard_scale=1.0,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def make_inputs(batch=6):
    return dict(
        x_wsi=torch.randn(batch, 9, 16),
        x_omic1=torch.randn(batch, 5),
        x_omic2=torch.randn(batch, 7),
        x_omic3=torch.randn(batch, 3),
    )


def build(**overrides):
    torch.manual_seed(5)
    model = ArchetypalTransportComposition(make_args(**overrides), omic_input_dim=15)
    return model


def test_composition_is_a_convex_combination_by_construction():
    """β 必须精确落在单纯形上——这是凸组合来自运输边缘条件而非 softmax 的体现。"""
    model = build()
    model.train()
    model(**make_inputs())
    composition = model.last_explanations["composition"]

    assert torch.all(composition >= 0)
    assert torch.allclose(
        composition.sum(dim=1), torch.ones(composition.size(0)), atol=1e-5
    )


def test_row_marginals_equal_token_masses_and_total_mass_is_one():
    model = build()
    model.train()
    model(**make_inputs())
    plan = model.last_explanations["transport_plan"]

    # 总质量恒为 1，因此 β 自动在单纯形上。
    assert torch.allclose(
        plan.sum(dim=(1, 2)), torch.ones(plan.size(0)), atol=1e-5
    )
    # 行边缘 = token 质量。
    assert torch.allclose(
        plan.sum(dim=2), model.last_explanations["token_mass"], atol=1e-6
    )


def test_additive_attribution_is_exact_at_both_levels():
    """archetype 级与 token×archetype 级的贡献必须精确求和到 logits。"""
    model = build()
    model.train()
    logits, _ = model(**make_inputs())
    explanations = model.last_explanations

    archetype_sum = explanations["archetype_contribution"].sum(dim=1)
    token_sum = explanations["token_contribution"].sum(dim=1)

    assert torch.allclose(logits, archetype_sum, atol=1e-5)
    assert torch.allclose(logits, token_sum, atol=1e-5)
    # 不设自由 bias，故完备性残差恒为 0（仅受浮点精度限制）。
    assert explanations["completeness_error"].max() < 1e-4


def test_prediction_stays_inside_the_archetype_hazard_convex_hull():
    """有界外推：任何预测都落在 K 条 archetype hazard 曲线的凸包内。"""
    model = build()
    model.eval()
    logits, _ = model(**make_inputs())
    hazards = model.last_explanations["archetype_hazards"]

    lower = hazards.min(dim=0).values
    upper = hazards.max(dim=0).values
    assert torch.all(logits >= lower[None, :] - 1e-5)
    assert torch.all(logits <= upper[None, :] + 1e-5)


def test_deletion_counterfactual_matches_an_actual_recomputation():
    """闭式删除反事实必须与真正重算一致，否则不能声称无需重解运输。"""
    model = build()
    model.eval()
    inputs = make_inputs()
    model(**inputs)
    closed_form = model.deletion_counterfactual(token_index=2)

    explanations = model.last_explanations
    plan = explanations["transport_plan"]
    hazards = explanations["archetype_hazards"]
    # 显式重算：去掉该 token 的质量后重新归一化列边缘。
    kept = torch.cat([plan[:, :2], plan[:, 3:]], dim=1)
    recomputed_composition = kept.sum(dim=1)
    recomputed_composition = recomputed_composition / recomputed_composition.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    recomputed = recomputed_composition @ hazards

    assert torch.allclose(closed_form, recomputed, atol=1e-4)


def test_missing_modality_only_removes_its_own_token_mass():
    """缺失模态的 token 质量为 0，但 β 仍精确在单纯形上。"""
    model = build()
    model.eval()
    inputs = make_inputs(batch=4)
    wsi_available = torch.tensor([True, False, True, False])
    model(**inputs, wsi_available=wsi_available, omics_available=torch.ones(4).bool())

    explanations = model.last_explanations
    token_mass = explanations["token_mass"]
    composition = explanations["composition"]

    # 前 4 个 token 是 WSI slot；缺失 WSI 的患者其质量必须为 0。
    assert torch.allclose(token_mass[1, :4], torch.zeros(4), atol=1e-6)
    assert torch.allclose(token_mass[3, :4], torch.zeros(4), atol=1e-6)
    assert torch.allclose(
        composition.sum(dim=1), torch.ones(4), atol=1e-5
    )
    assert not explanations["degenerate_patients"].any()


def test_patient_without_any_modality_falls_back_to_uniform_composition():
    model = build()
    model.eval()
    inputs = make_inputs(batch=3)
    none_available = torch.tensor([True, False, True])
    model(
        **inputs,
        wsi_available=none_available,
        omics_available=none_available,
    )
    explanations = model.last_explanations

    assert bool(explanations["degenerate_patients"][1])
    assert torch.allclose(
        explanations["composition"][1],
        torch.full((model.num_archetypes,), 1.0 / model.num_archetypes),
        atol=1e-5,
    )


def test_only_balance_and_rank_are_auxiliary_losses():
    """v4.2 的辅助损失只有两项：塌缩抑制与可选排序。"""
    model = build()
    model.train()
    logits, aux_loss = model(
        **make_inputs(),
        y=torch.tensor([0, 1, 2, 3, 1, 2]),
        c=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    )
    losses = model.last_training_losses

    expected = (
        model.act_lambda_balance * losses["act_balance"]
        + model.act_lambda_rank * losses["act_rank"]
    )
    assert torch.allclose(aux_loss, expected, atol=1e-6)
    assert torch.isfinite(logits).all()
    # 归因与反事实不产生任何损失项。
    assert not any("attribution" in key for key in losses)
    assert not any("deletion" in key for key in losses)


def test_archetype_differentiation_diagnostics_are_reported():
    """接可加归因这一卖点之前，必须能判断 archetype 是否真的分化开。"""
    model = build()
    model.train()
    model(**make_inputs())
    losses = model.last_training_losses

    for key in (
        "act_archetype_cosine",
        "act_hazard_spread",
        "act_effective_archetypes",
        "act_composition_dispersion",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])

    # 正交初始化下顶点方向不应一开始就塌缩。
    assert losses["act_archetype_cosine"] < 0.9
    assert losses["act_effective_archetypes"] > 1.0


def test_gradients_reach_archetypes_and_hazard_curves():
    model = build()
    model.train()
    logits, aux_loss = model(**make_inputs())
    (logits.sum() + aux_loss).backward()

    assert model.archetype_embedding.grad is not None
    assert torch.isfinite(model.archetype_embedding.grad).all()
    assert model.archetype_hazard_logits.grad is not None
    assert torch.isfinite(model.archetype_hazard_logits.grad).all()


def test_rejects_degenerate_hyperparameters():
    for overrides, field in (
        ({"act_num_archetypes": 1}, "act_num_archetypes"),
        ({"act_epsilon": 0.0}, "act_epsilon"),
        ({"act_hazard_scale": 0.0}, "act_hazard_scale"),
        ({"act_lambda_balance": -1.0}, "act_lambda_balance"),
        ({"act_rank_max_pairs": 0}, "act_rank_max_pairs"),
    ):
        with pytest.raises(ValueError, match=field):
            build(**overrides)
