from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.archetypal_risk_composition.model import (
    ArchetypalRiskComposition,
    CohortArchetypeBank,
)
from survot_rank.training.extended_args import process_args_extended
from survot_rank.training.model_factory import METHOD_ALIASES, METHOD_REGISTRY, get_model


def make_args(**overrides):
    args = SimpleNamespace(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=3,
        slot_num_omics=3,
        slot_iters=2,
        arc_num_archetypes=4,
        arc_bank_size=12,
        arc_temperature=0.25,
        arc_beta_init_scale=1.5,
        arc_lambda_recon=0.05,
        arc_lambda_align=0.05,
        arc_lambda_balance=0.01,
        arc_lambda_volume=0.01,
        arc_lambda_rank=0.10,
        arc_rank_margin=0.0,
        arc_rank_max_pairs=128,
        arc_seed_anchors=0,
        arc_freeze_state_encoder=1,
        arc_lambda_sharpness=0.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_inputs(batch=3):
    return {
        "x_wsi": torch.randn(batch, 6, 16),
        "x_omics": torch.randn(batch, 5, 20),
        "y": torch.tensor([0, 2, 3][:batch]).long(),
        "c": torch.tensor([0.0, 1.0, 0.0][:batch]),
    }


def test_arcsurv_forward_backward_and_simplex_are_finite():
    torch.manual_seed(0)
    model = ArchetypalRiskComposition(make_args(), omic_input_dim=20)
    model.train()
    logits, aux_loss = model(**make_inputs())
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert torch.allclose(
        model.last_composition.sum(dim=1), torch.ones(3), atol=1e-6
    )
    assert model.wsi_archetypes.memory_count.item() == 3
    assert model.omic_archetypes.memory_count.item() == 3
    assert torch.isfinite(model.last_training_losses["arc_simplex_volume"])
    assert model.last_training_losses["arc_composition_variance"] > 0
    (logits.sum() + aux_loss).backward()
    assert model.archetype_hazard_logits.grad is not None
    assert torch.isfinite(model.archetype_hazard_logits.grad).all()


def test_arcsurv_archetypes_are_convex_combinations_of_memory():
    torch.manual_seed(1)
    model = ArchetypalRiskComposition(make_args(), omic_input_dim=20)
    model.train()
    model(**make_inputs())
    parameters = model.archetype_parameters()
    beta = parameters["shared_beta"]
    assert torch.all(beta >= 0)
    assert torch.allclose(beta.sum(dim=1), torch.ones(4), atol=1e-6)
    expected = beta @ model.shared_archetypes.memory[:3]
    assert torch.allclose(parameters["shared_archetypes"], expected, atol=1e-6)
    assert model.wsi_archetypes is model.omic_archetypes


def test_arcsurv_missing_modalities_and_eval_do_not_update_memory():
    torch.manual_seed(2)
    model = ArchetypalRiskComposition(make_args(), omic_input_dim=20)
    model.train()
    inputs = make_inputs()
    inputs.update(
        {
            "wsi_available": torch.tensor([1, 1, 0]),
            "omics_available": torch.tensor([1, 0, 1]),
        }
    )
    logits, aux_loss = model(**inputs)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    # One shared bank stores one fused support point for every patient with at
    # least one available modality.
    assert model.shared_archetypes.memory_count.item() == 3

    model.eval()
    before = (
        model.wsi_archetypes.memory_count.item(),
        model.omic_archetypes.memory_count.item(),
    )
    eval_logits, eval_aux = model(**inputs)
    assert eval_logits.shape == (3, 4)
    assert eval_aux.item() == 0.0
    assert before == (
        model.wsi_archetypes.memory_count.item(),
        model.omic_archetypes.memory_count.item(),
    )


def _memory_snapshot(model):
    return (
        model.shared_archetypes.memory.clone(),
        model.shared_archetypes.memory_priority.clone(),
        model.shared_archetypes.memory_seen.clone(),
    )


def test_arcsurv_memory_is_frozen_after_the_bank_update_window():
    """原型库在 warmup 期间持续更新，warmup 结束后冻结。

    原实现只在 epoch 0 建库并冻结，那时编码器还没被生存目标塑形。
    """
    torch.manual_seed(7)
    model = ArchetypalRiskComposition(
        make_args(arc_warmup_epochs=3, arc_bank_update_epochs=-1),
        omic_input_dim=20,
    )
    assert model.arc_bank_update_epochs == 3
    model.train()

    # 窗口内必须仍在更新。
    model(**make_inputs(), cur_epoch=0)
    inside_window = _memory_snapshot(model)
    model(**make_inputs(), cur_epoch=1)
    assert not all(
        torch.equal(left, right)
        for left, right in zip(inside_window, _memory_snapshot(model))
    )

    # 窗口结束后必须冻结。
    model(**make_inputs(), cur_epoch=3)
    assert model.state_encoder_frozen.item()
    assert not any(parameter.requires_grad for parameter in model.wsi_mlp.parameters())
    frozen = _memory_snapshot(model)
    model(**make_inputs(), cur_epoch=4)
    assert all(
        torch.equal(left, right)
        for left, right in zip(frozen, _memory_snapshot(model))
    )


def test_arcsurv_zero_warmup_reproduces_epoch_zero_only_bank():
    """arc_bank_update_epochs=0 时退回旧行为：只在 epoch 0 建库。"""
    torch.manual_seed(7)
    model = ArchetypalRiskComposition(
        make_args(arc_warmup_epochs=0, arc_bank_update_epochs=0),
        omic_input_dim=20,
    )
    model.train()
    model(**make_inputs(), cur_epoch=0)
    before = _memory_snapshot(model)
    model(**make_inputs(), cur_epoch=1)
    assert all(
        torch.equal(left, right)
        for left, right in zip(before, _memory_snapshot(model))
    )


def test_arcsurv_pathway_inputs_match_blca_configuration():
    torch.manual_seed(3)
    args = make_args(
        rna_format="Pathways",
        omic_sizes=[5, 7, 3],
    )
    model = ArchetypalRiskComposition(args, omic_input_dim=None)
    model.train()
    inputs = {
        "x_wsi": torch.randn(3, 6, 16),
        "x_omic1": torch.randn(3, 5),
        "x_omic2": torch.randn(3, 7),
        "x_omic3": torch.randn(3, 3),
        "y": torch.tensor([0, 2, 3]).long(),
        "c": torch.tensor([0.0, 1.0, 0.0]),
    }
    logits, aux_loss = model(**inputs)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)


def test_arcsurv_explanation_reconstructs_logits_from_shared_archetypes():
    torch.manual_seed(31)
    model = ArchetypalRiskComposition(make_args(), omic_input_dim=20)
    model.eval()
    with torch.no_grad():
        logits, _ = model(**make_inputs())
    explanation = model.explain_last_batch()
    reconstructed = (
        explanation["archetype_logit_contribution"].sum(dim=1)
        + model.hazard_bias
    )
    assert torch.allclose(reconstructed, logits, atol=1e-6)
    assert explanation["bank_support_weights"].shape[0] == model.num_archetypes


def test_arcsurv_is_registered_and_parser_accepts_alias():
    assert "archetypal_risk_composition" in METHOD_REGISTRY
    assert METHOD_ALIASES["arcsurv"] == "archetypal_risk_composition"
    parsed = process_args_extended(["--survot_method", "arcsurv"])
    assert parsed.survot_method == "arcsurv"
    model = get_model("arcsurv", make_args(), omic_input_dim=20)
    assert type(model).__name__ == "ArchetypalRiskComposition"


def test_arcsurv_priority_reservoir_is_order_robust_and_first_epoch_only():
    states = torch.arange(96, dtype=torch.float32).view(6, 16)
    first = CohortArchetypeBank(16, num_archetypes=3, bank_size=4, temperature=0.25)
    second = CohortArchetypeBank(16, num_archetypes=3, bank_size=4, temperature=0.25)
    first.train()
    second.train()

    first.update(states)
    second.update(states.flip(0))
    assert first.memory_seen.item() == 6
    assert second.memory_seen.item() == 6
    assert torch.equal(first.memory_priority, second.memory_priority)
    assert torch.equal(first.memory, second.memory)

    frozen = first.memory.clone()
    first.update(torch.randn(3, 16), allow_update=False)
    assert torch.equal(first.memory, frozen)
    assert first.memory_seen.item() == 6


def test_arcsurv_simplex_volume_penalizes_collapse_and_backpropagates():
    collapsed = torch.zeros(4, 8, requires_grad=True)
    spread = torch.zeros(4, 8)
    spread[1, 0] = 1.0
    spread[2, 1] = 1.0
    spread[3, 2] = 1.0

    collapsed_loss = ArchetypalRiskComposition._simplex_volume_loss(collapsed)
    spread_loss = ArchetypalRiskComposition._simplex_volume_loss(spread)
    assert spread_loss < collapsed_loss
    collapsed_loss.backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()


def test_arcsurv_beta_initialization_breaks_the_uniform_composition_symmetry():
    torch.manual_seed(23)
    bank = CohortArchetypeBank(
        16,
        num_archetypes=4,
        bank_size=12,
        temperature=0.25,
        beta_init_scale=1.5,
    )
    bank.train()
    states = torch.randn(12, 16)
    bank.update(states)

    composition, _, archetypes = bank(states)
    assert torch.pdist(archetypes).min() > 1e-3
    assert composition.var(dim=1, unbiased=False).mean() > 1e-5
    assert not torch.allclose(
        composition,
        torch.full_like(composition, 1.0 / bank.num_archetypes),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"arc_temperature": float("nan")}, "arc_temperature"),
        ({"arc_beta_init_scale": float("inf")}, "arc_beta_init_scale"),
        ({"arc_lambda_rank": float("nan")}, "arc_lambda_rank"),
        ({"arc_rank_max_pairs": 0}, "arc_rank_max_pairs"),
    ],
)
def test_arcsurv_rejects_invalid_objective_hyperparameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        ArchetypalRiskComposition(make_args(**overrides), omic_input_dim=20)
