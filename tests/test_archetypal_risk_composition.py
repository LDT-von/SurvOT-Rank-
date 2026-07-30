from types import SimpleNamespace

import torch

from survot_rank.research.methods.archetypal_risk_composition.model import (
    ArchetypalRiskComposition,
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
        arc_lambda_recon=0.05,
        arc_lambda_align=0.05,
        arc_lambda_balance=0.01,
        arc_lambda_rank=0.10,
        arc_rank_margin=0.0,
        arc_rank_max_pairs=128,
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
    (logits.sum() + aux_loss).backward()
    assert model.archetype_hazard_logits.grad is not None
    assert torch.isfinite(model.archetype_hazard_logits.grad).all()


def test_arcsurv_archetypes_are_convex_combinations_of_memory():
    torch.manual_seed(1)
    model = ArchetypalRiskComposition(make_args(), omic_input_dim=20)
    model.train()
    model(**make_inputs())
    parameters = model.archetype_parameters()
    wsi_beta = parameters["wsi_beta"]
    assert torch.all(wsi_beta >= 0)
    assert torch.allclose(wsi_beta.sum(dim=1), torch.ones(4), atol=1e-6)
    expected = wsi_beta @ model.wsi_archetypes.memory[:3]
    assert torch.allclose(parameters["wsi_archetypes"], expected, atol=1e-6)


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
    assert model.wsi_archetypes.memory_count.item() == 2
    assert model.omic_archetypes.memory_count.item() == 2

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


def test_arcsurv_is_registered_and_parser_accepts_alias():
    assert "archetypal_risk_composition" in METHOD_REGISTRY
    assert METHOD_ALIASES["arcsurv"] == "archetypal_risk_composition"
    parsed = process_args_extended(["--survot_method", "arcsurv"])
    assert parsed.survot_method == "arcsurv"
    model = get_model("arcsurv", make_args(), omic_input_dim=20)
    assert type(model).__name__ == "ArchetypalRiskComposition"
