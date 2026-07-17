from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.v70_patient_specific_prognostic_circuits.model import (
    V70PatientSpecificPrognosticCircuits,
)
from survot_rank.training.model_factory import METHOD_ALIASES, METHOD_REGISTRY, get_model


def make_args(**overrides):
    args = SimpleNamespace(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=8,
        slot_num_omics=8,
        slot_iters=2,
        pspc_max_modules=5,
        pspc_heads=2,
        pspc_layers=2,
        pspc_dropout=0.0,
        pspc_gate_temperature=2.0 / 3.0,
        pspc_gate_gamma=-0.1,
        pspc_gate_zeta=1.1,
        pspc_gate_threshold=0.5,
        pspc_edge_temperature=0.75,
        pspc_edge_threshold=0.5,
        pspc_edge_rank=2,
        pspc_lambda_node_sparse=0.01,
        pspc_lambda_edge_sparse=0.005,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_inputs(batch=3):
    return {
        "x_wsi": torch.randn(batch, 7, 16),
        "x_omics": torch.randn(batch, 6, 20),
    }


def test_v70_forward_backward_uses_only_two_structural_auxiliary_terms():
    torch.manual_seed(3)
    model = V70PatientSpecificPrognosticCircuits(make_args(), omic_input_dim=20)
    model.train()
    logits, aux_loss = model(**make_inputs())

    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
    assert set(model.last_training_losses) == {
        "node_sparse",
        "edge_sparse",
        "auxiliary",
    }
    expected = (
        0.01 * model.last_training_losses["node_sparse"]
        + 0.005 * model.last_training_losses["edge_sparse"]
    )
    assert torch.allclose(aux_loss, expected)

    (logits.sum() + aux_loss).backward()
    assert model.module_codes.grad is not None
    assert model.read_in.in_proj_weight.grad is not None
    assert model.context_to_gates.weight.grad is not None
    assert model.edge_basis.grad is not None


def test_v70_has_no_slot_attention_ot_or_fixed_event_queries():
    model = V70PatientSpecificPrognosticCircuits(make_args(), omic_input_dim=20)
    names = {name for name, _ in model.named_modules()}
    assert not hasattr(model, "slot_attention_wsi")
    assert not hasattr(model, "slot_attention_omic")
    assert not hasattr(model, "fusion")
    assert not hasattr(model, "event_queries")
    assert not any("slot_attention" in name for name in names)


def test_v70_eval_is_repeatable_and_exposes_a_valid_sparse_circuit():
    torch.manual_seed(7)
    model = V70PatientSpecificPrognosticCircuits(make_args(), omic_input_dim=20)
    model.eval()
    inputs = make_inputs()
    with torch.no_grad():
        logits_a, aux_a = model(**inputs)
        logits_b, aux_b = model(**inputs)
    explanation = model.explain_last_batch()

    assert torch.allclose(logits_a, logits_b)
    assert aux_a.item() == 0.0 and aux_b.item() == 0.0
    assert explanation["module_gates"].shape == (3, 5)
    assert explanation["adjacency"].shape == (3, 5, 5)
    assert explanation["edge_probability"].shape == (3, 5, 5)
    assert explanation["temporal_module_weights"].shape == (3, 5, 4)
    assert torch.all(explanation["active_module_count"] >= 1)
    assert torch.allclose(
        explanation["temporal_module_weights"].sum(dim=1),
        torch.ones(3, 4),
        atol=1e-5,
    )

    row_sums = explanation["adjacency"].sum(dim=-1)
    active = explanation["module_gates"] > 0.5
    assert torch.allclose(row_sums[active], torch.ones_like(row_sums[active]), atol=1e-5)
    assert torch.allclose(row_sums[~active], torch.zeros_like(row_sums[~active]), atol=1e-5)


def test_v70_circuit_generator_is_patient_conditioned():
    torch.manual_seed(13)
    model = V70PatientSpecificPrognosticCircuits(make_args(), omic_input_dim=20)
    model.eval()
    with torch.no_grad():
        model(
            x_wsi=torch.zeros(1, 7, 16),
            x_omics=torch.zeros(1, 6, 20),
        )
        first = model.explain_last_batch()
        model(
            x_wsi=torch.randn(1, 7, 16) * 3.0,
            x_omics=torch.randn(1, 6, 20) * 3.0,
        )
        second = model.explain_last_batch()

    assert not torch.allclose(
        first["expected_open_probability"], second["expected_open_probability"]
    )
    assert not torch.allclose(first["edge_probability"], second["edge_probability"])


def test_v70_missing_modalities_pathways_and_token_masks_are_supported():
    torch.manual_seed(17)
    rna_model = V70PatientSpecificPrognosticCircuits(make_args(), omic_input_dim=20)
    rna_model.train()
    inputs = make_inputs()
    inputs["wsi_available"] = torch.tensor([1, 1, 0])
    inputs["omics_available"] = torch.tensor([1, 0, 0])
    inputs["wsi_token_mask"] = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    logits, aux_loss = rna_model(**inputs)
    assert torch.isfinite(logits).all() and torch.isfinite(aux_loss)

    pathway_args = make_args(rna_format="Pathways", omic_sizes=[3, 4])
    pathway_model = V70PatientSpecificPrognosticCircuits(pathway_args)
    pathway_model.eval()
    with torch.no_grad():
        pathway_logits, _ = pathway_model(
            x_wsi=torch.randn(2, 7, 16),
            x_omic1=torch.randn(2, 3),
            x_omic2=torch.randn(2, 4),
        )
    assert pathway_logits.shape == (2, 4)


def test_v70_rejects_wrong_token_mask_shape():
    model = V70PatientSpecificPrognosticCircuits(make_args(), omic_input_dim=20)
    with pytest.raises(ValueError, match="wsi_token_mask"):
        model(**make_inputs(), wsi_token_mask=torch.ones(3, 6))


def test_v70_registered_and_aliases_build():
    name = "v70_patient_specific_prognostic_circuits"
    assert name in METHOD_REGISTRY
    assert METHOD_ALIASES["70"] == name
    assert METHOD_ALIASES["pspc_surv"] == name
    model = get_model("pspc", make_args(), omic_input_dim=20)
    assert type(model).__name__ == "V70PatientSpecificPrognosticCircuits"
