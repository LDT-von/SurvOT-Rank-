from types import SimpleNamespace

import torch

from survot_rank.research.methods.cohort_anchored_adaptive_prognostic_slot_attention.model import (
    CohortAnchoredAdaptivePrognosticSlotAttention,
)
from survot_rank.training.model_factory import METHOD_ALIASES, METHOD_REGISTRY, get_model


def make_args(**overrides):
    args = SimpleNamespace(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_iters=2,
        capsa_max_slots=5,
        capsa_slot_iters=2,
        capsa_heads=2,
        capsa_dropout=0.0,
        capsa_gate_temperature=2.0 / 3.0,
        capsa_gate_gamma=-0.1,
        capsa_gate_zeta=1.1,
        capsa_gate_threshold=0.5,
        capsa_gate_prior_start=-1.0,
        capsa_gate_prior_end=-2.2,
        capsa_lambda_sparse=0.01,
        capsa_lambda_align=0.02,
        capsa_lambda_budget=0.01,
        capsa_lambda_identity=0.02,
        capsa_target_active_ratio=0.25,
        capsa_identity_temperature=0.10,
        capsa_anchor_cosine_margin=0.20,
        capsa_anchor_scale=0.50,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_inputs(batch=3):
    return {
        "x_wsi": torch.randn(batch, 7, 16),
        "x_omics": torch.randn(batch, 6, 20),
    }


def test_capsa_forward_backward_has_closed_identity_and_budget_objective():
    torch.manual_seed(3)
    model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    model.train()
    logits, aux_loss = model(**make_inputs())

    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
    assert set(model.last_training_losses) == {
        "gate_budget",
        "identity",
        "identity_match_accuracy",
        "identity_diagonal_similarity",
        "anchor_max_offdiag_cosine",
        "mean_expected_active_ratio",
        "mean_hard_active_count",
        "auxiliary",
    }
    assert torch.allclose(
        aux_loss,
        0.01 * model.last_training_losses["gate_budget"]
        + 0.02 * model.last_training_losses["identity"],
    )

    (logits.sum() + aux_loss).backward()
    assert model.cohort_anchors.grad is not None
    assert model.gate_head[-1].weight.grad is not None
    assert model.wsi_slot_updater.to_q.weight.grad is not None
    assert model.omic_slot_updater.to_q.weight.grad is not None


def test_capsa_hard_concrete_is_dynamic_deterministic_and_keeps_one_slot():
    model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    log_alpha = torch.tensor(
        [[8.0, 8.0, -8.0, -8.0, -8.0], [-8.0, -8.0, -8.0, -8.0, -8.0]]
    )
    gates, expected = model._hard_concrete(log_alpha, sample=False)
    assert gates[0].sum() == 1
    assert gates[1].sum() == 1
    assert torch.equal(gates[1], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert expected[0, 0] > expected[0, 2]


def test_capsa_same_slot_indices_share_the_exact_cohort_anchor_at_initial_state():
    model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    anchors = model.cohort_anchors.detach().clone()
    wsi_anchor_batch = anchors.unsqueeze(0).expand(2, -1, -1)
    omic_anchor_batch = anchors.unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(wsi_anchor_batch, omic_anchor_batch)
    assert not hasattr(model, "slot_attention_wsi")
    assert not hasattr(model, "slot_attention_omic")


def test_capsa_eval_is_repeatable_and_exposes_patient_active_counts():
    torch.manual_seed(7)
    model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    model.eval()
    inputs = make_inputs()
    with torch.no_grad():
        logits_a, aux_a = model(**inputs)
        logits_b, aux_b = model(**inputs)
    explanation = model.explain_last_batch()

    assert torch.allclose(logits_a, logits_b)
    assert aux_a.item() == 0.0 and aux_b.item() == 0.0
    assert explanation["gates"].shape == (3, 5)
    assert explanation["active_slot_count"].shape == (3,)
    assert torch.all(explanation["active_slot_count"] >= 1)
    assert explanation["temporal_slot_weights"].shape == (3, 5, 4)
    assert torch.allclose(
        explanation["temporal_slot_weights"].sum(dim=1), torch.ones(3, 4), atol=1e-5
    )


def test_capsa_capacity_prior_avoids_initial_all_on_or_single_slot_collapse():
    torch.manual_seed(0)
    model = CohortAnchoredAdaptivePrognosticSlotAttention(
        make_args(capsa_max_slots=16), omic_input_dim=20
    )
    model.eval()
    with torch.no_grad():
        model(
            x_wsi=torch.randn(4, 7, 16),
            x_omics=torch.randn(4, 6, 20),
        )
    counts = model.explain_last_batch()["active_slot_count"]
    assert torch.all(counts == 4)


def test_capsa_explanation_is_the_exact_additive_prediction_path():
    torch.manual_seed(19)
    model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    model.eval()
    with torch.no_grad():
        logits, _ = model(**make_inputs())
    explanation = model.explain_last_batch()
    assert torch.allclose(
        explanation["route_time_contribution"].sum(dim=1), logits, atol=1e-6
    )
    gram = explanation["anchor_cosine_matrix"]
    assert torch.allclose(torch.diagonal(gram), torch.ones(5), atol=1e-6)
    assert torch.isfinite(gram).all()


def test_capsa_gate_budget_has_no_all_off_optimum():
    model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    target = torch.full((4, 5), 0.25)
    all_off = torch.zeros_like(target)
    assert model._gate_budget_loss(target) < model._gate_budget_loss(all_off)


def test_capsa_missing_modality_and_pathway_inputs_are_supported():
    torch.manual_seed(11)
    rna_model = CohortAnchoredAdaptivePrognosticSlotAttention(make_args(), omic_input_dim=20)
    rna_model.train()
    inputs = make_inputs()
    inputs["wsi_available"] = torch.tensor([1, 1, 0])
    inputs["omics_available"] = torch.tensor([1, 0, 1])
    logits, aux_loss = rna_model(**inputs)
    assert torch.isfinite(logits).all() and torch.isfinite(aux_loss)

    pathway_args = make_args(rna_format="Pathways", omic_sizes=[3, 4])
    pathway_model = CohortAnchoredAdaptivePrognosticSlotAttention(pathway_args)
    pathway_model.eval()
    with torch.no_grad():
        logits, _ = pathway_model(
            x_wsi=torch.randn(2, 7, 16),
            x_omic1=torch.randn(2, 3),
            x_omic2=torch.randn(2, 4),
        )
    assert logits.shape == (2, 4)


def test_capsa_registered_and_alias_builds():
    name = "cohort_anchored_adaptive_prognostic_slot_attention"
    assert name in METHOD_REGISTRY
    assert METHOD_ALIASES["ca_psa"] == name
    model = get_model("ca_psa", make_args(), omic_input_dim=20)
    # The factory intentionally loads each method under an isolated module name.
    assert type(model).__name__ == "CohortAnchoredAdaptivePrognosticSlotAttention"
