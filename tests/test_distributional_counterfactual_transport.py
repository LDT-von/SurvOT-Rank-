from types import SimpleNamespace

import torch

from survot_rank.research.methods.distributional_counterfactual_transport.model import (
    DistributionalCounterfactualTransport,
)


def make_args():
    return SimpleNamespace(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=3,
        slot_num_omics=3,
        slot_iters=2,
        otehv2_eps=0.05,
        otehv2_iter=100,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.1,
        dct_num_stages=4,
        dct_lambda_ot=0.06,
        dct_lambda_rank=0.05,
        dct_lambda_anchor=0.03,
        dct_lambda_stage_risk=0.05,
        dct_stage_risk_margin=0.02,
        dct_anchor_margin=0.02,
        dct_anchor_momentum=0.90,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_coupling_projection_iters=1000,
        dct_coupling_projection_tol=1e-4,
        dct_lambda_coordinate=0.01,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.50,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=1,
    )


def test_train_fold_reference_uses_late_censoring_as_low_risk_set_context():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    low, high = model._stage_membership_weights(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert torch.all(high[:4].sum(dim=1) > 0)
    assert low[4].sum() > 0
    assert low[4, -1] > 0


def test_distributional_counterfactual_transport_uses_feasible_risk_anchored_paths():
    torch.manual_seed(0)
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(5, 6, 16),
        x_omics=torch.randn(5, 5, 20),
        event_time=torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert logits.shape == (5, 4)
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    aux_loss.backward()
    assert model.stage_pair_cost[-1].weight.grad is not None
    assert model.risk_anchor_seen.all()
    assert not hasattr(model, "risk_prototypes")

    # A second batch activates the survival-anchored contrastive geometry loss;
    # it never imposes a requested ordering on the model's CF risk predictions.
    model.zero_grad(set_to_none=True)
    _, aux_loss = model(
        x_wsi=torch.randn(5, 6, 16),
        x_omics=torch.randn(5, 5, 20),
        event_time=torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    aux_loss.backward()
    assert model.stage_pair_cost[-1].weight.grad is not None

    model.eval()
    with torch.no_grad():
        logits, _ = model(
            x_wsi=torch.randn(5, 6, 16),
            x_omics=torch.randn(5, 5, 20),
        )
    explanation = model.explain_last_batch()
    assert logits.shape == (5, 4)
    assert explanation["low_risk_counterfactual"].shape == (5,)
    assert explanation["high_risk_counterfactual"].shape == (5,)
    for key in (
        "factual_coupling_marginal_error",
        "low_coupling_marginal_error",
        "high_coupling_marginal_error",
    ):
        assert torch.all(explanation[key] < 1e-3), key


def test_competitive_semantic_slots_separate_opposite_token_evidence():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    tokens = torch.zeros(1, 2, 16)
    tokens[0, 0, 0] = 1.0
    tokens[0, 1, 0] = -1.0
    prototypes = torch.zeros(2, 16)
    prototypes[0, 0] = 1.0
    prototypes[1, 0] = -1.0
    _, weights = model._semantic_slots(tokens, prototypes)
    assert weights[0, 0, 0] > weights[0, 1, 0]
    assert weights[0, 1, 1] > weights[0, 0, 1]


def test_stage_risk_contrast_uses_observed_high_and_low_risk_sets():
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    low = torch.zeros(2, 4)
    high = torch.zeros(2, 4)
    low[1, 0] = 1.0
    high[0, 0] = 1.0
    separated = model._stage_risk_contrast_loss(
        torch.tensor([[4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0]]), low, high
    )
    reversed_loss = model._stage_risk_contrast_loss(
        torch.tensor([[-4.0, -4.0, -4.0, -4.0], [4.0, -4.0, -4.0, -4.0]]), low, high
    )
    assert separated == 0
    assert reversed_loss > 0
