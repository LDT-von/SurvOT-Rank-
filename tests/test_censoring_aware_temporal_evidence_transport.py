from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.censoring_aware_temporal_evidence_transport.model import (
    CensoringAwareTemporalEvidenceTransport,
)


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
        otehv2_eps=0.08,
        otehv2_iter=30,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        catet_num_stages=4,
        catet_prog_cost=0.2,
        catet_lambda_ot=0.04,
        catet_lambda_rank=0.08,
        catet_lambda_stage=0.04,
        catet_lambda_intervention=0.05,
        catet_keep_ratio=0.25,
        catet_intervention_margin=0.05,
        catet_intervention_cost=1.0,
        catet_plan_diversity_margin=0.01,
        catet_rank_margin=0.0,
        catet_rank_temperature=0.5,
        catet_ipcw_max_weight=10.0,
        catet_rank_max_pairs=128,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def reference():
    return (
        torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )


def batch():
    return {
        "x_wsi": torch.randn(4, 6, 16),
        "x_omics": torch.randn(4, 5, 20),
        "event_time": torch.tensor([4.0, 12.0, 7.0, 15.0]),
        "c": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }


def test_catet_final_has_true_stage_costs_balanced_retransport_and_gradients():
    torch.manual_seed(0)
    model = CensoringAwareTemporalEvidenceTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()
    logits, aux_loss = model(**batch())
    explanation = model.explain_last_batch()

    assert logits.shape == (4, 4)
    assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
    assert explanation["stage_slot_pair_evidence"].shape == (4, 4, 3, 3)
    assert explanation["stage_slot_pair_risk"].shape == (4, 4, 3, 3)
    assert not torch.allclose(
        explanation["stage_slot_pair_risk"][:, 0],
        explanation["stage_slot_pair_risk"][:, 1],
    )
    assert explanation["adjacent_stage_plan_l1"].mean() > 0

    for prefix in ("factual", "keep", "remove"):
        assert explanation[f"{prefix}_row_marginal_error"].max() < 5e-4
        assert explanation[f"{prefix}_col_marginal_error"].max() < 5e-4

    assert torch.allclose(
        model._risk_score(logits), explanation["factual_risk"], atol=1e-6
    )
    assert set(model.last_training_losses) == {
        "catet_ot",
        "catet_plan_diversity",
        "catet_adjacent_plan_l1",
        "catet_ipcw_rank",
        "catet_ipcw_pairs",
        "catet_censored_stage",
        "catet_sufficiency",
        "catet_comprehensiveness",
        "catet_gate_budget",
        "catet_factual_marginal_error",
        "catet_keep_marginal_error",
        "catet_remove_marginal_error",
        "catet_mean_sufficiency_gap",
        "catet_mean_comprehensiveness_gap",
        "catet_stage_probability_entropy",
        "catet_auxiliary",
        "catet_finite",
    }
    assert model.last_training_losses["catet_ipcw_pairs"] > 0
    assert model.last_training_losses["catet_finite"] == 1

    (logits.sum() + aux_loss).backward()
    assert model.temporal_evidence_gate[-1].weight.grad is not None
    assert model.stage_edge_risk[-1].weight.grad is not None
    assert torch.isfinite(model.stage_edge_risk[-1].weight.grad).all()


def test_catet_stage_nll_prefers_the_observed_event_stage():
    model = CensoringAwareTemporalEvidenceTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    times = torch.tensor([2.0, 8.0, 12.0, 16.0])
    censorship = torch.zeros(4)
    indices = torch.bucketize(times, model.catet_stage_edges[1:-1])
    correct = torch.full((4, 4), 0.01)
    correct[torch.arange(4), indices] = 0.97
    wrong = correct.flip(1)
    assert model._censored_stage_loss(correct, times, censorship) < model._censored_stage_loss(
        wrong, times, censorship
    )


def test_catet_training_reference_produces_nontrivial_ipcw_and_is_required_for_stage_loss():
    model = CensoringAwareTemporalEvidenceTransport(make_args(), omic_input_dim=20)
    probability = torch.full((2, 4), 0.25, requires_grad=True)
    zero = model._censored_stage_loss(
        probability, torch.tensor([2.0, 4.0]), torch.tensor([0.0, 1.0])
    )
    assert zero.item() == 0.0
    model.configure_train_reference(*reference())
    assert model.has_train_reference
    assert model._ipcw(torch.tensor([16.0])).item() > 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"catet_num_stages": 1}, "catet_num_stages"),
        ({"catet_intervention_cost": -1.0}, "catet_intervention_cost"),
        ({"catet_rank_temperature": 0.0}, "catet_rank_temperature"),
        ({"catet_rank_max_pairs": 0}, "catet_rank_max_pairs"),
    ],
)
def test_catet_rejects_invalid_final_objective(overrides, message):
    with pytest.raises(ValueError, match=message):
        CensoringAwareTemporalEvidenceTransport(
            make_args(**overrides), omic_input_dim=20
        )
