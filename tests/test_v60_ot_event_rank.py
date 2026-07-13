from types import SimpleNamespace

import torch

from survot_rank.research.methods.v60_ot_event_rank.model import (
    V60OTEventRank,
    masked_log_sinkhorn_plan,
)
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
        otehv2_eps=0.05,
        otehv2_iter=5,
        otehv2_warmup=0,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.1,
        lambda_otehv2_ot=0.06,
        lambda_otehv2_div=0.0,
        lambda_otehv2_event_surv=0.0,
        lambda_otehv2_recon=0.0,
        v60_num_events=4,
        v60_lambda_per_event=0.15,
        v60_lambda_rank=0.15,
        v60_rank_max_pairs=128,
        alpha_surv=0.0,
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


def test_masked_sinkhorn_respects_valid_marginals():
    cost = torch.rand(2, 3, 4)
    row_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    col_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    plan, distance = masked_log_sinkhorn_plan(cost, row_mask, col_mask, eps=0.1, max_iter=40)
    assert torch.isfinite(plan).all()
    assert torch.isfinite(distance).all()
    assert torch.all(plan[~(row_mask.unsqueeze(2) & col_mask.unsqueeze(1))] == 0)
    assert torch.allclose(plan.sum(2)[row_mask], torch.tensor([0.5, 0.5, 1.0]), atol=1e-3)


def test_v60_complete_batch_forward_backward_is_finite():
    torch.manual_seed(0)
    model = V60OTEventRank(make_args(), omic_input_dim=20)
    model.train()
    logits, aux_loss = model(**make_inputs())
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    (logits.sum() + aux_loss).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_v60_missing_modalities_use_fallback_without_ot_failure():
    torch.manual_seed(0)
    model = V60OTEventRank(make_args(), omic_input_dim=20)
    model.train()
    inputs = make_inputs()
    inputs.update(
        {
            "wsi_available": torch.tensor([1, 1, 0]),
            "omics_available": torch.tensor([1, 0, 1]),
        }
    )
    logits, aux_loss = model(**inputs)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)


def test_v60_invalid_rank_pairs_return_finite_zero_rank_term():
    model = V60OTEventRank(make_args(), omic_input_dim=20)
    logits = torch.randn(2, 4)
    loss = model._ranking_loss(logits, torch.tensor([2, 1]), torch.tensor([1.0, 1.0]))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_v60_registered_and_alias_60_builds():
    assert "v60_ot_event_rank" in METHOD_REGISTRY
    assert METHOD_ALIASES["60"] == "v60_ot_event_rank"
    model = get_model("60", make_args(), omic_input_dim=20)
    assert type(model).__name__ == "V60OTEventRank"
