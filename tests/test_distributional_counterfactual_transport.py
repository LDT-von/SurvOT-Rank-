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
        otehv2_iter=5,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.1,
        dct_num_stages=4,
        dct_lambda_ot=0.06,
        dct_lambda_rank=0.05,
        dct_lambda_cf=0.10,
        dct_lambda_proto=0.01,
        dct_cf_margin=0.05,
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


def test_distributional_counterfactual_transport_outputs_risk_changes():
    torch.manual_seed(0)
    model = DistributionalCounterfactualTransport(make_args(), omic_input_dim=20)
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(3, 6, 16),
        x_omics=torch.randn(3, 5, 20),
        event_time=torch.tensor([4.0, 12.0, 7.0]),
        c=torch.tensor([0.0, 1.0, 0.0]),
    )
    explanation = model.explain_last_batch()
    assert logits.shape == (3, 4)
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert explanation["low_risk_counterfactual"].shape == (3,)
    assert explanation["high_risk_counterfactual"].shape == (3,)
    aux_loss.backward()
    assert model.risk_prototypes.grad is not None
