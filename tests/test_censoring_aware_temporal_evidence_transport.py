from types import SimpleNamespace

import torch

from survot_rank.research.methods.censoring_aware_temporal_evidence_transport.model import (
    CensoringAwareTemporalEvidenceTransport,
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
        catet_num_stages=4,
        catet_prog_cost=0.2,
        catet_lambda_ot=0.04,
        catet_lambda_rank=0.08,
        catet_lambda_intervention=0.05,
        catet_keep_ratio=0.25,
        catet_intervention_margin=0.05,
        catet_rank_margin=0.0,
        catet_rank_max_pairs=128,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=1,
    )


def test_catet_changes_ot_evidence_and_exposes_intervention():
    torch.manual_seed(0)
    model = CensoringAwareTemporalEvidenceTransport(make_args(), omic_input_dim=20)
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(3, 6, 16),
        x_omics=torch.randn(3, 5, 20),
        event_time=torch.tensor([4.0, 12.0, 7.0]),
        c=torch.tensor([0.0, 1.0, 0.0]),
    )
    explanation = model.explain_last_batch()
    assert logits.shape == (3, 4)
    assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
    assert explanation["stage_slot_pair_evidence"].shape == (3, 4, 3, 3)
    assert explanation["stage_slot_pair_risk"].shape == (3, 4, 3, 3)
    assert torch.isfinite(explanation["removed_risk"]).all()
    aux_loss.backward()
    assert model.temporal_evidence_gate[-1].weight.grad is not None
    assert model.stage_edge_risk[-1].weight.grad is not None
