from types import SimpleNamespace

import torch

from survot_rank.research.methods.faithful_evidence_transport.model import (
    FaithfulEvidenceTransport,
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
        fet_num_stages=4,
        fet_lambda_sparse=0.01,
        fet_lambda_faith=0.05,
        fet_keep_ratio=0.25,
        fet_faith_margin=0.05,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.06,
        spt_lambda_rank=0.05,
        spt_lambda_stage=0.02,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=1,
    )


def test_faithful_transport_exposes_stage_slot_evidence():
    torch.manual_seed(0)
    model = FaithfulEvidenceTransport(make_args(), omic_input_dim=20)
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
    assert explanation["stage_slot_pair_evidence"].shape == (3, 4, 3, 3)
    aux_loss.backward()
    assert model.evidence_gate[-1].weight.grad is not None
