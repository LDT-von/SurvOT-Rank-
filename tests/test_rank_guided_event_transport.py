from types import SimpleNamespace

import torch

from survot_rank.research.methods.rank_guided_event_transport.model import (
    RankGuidedEventTransport,
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
        rg_num_events=4,
        rg_prog_cost=0.2,
        rg_lambda_ot=0.06,
        rg_lambda_rank=0.15,
        rg_lambda_stage=0.02,
        rg_rank_margin=0.0,
        rg_rank_max_pairs=128,
        rg_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        cur_epoch=1,
    )


def test_rank_guided_transport_has_compact_training_objective():
    torch.manual_seed(0)
    model = RankGuidedEventTransport(make_args(), omic_input_dim=20)
    model.train()
    outputs, aux_loss = model(
        x_wsi=torch.randn(3, 6, 16),
        x_omics=torch.randn(3, 5, 20),
        event_time=torch.tensor([4.0, 12.0, 7.0]),
        c=torch.tensor([0.0, 1.0, 0.0]),
    )
    assert outputs.shape == (3, 4)
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    aux_loss.backward()
    assert model.prognostic_pair_cost[1].weight.grad is not None
