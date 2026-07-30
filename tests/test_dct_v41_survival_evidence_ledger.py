import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.run_dct_v41_survival_evidence_ledger import (
    CANCERS,
    FOLDS,
    build_train_command,
    parse_cancers,
    parse_folds,
)
from survot_rank.research.methods.dct_v41_survival_evidence_ledger.model import (
    CrossLedgerCompletion,
    DCTV41SurvivalEvidenceLedger,
    SurvivalEvidenceLedger,
)
from survot_rank.training.model_factory import get_model


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
        otehv2_iter=20,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.1,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.10,
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.50,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=0,
        dct_lambda_etar=0.0,
        dct_etar_margin=0.02,
        dct_etar_uncertainty_weight=0.05,
        dct_etar_temperature=0.50,
        dct_etar_evidence_floor=0.10,
        dct_anchor_momentum=0.90,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
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
        v41_modality_dropout=0.35,
        v41_ledger_temperature=0.25,
        v41_missing_confidence_cap=0.65,
        v41_confidence_floor=0.05,
        v41_lambda_completion=0.05,
        v41_lambda_ledger=0.02,
        v41_lambda_survival=0.05,
        v41_lambda_private=0.02,
        v41_shared_rank=4,
    )


def _configure_reference(model):
    model.configure_train_reference(
        torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_ledger_conserves_every_tokens_nonnegative_precision_mass():
    torch.manual_seed(1)
    ledger = SurvivalEvidenceLedger(dim=12, num_slots=4, temperature=0.25)
    tokens = torch.randn(3, 7, 12)
    slots, assignment, slot_mass, confidence = ledger(tokens)

    normalized = ledger.token_norm(tokens)
    token_mass = (
        torch.nn.functional.softplus(
            ledger.token_precision(normalized).squeeze(-1)
        )
        + 1e-4
    )
    assert slots.shape == (3, 4, 12)
    assert assignment.shape == (3, 4, 7)
    assert torch.allclose(assignment.sum(dim=1), torch.ones(3, 7), atol=1e-6)
    assert torch.allclose(slot_mass.sum(dim=1), token_mass.sum(dim=1), atol=1e-5)
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))


def test_v41_replaces_v33_slots_and_factory_aliases_resolve():
    for method in (
        "dct_v41_survival_evidence_ledger",
        "dct_v41",
        "dct_v4_1",
    ):
        model = get_model(method, make_args(), omic_input_dim=20)
        assert model.__class__.__name__ == "DCTV41SurvivalEvidenceLedger"
        assert not hasattr(model, "slot_attention_wsi")
        assert not hasattr(model, "slot_attention_omic")
        assert not hasattr(model, "shared_wsi_prototypes")
        assert not hasattr(model, "shared_omic_prototypes")


def test_missing_modality_completion_is_finite_and_confidence_tempered():
    torch.manual_seed(2)
    model = DCTV41SurvivalEvidenceLedger(make_args(), omic_input_dim=20)
    model.eval()
    availability_wsi = torch.tensor([1.0, 1.0, 0.0, 0.0])
    availability_omic = torch.tensor([1.0, 0.0, 1.0, 0.0])

    with torch.no_grad():
        logits, aux_loss = model(
            x_wsi=torch.randn(4, 6, 16),
            x_omics=torch.randn(4, 5, 20),
            wsi_available=availability_wsi,
            omics_available=availability_omic,
        )

    explanation = model.explain_last_batch()
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(torch.as_tensor(aux_loss))
    assert torch.equal(explanation["wsi_available"], availability_wsi)
    assert torch.equal(explanation["omic_available"], availability_omic)
    assert torch.all(
        explanation["wsi_ledger_confidence"][2]
        <= make_args().v41_missing_confidence_cap
    )
    assert torch.all(
        explanation["omic_ledger_confidence"][1]
        <= make_args().v41_missing_confidence_cap
    )
    assert torch.allclose(
        explanation["wsi_ledger_confidence"][3],
        torch.full((3,), make_args().v41_confidence_floor),
    )
    assert torch.allclose(
        explanation["omic_ledger_confidence"][3],
        torch.full((3,), make_args().v41_confidence_floor),
    )
    assert torch.count_nonzero(explanation["wsi_ledger_assignment"][2:]) == 0
    assert torch.count_nonzero(
        explanation["omic_ledger_assignment"][[1, 3]]
    ) == 0
    assert explanation["wsi_recoverable_shared"].shape == (4, 3, 16)
    assert explanation["omic_recoverable_shared"].shape == (4, 3, 16)
    assert torch.all(explanation["wsi_private_uncertainty"] >= 0)
    assert torch.all(explanation["omic_private_uncertainty"] >= 0)


def test_ledger_confidence_changes_the_actual_ot_marginals():
    model = DCTV41SurvivalEvidenceLedger(make_args(), omic_input_dim=20)
    rows = torch.full((1, 4, 3), 1.0 / 3.0)
    cols = torch.full((1, 4, 3), 1.0 / 3.0)
    wsi_confidence = torch.tensor([[0.9, 0.3, 0.1]])
    omic_confidence = torch.tensor([[0.2, 0.8, 0.4]])

    tempered_rows, tempered_cols = model._temper_marginals(
        rows, cols, wsi_confidence, omic_confidence
    )
    assert torch.allclose(tempered_rows.sum(dim=-1), torch.ones(1, 4))
    assert torch.allclose(tempered_cols.sum(dim=-1), torch.ones(1, 4))
    assert torch.all(tempered_rows[..., 0] > tempered_rows[..., 1])
    assert torch.all(tempered_rows[..., 1] > tempered_rows[..., 2])
    assert torch.all(tempered_cols[..., 1] > tempered_cols[..., 2])
    assert torch.all(tempered_cols[..., 2] > tempered_cols[..., 0])


def test_selc_objective_is_active_and_backpropagates_through_new_ledgers():
    torch.manual_seed(1)
    model = DCTV41SurvivalEvidenceLedger(make_args(), omic_input_dim=20)
    _configure_reference(model)
    model.train()
    logits, aux_loss = model(
        x_wsi=torch.randn(5, 6, 16),
        x_omics=torch.randn(5, 5, 20),
        event_time=torch.tensor([2.0, 4.0, 7.0, 10.0, 13.0]),
        c=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    losses = model.last_training_losses

    assert logits.shape == (5, 4)
    assert torch.isfinite(aux_loss)
    assert losses["v41_missing_fraction"] > 0
    expected = (
        model.dct_lambda_ipcw_rank * losses["ipcw_rank"]
        + losses["v41_objective"]
    )
    assert torch.allclose(aux_loss, expected)
    for key in (
        "v41_completion",
        "v41_private_uncertainty",
        "v41_ledger",
        "v41_survival_consistency",
        "v41_objective",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])
    aux_loss.backward()
    assert model.wsi_ledger.token_key.weight.grad is not None
    assert model.omic_ledger.token_key.weight.grad is not None
    assert model.wsi_from_omic.source_shared[1].weight.grad is not None
    assert model.omic_from_wsi.source_shared[1].weight.grad is not None
    assert model.wsi_from_omic.target_shared[1].weight.grad is not None
    assert model.omic_from_wsi.target_shared[1].weight.grad is not None


def test_cross_ledger_separates_shared_evidence_and_private_uncertainty():
    torch.manual_seed(11)
    completion = CrossLedgerCompletion(
        dim=8,
        confidence_cap=0.7,
        shared_rank=3,
    )
    source = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    confidence = torch.full((2, 4), 0.9)

    shared_target, private_target = completion.decompose_target(target)
    assert torch.allclose(shared_target + private_target, target, atol=1e-6)

    outputs = completion(source, confidence)
    shared, _, completed_confidence, private_uncertainty, recoverability = outputs
    assert shared.shape == target.shape
    assert torch.all(private_uncertainty >= 0)
    assert torch.all((recoverability >= 0) & (recoverability <= 1))
    assert torch.all(completed_confidence <= 0.7)

    with torch.no_grad():
        completion.private_uncertainty.bias.fill_(5.0)
    high_uncertainty_confidence = completion(source, confidence)[2]
    assert torch.all(high_uncertainty_confidence < completed_confidence)

def test_v41_configs_and_runner_are_restricted_to_uni_four_cancers_three_folds():
    root = Path(__file__).resolve().parent.parent
    assert CANCERS == ("blca", "brca", "stad", "hnsc")
    assert FOLDS == (0, 2, 4)
    assert parse_cancers("blca,hnsc") == ["blca", "hnsc"]
    assert parse_folds("0,2,4") == [0, 2, 4]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_cancers("kirc")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_folds("1")

    for cancer in CANCERS:
        config_path = (
            root
            / "configs"
            / f"dct_v41_survival_evidence_ledger_{cancer}.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["data"]["study"] == cancer
        assert config["data"]["wsi_encoder"] == "uni"
        assert config["data"]["encoding_dim"] == 1024
        assert (
            config["train"]["survot_method"]
            == "dct_v41_survival_evidence_ledger"
        )

        for fold in FOLDS:
            command, _ = build_train_command(
                "python", cancer, fold, "0", "4", "/uni"
            )
            joined = " ".join(command)
            assert f"dct_v41_survival_evidence_ledger_{cancer}.yaml" in joined
            assert "--set wsi_encoder=uni" in joined
            assert f"--set k_start={fold}" in joined
            assert f"--set k_end={fold + 1}" in joined
