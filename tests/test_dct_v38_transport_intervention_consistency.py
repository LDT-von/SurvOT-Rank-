from types import SimpleNamespace

import torch

from scripts.run_dct_v38_transport_consistency import (
    COMMON_OVERRIDES,
    PROTOCOLS,
    VARIANTS,
    build_parser,
    build_train_command,
    parse_folds,
    parse_variants,
)
from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)
from survot_rank.training.model_factory import get_model, list_methods


def make_args(**overrides):
    values = dict(
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
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.10,
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.50,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=0,
        dct_lambda_etar=0.0,
        dct_lambda_ot=0.0,
        dct_lambda_rank=0.0,
        dct_lambda_anchor=0.0,
        dct_lambda_stage_risk=0.0,
        dct_lambda_coordinate=0.0,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=1.0,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.03,
        dct_v38_lambda_reconfiguration=0.02,
        dct_v38_direction_margin=0.02,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.05,
        dct_v38_alpha_mid=0.50,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=1,
        dct_v38_dose_every=1,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        dct_slot_init_mode="deterministic",
        dct_slot_eval_seed=91,
        cur_epoch=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reference():
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return times, censorship


def batch(seed):
    generator = torch.Generator().manual_seed(seed)
    times, censorship = reference()
    return {
        "x_wsi": torch.randn(8, 6, 16, generator=generator),
        "x_omics": torch.randn(8, 5, 20, generator=generator),
        "event_time": times,
        "c": censorship,
    }


def test_direction_and_dose_losses_reward_the_requested_order():
    model = DCTTransportInterventionConsistency(make_args(), omic_input_dim=20)
    factual = torch.tensor([-2.0, -1.5])
    ordered_direction, _, _ = model._direction_loss(
        factual,
        factual - 0.20,
        factual + 0.20,
    )
    reversed_direction, _, _ = model._direction_loss(
        factual,
        factual + 0.20,
        factual - 0.20,
    )
    assert ordered_direction < reversed_direction

    ordered_dose = model._dose_loss(
        factual,
        factual - 0.10,
        factual - 0.20,
        factual + 0.10,
        factual + 0.20,
    )
    reversed_dose = model._dose_loss(
        factual,
        factual + 0.10,
        factual + 0.20,
        factual - 0.10,
        factual - 0.20,
    )
    assert ordered_dose < reversed_dose


def test_v38_forward_uses_reoptimised_interventions_and_has_finite_gradients():
    torch.manual_seed(7)
    args = make_args()
    model = DCTTransportInterventionConsistency(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    # Epoch zero only establishes detached train-fold anchors.
    _, warmup_aux = model(**batch(1))
    assert warmup_aux.isfinite()
    assert model.risk_anchor_seen.all()
    assert model.last_training_losses["v38_total"] == 0

    args.cur_epoch = 1
    logits, aux_loss = model(**batch(2))
    diagnostics = model.last_training_losses
    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert diagnostics["v38_finite"] == 1
    assert diagnostics["v38_active_stage_fraction"] == 1
    assert diagnostics["v38_direction"] > 0
    assert diagnostics["v38_dose"] > 0
    assert diagnostics["v38_reconfiguration"] > 0
    assert diagnostics["v38_high_plan_shift"] > 0
    assert diagnostics["v38_low_plan_shift"] > 0
    assert torch.allclose(
        aux_loss.detach(),
        (
            model.dct_lambda_ipcw_rank * diagnostics["ipcw_rank"]
            + diagnostics["v38_total"]
        ),
        atol=1e-6,
    )

    aux_loss.backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert finite_gradients
    assert all(bool(value) for value in finite_gradients)
    assert model.stage_pair_cost[-1].weight.grad is not None
    assert model.event_hazard.weight.grad is not None


def test_v38_all_censored_batch_skips_structural_loss_without_nan():
    args = make_args(cur_epoch=2)
    model = DCTTransportInterventionConsistency(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()
    values = batch(3)
    values["c"] = torch.ones(8)
    _, aux_loss = model(**values)
    assert torch.isfinite(aux_loss)
    assert model.last_training_losses["v38_total"] == 0
    assert model.last_training_losses["v38_active_stage_fraction"] == 0


def test_v38_registry_and_screen_are_isolated_and_auditable():
    assert "dct_transport_intervention_consistency" in list_methods()
    factory_model = get_model("dct_v38", make_args(), omic_input_dim=20)
    assert factory_model.__class__.__name__ == "DCTTransportInterventionConsistency"
    assert set(VARIANTS) == {
        "base",
        "direction",
        "dose",
        "reconfiguration",
        "full",
    }
    assert set(PROTOCOLS) == {"highscore", "clean"}
    defaults = build_parser().parse_args([])
    assert defaults.mode == "plan"
    assert defaults.protocols == ["highscore"]
    assert defaults.variants == ["full"]
    assert parse_variants("direction,dose") == ["direction", "dose"]
    assert parse_folds("0,2") == [0, 2]

    command, result_dir = build_train_command(
        "python3",
        "blca",
        "highscore",
        "full",
        2,
        "0",
        "4",
        "/data1/TCGA-UNI2-h-features",
    )
    rendered = " ".join(command)
    assert "survot_method=dct_transport_intervention_consistency" in rendered
    assert "fit_bins_on_train=false" in rendered
    assert "dct_slot_init_mode=gaussian" in rendered
    assert "dct_v38_lambda_direction=0.05" in rendered
    assert "dct_v38_lambda_dose=0.03" in rendered
    assert "dct_v38_lambda_reconfiguration=0.02" in rendered
    assert result_dir.as_posix() == (
        "results/dct_v3.8_transport_consistency/highscore/full/blca"
    )
    assert COMMON_OVERRIDES["dct_lambda_ipcw_rank"] == 0.10
