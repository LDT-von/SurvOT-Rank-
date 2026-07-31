import argparse
from types import SimpleNamespace

import pytest
import torch

from scripts.run_dct_v382_mgptr import (
    COMMON_OVERRIDES,
    SCREEN_VARIANTS,
    _validate_selection,
    build_parser,
    build_train_command,
    parse_screen_variants,
    variant_settings,
)
from survot_rank.research.methods.dct_v382_prognostic_transport_reconstruction.model import (
    BudgetedAdaptiveAuxiliaryWeighter,
    DCTV382PrognosticTransportReconstruction,
)
from survot_rank.training.model_factory import METHOD_ALIASES, get_model, list_methods


def make_args(**overrides):
    values = dict(
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        alpha_surv=0.15,
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
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_geometry_reliability_strength=0.0,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=1.0,
        dct_v38_lambda_direction=0.0,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        dct_v38_direction_margin=0.02,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.05,
        dct_v38_alpha_mid=0.50,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=1,
        dct_v38_ramp_epochs=0,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.05,
        dct_v382_distill_weight=0.50,
        dct_v382_warmup_epochs=0,
        dct_v382_ramp_epochs=0,
        dct_v382_adaptive_aux_weights=False,
        dct_v382_adaptive_prior_fraction=0.25,
        dct_v382_adaptive_temperature=1.0,
        dct_v382_adaptive_kl_strength=0.01,
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
        cur_epoch=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reference():
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return times, censorship


def batch(seed=1):
    generator = torch.Generator().manual_seed(seed)
    times, censorship = reference()
    return {
        "x_wsi": torch.randn(8, 6, 16, generator=generator),
        "x_omics": torch.randn(8, 5, 20, generator=generator),
        "y": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "event_time": times,
        "c": censorship,
    }


def test_mgptr_only_is_finite_backpropagates_and_adds_no_sinkhorn_solve(
    monkeypatch,
):
    torch.manual_seed(7)
    model = DCTV382PrognosticTransportReconstruction(
        make_args(), omic_input_dim=20
    )
    model.configure_train_reference(*reference())
    model.train()

    calls = 0
    original = model._plans_from_cost_tensor

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_plans_from_cost_tensor", counted)
    logits, aux_loss = model(**batch())
    diagnostics = model.last_training_losses

    assert calls == 1
    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert diagnostics["v38_total"] == 0
    assert diagnostics["v382_mgptr_nll"] > 0
    assert diagnostics["v382_mgptr_reconstruction"] >= 0
    assert diagnostics["v382_mgptr_weighted"] > 0
    assert diagnostics["v382_finite"] == 1
    assert torch.allclose(
        aux_loss.detach(),
        (
            model.dct_lambda_ipcw_rank * diagnostics["ipcw_rank"]
            + diagnostics["v382_mgptr_weighted"]
        ),
        atol=1e-6,
    )

    aux_loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert model.stage_pair_cost[-1].weight.grad is not None
    assert model.event_hazard.weight.grad is not None
    assert model._v382_y is None
    assert model._v382_c is None


def test_mgptr_reconstruction_rewards_matching_observed_hazards():
    model = DCTV382PrognosticTransportReconstruction(
        make_args(), omic_input_dim=20
    )
    teacher = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5], [0.5, 0.0, -0.5, -1.0]]
    )
    matching = teacher.unsqueeze(0).repeat(3, 1, 1)
    shifted = matching + 2.0
    y = torch.tensor([1, 3])
    matching_loss = model._observed_horizon_kl(matching, teacher, y)
    shifted_loss = model._observed_horizon_kl(shifted, teacher, y)
    assert matching_loss == pytest.approx(0.0, abs=1e-6)
    assert shifted_loss > matching_loss


def test_zero_mgptr_weight_preserves_the_v38_parameterisation():
    torch.manual_seed(11)
    v382 = DCTV382PrognosticTransportReconstruction(
        make_args(dct_v382_lambda_mgptr=0.0), omic_input_dim=20
    )
    from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
        DCTTransportInterventionConsistency,
    )

    torch.manual_seed(11)
    v38 = DCTTransportInterventionConsistency(make_args(), omic_input_dim=20)
    assert sum(parameter.numel() for parameter in v382.parameters()) == sum(
        parameter.numel() for parameter in v38.parameters()
    )


def test_adaptive_full_learns_five_bounded_auxiliary_weights():
    torch.manual_seed(17)
    args = make_args(
        cur_epoch=0,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.03,
        dct_v38_lambda_reconfiguration=0.02,
        dct_v38_warmup_epochs=1,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.05,
        dct_v382_adaptive_aux_weights=True,
    )
    model = DCTV382PrognosticTransportReconstruction(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    # Establish detached risk anchors before all structural terms activate.
    warmup_logits, warmup_aux = model(**batch(7))
    assert torch.isfinite(warmup_logits).all()
    assert torch.isfinite(warmup_aux)

    args.cur_epoch = 2
    logits, aux_loss = model(**batch(8))
    diagnostics = model.last_training_losses
    names = ("ipcw_rank", "direction", "dose", "reconfiguration", "mgptr")

    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert diagnostics["v382_adaptive_enabled"] == 1
    assert diagnostics["v382_adaptive_active_terms"] == 5
    learned_weights = torch.stack(
        tuple(diagnostics[f"v382_adaptive_weight_{name}"] for name in names)
    )
    assert learned_weights.sum() == pytest.approx(0.25, abs=1e-6)
    assert torch.allclose(
        learned_weights,
        torch.tensor([0.10, 0.05, 0.03, 0.02, 0.05]),
        atol=1e-6,
    )
    assert torch.allclose(
        aux_loss.detach(), diagnostics["v382_adaptive_total"], atol=1e-6
    )

    aux_loss.backward()
    logits_parameter = model.adaptive_auxiliary_weighter.allocation_logits
    assert logits_parameter.grad is not None
    assert torch.isfinite(logits_parameter.grad).all()


def test_adaptive_weight_budget_and_prior_floor_cannot_collapse():
    base = {"ipcw_rank": 0.10, "direction": 0.05, "mgptr": 0.05}
    weighter = BudgetedAdaptiveAuxiliaryWeighter(
        base, prior_fraction=0.25, temperature=1.0, kl_strength=0.01
    )
    with torch.no_grad():
        weighter.allocation_logits.copy_(torch.tensor([20.0, -20.0, -20.0]))
    total, weights, kl = weighter(
        {name: torch.tensor(1.0, requires_grad=True) for name in base}
    )
    assert torch.isfinite(total)
    assert torch.isfinite(kl)
    assert sum(float(value.detach()) for value in weights.values()) == pytest.approx(
        sum(base.values()), abs=1e-6
    )
    for name, initial in base.items():
        assert float(weights[name].detach()) >= 0.25 * initial - 1e-6


def test_v382_registry_aliases_and_defaults():
    method = "dct_v382_prognostic_transport_reconstruction"
    assert method in list_methods()
    assert METHOD_ALIASES["dct_v382"] == method
    assert METHOD_ALIASES["dct_v3_8_2"] == method
    factory_model = get_model("dct_v382", make_args(), omic_input_dim=20)
    assert factory_model.__class__.__name__ == (
        "DCTV382PrognosticTransportReconstruction"
    )
    assert COMMON_OVERRIDES["dct_v382_lambda_mgptr"] == 0.05


def test_v382_launcher_supports_mgptr_and_later_selected_combination():
    assert set(SCREEN_VARIANTS) == {
        "base",
        "mgptr",
        "selected",
        "selected_mgptr",
        "fixed_full",
        "adaptive_full",
    }
    assert parse_screen_variants("base,mgptr") == ["base", "mgptr"]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_screen_variants("unknown")

    mgptr_label, mgptr = variant_settings("mgptr", None)
    assert mgptr_label == "mgptr"
    assert mgptr["dct_v382_lambda_mgptr"] == 0.05
    assert mgptr["dct_v38_lambda_direction"] == 0.0
    assert mgptr["dct_v38_lambda_dose"] == 0.0
    assert mgptr["dct_v38_lambda_reconfiguration"] == 0.0

    selected_label, selected = variant_settings(
        "selected_mgptr", "direction_dose"
    )
    assert selected_label == "selected_direction_dose_mgptr"
    assert selected["dct_v38_lambda_direction"] == 0.05
    assert selected["dct_v38_lambda_dose"] == 0.03
    assert selected["dct_v38_lambda_reconfiguration"] == 0.0
    assert selected["dct_v382_lambda_mgptr"] == 0.05

    adaptive_label, adaptive = variant_settings("adaptive_full", None)
    assert adaptive_label == "adaptive_full"
    assert adaptive["dct_v38_lambda_direction"] == 0.05
    assert adaptive["dct_v38_lambda_dose"] == 0.03
    assert adaptive["dct_v38_lambda_reconfiguration"] == 0.02
    assert adaptive["dct_v382_lambda_mgptr"] == 0.05
    assert adaptive["dct_v382_adaptive_aux_weights"] is True
    assert adaptive["dct_v38_dose_every"] == 1


def test_v382_default_plan_is_the_blca_brca_adaptive_screen():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.mode == "plan"
    assert defaults.cancers == ["blca", "brca"]
    assert defaults.folds == [0]
    assert defaults.protocols == ["robust"]
    assert defaults.variants == ["adaptive_full"]
    assert defaults.max_epochs == 20
    _validate_selection(parser, defaults)

    invalid = parser.parse_args(["--variants", "selected_mgptr"])
    with pytest.raises(SystemExit):
        _validate_selection(parser, invalid)


def test_v382_twenty_epoch_results_are_isolated_and_auditable():
    command, result_dir = build_train_command(
        "python3",
        "blca",
        "robust",
        "selected_mgptr",
        0,
        "0",
        "4",
        "/data1/TCGA-UNI2-h-features",
        selected_v38_variant="direction_reconfiguration",
        max_epochs=20,
    )
    rendered = " ".join(command)
    assert (
        "survot_method=dct_v382_prognostic_transport_reconstruction" in rendered
    )
    assert "dct_v38_lambda_direction=0.05" in rendered
    assert "dct_v38_lambda_dose=0.0" in rendered
    assert "dct_v38_lambda_reconfiguration=0.02" in rendered
    assert "dct_v382_lambda_mgptr=0.05" in rendered
    assert "max_epochs=20" in rendered
    assert result_dir.as_posix() == (
        "results/dct_v3.8.2_20ep/robust/"
        "selected_direction_reconfiguration_mgptr/blca"
    )

    adaptive_command, adaptive_dir = build_train_command(
        "python3",
        "brca",
        "robust",
        "adaptive_full",
        0,
        "0",
        "4",
        "/data1/TCGA-UNI2-h-features",
        max_epochs=20,
    )
    adaptive_rendered = " ".join(adaptive_command)
    assert "dct_v382_adaptive_aux_weights=true" in adaptive_rendered
    assert "dct_v38_dose_every=1" in adaptive_rendered
    assert adaptive_dir.as_posix() == (
        "results/dct_v3.8.2_20ep/robust/adaptive_full/brca"
    )
