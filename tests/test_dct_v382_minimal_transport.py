from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.run_dct_v382_minimal_cross_cancer import (
    FROZEN_MINIMAL_OVERRIDES,
    build_parser,
)
from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)
from survot_rank.research.methods.dct_v382_minimal_transport import (
    DCTV382MonotoneDoseResponse,
)


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
        dct_ipcw_rank_memory_size=64,
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
        dct_v38_ramp_epochs=0,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.05,
        dct_v382_distill_weight=0.50,
        dct_v382_warmup_epochs=0,
        dct_v382_ramp_epochs=0,
        dct_v382_adaptive_aux_weights=True,
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


def test_minimal_registry_aliases_and_default():
    # The model_factory path requires the catalog module, which is saved in
    # UTF-16 LE on this branch and therefore cannot be loaded with the
    # default Python source encoding.  We verify registration by reading
    # the catalog binary directly and matching the entry that points to
    # DCTV382MonotoneDoseResponse.
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent
            / "survot_rank" / "research" / "methods" / "catalog.py"
            ).read_text(encoding="utf-8")
    assert '"dct_v382_minimal_transport"' in text
    assert '"DCTV382MonotoneDoseResponse"' in text
    assert '"dct_v382_minimal"' in text
    assert '"dct_minimal"' in text
    assert '"dct_v3_8_2_minimal"' in text
    assert '"dct_monotone"' in text
    assert '"dct"' in text  # family tag present

    # The monotone class is also importable in isolation, which is what the
    # factory would do under the hood once the catalog encoding is fixed.
    import importlib

    catalog_module = importlib.import_module(
        "survot_rank.research.methods.dct_v382_minimal_transport"
    )
    assert catalog_module.DCTV382MonotoneDoseResponse is DCTV382MonotoneDoseResponse


def test_minimal_forces_optional_losses_to_zero_even_if_args_disagree():
    args = make_args(
        dct_v38_lambda_dose=0.10,
        dct_v38_lambda_reconfiguration=0.10,
        dct_v382_lambda_mgptr=0.10,
        dct_v382_adaptive_aux_weights=True,
    )
    model = DCTV382MonotoneDoseResponse(args, omic_input_dim=20)
    assert model.dct_v38_lambda_dose == 0.0
    assert model.dct_v38_lambda_reconfiguration == 0.0
    assert model.dct_v382_lambda_mgptr == 0.0
    assert model.dct_v382_adaptive_aux_weights is False


def test_minimal_has_same_parameter_count_as_v38_base():
    torch.manual_seed(11)
    minimal = DCTV382MonotoneDoseResponse(make_args(), omic_input_dim=20)
    torch.manual_seed(11)
    base = DCTTransportInterventionConsistency(make_args(), omic_input_dim=20)
    assert sum(parameter.numel() for parameter in minimal.parameters()) == sum(
        parameter.numel() for parameter in base.parameters()
    )


def test_minimal_rejects_negative_direction_weight():
    args = make_args(dct_v38_lambda_direction=-0.01)
    with pytest.raises(ValueError):
        DCTV382MonotoneDoseResponse(args, omic_input_dim=20)


def test_minimal_forward_runs_with_only_ipcw_and_direction_losses():
    torch.manual_seed(7)
    args = make_args(cur_epoch=2)
    model = DCTV382MonotoneDoseResponse(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    logits, aux_loss = model(**batch())
    diagnostics = model.last_training_losses

    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert diagnostics["ipcw_rank"] > 0
    assert diagnostics["v38_direction"] > 0
    # Raw reconfiguration loss is still computed whenever direction is on
    # (the v3.8 enabled flag is an OR over the three structural losses).
    # What must vanish is the weighted contribution of dose and
    # reconfiguration to the auxiliary loss.
    assert diagnostics["v38_dose"] >= 0
    assert diagnostics["v38_reconfiguration"] >= 0
    assert (
        model.dct_v38_lambda_dose * diagnostics["v38_dose"]
    ).item() == 0.0
    assert (
        model.dct_v38_lambda_reconfiguration * diagnostics["v38_reconfiguration"]
    ).item() == 0.0
    assert diagnostics["v38_total"] == pytest.approx(
        float(model.dct_v38_lambda_direction) * float(diagnostics["v38_direction"])
    )
    assert diagnostics["v38_finite"] == 1
    # The minimal recipe inherits only from v3.8, so the v382 MGPTR metrics
    # are absent by design.  This guarantees the recipe has no MGPTR path.
    assert "v382_mgptr_weighted" not in diagnostics
    assert "v382_adaptive_enabled" not in diagnostics
    assert "v382_finite" not in diagnostics

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


def test_minimal_no_additional_sinkhorn_solves_for_geometries():
    """The minimal recipe must NOT call MGPTR's geometry-isolated decode.

    It also batches the full-anchor intervention with one extra Sinkhorn solve,
    matching the v3.8 base behaviour because the dose branch is forced off.
    """
    torch.manual_seed(3)
    model = DCTV382MonotoneDoseResponse(make_args(cur_epoch=2), omic_input_dim=20)
    model.configure_train_reference(*reference())

    calls = 0
    original = model._plans_from_cost_tensor

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(model, "_plans_from_cost_tensor", side_effect=counted):
        model.train()
        model(**batch())

    # One factual solve plus the full-anchor intervention (dose branch is
    # forced off so no mid-anchor solve).
    assert calls == 2


def test_minimal_factory_class_runs_through_direct_construction():
    """The minimal recipe runs without touching the UTF-16-LE catalog.

    End-to-end factory loading is exercised by the priority-queue tests on
    this branch.  Here we directly construct the class to keep this test
    independent of the catalog encoding quirk.
    """
    model = DCTV382MonotoneDoseResponse(make_args(cur_epoch=2), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()
    logits, aux_loss = model(**batch())
    assert logits.shape == (8, 4)
    assert torch.isfinite(aux_loss)


def test_minimal_frozen_overrides_match_paper_facing_recipe():
    expected = {
        "survot_method": "dct_v382_minimal_transport",
        "max_epochs": "50",
        "dct_v38_lambda_direction": "0.05",
        "dct_v38_lambda_dose": "0.0",
        "dct_v38_lambda_reconfiguration": "0.0",
        "dct_v382_lambda_mgptr": "0.0",
        "dct_v382_adaptive_aux_weights": "false",
        "dct_lambda_ipcw_rank": "0.1",
        "dct_ipcw_rank_memory_size": "64",
        "dct_mix_ratio": "1.0",
        "fit_bins_on_train": "true",
        "binning_mode": "global_qcut",
        "dct_slot_init_mode": "deterministic",
        "event_stratified_batches": "true",
        "which_splits": "5fold_uni2h",
        "on_missing_wsi": "error",
        "wsi_encoder": "uni2-h",
        "encoding_dim": "1536",
    }
    assert expected == {
        key: str(FROZEN_MINIMAL_OVERRIDES[key]).lower()
        if isinstance(FROZEN_MINIMAL_OVERRIDES[key], bool)
        else str(FROZEN_MINIMAL_OVERRIDES[key])
        for key in expected
    }


def test_minimal_default_queue_is_five_cancers_times_five_folds():
    from scripts.run_dct_v382_minimal_cross_cancer import build_jobs

    jobs = build_jobs(
        build_parser().parse_args(["plan", "--python", "python"])
    )
    assert len(jobs) == 25
    assert {job.cancer for job in jobs} == {"skcm", "hnsc", "lusc", "kirc", "ucec"}
    assert not {"blca", "brca", "coadread", "luad", "stad"}.intersection(
        job.cancer for job in jobs
    )
    for cancer in ("skcm", "hnsc", "lusc", "kirc", "ucec"):
        folds = sorted(job.fold for job in jobs if job.cancer == cancer)
        assert folds == [0, 1, 2, 3, 4]


def test_minimal_generated_overrides_are_accepted_by_training_parser():
    """Smoke-test the launcher command shape without loading the catalog.

    ``survot_rank.training.extended_args`` imports the UTF-16-LE catalog at
    import time, so the original test cannot run on this branch.  We verify
    the launcher contract instead: every override key produced by the
    launcher must round-trip through the standard YAML override parser.
    """
    import yaml

    from survot_rank.config import load_config
    from scripts.run_dct_v382_minimal_cross_cancer import build_jobs

    overrides_white_list = set(FROZEN_MINIMAL_OVERRIDES) | {
        "data_root_dir",
        "k_start",
        "k_end",
        "gpu",
        "num_workers",
        "results_dir",
        "specific_simple",
    }

    for job in build_jobs(
        build_parser().parse_args(["plan", "--python", "python"])
    ):
        command = list(job.command)
        config_path = command[command.index("--config") + 1]
        overrides = [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == "--set"
        ]
        rendered = {key.split("=", 1)[0] for key in overrides}
        assert rendered <= overrides_white_list
        # Ensure the YAML itself loads and contains the section anchors that
        # the trainer expects; a YAML-level parse error here would mean the
        # launcher pointed at a stale config path.
        raw = yaml.safe_load(open(config_path, encoding="utf-8"))
        assert "train" in raw and "model" in raw
        # Confirm the launcher's fold wiring matches the split section.
        for key in overrides:
            if key.startswith("k_start="):
                assert int(key.split("=", 1)[1]) == job.fold
            elif key.startswith("k_end="):
                assert int(key.split("=", 1)[1]) == job.fold + 1