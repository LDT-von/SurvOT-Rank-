from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import run_dct_v310_experiments as experiments
from scripts import run_dct_v310_final_cross_cancer as final
from survot_rank.research.methods.catalog import METHOD_ALIASES, PRIMARY_METHOD
from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)
from survot_rank.research.methods.dct_v310_directional_regularized_transport import (
    DCTV310DirectionalRegularizedTransport,
)
from survot_rank.training.model_factory import get_model
from survot_rank.training.train_runner import compose_batch_objective, init_loss_function


def make_args(**overrides):
    values = dict(
        bag_loss="nll_surv",
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
        dct_lambda_ipcw_rank=0.99,
        dct_ipcw_rank_margin=0.99,
        dct_ipcw_rank_temperature=0.99,
        dct_ipcw_max_weight=99.0,
        dct_ipcw_rank_memory_size=1,
        dct_lambda_etar=0.99,
        dct_lambda_listwise=0.99,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.99,
        dct_evidence_mass_floor=0.99,
        dct_evidence_marginal_strength=0.0,
        dct_geometry_reliability_strength=0.99,
        dct_geometry_reliability_temperature=0.25,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.25,
        dct_v38_lambda_direction=0.99,
        dct_v38_lambda_dose=0.99,
        dct_v38_lambda_reconfiguration=0.99,
        dct_v38_direction_margin=0.99,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.99,
        dct_v38_alpha_mid=0.25,
        dct_v38_alpha_full=0.75,
        dct_v38_warmup_epochs=5,
        dct_v38_ramp_epochs=10,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.99,
        dct_v382_adaptive_aux_weights=True,
        dct_fixed_coupling=True,
        dct_random_anchors=True,
        dct_perm_labels_seed=7,
        dct_stage_jitter_fraction=0.3,
        dct_freeze_source_prototype="",
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        dct_slot_init_mode="gaussian",
        dct_slot_eval_seed=91,
        cur_epoch=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reference():
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return times, censorship


def batch(size=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    times, censorship = reference()
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return {
        "x_wsi": torch.randn(size, 6, 16, generator=generator),
        "x_omics": torch.randn(size, 5, 20, generator=generator),
        "y": y[:size],
        "event_time": times[:size],
        "c": censorship[:size],
    }


def override(job, key):
    command = list(job.command)
    prefix = f"{key}="
    for index, item in enumerate(command[:-1]):
        if item == "--set" and command[index + 1].startswith(prefix):
            return command[index + 1][len(prefix):]
    return None


def test_v310_is_primary_registered_method_and_factory_alias():
    key = "dct_v310_directional_regularized_transport"
    assert PRIMARY_METHOD == key
    assert METHOD_ALIASES["dct_v310"] == key
    assert METHOD_ALIASES["dct_reg"] == key
    model = get_model("dct_reg", make_args(), omic_input_dim=20)
    # The factory intentionally loads model.py under an isolated module name,
    # so class identity differs from the package import even though the public
    # implementation is the same class.
    assert type(model).__name__ == "DCTV310DirectionalRegularizedTransport"


def test_v310_hostile_overrides_cannot_change_frozen_recipe():
    args = make_args()
    model = DCTV310DirectionalRegularizedTransport(args, omic_input_dim=20)

    assert model.objective_weights() == {
        "nll": 1.0,
        "ipcw_rank": 0.10,
        "direction": 0.05,
    }
    assert model.dct_lambda_ipcw_rank == 0.10
    assert model.dct_v38_lambda_direction == 0.05
    assert model.dct_lambda_etar == 0.0
    assert model.dct_lambda_listwise == 0.0
    assert model.dct_v38_lambda_dose == 0.0
    assert model.dct_v38_lambda_reconfiguration == 0.0
    assert model.dct_v382_lambda_mgptr == 0.0
    assert model.dct_v382_adaptive_aux_weights is False
    assert model.dct_v38_warmup_epochs == 0
    assert model.dct_v38_ramp_epochs == 0
    assert model.dct_fixed_coupling is False
    assert model.dct_random_anchors is False
    assert model.dct_perm_labels_seed == 0
    assert model.dct_stage_jitter_fraction == 0.0
    assert model.dct_evidence_cost_weight == 0.0
    assert model.dct_geometry_reliability_strength == 0.0
    assert model.dct_mix_ratio == 1.0
    assert args.dct_slot_init_mode == "deterministic"


def test_v310_rejects_non_nll_primary_loss():
    with pytest.raises(ValueError, match="bag_loss='nll_surv'"):
        DCTV310DirectionalRegularizedTransport(
            make_args(bag_loss="cox_surv"), omic_input_dim=20
        )


def test_v310_forward_auxiliary_loss_is_exact_two_term_sum():
    torch.manual_seed(7)
    model = DCTV310DirectionalRegularizedTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    logits, aux_loss = model(**batch())
    diagnostics = model.last_training_losses
    expected = 0.10 * diagnostics["ipcw_rank"] + 0.05 * diagnostics["v38_direction"]

    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert diagnostics["ipcw_rank"] > 0
    assert diagnostics["v38_direction"] > 0
    assert float(aux_loss.detach()) == pytest.approx(
        float(expected), rel=1e-6, abs=1e-7
    )
    assert diagnostics["v38_total"] == pytest.approx(
        0.05 * float(diagnostics["v38_direction"]), rel=1e-6
    )
    assert model.dct_v38_lambda_dose * diagnostics["v38_dose"] == 0
    assert model.dct_v38_lambda_reconfiguration * diagnostics["v38_reconfiguration"] == 0
    assert "v382_mgptr_weighted" not in diagnostics


def test_v310_shared_trainer_objective_is_exact_paper_formula():
    torch.manual_seed(19)
    args = make_args()
    model = DCTV310DirectionalRegularizedTransport(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    payload = batch()
    logits, auxiliary_loss = model(**payload)
    loss_fn = init_loss_function(args)
    raw_nll = loss_fn(
        logits,
        payload["y"],
        payload["event_time"],
        payload["c"],
    )
    total = compose_batch_objective(raw_nll, auxiliary_loss, payload["y"].shape[0])
    diagnostics = model.last_training_losses
    expected = (
        raw_nll / payload["y"].shape[0]
        + 0.10 * diagnostics["ipcw_rank"]
        + 0.05 * diagnostics["v38_direction"]
    )

    torch.testing.assert_close(total, expected)


def test_shared_trainer_objective_rejects_empty_batch():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        compose_batch_objective(torch.tensor(1.0), torch.tensor(2.0), 0)


def test_v310_final_launcher_is_six_cancers_by_five_folds_and_exact_objective():
    args = final.build_parser().parse_args(["plan", "--python", "python"])
    jobs = final.build_jobs(args)

    assert len(jobs) == 30
    assert {job.cancer for job in jobs} == {
        "blca", "skcm", "hnsc", "lusc", "kirc", "ucec"
    }
    for job in jobs:
        assert job.config.as_posix() == (
            "configs/dct_v310_directional_regularized_transport.yaml"
        )
        assert override(job, "study") == job.cancer
        assert override(job, "survot_method") == (
            "dct_v310_directional_regularized_transport"
        )
        assert override(job, "bag_loss") == "nll_surv"
        assert override(job, "dct_lambda_ipcw_rank") == "0.1"
        assert override(job, "dct_v38_lambda_direction") == "0.05"
        assert override(job, "dct_v38_lambda_dose") == "0.0"
        assert override(job, "dct_v38_lambda_reconfiguration") == "0.0"
        assert override(job, "dct_v382_lambda_mgptr") == "0.0"


def test_v310_default_experiment_queue_is_matched_two_by_two_ablation():
    args = experiments.build_parser().parse_args(["plan", "--python", "python"])
    jobs = experiments.build_jobs(args)
    assert len(jobs) == 20
    assert {job.variant for job in jobs} == set(experiments.DEFAULT_VARIANTS)
    assert {job.cancer for job in jobs} == {"blca"}

    expected = {
        "nll_only": ("0.0", "0.0"),
        "ipcw_only": ("0.1", "0.0"),
        "direction_only": ("0.0", "0.05"),
        "full": ("0.1", "0.05"),
    }
    for job in jobs:
        assert (
            override(job, "dct_lambda_ipcw_rank"),
            override(job, "dct_v38_lambda_direction"),
        ) == expected[job.variant]
        if job.variant == "full":
            assert override(job, "survot_method") == (
                "dct_v310_directional_regularized_transport"
            )
        else:
            assert override(job, "survot_method") == experiments.PARENT_METHOD


def test_v310_mechanism_control_queue_matches_documented_subset():
    args = experiments.build_parser().parse_args(
        [
            "plan",
            "--python",
            "python",
            "--cancers",
            "blca,ucec,lusc",
            "--folds",
            "1,2,4",
            "--variants",
            (
                "fixed_coupling,noisy_batch_mean_anchors,"
                "permuted_reference"
            ),
        ]
    )
    jobs = experiments.build_jobs(args)

    assert len(jobs) == 27
    assert {job.cancer for job in jobs} == {"blca", "ucec", "lusc"}
    assert {job.fold for job in jobs} == {1, 2, 4}
    assert {job.variant for job in jobs} == {
        "fixed_coupling",
        "noisy_batch_mean_anchors",
        "permuted_reference",
    }


def test_fixed_coupling_replays_current_batch_for_two_batch_sizes():
    args = make_args(
        dct_lambda_ipcw_rank=0.10,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        dct_fixed_coupling=True,
        dct_random_anchors=False,
        dct_perm_labels_seed=0,
        dct_stage_jitter_fraction=0.0,
        dct_v38_warmup_epochs=0,
        dct_v38_ramp_epochs=0,
        dct_slot_init_mode="deterministic",
    )
    model = DCTTransportInterventionConsistency(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    logits_large, loss_large = model(**batch(size=8, seed=3))
    logits_small, loss_small = model(**batch(size=3, seed=4))

    assert logits_large.shape == (8, 4)
    assert logits_small.shape == (3, 4)
    assert torch.isfinite(loss_large)
    assert torch.isfinite(loss_small)
    assert model._factual_plan_cache[0][0].shape[0] == 3


def test_fixed_coupling_keeps_current_factual_sinkhorn_unchanged():
    common = dict(
        dct_lambda_ipcw_rank=0.10,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        dct_random_anchors=False,
        dct_perm_labels_seed=0,
        dct_stage_jitter_fraction=0.0,
        dct_v38_warmup_epochs=0,
        dct_v38_ramp_epochs=0,
        dct_slot_init_mode="deterministic",
    )
    torch.manual_seed(13)
    fresh = DCTTransportInterventionConsistency(
        make_args(dct_fixed_coupling=False, **common), omic_input_dim=20
    )
    fixed = DCTTransportInterventionConsistency(
        make_args(dct_fixed_coupling=True, **common), omic_input_dim=20
    )
    fixed.load_state_dict(fresh.state_dict())
    fresh.configure_train_reference(*reference())
    fixed.configure_train_reference(*reference())
    fresh.eval()
    fixed.eval()

    payload = batch(seed=17)
    fresh(**payload)
    fixed(**payload)

    torch.testing.assert_close(
        fixed.last_explanations["stage_slot_pair_evidence"],
        fresh.last_explanations["stage_slot_pair_evidence"],
    )
