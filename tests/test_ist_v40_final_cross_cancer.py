from __future__ import annotations

from scripts import run_ist_v40_final_cross_cancer as final


def _args(*extra: str):
    return final.build_parser().parse_args(["plan", "--python", "python", *extra])


def _override(job: final.Job, key: str) -> str | None:
    prefix = f"{key}="
    for index, item in enumerate(job.command[:-1]):
        if item == "--set" and job.command[index + 1].startswith(prefix):
            return job.command[index + 1][len(prefix) :]
    return None


def test_default_queue_covers_six_current_complete_cancers_and_five_folds():
    jobs = final.build_jobs(_args())
    assert len(jobs) == 30
    assert list(dict.fromkeys(job.cancer for job in jobs)) == list(final.DEFAULT_CANCERS)
    for cancer in final.DEFAULT_CANCERS:
        assert [job.fold for job in jobs if job.cancer == cancer] == list(range(5))
    assert not {"brca", "coadread", "luad", "stad"}.intersection(
        job.cancer for job in jobs
    )


def test_final_ist_recipe_keeps_cost_feedback_and_drops_unhelpful_aux_losses():
    expected = {
        "survot_method": "intervention_stable_survival_transport",
        "max_epochs": "50",
        "ist_warmup_epochs": "5",
        "ist_ramp_epochs": "10",
        "ist_stability_strength": "0.1",
        "ist_stability_normalization": "raw_mass",
        "ist_feedback_mode": "legacy_product",
        "ist_lambda_plan": "0.0",
        "ist_lambda_attribution": "0.0",
        "ist_lambda_risk": "0.0",
        "fit_bins_on_train": "true",
        "binning_mode": "global_qcut",
        "which_splits": "5fold_uni2h",
        "on_missing_wsi": "error",
        "wsi_encoder": "uni2-h",
        "encoding_dim": "1536",
    }
    for job in final.build_jobs(_args()):
        assert {key: _override(job, key) for key in expected} == expected
        assert _override(job, "study") == job.cancer


def test_all_ten_cancers_can_be_requested_but_incomplete_ones_are_not_default():
    jobs = final.build_jobs(_args("--cancers", "all", "--folds", "0"))
    assert [job.cancer for job in jobs] == list(final.SUPPORTED_CANCERS)
    assert len(jobs) == 10


def test_blca_reuses_existing_ablation_b_result_directory_and_identity():
    job = final.build_jobs(_args("--cancers", "blca", "--folds", "0"))[0]
    assert job.result_dir.as_posix() == (
        "results/ist_surv_v4.0_staged_50ep/clean/abl_b_cost_only/blca"
    )
    assert _override(job, "specific_simple") == "ist_v40_abl_b_cost_only_blca_50ep"


def test_final_ist_queue_shares_gpu_lock_with_other_experiment_queues():
    from scripts import run_priority_experiment_queue as priority

    assert final.scheduler_lock_path("0", False) == priority._scheduler_lock_path("0", False)


def test_generated_overrides_are_accepted_and_select_the_correct_study():
    from survot_rank.config import apply_overrides, config_to_argv, load_config
    from survot_rank.training.extended_args import process_args_extended

    for job in final.build_jobs(_args()):
        command = list(job.command)
        config_path = command[command.index("--config") + 1]
        overrides = [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == "--set"
        ]
        config = apply_overrides(load_config(final.REPO_ROOT / config_path), overrides)
        parsed = process_args_extended(config_to_argv(config))
        assert parsed.study == job.cancer
        assert parsed.k_start == job.fold
        assert parsed.k_end == job.fold + 1


def test_repaired_recipe_is_opt_in_and_writes_to_an_isolated_directory():
    job = final.build_jobs(
        _args(
            "--recipe",
            "repaired",
            "--cancers",
            "blca",
            "--folds",
            "1",
        )
    )[0]
    assert job.result_dir.as_posix() == (
        "results/ist_surv_v4.0_repaired_50ep/clean/"
        "importance_instability/blca"
    )
    assert _override(job, "ist_stability_normalization") == "independence_lift"
    assert _override(job, "ist_feedback_mode") == (
        "importance_weighted_instability"
    )
    assert _override(job, "specific_simple") == (
        "ist_v40_repaired_importance_instability_blca_50ep"
    )


def test_repair_gate_defaults_to_blca_folds_1_2_4():
    from scripts import run_ist_v40_repair_gate as gate

    args = gate.build_parser().parse_args(["plan", "--python", "python"])
    jobs = final.build_jobs(args)
    assert [(job.cancer, job.fold) for job in jobs] == [
        ("blca", 1),
        ("blca", 2),
        ("blca", 4),
    ]
