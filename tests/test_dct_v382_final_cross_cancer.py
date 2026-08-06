from __future__ import annotations

from scripts import run_dct_v382_final_cross_cancer as final


def _args(*extra: str):
    return final.build_parser().parse_args(["plan", "--python", "python", *extra])


def _override(job: final.Job, key: str) -> str | None:
    prefix = f"{key}="
    for index, item in enumerate(job.command[:-1]):
        if item == "--set" and job.command[index + 1].startswith(prefix):
            return job.command[index + 1][len(prefix) :]
    return None


def test_default_queue_is_one_frozen_method_on_five_complete_cancers():
    jobs = final.build_jobs(_args())

    assert len(jobs) == 25
    assert list(dict.fromkeys(job.cancer for job in jobs)) == list(final.DEFAULT_CANCERS)
    assert all([job.fold for job in jobs if job.cancer == cancer] == list(range(5)) for cancer in final.DEFAULT_CANCERS)
    assert not {"brca", "coadread", "luad", "stad", "blca"}.intersection(
        job.cancer for job in jobs
    )


def test_final_recipe_is_fixed_non_adaptive_and_keeps_ipcw():
    jobs = final.build_jobs(_args())
    expected = {
        "survot_method": "dct_v382_prognostic_transport_reconstruction",
        "max_epochs": "50",
        "dct_v382_warmup_epochs": "5",
        "dct_v382_ramp_epochs": "10",
        "dct_v382_lambda_mgptr": "0.05",
        "dct_v382_distill_weight": "0.5",
        "dct_v382_adaptive_aux_weights": "false",
        "dct_v38_lambda_direction": "0.05",
        "dct_v38_lambda_dose": "0.03",
        "dct_v38_lambda_reconfiguration": "0.02",
        "dct_lambda_ipcw_rank": "0.1",
        "dct_ipcw_rank_memory_size": "64",
        "fit_bins_on_train": "true",
        "binning_mode": "global_qcut",
        "dct_slot_init_mode": "deterministic",
        "event_stratified_batches": "true",
        "which_splits": "5fold_uni2h",
        "on_missing_wsi": "error",
        "wsi_encoder": "uni2-h",
        "encoding_dim": "1536",
    }
    for job in jobs:
        assert {key: _override(job, key) for key in expected} == expected


def test_all_cancers_are_selectable_but_incomplete_ones_are_not_defaults():
    args = _args("--cancers", "blca,brca,coadread,luad,stad")
    jobs = final.build_jobs(args)
    assert {job.cancer for job in jobs} == {"blca", "brca", "coadread", "luad", "stad"}
    assert len(jobs) == 25


def test_results_are_isolated_by_cancer_and_reuse_blca_fixed_full_identity():
    args = _args("--cancers", "blca,skcm", "--folds", "0")
    jobs = final.build_jobs(args)
    assert [job.result_dir.as_posix() for job in jobs] == [
        "results/dct_v3.8.2/robust/fixed_full/blca",
        "results/dct_v3.8.2/robust/fixed_full/skcm",
    ]
    assert _override(jobs[0], "specific_simple") == "dct_v382_robust_fixed_full_blca_50ep"
    assert _override(jobs[1], "specific_simple") == "dct_v382_robust_fixed_full_skcm_50ep"


def test_final_queue_shares_scheduler_and_task_locks_with_priority_queue():
    from scripts import run_priority_experiment_queue as priority

    job = final.build_jobs(_args("--cancers", "blca", "--folds", "0"))[0]
    assert final.scheduler_lock_path("0", False) == priority._scheduler_lock_path("0", False)
    assert final.task_lock_path(job) == job.result_dir / ".split_0.priority_queue.lock"


def test_all_generated_overrides_are_accepted_by_training_parser():
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
        assert parsed.k_start == job.fold
        assert parsed.k_end == job.fold + 1
