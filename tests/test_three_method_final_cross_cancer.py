from scripts import run_three_method_final_cross_cancer as final


def args(*extra):
    return final.build_parser().parse_args(["plan", "--python", "python", *extra])


def override(job, key):
    prefix = f"{key}="
    for index, item in enumerate(job.command[:-1]):
        if item == "--set" and job.command[index + 1].startswith(prefix):
            return job.command[index + 1][len(prefix) :]
    return None


def test_default_queue_is_three_final_methods_six_cancers_five_folds():
    jobs = final.build_jobs(args())
    assert len(jobs) == 90
    assert list(dict.fromkeys(job.method for job in jobs)) == list(final.METHOD_ORDER)
    assert list(dict.fromkeys(job.cancer for job in jobs)) == list(final.DEFAULT_CANCERS)


def test_final_recipes_freeze_the_repaired_mechanisms():
    jobs = final.build_jobs(args("--cancers", "blca", "--folds", "0"))
    by_method = {job.method: job for job in jobs}
    assert override(by_method["capsa_final"], "capsa_lambda_identity") == "0.02"
    assert override(by_method["capsa_final"], "capsa_target_active_ratio") == "0.25"
    assert override(by_method["arcsurv_final"], "arc_freeze_state_encoder") == "1"
    assert override(by_method["arcsurv_final"], "arc_seed_anchors") == "0"
    assert override(by_method["catet_final"], "catet_lambda_stage") == "0.04"
    assert override(by_method["catet_final"], "catet_intervention_cost") == "1.0"
    for job in jobs:
        assert override(job, "which_splits") == "5fold_uni2h"
        assert override(job, "on_missing_wsi") == "error"
        assert override(job, "encoding_dim") == "1536"
        assert override(job, "study") == "blca"


def test_every_generated_command_is_accepted_by_the_training_parser():
    from survot_rank.config import apply_overrides, config_to_argv, load_config
    from survot_rank.training.extended_args import process_args_extended

    jobs = final.build_jobs(args("--cancers", "blca", "--folds", "0"))
    for job in jobs:
        command = list(job.command)
        config_path = command[command.index("--config") + 1]
        overrides = [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == "--set"
        ]
        config = apply_overrides(load_config(final.REPO_ROOT / config_path), overrides)
        parsed = process_args_extended(config_to_argv(config))
        assert parsed.k_start == 0
        assert parsed.k_end == 1
        assert parsed.study == "blca"


def test_smoke_runs_only_first_selected_fold_in_isolated_directories():
    jobs = final.build_jobs(
        args("--methods", "capsa_final,catet_final", "--cancers", "blca", "--folds", "2,4"),
        smoke=True,
    )
    assert [(job.method, job.fold) for job in jobs] == [
        ("capsa_final", 2),
        ("catet_final", 2),
    ]
    assert all("three_method_final_smoke" in job.result_dir.as_posix() for job in jobs)
