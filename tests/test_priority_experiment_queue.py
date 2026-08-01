from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts import run_priority_experiment_queue as queue


def _args(*extra: str):
    return queue.build_parser().parse_args(["plan", "--python", "python", *extra])


def _override(job: queue.Job, key: str) -> str | None:
    return queue._override_value(job, key)


def test_formal_queue_matches_requested_order_and_scope():
    jobs = queue.build_jobs(_args())

    assert len(jobs) == 31
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.STAGES)
    assert Counter(job.stage for job in jobs) == {
        "v33_blca_uni5": 5,
        "v38_lusc_screen": 8,
        "v382_blca_fold124": 3,
        "v383_blca_fold124": 3,
        "v39_blca_fold124": 3,
        "v40_blca_fold124": 3,
        "v41_blca_fold124": 3,
        "arcsurv_blca_fold124": 3,
    }

    by_stage = {
        stage: [job for job in jobs if job.stage == stage] for stage in queue.STAGES
    }
    assert [job.fold for job in by_stage["v33_blca_uni5"]] == [0, 1, 2, 3, 4]
    assert {job.fold for job in by_stage["v38_lusc_screen"]} == {0}
    for stage in queue.STAGES[2:]:
        assert [job.fold for job in by_stage[stage]] == [1, 2, 4]

    assert all(_override(job, "on_missing_wsi") == "error" for job in jobs)
    assert all(_override(job, "max_epochs") == "50" for job in by_stage["v33_blca_uni5"])
    assert all(_override(job, "max_epochs") == "50" for job in by_stage["v38_lusc_screen"])
    for stage in queue.STAGES[2:]:
        assert all(_override(job, "max_epochs") == "30" for job in by_stage[stage])


def test_encoder_protocols_and_v40_recipe_are_explicit():
    jobs = queue.build_jobs(_args())
    by_stage = {
        stage: [job for job in jobs if job.stage == stage] for stage in queue.STAGES
    }

    for stage in ("v33_blca_uni5", "v41_blca_fold124", "arcsurv_blca_fold124"):
        assert {job.encoder for job in by_stage[stage]} == {"uni"}
        assert {_override(job, "encoding_dim") for job in by_stage[stage]} == {"1024"}
    for stage in queue.STAGES[1:6]:
        assert {job.encoder for job in by_stage[stage]} == {"uni2-h"}
        assert {job.which_splits for job in by_stage[stage]} == {"5fold_uni2h"}

    v40 = by_stage["v40_blca_fold124"][0]
    assert _override(v40, "fit_bins_on_train") == "true"
    assert _override(v40, "ist_stability_strength") == "0.1"
    assert _override(v40, "ist_lambda_plan") == "0.05"
    assert _override(v40, "ist_lambda_attribution") == "0.05"

    v41 = by_stage["v41_blca_fold124"]
    assert [job.fold for job in v41] == [1, 2, 4]


def test_smoke_uses_first_fold_and_one_epoch_per_stage():
    jobs = queue.build_jobs(_args(), smoke=True)

    assert len(jobs) == 15
    assert Counter(job.stage for job in jobs)["v38_lusc_screen"] == 8
    assert all(
        count == 1
        for stage, count in Counter(job.stage for job in jobs).items()
        if stage != "v38_lusc_screen"
    )
    assert all(_override(job, "max_epochs") == "1" for job in jobs)
    assert all(_override(job, "max_smoke_batches") == "2" for job in jobs)


def test_stage_resume_filter_and_nested_completion(tmp_path: Path):
    args = _args("--from-stage", "v39_blca_fold124")
    jobs = queue.build_jobs(args)
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.STAGES[4:])

    job = jobs[0]
    result_dir = tmp_path / "result"
    nested = result_dir / "experiment" / f"split_{job.fold}_results_final.pkl"
    nested.parent.mkdir(parents=True)
    nested.touch()
    replaced = queue.Job(
        stage=job.stage,
        label=job.label,
        fold=job.fold,
        command=job.command,
        result_dir=result_dir,
        config=job.config,
        encoder=job.encoder,
        cancer=job.cancer,
        which_splits=job.which_splits,
    )
    assert queue._completion(replaced) == nested


def test_all_generated_overrides_are_accepted_by_training_parser():
    from survot_rank.config import apply_overrides, config_to_argv, load_config
    from survot_rank.training.extended_args import process_args_extended

    for job in queue.build_jobs(_args()):
        command = list(job.command)
        config_path = command[command.index("--config") + 1]
        overrides = [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == "--set"
        ]
        config = apply_overrides(load_config(queue.REPO_ROOT / config_path), overrides)
        parsed = process_args_extended(config_to_argv(config))
        assert parsed.k_start == job.fold
        assert parsed.k_end == job.fold + 1
