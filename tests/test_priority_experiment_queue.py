from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts import run_priority_experiment_queue as queue


def _args(*extra: str):
    return queue.build_parser().parse_args(["plan", "--python", "python", *extra])


def _override(job: queue.Job, key: str) -> str | None:
    return queue._override_value(job, key)


def test_default_queue_is_the_mgptr_decision_plus_repro_and_staged_reruns():
    jobs = queue.build_jobs(_args())

    assert len(jobs) == 20
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.DEFAULT_STAGES)
    assert Counter(job.stage for job in jobs) == {
        "b0_mgptr_control": 3,
        "b1_mgptr": 3,
        "v33_blca_legacy_repro": 5,
        "v40_staged_rerun": 3,
        "v41_staged_rerun": 3,
        "arcsurv_staged_rerun": 3,
    }

    b0 = [job for job in jobs if job.stage == "b0_mgptr_control"]
    b1 = [job for job in jobs if job.stage == "b1_mgptr"]
    assert [job.fold for job in b0] == [1, 2, 4]
    assert [job.fold for job in b1] == [1, 2, 4]

    # B0 与 B1 的唯一训练差异必须是 MGPTR 权重，否则差值无法归因。
    # 运行标识 (specific_simple) 与结果目录必须不同，不算训练差异。
    identity_keys = ("specific_simple=", "results_dir=")

    def training_overrides(job: queue.Job) -> set[str]:
        return {
            job.command[index + 1]
            for index, item in enumerate(job.command[:-1])
            if item == "--set"
            and not job.command[index + 1].startswith(identity_keys)
        }

    for left, right in zip(b0, b1):
        only_b0 = training_overrides(left) - training_overrides(right)
        only_b1 = training_overrides(right) - training_overrides(left)
        assert only_b0 == {"dct_v382_lambda_mgptr=0.0"}
        assert only_b1 == {"dct_v382_lambda_mgptr=0.05"}
        assert _override(left, "specific_simple") != _override(right, "specific_simple")

    # 三个 v3.8 干预损失全部关闭，50ep，robust 协议。
    for job in b0 + b1:
        assert _override(job, "dct_v38_lambda_direction") == "0.0"
        assert _override(job, "dct_v38_lambda_dose") == "0.0"
        assert _override(job, "dct_v38_lambda_reconfiguration") == "0.0"
        assert _override(job, "dct_v382_adaptive_aux_weights") == "false"
        assert _override(job, "dct_v382_warmup_epochs") == "5"
        assert _override(job, "dct_v382_ramp_epochs") == "10"
        assert _override(job, "fit_bins_on_train") == "true"
        assert _override(job, "max_epochs") == "50"
        assert job.which_splits == "5fold_uni2h"

    # B0 与 B1 结果目录必须分开。
    assert len({job.result_dir for job in b0} | {job.result_dir for job in b1}) == 2


def test_legacy_repro_uses_recovered_split_and_original_protocol():
    jobs = [
        job
        for job in queue.build_jobs(_args())
        if job.stage == "v33_blca_legacy_repro"
    ]

    assert [job.fold for job in jobs] == [0, 1, 2, 3, 4]
    for job in jobs:
        assert job.which_splits == "5fold_legacy"
        assert job.encoder == "uni"
        assert _override(job, "encoding_dim") == "1024"
        assert _override(job, "max_epochs") == "50"
        # 复现历史分数必须沿用当时的分箱协议，即不设 fit_bins_on_train。
        assert _override(job, "fit_bins_on_train") is None
        assert "dct_v3.3_score_first_blca_legacy_repro" in job.result_dir.as_posix()


def test_full_queue_still_covers_every_historical_stage():
    jobs = queue.build_jobs(_args("--stages", "all"))

    assert len(jobs) == 51
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.STAGES)
    assert Counter(job.stage for job in jobs) == {
        "b0_mgptr_control": 3,
        "b1_mgptr": 3,
        "v33_blca_legacy_repro": 5,
        "v40_staged_rerun": 3,
        "v41_staged_rerun": 3,
        "arcsurv_staged_rerun": 3,
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
    historical = queue.STAGES[queue.STAGES.index("v382_blca_fold124"):]
    for stage in historical:
        assert [job.fold for job in by_stage[stage]] == [1, 2, 4]

    assert all(_override(job, "on_missing_wsi") == "error" for job in jobs)
    assert all(_override(job, "max_epochs") == "50" for job in by_stage["v33_blca_uni5"])
    # LUSC 八变体筛选按记录使用 20ep。
    assert all(_override(job, "max_epochs") == "20" for job in by_stage["v38_lusc_screen"])
    for stage in historical:
        assert all(_override(job, "max_epochs") == "30" for job in by_stage[stage])


def test_encoder_protocols_and_v40_recipe_are_explicit():
    jobs = queue.build_jobs(_args("--stages", "all"))
    by_stage = {
        stage: [job for job in jobs if job.stage == stage] for stage in queue.STAGES
    }

    uni_stages = (
        "v33_blca_legacy_repro",
        "v33_blca_uni5",
        "v41_blca_fold124",
        "arcsurv_blca_fold124",
    )
    for stage in uni_stages:
        assert {job.encoder for job in by_stage[stage]} == {"uni"}
        assert {_override(job, "encoding_dim") for job in by_stage[stage]} == {"1024"}
    uni2h_stages = (
        "b0_mgptr_control",
        "b1_mgptr",
        "v38_lusc_screen",
        "v382_blca_fold124",
        "v383_blca_fold124",
        "v39_blca_fold124",
        "v40_blca_fold124",
    )
    for stage in uni2h_stages:
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
    jobs = queue.build_jobs(_args("--stages", "all"), smoke=True)

    assert Counter(job.stage for job in jobs)["v38_lusc_screen"] == 8
    assert all(
        count == 1
        for stage, count in Counter(job.stage for job in jobs).items()
        if stage != "v38_lusc_screen"
    )
    assert all(_override(job, "max_epochs") == "1" for job in jobs)
    assert all(_override(job, "max_smoke_batches") == "2" for job in jobs)


def test_default_smoke_covers_one_fold_per_default_stage():
    jobs = queue.build_jobs(_args(), smoke=True)

    assert Counter(job.stage for job in jobs) == {
        stage: 1 for stage in queue.DEFAULT_STAGES
    }


def test_staged_reruns_enable_delayed_activation_at_fifty_epochs():
    """三个重跑阶段的共同改动：辅助约束不再从第一轮生效，且统一 50ep。"""
    jobs = queue.build_jobs(_args())
    staged = {
        "v40_staged_rerun": ("ist_warmup_epochs", "ist_ramp_epochs"),
        "v41_staged_rerun": ("v41_warmup_epochs", "v41_ramp_epochs"),
        "arcsurv_staged_rerun": ("arc_warmup_epochs", "arc_ramp_epochs"),
    }
    for stage, (warmup_key, ramp_key) in staged.items():
        stage_jobs = [job for job in jobs if job.stage == stage]
        assert [job.fold for job in stage_jobs] == [1, 2, 4]
        for job in stage_jobs:
            assert _override(job, warmup_key) == "5"
            assert _override(job, ramp_key) == "10"
            assert _override(job, "max_epochs") == "50"
            # 结果目录必须与修复前的旧结果隔离。
            assert "staged" in job.result_dir.as_posix()

    # 具体针对性修复：v4.1 降低模态删除强度，ArcSurv 提高 batch 以获得排序信号。
    v41 = next(job for job in jobs if job.stage == "v41_staged_rerun")
    assert _override(v41, "v41_modality_dropout") == "0.2"
    arc = next(job for job in jobs if job.stage == "arcsurv_staged_rerun")
    assert _override(arc, "batch_size") == "8"


def test_stage_resume_filter_and_nested_completion(tmp_path: Path):
    args = _args("--stages", "all", "--from-stage", "v39_blca_fold124")
    jobs = queue.build_jobs(args)
    expected = queue.STAGES[queue.STAGES.index("v39_blca_fold124"):]
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(expected)

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

    for job in queue.build_jobs(_args("--stages", "all")):
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


def test_only_v33_legacy_stage_uses_the_recovered_split():
    """旧划分有已知缺陷，必须严格限定在 v3.3 复现这一个阶段。"""
    jobs = queue.build_jobs(_args("--stages", "all"))

    legacy_stages = {job.stage for job in jobs if job.which_splits == "5fold_legacy"}
    assert legacy_stages == {"v33_blca_legacy_repro"}

    for job in jobs:
        if job.stage != "v33_blca_legacy_repro":
            assert job.which_splits in {"5fold", "5fold_uni2h"}
        assert job.cancer in {"blca", "lusc"}
