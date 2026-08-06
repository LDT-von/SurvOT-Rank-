from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts import run_priority_experiment_queue as queue


def _args(*extra: str):
    return queue.build_parser().parse_args(["plan", "--python", "python", *extra])


def _override(job: queue.Job, key: str) -> str | None:
    return queue._override_value(job, key)


def test_default_queue_is_the_baseline_plus_single_variable_ablations():
    """默认队列 = v3.8.2 自适应对照 + v4.0 机制三档 + A 组 clean 底座。

    ArcSurv 与 v4.1 需要先修原型使用塌缩与补全损失下界，因此不入默认队列；
    MGPTR 单项、v3.8.3、v3.9 已判定停止。
    """
    jobs = queue.build_jobs(_args())

    assert len(jobs) == 21
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.DEFAULT_STAGES)
    assert Counter(job.stage for job in jobs) == {
        "v382_fixed_full": 3,
        "v382_adaptive_full": 3,
        "v40_abl_a_factual": 3,
        "v40_abl_b_cost_only": 3,
        "v40_staged_rerun": 3,
        "v33_clean_baseline": 3,
        "v33_clean_no_ipcw_rank": 3,
    }
    assert all(_override(job, "max_epochs") == "50" for job in jobs)
    assert all(job.fold in (1, 2, 4) for job in jobs)


def test_next_queue_is_four_task_gate_with_isolated_repair_outputs():
    jobs = queue.build_jobs(_args("--stages", "next"))

    assert len(jobs) == 4
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.NEXT_STAGES)
    assert Counter(job.stage for job in jobs) == {
        "v382_fixed_full_fold03": 2,
        "arcsurv_repaired_gate": 1,
        "v41_repaired_gate": 1,
    }

    fixed = [job for job in jobs if job.stage == "v382_fixed_full_fold03"]
    assert [job.fold for job in fixed] == [0, 3]
    assert {job.result_dir.as_posix() for job in fixed} == {
        "results/dct_v3.8.2/robust/fixed_full/blca"
    }
    assert all(_override(job, "dct_v382_adaptive_aux_weights") == "false" for job in fixed)

    arc = next(job for job in jobs if job.stage == "arcsurv_repaired_gate")
    assert arc.fold == 1
    assert "repaired_50ep" in arc.result_dir.as_posix()
    assert _override(arc, "arc_distance_reduction") == "scaled"
    assert _override(arc, "arc_anchor_logit") == "6.0"
    assert _override(arc, "arc_lambda_sharpness") == "0.02"

    v41 = next(job for job in jobs if job.stage == "v41_repaired_gate")
    assert v41.fold == 2
    assert "repaired_50ep" in v41.result_dir.as_posix()
    assert _override(v41, "v41_min_log_variance") == "-4.0"

    assert all(_override(job, "max_epochs") == "50" for job in jobs)
    assert all(_override(job, "fit_bins_on_train") == "true" for job in jobs)


def test_v382_adaptive_control_isolates_the_adaptive_weight_flag():
    """此前 base/mgptr 两次都设 adaptive=False，因此测不出自适应权重。"""
    jobs = queue.build_jobs(_args("--stages", "v382_fixed_full,v382_adaptive_full"))
    fixed = [job for job in jobs if job.stage == "v382_fixed_full"]
    adaptive = [job for job in jobs if job.stage == "v382_adaptive_full"]
    identity_keys = ("specific_simple=", "results_dir=")

    def training_overrides(job: queue.Job) -> set[str]:
        return {
            job.command[index + 1]
            for index, item in enumerate(job.command[:-1])
            if item == "--set"
            and not job.command[index + 1].startswith(identity_keys)
        }

    for left, right in zip(fixed, adaptive):
        assert left.fold == right.fold
        assert training_overrides(right) - training_overrides(left) == {
            "dct_v382_adaptive_aux_weights=true"
        }
        assert training_overrides(left) - training_overrides(right) == {
            "dct_v382_adaptive_aux_weights=false"
        }

    # 两者都是 full：v3.8 三个干预损失启用，MGPTR 权重相同。
    for job in fixed + adaptive:
        assert _override(job, "dct_v382_lambda_mgptr") == "0.05"
        assert _override(job, "dct_v38_lambda_direction") == "0.05"
        assert _override(job, "dct_v38_lambda_dose") == "0.03"
        assert _override(job, "dct_v38_lambda_reconfiguration") == "0.02"
        assert _override(job, "fit_bins_on_train") == "true"
        assert _override(job, "max_epochs") == "50"
    assert fixed[0].result_dir != adaptive[0].result_dir


def test_mgptr_control_pair_isolates_the_weight():
    jobs = queue.build_jobs(_args("--stages", "b0_mgptr_control,b1_mgptr"))

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
        for job in queue.build_jobs(_args("--stages", "v33_blca_legacy_repro"))
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

    assert len(jobs) == 82
    assert list(dict.fromkeys(job.stage for job in jobs)) == list(queue.STAGES)
    assert Counter(job.stage for job in jobs) == {
        "v382_fixed_full_fold03": 2,
        "arcsurv_repaired_gate": 1,
        "v41_repaired_gate": 1,
        "b0_mgptr_control": 3,
        "b1_mgptr": 3,
        "v33_blca_legacy_repro": 5,
        "v33_clean_baseline": 3,
        "v33_clean_no_ipcw_rank": 3,
        "v40_staged_rerun": 3,
        "v41_staged_rerun": 3,
        "arcsurv_staged_rerun": 3,
        "v382_fixed_full": 3,
        "v382_adaptive_full": 3,
        "v40_abl_a_factual": 3,
        "v40_abl_b_cost_only": 3,
        "v41_no_modality_dropout": 3,
        "v41_no_ipcw_rank": 3,
        "v42_act_surv": 3,
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
    jobs = queue.build_jobs(
        _args(
            "--stages",
            "v40_staged_rerun,v41_staged_rerun,arcsurv_staged_rerun",
        )
    )
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


def test_v33_clean_baseline_makes_the_binning_protocol_traceable():
    """冻结 YAML 未设 fit_bins_on_train（默认 False），clean 必须显式覆盖。

    否则跑出来的是 leaky 分箱，而 A 组基线是所有后续判定的参照。
    """
    jobs = queue.build_jobs(
        _args("--stages", "v33_clean_baseline,v33_clean_no_ipcw_rank")
    )
    baseline = [job for job in jobs if job.stage == "v33_clean_baseline"]
    no_rank = [job for job in jobs if job.stage == "v33_clean_no_ipcw_rank"]

    for job in baseline + no_rank:
        assert _override(job, "fit_bins_on_train") == "true"
        assert _override(job, "binning_mode") == "global_qcut"
        assert _override(job, "max_epochs") == "50"
        assert job.encoder == "uni"
        assert job.which_splits == "5fold"

    # 唯一变量 = IPCW pairwise rank 权重。
    assert _override(baseline[0], "dct_lambda_ipcw_rank") is None
    assert _override(no_rank[0], "dct_lambda_ipcw_rank") == "0.0"
    assert baseline[0].result_dir != no_rank[0].result_dir


def test_v40_three_way_ablation_separates_cost_feedback_from_aux_loss():
    """A/B/C 三档：B-A = cost 回写净效果，C-B = 辅助损失净效果。

    不设「关回写但留辅助损失」这一档：实测 plan≈5e-5、attribution≈1e-15、
    risk 权重为 0，辅助损失本身几乎不工作，那一档等价于 A。
    """
    stages = ("v40_abl_a_factual", "v40_abl_b_cost_only", "v40_staged_rerun")
    jobs = queue.build_jobs(_args("--stages", ",".join(stages)))
    by_stage = {
        stage: [job for job in jobs if job.stage == stage] for stage in stages
    }

    expected = {
        # (strength, plan, attribution, risk)
        "v40_abl_a_factual": ("0.0", "0.0", "0.0", "0.0"),
        "v40_abl_b_cost_only": ("0.1", "0.0", "0.0", "0.0"),
        "v40_staged_rerun": ("0.1", "0.05", "0.05", "0.0"),
    }
    for stage, (strength, plan, attribution, risk) in expected.items():
        for job in by_stage[stage]:
            assert _override(job, "ist_stability_strength") == strength
            assert _override(job, "ist_lambda_plan") == plan
            assert _override(job, "ist_lambda_attribution") == attribution
            assert _override(job, "ist_lambda_risk") == risk
            assert _override(job, "ist_warmup_epochs") == "5"
            assert _override(job, "ist_ramp_epochs") == "10"
            assert _override(job, "fit_bins_on_train") == "true"
            assert _override(job, "max_epochs") == "50"

    # 三档的结果目录必须互不相同。
    assert len({by_stage[stage][0].result_dir for stage in stages}) == 3


def test_single_variable_ablations_change_exactly_one_key():
    stages = (
        "v41_staged_rerun",
        "v41_no_modality_dropout",
        "v41_no_ipcw_rank",
    )
    jobs = queue.build_jobs(_args("--stages", ",".join(stages)))
    by_stage = {
        stage: [job for job in jobs if job.stage == stage] for stage in stages
    }
    identity_keys = ("specific_simple=", "results_dir=")

    def training_overrides(job: queue.Job) -> set[str]:
        return {
            job.command[index + 1]
            for index, item in enumerate(job.command[:-1])
            if item == "--set"
            and not job.command[index + 1].startswith(identity_keys)
        }

    def sole_difference(baseline_stage: str, variant_stage: str) -> set[str]:
        differences = set()
        for base, variant in zip(by_stage[baseline_stage], by_stage[variant_stage]):
            assert base.fold == variant.fold
            differences |= training_overrides(variant) - training_overrides(base)
        return differences

    # v4.1：只关掉人为删模态。
    assert sole_difference("v41_staged_rerun", "v41_no_modality_dropout") == {
        "v41_modality_dropout=0.0"
    }
    # v4.1：只关掉继承的 pairwise IPCW rank。
    assert sole_difference("v41_staged_rerun", "v41_no_ipcw_rank") == {
        "dct_lambda_ipcw_rank=0.0"
    }

    # 每个消融的结果目录都必须与其基线隔离。
    for variant_stage in ("v41_no_modality_dropout", "v41_no_ipcw_rank"):
        assert (
            by_stage["v41_staged_rerun"][0].result_dir
            != by_stage[variant_stage][0].result_dir
        )


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


def test_v42_is_implemented_but_kept_out_of_the_default_queue():
    """v4.2 的第二卖点依赖 archetype 真的分化开，须等 ArcSurv 重跑确认后再启动。"""
    assert "v42_act_surv" in queue.STAGES
    assert "v42_act_surv" not in queue.DEFAULT_STAGES

    default_stages = {job.stage for job in queue.build_jobs(_args())}
    assert "v42_act_surv" not in default_stages

    jobs = [
        job
        for job in queue.build_jobs(_args("--stages", "v42_act_surv"))
        if job.stage == "v42_act_surv"
    ]
    assert [job.fold for job in jobs] == [1, 2, 4]
    for job in jobs:
        assert _override(job, "survot_method") == "archetypal_transport_composition"
        assert _override(job, "max_epochs") == "50"
        assert job.encoder == "uni"
        assert job.which_splits == "5fold"
