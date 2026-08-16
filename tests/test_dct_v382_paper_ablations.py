from __future__ import annotations

from scripts import run_dct_v382_paper_ablations as ablations


def _override(job: ablations.Job, key: str) -> str | None:
    prefix = f"{key}="
    for index, item in enumerate(job.command[:-1]):
        if item == "--set" and job.command[index + 1].startswith(prefix):
            return job.command[index + 1][len(prefix) :]
    return None


def _args(*extra: str):
    return ablations.build_parser().parse_args(["plan", "--python", "python", *extra])


def test_default_queue_is_five_single_variable_ablations_on_three_blca_folds():
    jobs = ablations.build_jobs(_args())
    assert len(jobs) == 15
    assert list(dict.fromkeys(job.variant for job in jobs)) == list(
        ablations.DEFAULT_ABLATIONS
    )
    assert {(job.cancer, job.fold) for job in jobs} == {
        ("blca", 1),
        ("blca", 2),
        ("blca", 4),
    }


def test_each_ablation_changes_only_its_registered_term_from_frozen_full():
    frozen = ablations.base.FINAL_OVERRIDES
    for job in ablations.build_jobs(_args()):
        changed = ablations.ABLATIONS[job.variant]
        for key, value in frozen.items():
            expected = changed.get(key, value)
            rendered = str(expected).lower() if isinstance(expected, bool) else str(expected)
            assert _override(job, key) == rendered
        assert job.result_dir.as_posix().endswith(f"/{job.variant}/blca")


def test_ablation_selection_can_be_narrowed_without_changing_defaults():
    jobs = ablations.build_jobs(_args("--ablations", "no_mgptr,no_ipcw_rank"))
    assert [job.variant for job in jobs] == ["no_mgptr"] * 3 + ["no_ipcw_rank"] * 3
