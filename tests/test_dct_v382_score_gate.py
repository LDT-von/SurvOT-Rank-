from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import run_dct_v382_score_gate as gate
from scripts import summarize_dct_v382_score_gate as summary


def _args(*extra: str):
    return gate.build_parser().parse_args(["plan", "--python", "python", *extra])


def _override(job: gate.Job, key: str) -> str | None:
    prefix = f"{key}="
    for index, item in enumerate(job.command[:-1]):
        if item == "--set" and job.command[index + 1].startswith(prefix):
            return job.command[index + 1][len(prefix) :]
    return None


def test_default_is_four_single_variable_versions_on_three_cancers_and_folds():
    jobs = gate.build_jobs(_args())
    assert len(jobs) == 4 * 3 * 3
    assert list(dict.fromkeys(job.variant for job in jobs)) == list(gate.PHASE1_VARIANTS)
    assert {job.cancer for job in jobs} == {"blca", "kirc", "skcm"}
    assert {job.fold for job in jobs} == {1, 2, 4}
    assert all(len(gate.VARIANTS[name].overrides) == 1 for name in gate.PHASE1_VARIANTS)


def test_registered_version_changes_only_declared_overrides():
    for job in gate.build_jobs(_args("--variants", "all")):
        changed = gate.VARIANTS[job.variant].overrides
        for key, value in gate.base.FINAL_OVERRIDES.items():
            expected = changed.get(key, value)
            rendered = str(expected).lower() if isinstance(expected, bool) else str(expected)
            assert _override(job, key) == rendered
        for key, value in changed.items():
            assert _override(job, key) == str(value).lower()


def test_plan_explains_use_supported_claim_and_limit(capsys):
    jobs = gate.build_jobs(_args("--variants", "patches4096", "--folds", "1", "--cancers", "skcm"))
    gate.print_plan(jobs)
    output = capsys.readouterr().out
    assert "USE:" in output
    assert "CAN SUPPORT:" in output
    assert "CANNOT SUPPORT:" in output
    assert "input-information limited" in output


def _frame(values: dict[str, list[float]], variant: str) -> pd.DataFrame:
    rows = []
    for cancer, scores in values.items():
        for fold, score in zip((1, 2, 4), scores):
            rows.append(
                {
                    "variant": variant,
                    "cancer": cancer,
                    "fold": fold,
                    "status": "ok",
                    "best_cindex": score,
                    "last5_cindex": score - 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_promotion_requires_macro_gain_skcm_gain_and_no_cancer_regression():
    control = _frame({"blca": [.70] * 3, "kirc": [.80] * 3, "skcm": [.66] * 3}, "control")
    passing = _frame({"blca": [.702] * 3, "kirc": [.806] * 3, "skcm": [.670] * 3}, "candidate")
    decision = summary.promotion(passing, control)
    assert decision["promote"] is True

    regresses_blca = _frame({"blca": [.690] * 3, "kirc": [.820] * 3, "skcm": [.680] * 3}, "candidate")
    assert summary.promotion(regresses_blca, control)["promote"] is False


def test_fold_record_reads_best_epoch_and_stability(tmp_path: Path):
    cancer_dir = tmp_path / "skcm"
    cancer_dir.mkdir()
    pd.DataFrame(
        {"epoch": [0, 1, 2, 3, 4], "val_cindex": [.60, .64, .63, .62, .61]}
    ).to_csv(cancer_dir / "epoch_curve_fold1.csv", index=False)
    row = summary.fold_record(tmp_path, "candidate", "skcm", 1)
    assert row["status"] == "ok"
    assert row["best_epoch"] == 1
    assert row["best_cindex"] == 0.64
    assert row["last5_cindex"] == 0.62
