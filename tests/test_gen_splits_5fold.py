from pathlib import Path

import pandas as pd

from tools.gen_splits_5fold import audit_existing_splits, gen


def test_gen_preserves_full_unique_cohort_without_train_val_overlap(tmp_path: Path):
    study = "test"
    data_path = tmp_path / "dataset_csv"
    clinical_dir = data_path / "clinical" / "all"
    output_dir = data_path / "splits" / "5fold"
    clinical_dir.mkdir(parents=True)

    rows = [
        {
            "case id": f"TCGA-AA-{index:04d}",
            "survival_months_dss": float(index + 1),
            "censorship_dss": index % 2,
        }
        for index in range(40)
    ]
    # Duplicate clinical rows must not create patient leakage across folds.
    rows.append(dict(rows[3]))
    pd.DataFrame(rows).to_csv(clinical_dir / f"{study}.csv")

    gen(
        study=study,
        data_path=str(data_path),
        label_col="survival_months_dss",
        censor_col="censorship_dss",
        n_folds=5,
        seed=42,
        out_dir=str(output_dir),
    )

    expected = {row["case id"] for row in rows}
    validation_counts = {case_id: 0 for case_id in expected}

    for fold in range(5):
        split = pd.read_csv(output_dir / study / f"fold_{fold}.csv")
        assert list(split.columns) == ["train", "val"]

        train = set(split["train"].dropna())
        val = set(split["val"].dropna())
        assert len(train) == 32
        assert len(val) == 8
        assert train.isdisjoint(val)
        assert train | val == expected

        for case_id in val:
            validation_counts[case_id] += 1

    assert set(validation_counts.values()) == {1}

    audit = audit_existing_splits(
        study=study,
        data_path=str(data_path),
        label_col="survival_months_dss",
        censor_col="censorship_dss",
        n_folds=5,
        split_root=str(output_dir),
    )
    assert audit["ok"] is True
    assert audit["eligible_cases"] == 40
    assert max(audit["validation_event_counts"]) - min(
        audit["validation_event_counts"]
    ) <= 1


def test_gen_balances_time_within_event_status_for_sparse_events(tmp_path: Path):
    study = "sparse"
    data_path = tmp_path / "dataset_csv"
    clinical_dir = data_path / "clinical" / "all"
    output_dir = data_path / "splits" / "5fold"
    clinical_dir.mkdir(parents=True)

    rows = []
    for index in range(100):
        is_event = index < 10
        rows.append(
            {
                "case id": f"TCGA-BB-{index:04d}",
                "survival_months_dss": float(index + 1),
                "censorship_dss": 0 if is_event else 1,
            }
        )
    pd.DataFrame(rows).to_csv(clinical_dir / f"{study}.csv")

    gen(
        study=study,
        data_path=str(data_path),
        label_col="survival_months_dss",
        censor_col="censorship_dss",
        n_folds=5,
        seed=42,
        out_dir=str(output_dir),
    )
    audit = audit_existing_splits(
        study=study,
        data_path=str(data_path),
        label_col="survival_months_dss",
        censor_col="censorship_dss",
        n_folds=5,
        split_root=str(output_dir),
    )
    assert audit["ok"] is True
    assert audit["validation_event_counts"] == [2, 2, 2, 2, 2]
