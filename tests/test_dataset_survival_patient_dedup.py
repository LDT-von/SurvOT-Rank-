from pathlib import Path

import pandas as pd
import pytest

from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
    SurvivalDatasetFactory,
)


def _factory_for_clinical_csv(data_path: Path) -> SurvivalDatasetFactory:
    factory = SurvivalDatasetFactory.__new__(SurvivalDatasetFactory)
    factory.data_path = str(data_path)
    factory.study = "test"
    factory.label_col = "survival_months_dss"
    factory.censorship_var = "censorship_dss"
    factory.use_clinical_modality = False
    factory.clinical_feature_cols = None
    factory.binning_mode = "legacy_equal_width"
    factory.n_bins = 4
    return factory


def _write_clinical_csv(data_path: Path, rows: list[dict]) -> None:
    clinical_dir = data_path / "clinical" / "all"
    clinical_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(clinical_dir / "test.csv", index=False)


def test_clinical_loader_deduplicates_identical_patient_rows(tmp_path: Path):
    rows = [
        {
            "case id": f"TCGA-AA-{index:04d}",
            "survival_months_dss": float(index + 1),
            "censorship_dss": index % 2,
            "wsi": f"slide-{index}.svs",
        }
        for index in range(8)
    ]
    rows.append(dict(rows[3]))
    _write_clinical_csv(tmp_path, rows)

    factory = _factory_for_clinical_csv(tmp_path)
    factory._setup_clinical_data()

    assert len(factory.clinical_df) == 8
    assert factory.clinical_df["case id"].is_unique


def test_clinical_loader_rejects_conflicting_duplicate_patient_rows(tmp_path: Path):
    rows = [
        {
            "case id": f"TCGA-AA-{index:04d}",
            "survival_months_dss": float(index + 1),
            "censorship_dss": index % 2,
            "wsi": f"slide-{index}.svs",
        }
        for index in range(8)
    ]
    conflicting = dict(rows[3])
    conflicting["survival_months_dss"] = 99.0
    rows.append(conflicting)
    _write_clinical_csv(tmp_path, rows)

    factory = _factory_for_clinical_csv(tmp_path)
    with pytest.raises(ValueError, match="Conflicting clinical rows"):
        factory._setup_clinical_data()
