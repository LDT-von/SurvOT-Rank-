"""Regression tests for leakage-safe and diagnosable survival evaluation."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sksurv.util import Surv

_SLOTSPE_RUNTIME = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "survot_rank", "research", "legacy", "slotspe_runtime",
)
if _SLOTSPE_RUNTIME not in sys.path:
    sys.path.insert(0, _SLOTSPE_RUNTIME)

from utils.core_utils import (  # noqa: E402
    _calculate_metrics,
    _extract_survival_metadata,
    _record_metric_error,
    _select_valid_metric_time_grid,
)


def test_ipcw_reference_can_be_restricted_to_training_fold():
    factory = SimpleNamespace(
        censorship_var="censorship_dss",
        label_col="survival_months_dss",
        clinical_df=pd.DataFrame({
            "censorship_dss": [0, 1, 0],
            "survival_months_dss": [2.0, 4.0, 99.0],
        }),
    )
    train_labels = factory.clinical_df.iloc[:2]

    survival_train = _extract_survival_metadata(factory, train_labels)

    assert survival_train["time"].tolist() == [2.0, 4.0]
    assert survival_train["event"].tolist() == [True, False]


def test_metric_grid_discards_out_of_followup_columns_without_misaligning_predictions():
    survival_train = Surv.from_arrays(
        event=np.array([True, False, True, False]),
        time=np.array([1.0, 3.0, 5.0, 7.0]),
    )
    validation_times = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    predictions = np.arange(20, dtype=float).reshape(5, 4)

    times, selected = _select_valid_metric_time_grid(
        validation_times,
        survival_train,
        bins=np.array([-np.inf, 1.0, 4.0, 10.0, np.inf]),
        survival_predictions=predictions,
    )

    assert np.allclose(times, [2.0001, 4.0, 5.9999])
    assert np.array_equal(selected, predictions[:, [0, 2, 3]])


def test_metric_error_is_persisted_for_audit(tmp_path):
    output = tmp_path / "metric_diagnostics_fold1.log"

    _record_metric_error("cindex_ipcw", ValueError("test failure"), output)

    text = output.read_text(encoding="utf-8")
    assert "cindex_ipcw unavailable" in text
    assert "ValueError: test failure" in text


def test_metric_calculation_uses_training_reference_and_valid_time_subset(tmp_path):
    train_labels = pd.DataFrame({
        "censorship_dss": [0, 1, 0, 1, 0, 1, 0, 1],
        "survival_months_dss": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    val_labels = pd.DataFrame({
        "censorship_dss": [0, 1, 0, 1, 0],
        "survival_months_dss": [2.0, 3.0, 4.0, 5.0, 6.0],
    })
    factory = SimpleNamespace(
        censorship_var="censorship_dss",
        label_col="survival_months_dss",
        bins=np.array([-np.inf, 2.5, 4.5, 6.5, np.inf]),
        clinical_df=pd.concat([train_labels, val_labels], ignore_index=True),
    )
    loader = SimpleNamespace(dataset=SimpleNamespace(label_df=val_labels))
    survival_train = _extract_survival_metadata(factory, train_labels)

    metrics = _calculate_metrics(
        loader,
        factory,
        survival_train,
        all_risk_scores=np.linspace(-1.0, 1.0, len(val_labels)),
        all_censorships=val_labels["censorship_dss"].to_numpy(),
        all_event_times=val_labels["survival_months_dss"].to_numpy(),
        all_risk_by_bin_scores=np.tile([0.9, 0.8, 0.7, 0.6], (len(val_labels), 1)),
        metric_error_path=tmp_path / "metric_diagnostics_fold0.log",
    )

    assert all(np.isfinite(metric) for metric in metrics[:2])
    assert np.all(np.isfinite(metrics[2]))
    assert np.isfinite(metrics[3])
    assert np.isfinite(metrics[4])
    assert not (tmp_path / "metric_diagnostics_fold0.log").exists()
