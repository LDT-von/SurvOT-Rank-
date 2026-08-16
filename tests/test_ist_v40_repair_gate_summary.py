from __future__ import annotations

import csv

from scripts import summarize_ist_v40_repair_gate as summary


def _curve(tmp_path, fold: int, best: float):
    path = tmp_path / "nested" / f"epoch_curve_fold{fold}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "val_cindex",
                "val_cindex_ipcw",
                "val_IBS",
                "val_iauc",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "val_cindex": best,
                "val_cindex_ipcw": best,
                "val_IBS": 0.2,
                "val_iauc": best,
            }
        )
    return path


def test_gate_passes_only_with_mean_gain_and_two_improved_folds(tmp_path):
    curves = {
        1: _curve(tmp_path, 1, 0.7000),
        2: _curve(tmp_path, 2, 0.6800),
        4: _curve(tmp_path, 4, 0.7600),
    }
    report = summary.evaluate_gate(curves)
    assert report["passed"] is True
    assert report["improved_folds"] == 2


def test_gate_stops_when_only_the_mean_condition_passes(tmp_path):
    curves = {
        1: _curve(tmp_path, 1, 0.6800),
        2: _curve(tmp_path, 2, 0.6500),
        4: _curve(tmp_path, 4, 0.8200),
    }
    report = summary.evaluate_gate(curves)
    assert report["repaired_mean"] >= report["threshold"]
    assert report["improved_folds"] == 1
    assert report["passed"] is False


def test_gate_is_incomplete_until_all_three_curves_exist(tmp_path):
    curves = {
        1: _curve(tmp_path, 1, 0.8000),
        2: _curve(tmp_path, 2, 0.8000),
    }
    report = summary.evaluate_gate(curves)
    assert report["missing"] == [4]
    assert report["passed"] is False
