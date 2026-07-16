import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_dct_v3_score_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("dct_diagnostics_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _write_curve(path: Path, cindices: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc"])
        writer.writeheader()
        for epoch, cindex in enumerate(cindices):
            writer.writerow({"epoch": epoch, "val_cindex": cindex, "val_cindex_ipcw": 0.5, "val_IBS": 0.2, "val_iauc": 0.6})


def test_summarize_curve_uses_best_and_stability_windows():
    rows = [{"epoch": epoch, "val_cindex": score, "val_cindex_ipcw": 0.5, "val_IBS": 0.2, "val_iauc": 0.6} for epoch, score in enumerate([0.60, 0.70, 0.65, 0.62, 0.61, 0.60])]
    result = MODULE.summarize_curve(rows)
    assert result["best_epoch"] == 1
    assert result["best_val_cindex"] == 0.70
    assert round(result["best3_val_cindex"], 4) == round((0.60 + 0.70 + 0.65) / 3, 4)
    assert round(result["last5_val_cindex"], 4) == round((0.70 + 0.65 + 0.62 + 0.61 + 0.60) / 5, 4)
    assert result["best_near_end"] is False


def test_collect_rows_marks_missing_expected_fold(tmp_path):
    _write_curve(tmp_path / "full" / "nested" / "epoch_curve_fold0.csv", [0.60, 0.61, 0.62])
    rows = MODULE.collect_rows(tmp_path, [0, 2])
    assert len(rows) == 2
    assert rows[0]["variant"] == "full"
    assert rows[0]["fold"] == 0
    assert rows[0]["status"] == "ok"
    assert rows[1] == {"variant": "full", "fold": 2, "status": "missing"}
