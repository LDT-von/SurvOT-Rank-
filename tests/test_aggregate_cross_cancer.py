"""tools/aggregate_cross_cancer.py 的单元测试。

覆盖需求文档（requirements.md）需求 2 的 AC3-AC7：
- AC3/AC4: 汇总数据列固定为 val_cindex/val_cindex_ipcw/val_IBS/val_iauc，
  通过读取 mean/std 索引行获取并输出各指标均值与标准差。
- AC5: CSV 与 Markdown 两种格式均包含各癌种原始指标值以及 mean/std 汇总行。
- AC6: 缺少 mean 行（或目录/文件缺失）的癌种被标注为缺失/无效，不中断其余
  癌种的处理。
- AC7: 输出行严格按字母序排列（blca, brca, coadread, hnsc, stad）。

使用 pytest tmp_path fixture 构造合成的结果目录树，不依赖真实训练结果，也不
写入仓库内任何文件。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.aggregate_cross_cancer import main


def _write_summary_csv(path: Path, mean_values: dict, std_values: dict) -> None:
    """写出一个"合法"的 summary.csv：包含 fold 0-4 行 + mean 行 + std 行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc"]
    rows = {}
    for fold in range(5):
        rows[fold] = {col: mean_values[col] for col in columns}
    rows["mean"] = {col: mean_values[col] for col in columns}
    rows["std"] = {col: std_values[col] for col in columns}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.to_csv(path)


def _write_summary_csv_without_mean(path: Path) -> None:
    """写出一个"格式无效"的 summary.csv：只有 fold 行，没有 mean/std 行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc"]
    rows = {fold: {col: 0.5 for col in columns} for fold in range(5)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.to_csv(path)


@pytest.fixture()
def synthetic_results_root(tmp_path: Path) -> Path:
    """构造合成的跨癌种结果目录树：

    - blca: 合法 summary.csv（mean/std 齐全）
    - coadread: 合法 summary.csv，嵌套在更深的子目录里（测试 rglob）
    - hnsc: summary.csv 存在但缺 mean/std 行 -> invalid
    - brca: 目录存在但内部没有任何 summary.csv -> missing
    - stad: 目录完全不存在 -> missing
    """
    root = tmp_path

    _write_summary_csv(
        root / "blca" / "run1" / "summary.csv",
        mean_values={"val_cindex": 0.70, "val_cindex_ipcw": 0.68, "val_IBS": 0.30, "val_iauc": 0.65},
        std_values={"val_cindex": 0.02, "val_cindex_ipcw": 0.02, "val_IBS": 0.01, "val_iauc": 0.02},
    )
    _write_summary_csv(
        root / "coadread" / "nested" / "dir" / "summary.csv",
        mean_values={"val_cindex": 0.72, "val_cindex_ipcw": 0.70, "val_IBS": 0.28, "val_iauc": 0.67},
        std_values={"val_cindex": 0.03, "val_cindex_ipcw": 0.03, "val_IBS": 0.02, "val_iauc": 0.02},
    )
    _write_summary_csv_without_mean(root / "hnsc" / "run1" / "summary.csv")
    (root / "brca" / "run1").mkdir(parents=True, exist_ok=True)
    # stad: 不创建任何目录

    return root


def _read_csv_rows(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


class TestCrossCancerAggregation:
    def test_output_files_created(self, synthetic_results_root: Path) -> None:
        exit_code = main(["--results-root", str(synthetic_results_root)])
        assert exit_code == 0

        csv_path = synthetic_results_root / "cross_cancer_summary.csv"
        md_path = synthetic_results_root / "cross_cancer_summary.md"
        assert csv_path.exists()
        assert md_path.exists()

    def test_csv_has_fixed_metric_columns(self, synthetic_results_root: Path) -> None:
        # 需求2 AC3: 汇总数据列固定为四项指标
        main(["--results-root", str(synthetic_results_root)])
        csv_path = synthetic_results_root / "cross_cancer_summary.csv"
        df = pd.read_csv(csv_path)
        for col in ["val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc"]:
            assert col in df.columns
            assert f"{col}_std" in df.columns
        assert "study" in df.columns
        assert "status" in df.columns

    def test_ok_studies_have_correct_values(self, synthetic_results_root: Path) -> None:
        # 需求2 AC4: 通过读取 mean/std 索引行获取并输出各指标均值与标准差
        main(["--results-root", str(synthetic_results_root)])
        rows = _read_csv_rows(synthetic_results_root / "cross_cancer_summary.csv")

        blca_row = next(r for r in rows if r["study"] == "blca")
        assert blca_row["status"] == "ok"
        assert blca_row["val_cindex"] == pytest.approx(0.70)
        assert blca_row["val_cindex_ipcw"] == pytest.approx(0.68)
        assert blca_row["val_IBS"] == pytest.approx(0.30)
        assert blca_row["val_iauc"] == pytest.approx(0.65)
        assert blca_row["val_cindex_std"] == pytest.approx(0.02)

        coadread_row = next(r for r in rows if r["study"] == "coadread")
        assert coadread_row["status"] == "ok"
        assert coadread_row["val_cindex"] == pytest.approx(0.72)

    def test_missing_and_invalid_studies_flagged_without_interrupting(
        self, synthetic_results_root: Path
    ) -> None:
        # 需求2 AC6: 缺 mean 行 / 目录缺失 / summary.csv 缺失均应被标注，
        # 且不能中断其余 study 的处理（blca/coadread 仍应正常出现在结果中）。
        main(["--results-root", str(synthetic_results_root)])
        rows = _read_csv_rows(synthetic_results_root / "cross_cancer_summary.csv")

        hnsc_row = next(r for r in rows if r["study"] == "hnsc")
        assert hnsc_row["status"] == "invalid"
        assert hnsc_row["reason"] == "missing_mean_row"

        brca_row = next(r for r in rows if r["study"] == "brca")
        assert brca_row["status"] == "missing"
        assert brca_row["reason"] == "summary_csv_not_found"

        stad_row = next(r for r in rows if r["study"] == "stad")
        assert stad_row["status"] == "missing"
        assert stad_row["reason"] == "directory_not_found"

        # 其余 study 的处理没有被中断：blca/coadread 仍然是 status=ok
        blca_row = next(r for r in rows if r["study"] == "blca")
        coadread_row = next(r for r in rows if r["study"] == "coadread")
        assert blca_row["status"] == "ok"
        assert coadread_row["status"] == "ok"

    def test_per_study_row_order_is_alphabetical(self, synthetic_results_root: Path) -> None:
        # 需求2 AC7: 输出行严格按字母序排列
        main(["--results-root", str(synthetic_results_root)])
        rows = _read_csv_rows(synthetic_results_root / "cross_cancer_summary.csv")

        per_study_rows = [r for r in rows if r["status"] in ("ok", "invalid", "missing")]
        studies_in_order = [r["study"] for r in per_study_rows]
        assert studies_in_order == ["blca", "brca", "coadread", "hnsc", "stad"]

    def test_aggregate_mean_std_rows_present_and_computed_from_ok_only(
        self, synthetic_results_root: Path
    ) -> None:
        # 需求2 AC4/AC5: CSV 中应包含 mean/std 汇总行，且只统计 status=ok 的
        # 癌种（blca, coadread）。
        main(["--results-root", str(synthetic_results_root)])
        rows = _read_csv_rows(synthetic_results_root / "cross_cancer_summary.csv")

        mean_row = next(r for r in rows if r["study"] == "mean")
        std_row = next(r for r in rows if r["study"] == "std")

        expected_mean_cindex = (0.70 + 0.72) / 2
        assert mean_row["val_cindex"] == pytest.approx(expected_mean_cindex)

        # 样本标准差（ddof=1），两个样本 0.70 与 0.72。
        import statistics

        expected_std_cindex = statistics.stdev([0.70, 0.72])
        assert std_row["val_cindex"] == pytest.approx(expected_std_cindex)

        # mean/std 汇总行应出现在所有 per-study 行之后。
        study_order = [r["study"] for r in rows]
        assert study_order.index("mean") > study_order.index("stad")
        assert study_order.index("std") > study_order.index("mean")

    def test_markdown_reflects_status_distinctions(self, synthetic_results_root: Path) -> None:
        # 需求2 AC5: Markdown 输出也应包含各癌种原始指标值与 mean/std 汇总行，
        # 并标注缺失/无效癌种。
        main(["--results-root", str(synthetic_results_root)])
        md_text = (synthetic_results_root / "cross_cancer_summary.md").read_text(encoding="utf-8")

        assert "val_cindex" in md_text
        assert "val_cindex_ipcw" in md_text
        assert "val_IBS" in md_text
        assert "val_iauc" in md_text

        # 每个 study 都应出现在表格中，且非 ok 的行能看到对应状态标注。
        for study in ["blca", "brca", "coadread", "hnsc", "stad"]:
            assert f"| {study} |" in md_text
        assert "| ok |" in md_text
        assert "| invalid |" in md_text
        assert "| missing |" in md_text

        # mean/std 汇总行也应出现。
        assert "| mean |" in md_text
        assert "| std |" in md_text

    def test_only_writes_under_results_root(self, tmp_path: Path, synthetic_results_root: Path) -> None:
        # 回归保护：确认输出文件都落在 results_root 下，没有写到别处。
        before = set(p for p in synthetic_results_root.rglob("*") if p.is_file())
        main(["--results-root", str(synthetic_results_root)])
        after = set(p for p in synthetic_results_root.rglob("*") if p.is_file())
        new_files = after - before
        assert new_files == {
            synthetic_results_root / "cross_cancer_summary.csv",
            synthetic_results_root / "cross_cancer_summary.md",
        }
