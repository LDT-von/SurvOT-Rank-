#!/usr/bin/env python3
"""Extract per-fold best validation metrics from SurvOT-Rank batch training logs.

For every <exp>.log in a batch_runs/<timestamp>/ directory this script:

  1. Splits the file at every "[Fold N] start" marker, so each fold's val lines
     end up in their own bucket.
  2. Parses every line of the form
        [Epoch K] val cindex=<c> ipcw=<i> IBS=<b> iauc=<a>
     and keeps the best epoch per fold by val cindex (and a second best by ipcw).
  3. Computes 5-fold mean +/- std for cindex, IPCW, IBS, iauc across the folds
     that have completed so far.
  4. Writes everything to BEST_RESULTS.md (markdown table) and
     best_results.csv (machine-readable).

It is safe to re-run while the batch is still going: any in-progress fold
contributes whatever val lines are present up to that point.

Usage:
    python scripts/extract_best_results.py [--run-dir PATH] [--out PATH]

Default --run-dir is the most recent results/batch_runs/<timestamp>/ subdir.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Regex matches: [Epoch 29] val cindex=0.6277 ipcw=0.5811 IBS=0.2318 iauc=0.7592
VAL_RE = re.compile(
    r"\[Epoch\s+(\d+)\]\s+val\s+cindex=([-\d.]+)"
    r"\s+ipcw=([-\d.]+)\s+IBS=([-\d.]+)\s+iauc=([-\d.]+)"
)
FOLD_START_RE = re.compile(r"\[Fold\s+(\d+)\]\s+start")


@dataclass
class FoldMetrics:
    """Per-fold best (by val cindex) and final-epoch snapshots.

    `best_*` capture the epoch that maximised val cindex within the fold,
    which is what reviewers typically want for headline numbers. `final_*`
    capture the last epoch actually emitted (==29 once the fold finishes,
    <29 while still running), which is the model's state at the time of the
    last checkpoint and is useful for spotting whether early-best came from
    noise.
    """
    fold: int
    best_epoch: int = -1
    best_cindex: float = float("-inf")
    best_ipcw: float = float("-inf")
    best_ibs: float = float("nan")
    best_iauc: float = float("nan")
    final_epoch: int = -1
    final_cindex: float = float("nan")
    final_ipcw: float = float("nan")
    final_ibs: float = float("nan")
    final_iauc: float = float("nan")

    @property
    def is_complete(self) -> bool:
        return self.final_epoch >= 29


@dataclass
class ExpReport:
    name: str
    log_path: Path
    folds: dict[int, FoldMetrics] = field(default_factory=dict)
    completed_folds: list[int] = field(default_factory=list)
    error: str | None = None

    @property
    def best_cindex(self) -> FoldMetrics | None:
        if not self.folds:
            return None
        # Only consider folds that actually saw a best cindex (>=0).
        valid = [m for m in self.folds.values() if m.best_cindex == m.best_cindex]
        return max(valid, key=lambda m: m.best_cindex) if valid else None

    @property
    def best_ipcw(self) -> FoldMetrics | None:
        if not self.folds:
            return None
        valid = [m for m in self.folds.values() if m.best_ipcw == m.best_ipcw]
        return max(valid, key=lambda m: m.best_ipcw) if valid else None


def split_into_fold_blocks(text: str) -> Iterable[tuple[int, str]]:
    """Yield (fold_number, fold_text) for every completed fold in the log."""
    matches = list(FOLD_START_RE.finditer(text))
    for i, m in enumerate(matches):
        fold = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield fold, text[start:end]


def parse_log(path: Path) -> ExpReport:
    # tqdm emits '\r' so convert them to newlines first.
    text = path.read_text(errors="replace").replace("\r", "\n")
    report = ExpReport(name=path.stem, log_path=path)

    # Detect fatal training errors: a Python traceback at the top of the log
    # means the run never produced any val lines, so surface that to the user.
    if "Traceback (most recent call last):" in text[:2000]:
        # The exception summary is typically the LAST non-blank, non-File,
        # non-frame-body line in the traceback header. Frame body lines look
        # like `return _run_code(code, main_globals, None,` — we skip them
        # by requiring the line to contain no '(' or to look like an actual
        # Python exception / statement.
        candidate: str | None = None
        for raw_line in text.splitlines()[:60]:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped == "Traceback (most recent call last):":
                continue
            if stripped.startswith("File "):
                continue
            # Skip indented frame bodies that just happen to be a single
            # line (e.g. "return foo(bar,").
            if "(" in stripped and ")" not in stripped.split("(")[-1]:
                continue
            if stripped.startswith("assert ") or stripped.startswith("raise "):
                candidate = stripped
                break
            # Bare exception class names (AssertionError, KeyError: 'foo', …).
            if (
                stripped[0].isalpha()
                and stripped[0].isupper()
                and len(stripped) < 200
            ):
                candidate = stripped
                break
        report.error = candidate or "Traceback detected (could not parse exception)"
        return report

    for fold_num, fold_text in split_into_fold_blocks(text):
        metrics = FoldMetrics(fold=fold_num)
        for line in fold_text.splitlines():
            m = VAL_RE.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            cindex = float(m.group(2))
            ipcw = float(m.group(3))
            ibs = float(m.group(4))
            iauc = float(m.group(5))

            # Final-epoch snapshot is always the latest epoch seen in this fold.
            if epoch >= metrics.final_epoch:
                metrics.final_epoch = epoch
                metrics.final_cindex = cindex
                metrics.final_ipcw = ipcw
                metrics.final_ibs = ibs
                metrics.final_iauc = iauc

            # Best-by-cindex snapshot.
            if cindex > metrics.best_cindex:
                metrics.best_epoch = epoch
                metrics.best_cindex = cindex
                metrics.best_ipcw = ipcw
                metrics.best_ibs = ibs
                metrics.best_iauc = iauc

        if metrics.final_epoch >= 0:
            prior = report.folds.get(fold_num)
            # Keep whichever record has more training progress so re-runs
            # while a fold is still running never downgrade its numbers.
            if prior is None or metrics.final_epoch >= prior.final_epoch:
                report.folds[fold_num] = metrics

    report.completed_folds = sorted(
        f for f, m in report.folds.items() if m.is_complete
    )
    return report


def latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.glob("[0-9]*") if p.is_dir()]
    if not candidates:
        return root
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _agg(reports: list[ExpReport], attr: str, source: str) -> tuple[float | None, float | None]:
    """Compute mean/std across all folds that contributed to `source`.

    `source` is one of "best" or "final". We only average folds that have a
    real number for that metric, so partially-trained experiments degrade
    gracefully.
    """
    vals: list[float] = []
    for r in reports:
        for m in r.folds.values():
            v = getattr(m, f"{source}_{attr}")
            if v == v:  # not NaN
                vals.append(v)
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _fmt(v: float | None, std: float | None, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if std is None:
        return f"{v:.{decimals}f}"
    return f"{v:.{decimals}f} ± {std:.{decimals}f}"


def render_markdown(reports: list[ExpReport], run_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# SurvOT-Rank Batch — Per-Fold Best Validation Results")
    lines.append("")
    lines.append(f"Run dir: `{run_dir}`  ")
    lines.append(f"Generated: {os.popen('date').read().strip()}  ")
    lines.append(
        f"Experiments parsed: **{len(reports)}**. "
        "Definitions used everywhere below:"
    )
    lines.append("")
    lines.append(
        "- **Best fold-K** = the epoch within fold K whose val cindex is the "
        "highest. Its companion metrics (IPCW / IBS / iAUC) are whatever that "
        "same epoch recorded. This is the headline number we want to compare "
        "across folds and across experiments."
    )
    lines.append(
        "- **Mean ± std** = the 5-fold mean and population std of those per-fold "
        "best values (across the folds that have data so far; partial folds are "
        "still counted in the mean)."
    )
    lines.append("")

    # ---- Failures section ----
    failures = [r for r in reports if r.error]
    if failures:
        lines.append("## ⚠️ Experiments that failed to start")
        lines.append("")
        for r in failures:
            lines.append(f"- **`{r.name}`**: `{r.error}`")
            lines.append(
                f"  Full log: `{r.log_path}` — this experiment produced no "
                "val lines, so no per-fold detail is shown below."
            )
        lines.append("")

    # ---- Summary table: one row per experiment ----
    lines.append("## Per-experiment summary")
    lines.append("")
    lines.append(
        "| # | Experiment | Folds done / total | Mean best cindex ± std "
        "| Mean best IPCW ± std | Mean best IBS ± std | Mean best iAUC ± std |"
    )
    lines.append(
        "|---|------------|--------------------|---------------------|"
        "---------------------|---------------------|----------------------|"
    )
    for idx, r in enumerate(reports, start=1):
        n_total = len(r.folds)
        n_done = len(r.completed_folds)
        if r.error:
            status = f"0 / {n_total or 5} (FAILED)"
        elif n_total == 0:
            status = "0 / 0 (no data)"
        else:
            status = f"{n_done} / {n_total}"

        cmean, cstd = _agg([r], "cindex", "best")
        imean, istd = _agg([r], "ipcw", "best")
        bmean, bstd = _agg([r], "ibs", "best")
        amean, astc = _agg([r], "iauc", "best")
        lines.append(
            f"| {idx} | `{r.name}` | {status} "
            f"| {_fmt(cmean, cstd)} | {_fmt(imean, istd)} "
            f"| {_fmt(bmean, bstd)} | {_fmt(amean, astc)} |"
        )

    # ---- Per-experiment per-fold detail block ----
    lines.append("")
    lines.append("## Per-experiment per-fold detail")
    lines.append("")
    lines.append(
        "Each row is the **best epoch within that fold** (highest val cindex "
        "seen so far). `Status` tells you whether the fold finished all 30 "
        "epochs (`✅ done`) or is still running (`⏳ ep K/29`); partial folds "
        "are still included in the summary's mean ± std."
    )
    lines.append("")
    for r in reports:
        lines.append(f"### `{r.name}`")
        lines.append("")
        lines.append(
            "| Fold | Status | Best epoch | best cindex | best IPCW "
            "| best IBS | best iAUC |"
        )
        lines.append(
            "|------|--------|------------|-------------|-----------|"
            "----------|-----------|"
        )
        for fold in sorted(r.folds):
            m = r.folds[fold]
            status = "✅ done" if m.is_complete else f"⏳ ep {m.final_epoch}/29"
            lines.append(
                f"| {fold} | {status} | {m.best_epoch}/29 "
                f"| {m.best_cindex:.4f} | {m.best_ipcw:.4f} "
                f"| {m.best_ibs:.4f} | {m.best_iauc:.4f} |"
            )
        if not r.folds:
            lines.append(
                "| — | no val lines parsed yet | — | — | — | — | — |"
            )
    return "\n".join(lines) + "\n"


def write_csv(reports: list[ExpReport], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "experiment", "fold", "completed",
            "best_epoch", "best_cindex", "best_ipcw", "best_ibs", "best_iauc",
            "final_epoch", "final_cindex", "final_ipcw", "final_ibs", "final_iauc",
        ])
        for r in reports:
            for fold in sorted(r.folds):
                m = r.folds[fold]
                w.writerow([
                    r.name, fold, m.is_complete,
                    m.best_epoch, f"{m.best_cindex:.4f}", f"{m.best_ipcw:.4f}",
                    f"{m.best_ibs:.4f}", f"{m.best_iauc:.4f}",
                    m.final_epoch, f"{m.final_cindex:.4f}",
                    f"{m.final_ipcw:.4f}", f"{m.final_ibs:.4f}",
                    f"{m.final_iauc:.4f}",
                ])


def main(argv: list[str]) -> int:
    project_dir = Path(__file__).resolve().parents[1]
    default_root = project_dir / "results" / "batch_runs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Directory containing <exp>.log files. "
             "Defaults to the most recent results/batch_runs/<timestamp>/ subdir.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Markdown output path. Defaults to <run-dir>/BEST_RESULTS.md.",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir or latest_run_dir(default_root)
    if not run_dir.is_dir():
        print(f"[extract_best_results] run dir not found: {run_dir}", file=sys.stderr)
        return 1

    logs = sorted(run_dir.glob("*.log"))
    if not logs:
        print(f"[extract_best_results] no .log files in {run_dir}", file=sys.stderr)
        return 1

    reports = [parse_log(p) for p in logs]

    out_md = args.out or (run_dir / "BEST_RESULTS.md")
    out_md.write_text(render_markdown(reports, run_dir))
    write_csv(reports, run_dir / "best_results.csv")

    print(f"[extract_best_results] parsed {len(reports)} log files in {run_dir}")
    for r in reports:
        n_done = len(r.completed_folds)
        n_total = len(r.folds)
        if r.error:
            print(f"  - {r.name:30s} FAILED to start: {r.error}")
            continue
        if n_total == 0:
            print(f"  - {r.name:30s} (no val lines yet)")
            continue
        cmean, _ = _agg([r], "cindex", "best")
        if cmean is None:
            print(f"  - {r.name:30s} {n_done}/{n_total} folds done")
            continue
        print(
            f"  - {r.name:30s} {n_done}/{n_total} folds done, "
            f"mean best cindex = {cmean:.4f}"
        )
    print(f"[extract_best_results] wrote {out_md}")
    print(f"[extract_best_results] wrote {run_dir / 'best_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))