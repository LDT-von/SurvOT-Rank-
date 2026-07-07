#!/usr/bin/env python3
"""
aggregate_multicancer.py 鈥?aggregate BLCA/BRCA multi-cancer V45 results

Reads:  /data1/sweep_results_30ep/multicancer_v1/{study}/blca|SurvOTRank_otehv2_rankevent/*/summary.csv
Writes: /data1/sweep_results_30ep/multicancer_v1/summary_table.md (markdown table)
        /data1/sweep_results_30ep/multicancer_v1/summary_table.csv (csv)
"""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path("/data1/sweep_results_30ep/multicancer_v1")
STUDIES = ["blca", "brca"]


def find_summary(study_dir: Path) -> Path | None:
    # runall_v45mc_XXX produces a run dir with summary.csv at deepest leaf
    candidates = list(study_dir.rglob("summary.csv"))
    return candidates[0] if candidates else None


def main():
    rows = []
    for s in STUDIES:
        d = ROOT / s
        if not d.is_dir():
            print(f"[{s}] missing dir")
            continue
        csv = find_summary(d)
        if not csv:
            print(f"[{s}] NO summary.csv")
            continue
        try:
            df = pd.read_csv(csv, index_col=0)
            if "mean" not in df.index:
                print(f"[{s}] no mean row in {csv}")
                continue
            m = df.loc["mean"]
            std = df.loc["std"] if "std" in df.index else None
            row = {"study": s, "csv": str(csv)}
            for col in df.columns:
                row[col] = m[col]
                if std is not None:
                    row[f"{col}_std"] = std[col]
            rows.append(row)
            print(f"[{s}] mean cindex={m.get('val_cindex', float('nan')):.4f} "
                  f"ipcw={m.get('val_cindex_ipcw', float('nan')):.4f} "
                  f"IBS={m.get('val_IBS', float('nan')):.4f}")
        except Exception as e:
            print(f"[{s}] ERR: {e}")
    if not rows:
        print("no results yet")
        return
    res = pd.DataFrame(rows)
    cols = [c for c in res.columns if not c.endswith("_std")]
    std_cols = [c for c in res.columns if c.endswith("_std")]

    # csv
    res.to_csv(ROOT / "summary_table.csv", index=False)
    # md
    md_lines = ["| study | val_cindex | val_cindex_ipcw | val_IBS | val_iauc | val_loss | n_folds |",
                "|-------|------------|------------------|---------|----------|----------|---------|"]
    for _, r in res.iterrows():
        n = 5  # canonical 5 fold
        # fallback: try to count from existing fold_*_results.pkl if needed
        md_lines.append(
            f"| {r['study']} | {r.get('val_cindex', 0):.4f}卤{r.get('val_cindex_std', 0):.4f} "
            f"| {r.get('val_cindex_ipcw', 0):.4f}卤{r.get('val_cindex_ipcw_std', 0):.4f} "
            f"| {r.get('val_IBS', 0):.4f}卤{r.get('val_IBS_std', 0):.4f} "
            f"| {r.get('val_iauc', 0):.4f}卤{r.get('val_iauc_std', 0):.4f} "
            f"| {r.get('val_loss', 0):.4f}卤{r.get('val_loss_std', 0):.4f} "
            f"| {n} |"
        )
    (ROOT / "summary_table.md").write_text("\n".join(md_lines) + "\n")

    print("")
    print("=" * 60)
    print("Multi-cancer V45 summary")
    print("=" * 60)
    print("\n".join(md_lines))
    print("=" * 60)
    print(f"saved: {ROOT / 'summary_table.md'}")
    print(f"saved: {ROOT / 'summary_table.csv'}")


if __name__ == "__main__":
    main()