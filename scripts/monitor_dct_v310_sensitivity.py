#!/usr/bin/env python3
"""DCT v3.10 legacy_uni2h_sensitivity 全队列监控
覆盖 6 cancer × 2 lambda × 5 folds = 60 runs
用法: python scripts/monitor_dct_v310_sensitivity.py
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

BASE = Path("results/dct_v310_legacy_uni2h_sensitivity")
CANCERS = ["blca", "skcm", "hnsc", "lusc", "kirc", "ucec"]
LAMBDA_MAP = {"lambda_d0": "0.00", "lambda_d0.05": "0.05"}
FOLDS = range(5)
# Treat CSV as stale if older than this
STALE_THRESHOLD = timedelta(minutes=10)


def scan_run(run_base: Path):
    if not run_base.exists():
        return {}
    found = {}
    now = datetime.now()
    for run_dir in run_base.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        fold_num = None
        for c in CANCERS:
            if f"_{c}_f" in name:
                fold_num = int(name.split(f"_{c}_f")[1])
                break
        if fold_num is None:
            continue
        csvs = list(run_dir.glob("epoch_curve_fold*.csv"))
        if not csvs:
            continue
        csv_path = csvs[0]
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        is_stale = (now - mtime) > STALE_THRESHOLD
        df = pd.read_csv(csv_path)
        best_idx = df["val_cindex"].idxmax()
        best = df.iloc[best_idx]
        max_ep = int(df["epoch"].max())
        found[fold_num] = {
            "status": "DONE" if max_ep >= 49 else f"EP{max_ep}",
            "best_cindex": round(float(best["val_cindex"]), 4),
            "best_ep": int(best["epoch"]),
            "last_ep": max_ep,
            "stale": is_stale,
        }
    return found


def fmt_cell(r, f):
    st = r.get("status", "-")
    ci = r.get("best_cindex", "-")
    ep = r.get("best_ep", "-")
    stale = r.get("stale", False)
    if st == "DONE":
        return f"{ci:.4f}@{int(ep)}"
    elif st == "NOT_STARTED":
        return "--wait--"
    elif st and st != "-":
        flag = " [OLD]" if stale else ""
        return f"~{ci:.3f}@{st}{flag}"
    return "----"


def build_rows():
    rows = []
    for ld_name, lam in LAMBDA_MAP.items():
        for cancer in CANCERS:
            run_base = BASE / ld_name / cancer / "blca" / "SurvOTRank_dct_v310_directional_regularized_transport"
            found = scan_run(run_base)
            for f in FOLDS:
                if f in found:
                    rows.append({"λ": lam, "cancer": cancer, "fold": f, **found[f]})
                else:
                    rows.append({"λ": lam, "cancer": cancer, "fold": f,
                                 "status": "NOT_STARTED", "best_cindex": "-", "best_ep": "-"})
    return pd.DataFrame(rows)


def main():
    print(f"\n{'='*96}")
    print(f"  DCT v3.10 legacy_uni2h_sensitivity  全队列监控  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*96}")

    df_all = build_rows()
    W = 96

    for lam in ["0.00", "0.05"]:
        label = "direction loss = OFF" if lam == "0.00" else "direction loss = ON"
        print(f"\n{'='*W}")
        print(f"  λ = {lam}  [{label}]")
        print(f"{'='*W}")
        sub = df_all[df_all["λ"] == lam].sort_values(["cancer","fold"])

        stats = {}
        for cancer in CANCERS:
            row_data = {r["fold"]: r for _, r in sub[sub["cancer"] == cancer].iterrows()}
            ci_vals = [row_data[f]["best_cindex"] for f in FOLDS
                       if isinstance(row_data.get(f, {}).get("best_cindex"), float)]
            done = sum(1 for f in FOLDS if row_data.get(f, {}).get("status") == "DONE")
            stats[cancer] = {
                "row_data": row_data,
                "done": done,
                "mean_ci": round(sum(ci_vals)/len(ci_vals), 4) if ci_vals else None,
            }

        header = f"{'Cancer':<8}" + "  " + "  ".join(f"{'F'+str(f):>10}" for f in FOLDS) + f"  {'Done':>5}  {'Mean':>8}"
        print(header)
        print("-" * W)
        for cancer in CANCERS:
            s = stats[cancer]
            line = f"{cancer:<8}" + "  "
            for f in FOLDS:
                r = s["row_data"].get(f, {})
                line += f"{fmt_cell(r, f):>10}  "
            m = s["mean_ci"]
            print(f"{line}{s['done']:>5}  {m:>8.4f}" if m is not None else f"{line}{s['done']:>5}  {'----':>8}")

        total_done = sum(s["done"] for s in stats.values())
        valid = [s["mean_ci"] for s in stats.values() if s["mean_ci"] is not None]
        ov = f"{sum(valid)/len(valid):.4f}" if valid else "----"
        print(f"\n  Completed folds: {total_done}/30   Overall mean c-index: {ov}")

    # Delta table
    print(f"\n{'='*W}")
    print("  Δ c-index  (λ=0.05 − λ=0.00)  per cancer  (only cancers with both λ done)")
    print(f"{'='*W}")
    print(f"{'Cancer':<8}  {'λ=0.00':>12}  {'λ=0.05':>12}  {'Δ':>8}  {'n':>4}")
    print("-" * 50)

    delta_rows = []
    for cancer in CANCERS:
        def get_vals(lam_):
            sub_ = df_all[df_all["λ"] == lam_]
            rd = {r["fold"]: r for _, r in sub_[sub_["cancer"] == cancer].iterrows()}
            return [rd[f]["best_cindex"] for f in FOLDS
                    if isinstance(rd.get(f, {}).get("best_cindex"), float)]
        d00 = get_vals("0.00")
        d05 = get_vals("0.05")
        if d00 and d05:
            n = min(len(d00), len(d05))
            delta = (sum(d05[:n])/n) - (sum(d00[:n])/n)
            delta_rows.append((cancer, round(sum(d00)/len(d00), 4),
                               round(sum(d05)/len(d05), 4), round(delta, 4), n))
        else:
            delta_rows.append((cancer, None, None, None, 0))

    for cancer, m00, m05, delta, n in delta_rows:
        if m00 is not None:
            flag = " ★" if delta > 0 else " ○" if delta < 0 else " ~"
            print(f"{cancer:<8}  {m00:>12.4f}  {m05:>12.4f}  {delta:>+8.4f}  {n:>4}{flag}")
        else:
            print(f"{cancer:<8}  {'--':>12}  {'--':>12}  {'--':>8}  {'--':>4}")

    valid_d = [(c, m00, m05, d, n) for c, m00, m05, d, n in delta_rows if m00 is not None]
    if valid_d:
        ov00 = sum(m for _, m, _, _, _ in valid_d) / len(valid_d)
        ov05 = sum(m for _, _, m, _, _ in valid_d) / len(valid_d)
        ovdelta = sum(d for _, _, _, d, _ in valid_d) / len(valid_d)
        flag = " ★" if ovdelta > 0 else " ○"
        print(f"\n  Overall   {ov00:>12.4f}  {ov05:>12.4f}  {ovdelta:>+8.4f}  {len(valid_d)}{flag}")

    # Active runs (non-stale, not done)
    active = []
    for _, row in df_all.iterrows():
        if row.get("status") == "DONE" or row.get("status") == "NOT_STARTED":
            continue
        if not row.get("stale", True):
            active.append(f"  {row['cancer'].upper()} λ={row['λ']} fold{row['fold']} @EP{row.get('last_ep','?')}/{row.get('best_ep','?')}")
    if active:
        print(f"\n{'='*W}")
        print(f"  正在训练  ({len(active)} active)")
        for a in active:
            print(a)


if __name__ == "__main__":
    main()
