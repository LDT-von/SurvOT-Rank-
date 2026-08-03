#!/usr/bin/env python3
"""监控优先实验队列进度"""

import csv
import pickle
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

STAGE_DEFS = OrderedDict([
    ("v33_blca_uni5", {
        "label": "DCT v3.3 Score-First BLCA UNI",
        "dirs": ["dct_v3.3_score_first_blca_uni_rep"],
        "folds": range(5),
        "max_epochs": 50,
    }),
    ("v38_lusc_screen", {
        "label": "DCT v3.8 LUSC UNI2-h 8变体",
        "dirs": None,  # dynamic: robust/<variant>/lusc
        "folds": [0],
        "max_epochs": 20,
        "variants": True,
    }),
    ("v382_blca_fold124", {
        "label": "DCT v3.8.2 MGPTR BLCA UNI2-h",
        "dirs": ["dct_v3.8.2_30ep"],
        "folds": [1, 2, 4],
        "max_epochs": 30,
    }),
    ("v383_blca_fold124", {
        "label": "DCT v3.8.3 Centered BLCA UNI2-h",
        "dirs": ["dct_v3.8.3_intervention_consistency_centered_30ep"],
        "folds": [1, 2, 4],
        "max_epochs": 30,
    }),
    ("v39_blca_fold124", {
        "label": "DCT v3.9 Risk-Simplex BLCA UNI2-h",
        "dirs": ["dct_v3.9_risk_simplex_transport_30ep"],
        "folds": [1, 2, 4],
        "max_epochs": 30,
    }),
    ("v40_blca_fold124", {
        "label": "IST-Surv v4.0 BLCA UNI2-h",
        "dirs": ["ist_surv_v4.0_30ep"],
        "folds": [1, 2, 4],
        "max_epochs": 30,
    }),
    ("v41_blca_fold124", {
        "label": "DCT v4.1 Evidence Ledger BLCA UNI",
        "dirs": ["dct_v4.1_survival_evidence_ledger_30ep"],
        "folds": [1, 2, 4],
        "max_epochs": 30,
    }),
    ("arcsurv_blca_fold124", {
        "label": "ArcSurv BLCA UNI",
        "dirs": ["archetypal_risk_composition_30ep"],
        "folds": [1, 2, 4],
        "max_epochs": 30,
    }),
])

V38_VARIANTS = [
    "base", "direction", "dose", "reconfiguration",
    "direction_dose", "direction_reconfiguration",
    "dose_reconfiguration", "full",
]


def _find_result_dirs(stage_name: str, stage_def: dict) -> list[tuple[str, Path, list[int]]]:
    """Return list of (variant_label, base_dir, folds) for this stage."""
    out = []
    if stage_name == "v38_lusc_screen":
        robust = RESULTS / "dct_v3.8_transport_consistency_20ep" / "robust_uni2h"
        for variant in V38_VARIANTS:
            d = robust / variant / "lusc"
            out.append((variant, d, [0]))  # always include, even if dir missing
    else:
        for base in stage_def["dirs"]:
            d = RESULTS / base / "blca"
            if not d.exists():
                # try deeper: some versions (e.g. v4.0) nest under protocol/variant/
                deeper = sorted(RESULTS.rglob(base + "/**/blca"))
                if deeper:
                    d = deeper[0]
            out.append(("", d, list(stage_def["folds"])))
    return out


def _parse_epoch_csv(path: Path) -> dict:
    """Return {best_epoch, best_cindex, last_epoch, last_cindex, running}."""
    best_ep, best_ci = None, -1.0
    last_ep, last_ci = None, -1.0
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                ep = int(r["epoch"])
                ci = float(r.get("val_cindex", r.get("val_c_index", 0)))
                last_ep = ep
                last_ci = ci
                if ci > best_ci:
                    best_ci = ci
                    best_ep = ep
    except Exception:
        return {"running": False}
    return {
        "running": last_ep is not None,
        "best_epoch": best_ep,
        "best_cindex": round(best_ci, 4) if best_ci > 0 else None,
        "last_epoch": last_ep,
        "last_cindex": round(last_ci, 4),
    }


def _compute_cindex(patient_dict: dict) -> float | None:
    """Compute Harrell's C-index from per-patient {risk, time, censor} dict."""
    import numpy as np

    patients = []
    for pid, info in patient_dict.items():
        if not isinstance(info, dict):
            continue
        t = info.get("time")
        c = info.get("censor")  # 1=censored, 0=event
        r = info.get("risk")
        if t is None or r is None:
            continue
        patients.append((float(t), float(c), float(r)))

    if len(patients) < 2:
        return None

    patients.sort(key=lambda x: x[0])
    concordant = 0
    comparable = 0

    for i in range(len(patients)):
        ti, ci, ri = patients[i]
        if ci >= 1.0:  # censored
            continue
        for j in range(i + 1, len(patients)):
            tj, cj, rj = patients[j]
            if tj <= ti:
                continue
            comparable += 1
            if ri > rj:
                concordant += 1
            elif abs(ri - rj) < 1e-10:
                concordant += 0.5

    if comparable == 0:
        return None
    return concordant / comparable


def _parse_final_pkl(path: Path) -> dict:
    """Return {done, cindex, epoch} from split results pkl."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        # try summary format first
        if isinstance(data, dict):
            ci = data.get("c_index", data.get("val_cindex"))
            ep = data.get("best_epoch", data.get("epoch"))
            if ci is not None:
                return {
                    "done": True,
                    "cindex": round(float(ci), 4),
                    "epoch": int(ep) if ep is not None else None,
                }
            # might be per-patient dict, try to get epoch from epoch curve
            fold = _extract_fold_from_path(path)
            epoch_csv = _find_epoch_csv(path.parent.parent.parent, fold) if fold is not None else None
            if epoch_csv:
                info = _parse_epoch_csv(epoch_csv)
                ep = info.get("best_epoch")
            else:
                ep = None
            ci = _compute_cindex(data)
            return {
                "done": True,
                "cindex": round(ci, 4) if ci is not None else None,
                "epoch": ep,
            }
    except Exception:
        return {"done": False}
    return {"done": False}


def _extract_fold_from_path(path: Path) -> int | None:
    """Extract fold number from split_N_results_final.pkl path."""
    import re
    m = re.search(r"split_(\d)_", str(path))
    return int(m.group(1)) if m else None


def _find_final_pkl(base_dir: Path, fold: int) -> Path | None:
    """Recursively find split_N_results_final.pkl for a given fold."""
    matches = list(base_dir.rglob(f"split_{fold}_results_final.pkl"))
    return matches[0] if matches else None


def _find_epoch_csv(base_dir: Path, fold: int) -> Path | None:
    """Recursively find epoch_curve_foldN.csv for a given fold."""
    matches = list(base_dir.rglob(f"epoch_curve_fold{fold}.csv"))
    return matches[0] if matches else None


def scan() -> list[dict]:
    results = []
    for stage_name, stage_def in STAGE_DEFS.items():
        dir_entries = _find_result_dirs(stage_name, stage_def)
        for variant_label, base_dir, folds in dir_entries:
            for fold in folds:
                entry = {
                    "stage": stage_name,
                    "variant": variant_label,
                    "fold": fold,
                    "max_epochs": stage_def["max_epochs"],
                    "status": "pending",
                    "cindex": None,
                    "epoch": None,
                    "running_epoch": None,
                    "running_cindex": None,
                    "best_running_epoch": None,
                    "best_running_cindex": None,
                }
                if not base_dir.exists():
                    results.append(entry)
                    continue

                pkl = _find_final_pkl(base_dir, fold)
                if pkl:
                    info = _parse_final_pkl(pkl)
                    if info.get("done"):
                        entry["status"] = "done"
                        entry["cindex"] = info.get("cindex")
                        entry["epoch"] = info.get("epoch")
                        results.append(entry)
                        continue

                csv_path = _find_epoch_csv(base_dir, fold)
                if csv_path:
                    info = _parse_epoch_csv(csv_path)
                    if info.get("running"):
                        entry["status"] = "running"
                        entry["running_epoch"] = info["last_epoch"]
                        entry["running_cindex"] = info["last_cindex"]
                        entry["best_running_epoch"] = info.get("best_epoch")
                        entry["best_running_cindex"] = info.get("best_cindex")
                results.append(entry)
    return results


def _cindex_str(entry: dict, full: bool = False) -> str:
    """Format a C-Index cell."""
    if entry["status"] == "done":
        return f"  {entry['cindex']:.4f}  " if entry["cindex"] else "  N/A   "
    if entry["status"] == "running":
        best = entry.get("best_running_cindex")
        best_ep = entry.get("best_running_epoch")
        cur = entry.get("running_cindex")
        cur_ep = entry.get("running_epoch")
        if full and cur_ep is not None:
            if best and best != -1.0:
                return f"{best:.4f}({best_ep})"
            return f"{cur:.3f}({cur_ep})"
        if best and best != -1.0:
            return f" {best:.4f}  "
        return f" e{cur_ep}  "
    return "   ···   "


def _color(status: str, text: str) -> str:
    """Simple ANSI color."""
    codes = {"done": "\033[32m", "running": "\033[33m", "pending": "\033[90m"}
    reset = "\033[0m"
    return f"{codes.get(status, '')}{text}{reset}"


def main():
    entries = scan()
    now = datetime.now().strftime("%H:%M:%S")

    # aggregate stats
    by_stage = OrderedDict()
    for e in entries:
        by_stage.setdefault(e["stage"], []).append(e)

    done_total = sum(1 for e in entries if e["status"] == "done")
    running_total = sum(1 for e in entries if e["status"] == "running")
    total = len(entries)
    # exclude missing dirs
    missing = sum(1 for e in entries if e.get("missing"))

    print(f"\n{'='*82}")
    print(f"  优先级实验队列监控  ({now})    完成: {done_total}/{total}  进行中: {running_total}")
    print(f"{'='*82}")

    for idx, (stage_name, rows) in enumerate(by_stage.items(), 1):
        stage_def = STAGE_DEFS[stage_name]
        label = stage_def["label"]
        max_ep = stage_def["max_epochs"]
        folds_wanted = stage_def.get("folds", [])

        done = [r for r in rows if r["status"] == "done"]
        running = [r for r in rows if r["status"] == "running"]
        pending = [r for r in rows if r["status"] == "pending"]

        if not rows:
            continue

        # stage header
        status_icon = "✅" if len(done) == len(rows) else ("🔄" if running else ("⏳" if rows else "  "))
        print(f"\n  {status_icon} [{idx}] {label} ({max_ep}ep)")

        # variant × fold table
        if stage_name == "v38_lusc_screen":
            done_cis = [r["cindex"] for r in done if r["cindex"] is not None]
            avg_str = f"  Mean: {sum(done_cis)/len(done_cis):.4f}" if done_cis else ""
            print(f"     {len(done)}/8 variants done{avg_str}")
            print(f"       {'Variant':<24} {'C-Index':>10} {'Epoch':>6}  Status")
            print(f"       {'-'*55}")
            # sort: done first (by cindex desc), then running, then pending
            def _sort_key(r):
                if r["status"] == "done":
                    return (0, -(r["cindex"] or 0))
                elif r["status"] == "running":
                    return (1, 0)
                return (2, 0)
            for r in sorted(rows, key=_sort_key):
                tag = r["variant"]
                if r["status"] == "done":
                    ci = f"{r['cindex']:.4f}" if r["cindex"] is not None else "N/A"
                    ep = r["epoch"] if r["epoch"] is not None else "?"
                    print(f"       {tag:<24}  {ci:>10}  {ep:>5}  ✅")
                elif r["status"] == "running":
                    cur = r.get("running_cindex")
                    cur_ep = r.get("running_epoch", "?")
                    best = r.get("best_running_cindex")
                    best_ep = r.get("best_running_epoch")
                    cur_str = f"{cur:.4f}" if cur is not None else "N/A"
                    info = f"{cur_str}    e{cur_ep}/{max_ep}"
                    if best is not None and best != -1.0:
                        info += f"  best:{best:.4f}@{best_ep}"
                    print(f"       {tag:<24}  {_color('running', info)}")
                else:
                    print(f"       {tag:<24}  {'···':>10}  {'···':>6}  ⏳")
        else:
            # fold table
            cindices = []
            print(f"       {'Fold':<6} {'Status':<10} {'C-Index':>10} {'Epoch'}")
            print(f"       {'-'*40}")
            for r in sorted(rows, key=lambda x: x["fold"]):
                if r["status"] == "done":
                    ci = r["cindex"] if r["cindex"] is not None else float("nan")
                    ep = r["epoch"]
                    if not (isinstance(ci, float) and ci == ci):  # NaN check
                        cindices.append(None)
                        ep_str = f"{ep}" if ep is not None else "?"
                        print(f"       {r['fold']:<6} {'done':<10} {'N/A':>10}    {ep_str:>5}")
                    else:
                        cindices.append(ci)
                        ep_str = f"{ep}" if ep is not None else "?"
                        print(f"       {r['fold']:<6} {'done':<10} {ci:.4f}    {ep_str:>5}")
                elif r["status"] == "running":
                    cur = r.get("running_cindex")
                    cur_ep = r.get("running_epoch", "?")
                    best = r.get("best_running_cindex")
                    best_ep = r.get("best_running_epoch")
                    cur_str = f"{cur:.4f}" if cur is not None else "N/A"
                    info = f"{cur_str}    e{cur_ep}/{max_ep}"
                    if best is not None and best != -1.0:
                        info += f" (best:{best:.4f}@{best_ep})"
                    print(f"       {r['fold']:<6} {_color('running', 'running  ')} {info}")
                else:
                    print(f"       {r['fold']:<6} pending       ···")
            if cindices:
                valid = [c for c in cindices if c is not None]
                if valid:
                    avg = sum(valid) / len(valid)
                    print(f"       {'-'*40}")
                    print(f"       {'Mean':<6} {' ':<10} {avg:.4f}   ({len(valid)}/{len(folds_wanted)} folds)")
                else:
                    print(f"       {'-'*40}")
                    print(f"       {'Mean':<6} {' ':<10} N/A   ({len(cindices)}/{len(folds_wanted)} folds)")

    # running process check
    import subprocess
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True, text=True,
    )
    train_lines = [l for l in result.stdout.split("\n") if "survot_rank" in l and "cli" in l and "train" in l]
    if train_lines:
        print(f"\n  [进程] {len(train_lines)} 个训练进程在运行")

    missing = sum(1 for e in entries if e.get("missing"))
    if missing:
        print(f"\n  [注意] {missing} 个变体目录不存在")

    print()


if __name__ == "__main__":
    main()
