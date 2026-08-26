#!/usr/bin/env python3
"""Build DCT v3.3 full reproducibility archive — no re-training."""

import csv, json, hashlib, os, shutil, subprocess, tarfile, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/home/ubuntu/SurvOT-Rank")
ARCHIVE_DIR = PROJECT_ROOT / "reproducibility_archives" / "dct_v3_3_score_first_5fold"
RESULTS_DIRS = {
    "v3.3_f0_f1_f3_f4": PROJECT_ROOT / "results/dct_v3_score_first_diagnostics/full/blca/SurvOTRank_distributional_counterfactual_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v3_score_first_full",
    "v3.3_f2_nan_fix": PROJECT_ROOT / "results/dct_v3_3_fold2_nan_fix/blca/SurvOTRank_distributional_counterfactual_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v3_3_fold2_nan_fix",
}
CONFIG_FILE = PROJECT_ROOT / "configs/diagnostics/dct_v3_score_blca.yaml"
SPLIT_DIR = Path("/home/ubuntu/newSlotSPE/SlotSPE/dataset_csv/splits/5fold/blca")
EXPERIMENT_SUMMARY = PROJECT_ROOT / "EXPERIMENT_SUMMARY.md"

FOLDS = [0, 1, 2, 3, 4]
CHECKPOINT_SIZE_MB = 116  # ~116 MB each, skip for git archive
SKIP_CHECKPOINTS = True   # too large for git (5 × 116MB = 580MB)

# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def run(cmd: str | list, cwd=None, timeout=30) -> str:
    return subprocess.check_output(
        cmd, shell=isinstance(cmd, str), cwd=cwd, text=True,
        stderr=subprocess.STDOUT, timeout=timeout)

def read_epoch_curve(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for r in reader:
            if len(r) < 6:
                continue
            rows.append({
                "epoch": int(r[0]),
                "val_cindex": float(r[1]),
                "val_ipcw": float(r[2]),
                "val_ibs": float(r[3]),
                "val_iauc": float(r[4]),
                "train_loss": float(r[5]) if r[5].lower() != "nan" else float("nan"),
                "ot": float(r[6]) if len(r) > 6 and r[6].lower() != "nan" else 0,
                "ipcw_rank": float(r[7]) if len(r) > 7 and r[7].lower() != "nan" else 0,
                "ipcw_pairs": float(r[8]) if len(r) > 8 and r[8].lower() != "nan" else 0,
                "rank": float(r[9]) if len(r) > 9 and r[9].lower() != "nan" else 0,
                "anchor": float(r[10]) if len(r) > 10 and r[10].lower() != "nan" else 0,
                "stage_risk": float(r[11]) if len(r) > 11 and r[11].lower() != "nan" else 0,
                "coordinate": float(r[12]) if len(r) > 12 and r[12].lower() != "nan" else 0,
                "active_stage_fraction": float(r[13]) if len(r) > 13 and r[13].lower() != "nan" else 0,
                "anchor_coverage": float(r[14]) if len(r) > 14 and r[14].lower() != "nan" else 0,
            })
    return rows

def find_result_dir(fold: int) -> Path | None:
    if fold == 2:
        return RESULTS_DIRS["v3.3_f2_nan_fix"]
    return RESULTS_DIRS["v3.3_f0_f1_f3_f4"]

def main():
    print("=== DCT v3.3 Reproducibility Archive Builder ===")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. git info ------------------------------------------------------
    print("[1/7] Saving git info ...")
    git_dir = ARCHIVE_DIR / "01_git"
    git_dir.mkdir(exist_ok=True)
    for cid in ["535e020", "48ccc2e", "48c45ac", "afa9dd1"]:
        try:
            log = run(f"git log --format=fuller -1 {cid}", cwd=PROJECT_ROOT)
            (git_dir / f"commit_{cid}.log").write_text(log)
        except Exception:
            pass
    (git_dir / "git_log_oneline.txt").write_text(run("git log --oneline -20", cwd=PROJECT_ROOT))
    (git_dir / "git_status.txt").write_text(run("git status", cwd=PROJECT_ROOT))
    (git_dir / "git_diff_head.txt").write_text(run("git diff HEAD~1..HEAD", cwd=PROJECT_ROOT))
    (git_dir / "git_diff_535e020_to_head.txt").write_text(run("git diff 535e020..HEAD", cwd=PROJECT_ROOT))

    # ---- 2. config & params -----------------------------------------------
    print("[2/7] Saving config & parameters ...")
    cfg_dir = ARCHIVE_DIR / "02_config"
    cfg_dir.mkdir(exist_ok=True)
    shutil.copy(CONFIG_FILE, cfg_dir / "dct_v3_score_blca.yaml")
    for fold in FOLDS:
        d = find_result_dir(fold)
        if d:
            sf = d / "experiment_settings.txt"
            if sf.exists():
                shutil.copy(sf, cfg_dir / f"fold{fold}_experiment_settings.txt")

    args_dict = {
        "method": "SurvOTRank_distributional_counterfactual_transport",
        "study": "blca", "label_col": "survival_months_dss",
        "rna_format": "Pathways", "signature": "combine",
        "n_classes": 4, "num_patches": 2048, "encoding_dim": 1024,
        "max_epochs": 50, "batch_size": 8, "lr": 0.0005,
        "opt": "adamW", "reg": 0.0005, "scheduler": "cosine",
        "warmup_epochs": 5, "grad_clip_norm": 1.0, "grad_accum_steps": 1,
        "seed": 3, "gpu": 0, "num_workers": 4,
        "bag_loss": "nll_surv", "alpha_surv": 0.15,
        "slot_num_wsi": 8, "slot_num_omics": 8, "slot_iters": 3,
        "temperature": 0.01, "topk_ratio": 0.25, "top_k_method": "parallel_topk_st",
        "results_dir_main": "results/dct_v3_score_first_diagnostics/full",
        "results_dir_fold2": "results/dct_v3_3_fold2_nan_fix",
        "split_dir": str(SPLIT_DIR), "num_splits": 5, "k_start": 0, "k_end": 5,
        "otehv2_eps": 0.05, "otehv2_iter": 50, "otehv2_heads": 4,
        "otehv2_layers": 2, "otehv2_dropout": 0.15,
        "dct_num_stages": 4,
        "dct_lambda_ipcw_rank": 0.10, "dct_ipcw_rank_margin": 0.02,
        "dct_ipcw_rank_temperature": 0.50, "dct_ipcw_max_weight": 10.0,
        "dct_lambda_ot": 0.0, "dct_lambda_rank": 0.0,
        "dct_lambda_anchor": 0.0, "dct_lambda_stage_risk": 0.0,
        "dct_evidence_cost_weight": 0.0, "dct_lambda_coordinate": 0.0,
        "dct_evidence_mass_floor": 0.05, "dct_mix_ratio": 0.50,
    }
    (cfg_dir / "args.json").write_text(json.dumps(args_dict, indent=2))

    splits_info = {}
    for fold in FOLDS:
        fp = SPLIT_DIR / f"fold_{fold}.csv"
        splits_info[f"fold_{fold}.csv"] = sha256_file(fp)
    (cfg_dir / "split_files_sha256.json").write_text(json.dumps(splits_info, indent=2))

    # ---- 3. per-fold results (curves, logs, small files — skip checkpoints) 
    print("[3/7] Saving per-fold results ...")
    res_dir = ARCHIVE_DIR / "03_results"
    res_dir.mkdir(exist_ok=True)

    fold_summaries = {}
    all_curves: dict[int, list[dict]] = {}
    nan_found = False

    for fold in FOLDS:
        src = find_result_dir(fold)
        if not src:
            print(f"  WARNING: no result dir for fold {fold}")
            continue

        fold_dir = res_dir / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)

        # epoch curve
        curve_file = src / f"epoch_curve_fold{fold}.csv"
        if curve_file.exists():
            rows = read_epoch_curve(curve_file)
            all_curves[fold] = rows
            shutil.copy(curve_file, fold_dir / f"epoch_curve_fold{fold}.csv")

            for r in rows:
                for k, v in r.items():
                    if isinstance(v, float) and np.isnan(v):
                        nan_found = True
                        print(f"  NaN FOUND in fold{fold} epoch{r['epoch']} column {k}")

            best_row = max(rows, key=lambda r: r["val_cindex"])
            last5_rows = [r for r in rows if r["epoch"] >= len(rows) - 5]
            last5_cindex = np.mean([r["val_cindex"] for r in last5_rows]) if last5_rows else float("nan")
            gap_val = best_row["val_cindex"] - last5_cindex

            fold_summaries[fold] = {
                "epochs_completed": len(rows),
                "best_epoch": best_row["epoch"],
                "best_cindex": round(best_row["val_cindex"], 6),
                "best_ipcw": round(best_row["val_ipcw"], 6),
                "best_ibs": round(best_row["val_ibs"], 6),
                "best_iauc": round(best_row["val_iauc"], 6),
                "last_cindex": round(rows[-1]["val_cindex"], 6),
                "last5_mean_cindex": round(last5_cindex, 6),
                "gap": round(gap_val, 6),
                "total_epochs": len(rows),
            }
            print(f"  fold{fold}: best={fold_summaries[fold]['best_cindex']} @{fold_summaries[fold]['best_epoch']}, "
                  f"last5={fold_summaries[fold]['last5_mean_cindex']}, gap={fold_summaries[fold]['gap']}")

        # log files
        for pat in [f"log_start_{fold}_end_{fold+1}.txt"]:
            lf = src / pat
            if lf.exists():
                shutil.copy(lf, fold_dir / pat)

        # model_parameters.txt (small, always copy)
        mp = src / "model_parameters.txt"
        if mp.exists():
            shutil.copy(mp, fold_dir / "model_parameters.txt")

        # pkl results (small)
        for pkl_name in [f"split_{fold}_results.pkl", f"split_{fold}_results_final.pkl"]:
            pf = src / pkl_name
            if pf.exists():
                shutil.copy(pf, fold_dir / pkl_name)

        # checkpoints: record hash & location, skip copying (too large)
        ckpt_file = src / f"model_best_s{fold}.pth"
        if ckpt_file.exists():
            ckpt_sha = sha256_file(ckpt_file)
            fold_summaries[fold]["checkpoint_best_sha256"] = ckpt_sha
            fold_summaries[fold]["checkpoint_best_path"] = str(ckpt_file)
            fold_summaries[fold]["checkpoint_best_size_mb"] = round(ckpt_file.stat().st_size / (1024*1024), 1)
            if not SKIP_CHECKPOINTS:
                shutil.copy(ckpt_file, fold_dir / f"model_best_s{fold}.pth")

    # ---- 4. metadata ------------------------------------------------------
    print("[4/7] Saving reproducibility metadata ...")
    meta_dir = ARCHIVE_DIR / "04_metadata"
    meta_dir.mkdir(exist_ok=True)

    pip_freeze = run(f"{sys.executable} -m pip freeze", cwd=PROJECT_ROOT)
    (meta_dir / "pip_freeze_full.txt").write_text(pip_freeze)

    gpu_info = {
        "device": "NVIDIA GeForce RTX 5090",
        "cuda_version": "12.8",
        "driver": "570.195.03",
        "memory_total": "32609 MiB",
        "nvidia_smi": run("nvidia-smi", cwd=PROJECT_ROOT).strip(),
    }
    (meta_dir / "gpu_info.json").write_text(json.dumps(gpu_info, indent=2))

    # python env
    env_info = {
        "python_version": sys.version,
        "executable": sys.executable,
        "host": subprocess.check_output("hostname", shell=True, text=True).strip(),
    }
    (meta_dir / "env_info.json").write_text(json.dumps(env_info, indent=2))

    # ---- 5. summary & manifest --------------------------------------------
    print("[5/7] Computing summary statistics ...")
    best_values = [fold_summaries[f]["best_cindex"] for f in FOLDS if f in fold_summaries]
    last5_values = [fold_summaries[f]["last5_mean_cindex"] for f in FOLDS if f in fold_summaries]
    gap_values = [fold_summaries[f]["gap"] for f in FOLDS if f in fold_summaries]

    summary = {
        "experiment": "DCT v3.3 Score-First IPCW Ranking",
        "date": "2026-07-16 / 2026-07-17",
        "commits": ["535e020 (NaN fix)", "48ccc2e (result docs)", "48c45ac (score-first recipe)", "afa9dd1 (isolate results)"],
        "folds_completed": len(fold_summaries),
        "fold_details": fold_summaries,
        "aggregate": {
            "best_mean": round(np.mean(best_values), 6),
            "best_std": round(np.std(best_values), 6),
            "best_median": round(np.median(best_values), 6),
            "last5_mean": round(np.mean(last5_values), 6),
            "last5_std": round(np.std(last5_values), 6),
            "last5_median": round(np.median(last5_values), 6),
            "gap_mean": round(np.mean(gap_values), 6),
            "gap_std": round(np.std(gap_values), 6),
            "checkpoints_skipped": SKIP_CHECKPOINTS,
            "note": "Checkpoints (~116MB each) excluded from git archive. See manifest for on-disk paths and SHA256.",
        },
    }
    (ARCHIVE_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Copy EXPERIMENT_SUMMARY.md
    shutil.copy(EXPERIMENT_SUMMARY, ARCHIVE_DIR / "EXPERIMENT_SUMMARY.md")

    manifest = {
        "archive_version": "1.0",
        "experiment_id": "dct_v3_3_score_first_5fold",
        "created": subprocess.check_output("date -Iseconds", shell=True, text=True).strip(),
        "host": subprocess.check_output("hostname", shell=True, text=True).strip(),
        "git_commit_head": run("git rev-parse HEAD", cwd=PROJECT_ROOT).strip(),
        "contents": {
            "01_git": "Git commit logs, diff, status",
            "02_config": "YAML config, per-fold settings, split SHA256, args.json",
            "03_results": "Per-fold epoch curves, logs, pkl files (checkpoints excluded — see paths below)",
            "04_metadata": "pip freeze, GPU info, env info",
            "EXPERIMENT_SUMMARY.md": "Full experiment summary (copy)",
            "summary.json": "Aggregate statistics",
            "manifest.json": "This file",
        },
        "fold_summaries": fold_summaries,
        "aggregate": summary["aggregate"],
        "checkpoint_locations": {
            str(f): {
                "path": str(find_result_dir(f) / f"model_best_s{f}.pth") if find_result_dir(f) else None,
                "sha256": fold_summaries.get(f, {}).get("checkpoint_best_sha256"),
            }
            for f in FOLDS
        },
        "training_commands": {
            "fold_0_1_3_4": (
                "cd /home/ubuntu/SurvOT-Rank && "
                "PYTHON_BIN=/home/ubuntu/.conda/envs/trisurv/bin/python "
                "FOLDS=0,1,2,3,4 bash scripts/run_dct_v3_score_diagnostics.sh run full"
            ),
            "fold_2_nan_fix": (
                "cd /home/ubuntu/SurvOT-Rank && "
                "/home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train "
                "--config configs/distributional_counterfactual_transport_blca.yaml "
                "--set k_start=2 --set k_end=3 "
                "--set results_dir=results/dct_v3_3_fold2_nan_fix "
                "--set specific_simple=dct_v3_3_fold2_nan_fix"
            ),
        },
    }
    (ARCHIVE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- 6. NaN audit -----------------------------------------------------
    print("[6/7] NaN audit ...")
    nan_report = {"folds_checked": len(all_curves), "total_rows": 0, "nan_cells": 0, "details": []}
    for fold, rows in all_curves.items():
        for r in rows:
            nan_report["total_rows"] += 1
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    nan_report["nan_cells"] += 1
                    nan_report["details"].append({"fold": fold, "epoch": r["epoch"], "column": k})
    (ARCHIVE_DIR / "nan_audit.json").write_text(json.dumps(nan_report, indent=2))
    print(f"  NaN cells found: {nan_report['nan_cells']} / {nan_report['total_rows']} rows")

    # ---- 7. archive -------------------------------------------------------
    print("[7/7] Creating tar.gz archive ...")
    archive_path = PROJECT_ROOT / "reproducibility_archives" / "dct_v3_3_score_first_5fold.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(ARCHIVE_DIR, arcname="dct_v3_3_score_first_5fold")

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"\n=== DONE ===")
    print(f"  Archive: {archive_path}")
    print(f"  Size:    {size_mb:.1f} MB")
    print(f"  Best mean:  {summary['aggregate']['best_mean']}")
    print(f"  Last5 mean: {summary['aggregate']['last5_mean']}")
    print(f"  Gap mean:   {summary['aggregate']['gap_mean']}")
    print(f"  NaN cells:  {nan_report['nan_cells']}")
    if SKIP_CHECKPOINTS:
        print(f"  Note: checkpoints excluded (~{CHECKPOINT_SIZE_MB}MB each), SHA256 recorded in manifest")


if __name__ == "__main__":
    main()
