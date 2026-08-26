#!/usr/bin/env python3
import subprocess
import os
import sys
from pathlib import Path

CANCERS = ["brca", "luad", "lusc", "blca", "skcm"]
GPU = "0"
NUM_WORKERS = "4"
PYTHON_BIN = "/home/ubuntu/.conda/envs/trisurv/bin/python3"
REPO_ROOT = Path(__file__).resolve().parent.parent

processes = []

for cancer in CANCERS:
    log_file = REPO_ROOT / "logs" / f"dct_v35_r_{cancer}.log"
    cmd = [
        PYTHON_BIN, "-u", str(REPO_ROOT / "scripts" / "run_dct_v35_screen.py"),
        "run", "--variants", "r", "--cancers", cancer, "--folds", "0,2",
        "--gpu", GPU, "--num-workers", NUM_WORKERS
    ]
    print(f"Starting {cancer.upper()} -> {log_file}")
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=f)
    processes.append((cancer, p))

print(f"\nStarted {len(processes)} parallel training processes.")
print("Use 'nvidia-smi' to monitor GPU usage.")
print("Use 'tail -f logs/dct_v35_r_{cancer}.log' to monitor individual progress.")
