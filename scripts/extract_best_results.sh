#!/bin/bash
# Extract per-fold best validation metrics from the latest batch run.
# Usage:
#   bash scripts/extract_best_results.sh                # latest batch dir
#   bash scripts/extract_best_results.sh <run_dir>      # explicit dir

set -euo pipefail

PROJECT_DIR="/home/ubuntu/SurvOT-Rank"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"

if [ "${1:-}" != "" ]; then
    RUN_DIR="$1"
else
    BATCH_ROOT="$PROJECT_DIR/results/batch_runs"
    if ! ls -d "$BATCH_ROOT"/[0-9]* >/dev/null 2>&1; then
        echo "[extract_best_results] no batch_runs/<timestamp>/ directory under $BATCH_ROOT" >&2
        exit 1
    fi
    RUN_DIR=$(ls -1dt "$BATCH_ROOT"/[0-9]* | head -1)
fi

echo "Run dir: $RUN_DIR"
exec "$PYTHON" "$PROJECT_DIR/scripts/extract_best_results.py" --run-dir "$RUN_DIR"