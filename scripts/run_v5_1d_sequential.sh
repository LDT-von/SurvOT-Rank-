#!/usr/bin/env bash
# Wait for v5.3 BLCA 5-fold to complete, then auto-start v5.4.
# 1D ablation experiment: isolate the two components of v5.1's improvement.
#   v5.3: only remove IPCW ranking (lambda_rank: 0.10→0.00), rest = v5 baseline
#   v5.4: only KL×5 (lambda_balance: 0.01→0.05), rest = v5 baseline
set -euo pipefail

LOG_DIR=/data1/SurvOT-Rank/logs
mkdir -p "$LOG_DIR"
cd /data1/SurvOT-Rank

V53_PID=""
V54_PID=""

cleanup() {
    echo "$(date -Iseconds) wrapper exiting; v5.3=$V53_PID v5.4=$V54_PID"
}
trap cleanup EXIT

echo "$(date -Iseconds) launching v5.3 BLCA 5-fold (variant=v5_3, 1D: no IPCW ranking)"
PATH=/home/ubuntu/.conda/envs/trisurv/bin:$PATH \
    python3 scripts/run_act_surv_v5.py \
        --cancers blca --variant v5_3 --folds 0 1 2 3 4 \
        > "$LOG_DIR/v5_3_blca_5fold.log" 2>&1 &
V53_PID=$!
echo "$(date -Iseconds) v5.3 PID=$V53_PID"

# Poll for completion
while kill -0 "$V53_PID" 2>/dev/null; do
    sleep 60
    if [[ -f "$LOG_DIR/v5_3_blca_5fold.log" ]] \
        && grep -qE "Total jobs: 0|Done all folds|FAILED.*fold [0-9]+" \
            "$LOG_DIR/v5_3_blca_5fold.log"; then
        break
    fi
done

# Wait for v5.3 process to actually exit (give it a moment to flush)
wait "$V53_PID" 2>/dev/null || true
echo "$(date -Iseconds) v5.3 completed"

# Quick sanity: did all 5 folds complete?
if grep -q "FAILED" "$LOG_DIR/v5_3_blca_5fold.log"; then
    echo "$(date -Iseconds) v5.3 had FAILED lines — aborting v5.4 launch"
    exit 1
fi
if ! grep -qE "fold4.*Done|Fold 4.*complete" "$LOG_DIR/v5_3_blca_5fold.log"; then
    echo "$(date -Iseconds) v5.3 did NOT reach fold 4 — aborting v5.4 launch"
    exit 1
fi

echo "$(date -Iseconds) launching v5.4 BLCA 5-fold (variant=v5_4, 1D: KL balance x5)"
PATH=/home/ubuntu/.conda/envs/trisurv/bin:$PATH \
    python3 scripts/run_act_surv_v5.py \
        --cancers blca --variant v5_4 --folds 0 1 2 3 4 \
        > "$LOG_DIR/v5_4_blca_5fold.log" 2>&1 &
V54_PID=$!
echo "$(date -Iseconds) v5.4 PID=$V54_PID"

# Wait for v5.4 to finish (blocking so the wrapper stays alive during GPU work)
wait "$V54_PID" 2>/dev/null || true
echo "$(date -Iseconds) v5.4 completed"
