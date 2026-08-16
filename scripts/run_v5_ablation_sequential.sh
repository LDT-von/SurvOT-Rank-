#!/usr/bin/env bash
# Wait for v5.1 BLCA 5-fold to complete, then auto-start v5.2.
# v5.1 will write `v5_1_blca_5fold.log` (it stops on its own when last fold
# finishes); once we see the "Done" sentinel + no python3 process for v5.1,
# we kick off v5.2 and exit.
set -euo pipefail

LOG_DIR=/data1/SurvOT-Rank/logs
mkdir -p "$LOG_DIR"
cd /data1/SurvOT-Rank

V51_PID=""
V52_PID=""

cleanup() {
    echo "$(date -Iseconds) wrapper exiting; v5.1=$V51_PID v5.2=$V52_PID"
}
trap cleanup EXIT

echo "$(date -Iseconds) launching v5.1 BLCA 5-fold (variant=v5_1)"
PATH=/home/ubuntu/.conda/envs/trisurv/bin:$PATH \
    python3 scripts/run_act_surv_v5.py \
        --cancers blca --variant v5_1 --folds 0 1 2 3 4 \
        > "$LOG_DIR/v5_1_blca_5fold.log" 2>&1 &
V51_PID=$!
echo "$(date -Iseconds) v5.1 PID=$V51_PID"

# Poll for completion
while kill -0 "$V51_PID" 2>/dev/null; do
    sleep 60
    if [[ -f "$LOG_DIR/v5_1_blca_5fold.log" ]] \
        && grep -qE "Total jobs: 0|Done all folds|FAILED.*fold [0-9]+" \
            "$LOG_DIR/v5_1_blca_5fold.log"; then
        break
    fi
done

# Wait for v5.1 process to actually exit (give it a moment to flush)
wait "$V51_PID" 2>/dev/null || true
echo "$(date -Iseconds) v5.1 completed"

# Quick sanity: did all 5 folds complete?
if grep -q "FAILED" "$LOG_DIR/v5_1_blca_5fold.log"; then
    echo "$(date -Iseconds) v5.1 had FAILED lines — aborting v5.2 launch"
    exit 1
fi
if ! grep -qE "fold4.*Done|Fold 4.*complete" "$LOG_DIR/v5_1_blca_5fold.log"; then
    echo "$(date -Iseconds) v5.1 did NOT reach fold 4 — aborting v5.2 launch"
    exit 1
fi

echo "$(date -Iseconds) launching v5.2 BLCA 5-fold (variant=v5_2)"
PATH=/home/ubuntu/.conda/envs/trisurv/bin:$PATH \
    python3 scripts/run_act_surv_v5.py \
        --cancers blca --variant v5_2 --folds 0 1 2 3 4 \
        > "$LOG_DIR/v5_2_blca_5fold.log" 2>&1 &
V52_PID=$!
echo "$(date -Iseconds) v5.2 PID=$V52_PID"

# Wait for v5.2 to finish (blocking so the wrapper stays alive during GPU work)
wait "$V52_PID" 2>/dev/null || true
echo "$(date -Iseconds) v5.2 completed"