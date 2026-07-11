#!/bin/bash
# Sequential batch runner for all SurvOT-Rank experiments.
#
# Usage:
#   bash scripts/run_batch.sh                # run every experiment from scratch
#   bash scripts/run_batch.sh --resume      # skip experiments whose .log already
#                                            # shows a final [Fold 4] epoch line
#   bash scripts/run_batch.sh --start-from <name>   # skip everything up to <name>
#
# Unlike the previous version this script does NOT use `set -e`, so a single
# failing experiment (e.g. a missing 5-fold split dir) is recorded as FAILED
# and the batch keeps going.

set -u

PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
PROJECT_DIR="/home/ubuntu/SurvOT-Rank"
LOG_DIR="$PROJECT_DIR/results/batch_runs/$(date +%Y%m%d_%H%M%S)"

RESUME=0
START_FROM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --resume) RESUME=1; shift ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0 ;;
        *) echo "[run_batch] unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "SurvOT-Rank Batch Experiment Runner"
echo "=========================================="
echo "Started: $(date)"
echo "Log dir: $LOG_DIR"
echo "Resume:  $RESUME  Start-from: ${START_FROM:-<none>}"
echo ""

EXPERIMENTS=(
    "configs/v45_blca.yaml:v45_blca"
    "configs/v45_brca.yaml:v45_brca"
    "configs/v45_stad.yaml:v45_stad"
    "configs/v45_coadread.yaml:v45_coadread"
    "configs/v45_hnsc.yaml:v45_hnsc"
    "configs/v45_best_blca.yaml:v45_best_blca"
    "configs/ablation/abl_00_baseline.yaml:abl_00_baseline"
    "configs/ablation/abl_01_clinical.yaml:abl_01_clinical"
    "configs/ablation/abl_02_unified.yaml:abl_02_unified"
    "configs/ablation/abl_03_disentangle.yaml:abl_03_disentangle"
    "configs/ablation/abl_04_sinkhorn.yaml:abl_04_sinkhorn"
    "configs/ablation/abl_05_crossmodal.yaml:abl_05_crossmodal"
    "configs/ablation/abl_06_adaptive_iters.yaml:abl_06_adaptive_iters"
    "configs/ablation/abl_07_learnable_weights.yaml:abl_07_learnable_weights"
    "configs/ablation/abl_08_all_on.yaml:abl_08_all_on"
    "configs/ablation/abl_09_all_on_learnable.yaml:abl_09_all_on_learnable"
)

RESULTS_FILE="$LOG_DIR/results.txt"
{
    echo "Batch Run Results"
    echo "Started: $(date)"
    echo "=========================================="
    echo ""
} > "$RESULTS_FILE"

is_completed() {
    # A log file is "completed" if it has a [Fold 4] section AND a final
    # [Epoch 29] val line. Used by --resume.
    local log="$1"
    [ -f "$log" ] || return 1
    grep -qE "^\[Fold 4\] start" "$log" 2>/dev/null || return 1
    grep -qE "^\[Epoch 29\] val " "$log" 2>/dev/null || return 1
    return 0
}

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0
FAILED=0
SKIPPED=0
SKIPPING_UNTIL=""
if [ -n "$START_FROM" ]; then
    SKIPPING_UNTIL="$START_FROM"
fi

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config name <<< "$exp"
    CURRENT=$((CURRENT + 1))
    LOG_FILE="$LOG_DIR/${name}.log"

    # Honour --start-from <name>: skip every experiment until we reach it.
    if [ -n "$SKIPPING_UNTIL" ] && [ "$name" != "$SKIPPING_UNTIL" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$CURRENT/$TOTAL] $name: SKIPPED (before start-from '$START_FROM')"
        echo "[$CURRENT/$TOTAL] $name: SKIPPED (before start-from '$START_FROM')" \
            >> "$RESULTS_FILE"
        continue
    else
        SKIPPING_UNTIL=""
    fi

    # Honour --resume: skip experiments whose log already shows a complete run.
    if [ "$RESUME" = "1" ] && is_completed "$LOG_FILE"; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$CURRENT/$TOTAL] $name: SKIPPED (already completed in this log dir)"
        echo "[$CURRENT/$TOTAL] $name: SKIPPED (already completed in this log dir)" \
            >> "$RESULTS_FILE"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Running: $name"
    echo "=========================================="

    cd "$PROJECT_DIR"
    # `|| true` so non-zero exits never tear down the script even if `set -e`
    # is re-enabled by anything upstream; we explicitly read $? below.
    $PYTHON -m survot_rank.cli train --config "$config" \
        > "$LOG_FILE" 2>&1 || EXIT_CODE=$?
    EXIT_CODE=${EXIT_CODE:-0}

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$CURRENT/$TOTAL] $name: SUCCESS" | tee -a "$RESULTS_FILE"
        if grep -q "val cindex" "$LOG_FILE"; then
            METRICS=$(tail -20 "$LOG_FILE" | tr '\r' '\n' | grep "val cindex" | tail -1)
            echo "  Final: $METRICS" | tee -a "$RESULTS_FILE"
        fi
    else
        echo "[$CURRENT/$TOTAL] $name: FAILED (exit $EXIT_CODE)" | tee -a "$RESULTS_FILE"
        if grep -q "AssertionError" "$LOG_FILE"; then
            LAST_ERR=$(grep -m1 -E "assert .*\)" "$LOG_FILE" || true)
            [ -n "$LAST_ERR" ] && echo "  Reason: $LAST_ERR" | tee -a "$RESULTS_FILE"
        fi
        FAILED=$((FAILED + 1))
    fi
    # Make sure EXIT_CODE never carries into the next iteration.
    unset EXIT_CODE
done

echo ""
echo "=========================================="
echo "Batch Complete!"
echo "=========================================="
echo "Total: $TOTAL | Failed: $FAILED | Skipped: $SKIPPED"
echo "Results saved to: $RESULTS_FILE"
echo "Logs saved to: $LOG_DIR"
echo ""

cat "$RESULTS_FILE"