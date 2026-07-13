#!/usr/bin/env bash
# ============================================================
# 一键运行 V51 SlimBridge (newSlotSPE) + V60 OT Event Rank (SurvOT-Rank)
# 用法: bash scripts/run_v51_v60.sh [GPU]
# ============================================================
set -euo pipefail

PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
GPU="${1:-0}"
WORKERS=4
LOG_BASE="/data1/sweep_results_30ep/_logs"

mkdir -p "$LOG_BASE"

# ============================================
# Job 1: V51 SlimBridge (newSlotSPE)
# ============================================
echo "========================================"
echo " [1/2] V51 SlimBridge"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd /home/ubuntu/newSlotSPE
CUDA_VISIBLE_DEVICES="$GPU" SEEDS="3 5" bash run_v51_slimbridge.sh "$GPU" \
  >> "$LOG_BASE/v51_slimbridge_fold2.log" 2>&1
echo " Done:   $(date '+%Y-%m-%d %H:%M:%S')"

# ============================================
# Job 2: V60 OT Event Rank (SurvOT-Rank)
# ============================================
echo ""
echo "========================================"
echo " [2/2] V60 OT Event Rank"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd /home/ubuntu/SurvOT-Rank
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m survot_rank.cli train \
  --config configs/v60_ot_event_rank_blca.yaml \
  --set "gpu=0" --set "num_workers=$WORKERS" \
  -- --k_start 2 --k_end 3 \
  >> "$LOG_BASE/v60_ot_event_rank_fold2.log" 2>&1
echo " Done:   $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "===== V51 + V60 all done ====="
