#!/bin/bash
# DCT v3.6 + v3.7 联合筛选启动脚本
# v3.6: 4 cancers x 6 variants x fold 0,2 = 48 jobs
# v3.7: 10 cancers x highscore x fold 0-4 = 50 jobs
# 总计: 98 jobs，单 GPU 串行

set -euo pipefail
cd "$(dirname "$0")/.."

LOG="logs/dct_v36_v37_screen.log"
PYTHON="${PYTHON_BIN:-/home/ubuntu/.conda/envs/trisurv/bin/python3}"
GPU="${GPU:-0}"
WORKERS="${NUM_WORKERS:-4}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# === 等待 Recovery 任务全部结束 ===
log "等待 Recovery 任务完成..."
RECOVERY_DONE=0
while [ "$RECOVERY_DONE" -eq 0 ]; do
    running=$(ps -eo args --no-headers 2>/dev/null | grep -c "dct_brca_recovery" | grep -v grep || true)
    if [ "$running" -eq 0 ]; then
        RECOVERY_DONE=1
        log "Recovery 全部结束。"
    else
        sleep 30
    fi
done

# === 拉取最新代码 ===
log "拉取最新代码..."
git pull --ff-only origin main 2>&1 | tee -a "$LOG"

# === DCT v3.6 smoke ===
log "===== DCT v3.6 Listwise Smoke ====="
$PYTHON scripts/run_dct_v36_listwise_screen.py smoke \
    --cancers blca,brca \
    --variants all \
    --folds 0,2 \
    --gpu "$GPU" \
    --num-workers "$WORKERS" \
    2>&1 | tee -a "$LOG"

# === DCT v3.6 全量运行 ===
log "===== DCT v3.6 Listwise Full Run ====="
log "4 cancers x 6 variants x 2 folds = 48 jobs"
$PYTHON scripts/run_dct_v36_listwise_screen.py run \
    --cancers all \
    --variants all \
    --folds 0,2 \
    --gpu "$GPU" \
    --num-workers "$WORKERS" \
    2>&1 | tee -a "$LOG"

# === DCT v3.7 smoke ===
log "===== DCT v3.7 UNI2-h Smoke ====="
$PYTHON scripts/run_dct_v37_uni2h_screen.py smoke \
    --cancers blca,brca \
    --variants highscore \
    --folds 0,2 \
    --gpu "$GPU" \
    --num-workers "$WORKERS" \
    2>&1 | tee -a "$LOG"

# === DCT v3.7 全量运行 ===
log "===== DCT v3.7 UNI2-h Full Run ====="
log "10 cancers x highscore x 5 folds = 50 jobs"
$PYTHON scripts/run_dct_v37_uni2h_screen.py run \
    --variants highscore \
    --folds 0,1,2,3,4 \
    --gpu "$GPU" \
    --num-workers "$WORKERS" \
    2>&1 | tee -a "$LOG"

log "===== 全部完成 ====="
