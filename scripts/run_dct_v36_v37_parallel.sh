#!/bin/bash
# DCT v3.6 + v3.7 并行启动脚本
# 按癌种分担到独立进程，同 GPU 0 并行，每批最多 3 个进程
# v3.6: 4 cancers × 6 variants × fold 0,2   → 最多 3 并发，约 4h
# v3.7: 10 cancers × highscore × fold 0-4    → 最多 3 并发，约 7h
# 总计: ~11h（比串行 57h 快 5 倍）

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON_BIN:-/home/ubuntu/.conda/envs/trisurv/bin/python3}"
GPU="${GPU:-0}"
WORKERS=2                    # 并行时降低 worker 数避免 CPU 过载
MAX_PARALLEL=3               # 同时最多 3 个训练进程 (~6GB VRAM / 32GB)

LOG_DIR="logs/v36_v37_parallel"
mkdir -p "$LOG_DIR"

log() {
    local ts
    ts=$(date '+%H:%M:%S')
    printf "[%s] %s\n" "$ts" "$*" | tee -a "${LOG_DIR}/master.log"
}

# ── V3.6 癌种 (4) ──
V36_CANCERS=(blca brca luad lusc)
# ── V3.7 癌种 (10) ──
V37_CANCERS=(blca brca luad lusc skcm coadread kirc ucec hnsc stad)

# ── 等待 Recovery 结束 ──
wait_recovery() {
    log "等待 Recovery (bin_legacy) 完成..."
    while ps -eo args --no-headers 2>/dev/null | grep -q "dct_brca_recovery"; do
        sleep 30
    done
    log "Recovery 全部结束。"
}

# ── 串行 smoke ──
run_smokes() {
    log "===== V3.6 Smoke (blca,brca) ====="
    $PYTHON scripts/run_dct_v36_listwise_screen.py smoke \
        --cancers blca,brca --variants all --folds 0,2 \
        --gpu "$GPU" --num-workers "$WORKERS" \
        2>&1 | tee -a "${LOG_DIR}/master.log"

    log "===== V3.7 Smoke (blca,brca highscore) ====="
    $PYTHON scripts/run_dct_v37_uni2h_screen.py smoke \
        --cancers blca,brca --variants highscore --folds 0,2 \
        --gpu "$GPU" --num-workers "$WORKERS" \
        2>&1 | tee -a "${LOG_DIR}/master.log"
}

# ── 单癌种 v3.6 训练 ──
run_v36_cancer() {
    local cancer="$1"
    log "[v3.6] $cancer 开始 (6 variants × 2 folds)"
    $PYTHON scripts/run_dct_v36_listwise_screen.py run \
        --cancers "$cancer" --variants all --folds 0,2 \
        --gpu "$GPU" --num-workers "$WORKERS" \
        > "${LOG_DIR}/v36_${cancer}.log" 2>&1
    log "[v3.6] $cancer 完成"
}

# ── 单癌种 v3.7 训练 ──
run_v37_cancer() {
    local cancer="$1"
    log "[v3.7] $cancer 开始 (1 variant × 5 folds)"
    $PYTHON scripts/run_dct_v37_uni2h_screen.py run \
        --cancers "$cancer" --variants highscore --folds 0,1,2,3,4 \
        --gpu "$GPU" --num-workers "$WORKERS" \
        > "${LOG_DIR}/v37_${cancer}.log" 2>&1
    log "[v3.7] $cancer 完成"
}

# ── 并行批处理（限 MAX_PARALLEL 并发）──
run_parallel() {
    local runner_func="$1"
    shift
    local cancers=("$@")
    local running=0

    for cancer in "${cancers[@]}"; do
        # 等待空位
        while [ "$running" -ge "$MAX_PARALLEL" ]; do
            wait -n 2>/dev/null && running=$((running - 1)) || running=$((running - 1))
        done

        "$runner_func" "$cancer" &
        running=$((running + 1))
        sleep 2  # 错开启动，避免同时抢 CUDA 初始化
    done

    # 等待全部完成
    wait
    log "批次全部完成。"
}

# ── 统计完成的 epoch ──
show_progress() {
    while true; do
        sleep 120
        log "─── 进度快照 ───"
        for f in "${LOG_DIR}"/v3[67]_*.log; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .log)
            ep=$(grep -oP 'Epoch \d+/\d+' "$f" 2>/dev/null | tail -1 || echo "waiting")
            best=$(grep "best cindex" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "-")
            printf "  %-16s  %-16s  best=%-8s\n" "$name" "$ep" "$best" | tee -a "${LOG_DIR}/master.log"
        done
        pgrep -f "survot_rank.cli train" > /dev/null 2>&1 || break
    done
}

# ═══════════════════════════════════════
# 主流程
# ═══════════════════════════════════════

log "===== DCT v3.6 + v3.7 并行启动 ====="
log "GPU=$GPU  Workers=$WORKERS  Max并行=$MAX_PARALLEL"
log "v3.6: ${#V36_CANCERS[@]} 癌种 × 6 variants × 2 folds"
log "v3.7: ${#V37_CANCERS[@]} 癌种 × 1 variant × 5 folds"

wait_recovery

log "拉取最新代码..."
git pull --ff-only origin main 2>&1 | tee -a "${LOG_DIR}/master.log"

run_smokes

# 后台进度监控
show_progress &
PROGRESS_PID=$!

log "===== V3.6 并行 ($MAX_PARALLEL 并发) ====="
run_parallel run_v36_cancer "${V36_CANCERS[@]}"

log "===== V3.7 并行 ($MAX_PARALLEL 并发) ====="
run_parallel run_v37_cancer "${V37_CANCERS[@]}"

kill $PROGRESS_PID 2>/dev/null || true

log "===== 全部完成 ====="
log "日志: ${LOG_DIR}/"
log "master: ${LOG_DIR}/master.log"
log "v3.6: ${LOG_DIR}/v36_*.log"
log "v3.7: ${LOG_DIR}/v37_*.log"
