#!/bin/bash
# v3.3 Score-First: 10 癌种 × 5-fold = 50 jobs, 3路GPU并行
# 协议: UNI v1, NLL + 0.1 IPCW, global qcut, gaussian slots, alpha=0.15

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

MASTER_LOG="$ROOT/logs/v33_all_cancers.log"
mkdir -p "$ROOT/logs"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# ── 癌种 & config 映射 ──
declare -A CONFIGS
CONFIGS=(
    [blca]="configs/distributional_counterfactual_transport_blca.yaml"
    [brca]="configs/distributional_counterfactual_transport_brca.yaml"
    [coadread]="configs/distributional_counterfactual_transport_coadread.yaml"
    [hnsc]="configs/distributional_counterfactual_transport_hnsc.yaml"
    [kirc]="configs/distributional_counterfactual_transport_kirc.yaml"
    [luad]="configs/distributional_counterfactual_transport_luad.yaml"
    [lusc]="configs/distributional_counterfactual_transport_lusc.yaml"
    [skcm]="configs/distributional_counterfactual_transport_skcm.yaml"
    [stad]="configs/distributional_counterfactual_transport_stad.yaml"
    [ucec]="configs/distributional_counterfactual_transport_ucec.yaml"
)

CANCERS=(blca brca coadread hnsc kirc luad lusc skcm stad ucec)
FOLDS=(0 1 2 3 4)
GPU=0  # single GPU with parallel processes

# ── Python 环境 ──
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

doctor() {
    log "=== DOCTOR ==="
    $PYTHON_BIN -m survot_rank.cli doctor
    log "DOCTOR DONE"
}

smoke() {
    log "=== SMOKE (1 epoch, blca fold0) ==="
    $PYTHON_BIN -m survot_rank.cli train \
        --config "${CONFIGS[blca]}" \
        --set "gpu=$GPU" \
        --set "num_workers=4" \
        --set "max_epochs=1" \
        --set "folds=[0]" \
        --set "exp_name=dct_v33_all_smoke"
    log "SMOKE DONE"
}

run_fold() {
    local cancer="$1" fold="$2"
    local config="${CONFIGS[$cancer]}"
    local logfile="$ROOT/logs/v33_${cancer}_fold${fold}.log"

    log "  start ${cancer} fold${fold}"
    $PYTHON_BIN -m survot_rank.cli train \
        --config "$config" \
        --set "gpu=$GPU" \
        --set "num_workers=4" \
        --set "folds=[$fold]" \
        --set "exp_name=dct_v33_all" \
        > "$logfile" 2>&1
    local rc=$?

    # extract best
    local best="N/A"
    best=$(grep -oP 'best cindex:?\s*\K[0-9.]+' "$logfile" 2>/dev/null | tail -1 || echo "N/A")
    log "  done  ${cancer} fold${fold}  best=${best}  rc=${rc}"
    return $rc
}

# ── 主流程 ──
case "${1:-run}" in
    doctor)
        doctor
        ;;
    smoke)
        doctor
        smoke
        ;;
    run)
        log "=============================="
        log "v3.3 ALL CANCERS 5-FOLD"
        log "10 cancers × 5 folds = 50 jobs"
        log "3-way GPU parallel"
        log "=============================="

        # 构建所有 (cancer, fold) 任务
        TASKS=()
        for cancer in "${CANCERS[@]}"; do
            for fold in "${FOLDS[@]}"; do
                TASKS+=("$cancer $fold")
            done
        done
        TOTAL=${#TASKS[@]}

        # 3路并行的轮询队列
        MAX_PARALLEL=3
        RUNNING=0
        COMPLETED=0
        FAILED=0
        PIDS=()

        for task in "${TASKS[@]}"; do
            # 等有空闲 slot
            while [ $RUNNING -ge $MAX_PARALLEL ]; do
                # 检查是否有完成的
                for i in "${!PIDS[@]}"; do
                    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                        wait "${PIDS[$i]}" && ((COMPLETED++)) || ((FAILED++))
                        unset "PIDS[$i]"
                        ((RUNNING--))
                    fi
                done
                [ $RUNNING -ge $MAX_PARALLEL ] && sleep 5
            done

            read -r cancer fold <<< "$task"
            log "[$((COMPLETED + RUNNING + 1))/$TOTAL] launching ${cancer} fold${fold} (running=${RUNNING})"
            run_fold "$cancer" "$fold" &
            PIDS+=($!)
            ((RUNNING++))
        done

        # 等待所有完成
        for pid in "${PIDS[@]}"; do
            [ -n "$pid" ] && wait "$pid" && ((COMPLETED++)) || ((FAILED++))
        done

        log "=============================="
        log "ALL DONE — ${COMPLETED} completed, ${FAILED} failed out of ${TOTAL}"
        log "=============================="
        ;;
    *)
        echo "Usage: $0 {doctor|smoke|run}"
        exit 1
        ;;
esac
