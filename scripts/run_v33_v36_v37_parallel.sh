#!/bin/bash
# ============================================================
# 三路并行: v3.3(UNI剩余6癌种) | v3.6(listwise 48jobs) | v3.7(UNI2-h 50jobs)
# 所有结果隔离到 results/{dct_v33_all,dct_v36_listwise,dct_v37_uni2h}/
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

LOG_DIR="$ROOT/logs"
USE_ENV="${USE_ENV:-trisurv}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/.conda/envs/trisurv/bin/python}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# verify torch available
verify_python() {
    if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
        log "ERROR: $PYTHON_BIN has no torch, trying auto-detect..."
        PYTHON_BIN="/home/ubuntu/.conda/envs/trisurv/bin/python"
        $PYTHON_BIN -c "import torch" 2>/dev/null || {
            log "FATAL: cannot find Python with torch"
            exit 1
        }
    fi
    log "PYTHON_BIN=$PYTHON_BIN"
}

# ══════════════════════════════════════
# 路1: v3.3 Score-First — 剩余6癌种 (UNI v1)
# ══════════════════════════════════════
route_v33() {
    local MASTER="$LOG_DIR/v33_remaining.log"
    log "=== ROUTE 1: v3.3 剩余6癌种 5-fold ===" | tee -a "$MASTER"

    declare -A CONFIGS
    CONFIGS[coadread]="configs/distributional_counterfactual_transport_coadread.yaml"
    CONFIGS[hnsc]="configs/distributional_counterfactual_transport_hnsc.yaml"
    CONFIGS[kirc]="configs/distributional_counterfactual_transport_kirc.yaml"
    CONFIGS[skcm]="configs/distributional_counterfactual_transport_skcm.yaml"
    CONFIGS[stad]="configs/distributional_counterfactual_transport_stad.yaml"
    CONFIGS[ucec]="configs/distributional_counterfactual_transport_ucec.yaml"

    for cancer in coadread hnsc kirc skcm stad ucec; do
        for fold in 0 1 2 3 4; do
            local cfg="${CONFIGS[$cancer]}"
            local lf="$LOG_DIR/v33_${cancer}_fold${fold}.log"
            log "  R1: start ${cancer} fold${fold}" | tee -a "$MASTER"
            $PYTHON_BIN -m survot_rank.cli train \
                --config "$cfg" \
                --set "gpu=0" \
                --set "num_workers=4" \
                --set "folds=[$fold]" \
                --set "exp_name=dct_v33_all" \
                > "$lf" 2>&1
            local best
            best=$(grep -oP 'best cindex:?\s*\K[0-9.]+' "$lf" 2>/dev/null | tail -1 || echo "ERR")
            log "  R1: done  ${cancer} fold${fold}  best=${best}" | tee -a "$MASTER"
        done
    done
    log "=== ROUTE 1 COMPLETE ===" | tee -a "$MASTER"
}

# ══════════════════════════════════════
# 路2: v3.6 Listwise Transport (新 UNI2-h 代码)
# ══════════════════════════════════════
route_v36() {
    local MASTER="$LOG_DIR/v36_listwise.log"
    log "=== ROUTE 2: v3.6 listwise 4癌种×6variants×2folds ===" | tee -a "$MASTER"

    local CANCERS="blca,brca,luad,lusc"
    local VARIANTS="all"
    local FOLDS="0,2"

    log "  R2: smoke check (blca,brca) ..." | tee -a "$MASTER"
    $PYTHON_BIN scripts/run_dct_v36_listwise_screen.py smoke --cancers blca,brca --variants "$VARIANTS" --folds "$FOLDS" \
        --python "$PYTHON_BIN" \
        >> "$MASTER" 2>&1 || { log "  R2: smoke FAILED" | tee -a "$MASTER"; return 1; }

    log "  R2: full run ($CANCERS)" | tee -a "$MASTER"
    $PYTHON_BIN scripts/run_dct_v36_listwise_screen.py run --cancers "$CANCERS" --variants "$VARIANTS" --folds "$FOLDS" \
        --python "$PYTHON_BIN" \
        >> "$MASTER" 2>&1
    log "=== ROUTE 2 COMPLETE ===" | tee -a "$MASTER"
}

# ══════════════════════════════════════
# 路3: v3.7 UNI2-h highscore (新代码, 10癌种 5-fold)
# ══════════════════════════════════════
route_v37() {
    local MASTER="$LOG_DIR/v37_uni2h.log"
    log "=== ROUTE 3: v3.7 UNI2-h 10癌种×5fold ===" | tee -a "$MASTER"

    log "  R3: smoke check (blca,brca) ..." | tee -a "$MASTER"
    $PYTHON_BIN scripts/run_dct_v37_uni2h_screen.py smoke --variants highscore --cancers blca,brca --folds 0,2 \
        --python "$PYTHON_BIN" \
        >> "$MASTER" 2>&1 || { log "  R3: smoke FAILED" | tee -a "$MASTER"; return 1; }

    log "  R3: full run highscore 5-fold" | tee -a "$MASTER"
    $PYTHON_BIN scripts/run_dct_v37_uni2h_screen.py run --variants highscore --folds 0,1,2,3,4 \
        --python "$PYTHON_BIN" \
        >> "$MASTER" 2>&1
    log "=== ROUTE 3 COMPLETE ===" | tee -a "$MASTER"
}

# ══════════════════════════════════════
# 主入口
# ══════════════════════════════════════
main() {
    verify_python
    log "=============================="
    log " 三路并行训练启动"
    log " R1: v3.3 UNI 剩余6癌种 5-fold (30 jobs)"
    log " R2: v3.6 listwise 4癌种×6×2 (48 jobs)"
    log " R3: v3.7 UNI2-h 10癌种 5-fold (50 jobs)"
    log "=============================="

    route_v33 &
    local pid1=$!

    route_v36 &
    local pid2=$!

    route_v37 &
    local pid3=$!

    log "R1 PID=$pid1  R2 PID=$pid2  R3 PID=$pid3"
    log "监控: tail -f logs/v33_remaining.log logs/v36_listwise.log logs/v37_uni2h.log"

    local failed=0
    wait $pid1 || { log "R1 FAILED"; ((failed++)); }
    wait $pid2 || { log "R2 FAILED"; ((failed++)); }
    wait $pid3 || { log "R3 FAILED"; ((failed++)); }

    log "=============================="
    log " ALL DONE — ${failed} routes failed"
    log "=============================="
}

main "$@"
