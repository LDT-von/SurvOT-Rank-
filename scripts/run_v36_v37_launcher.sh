#!/bin/bash
set -e
cd /home/ubuntu/SurvOT-Rank
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
LOG_DIR="logs/v36_v37_parallel"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
GPU=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "Phase 1/4: v3.6 smoke (blca,brca fold0,2)..."
$PY scripts/run_dct_v36_listwise_screen.py smoke --variants all --cancers blca,brca --folds 0,2 --gpu $GPU
log "v3.6 smoke done"

log "Phase 2/4: v3.6 full (3路并行: blca | brca | luad+lusc)..."
$PY scripts/run_dct_v36_listwise_screen.py run --variants all --cancers blca --folds 0,2 --gpu $GPU > "$LOG_DIR/v36_blca.log" 2>&1 &
P1=$!
$PY scripts/run_dct_v36_listwise_screen.py run --variants all --cancers brca --folds 0,2 --gpu $GPU > "$LOG_DIR/v36_brca.log" 2>&1 &
P2=$!
$PY scripts/run_dct_v36_listwise_screen.py run --variants all --cancers luad,lusc --folds 0,2 --gpu $GPU > "$LOG_DIR/v36_luad_lusc.log" 2>&1 &
P3=$!
log "v3.6 PIDs: $P1 $P2 $P3"
wait $P1 $P2 $P3
log "v3.6 full done"

log "Phase 3/4: v3.7 smoke (blca,brca fold0,2)..."
$PY scripts/run_dct_v37_uni2h_screen.py smoke --variants highscore --cancers blca,brca --folds 0,2 --gpu $GPU
log "v3.7 smoke done"

log "Phase 4/4: v3.7 full 5-fold (3路并行)..."
$PY scripts/run_dct_v37_uni2h_screen.py run --variants highscore --cancers blca,brca,luad --folds 0,1,2,3,4 --gpu $GPU > "$LOG_DIR/v37_g1.log" 2>&1 &
P1=$!
$PY scripts/run_dct_v37_uni2h_screen.py run --variants highscore --cancers lusc,skcm,coadread,kirc --folds 0,1,2,3,4 --gpu $GPU > "$LOG_DIR/v37_g2.log" 2>&1 &
P2=$!
$PY scripts/run_dct_v37_uni2h_screen.py run --variants highscore --cancers ucec,hnsc,stad --folds 0,1,2,3,4 --gpu $GPU > "$LOG_DIR/v37_g3.log" 2>&1 &
P3=$!
log "v3.7 PIDs: $P1 $P2 $P3"
wait $P1 $P2 $P3
log "v3.7 full done"

log "===== ALL DONE ====="
