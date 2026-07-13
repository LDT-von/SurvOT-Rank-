#!/usr/bin/env bash
# 全量实验批量运行：5个跨癌种V45 + v45_best_blca + 10个消融
set -euo pipefail

GPU="${GPU:-0}"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_DIR="results/batch_runs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "[run_batch] === 开始全量实验批次 ${TIMESTAMP} ===" | tee "$LOG_DIR/progress.log"

# Phase 1: 跨癌种 V45 基线
PHASE1=(
  "configs/v45_blca.yaml"
  "configs/v45_brca.yaml"
  "configs/v45_stad.yaml"
  "configs/v45_coadread.yaml"
  "configs/v45_hnsc.yaml"
  "configs/v45_best_blca.yaml"
)

for cfg in "${PHASE1[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "===================================================================" | tee -a "$LOG_DIR/progress.log"
  echo "[run_batch] Phase1: $name (GPU=$GPU)" | tee -a "$LOG_DIR/progress.log"
  echo "===================================================================" | tee -a "$LOG_DIR/progress.log"
  $PYTHON -m survot_rank.cli train --config "$cfg" --set "gpu=$GPU" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  echo "[run_batch] $name done at $(date)" | tee -a "$LOG_DIR/progress.log"
done

# Phase 2: 消融实验
ABL_DIR="configs/ablation"
ABL_NAMES=(
  abl_00_baseline
  abl_01_clinical
  abl_02_unified
  abl_03_disentangle
  abl_04_sinkhorn
  abl_05_crossmodal
  abl_06_adaptive_iters
  abl_07_learnable_weights
  abl_08_all_on
  abl_09_all_on_learnable
)

for name in "${ABL_NAMES[@]}"; do
  cfg="$ABL_DIR/$name.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[run_batch] 跳过：$cfg 不存在" | tee -a "$LOG_DIR/progress.log"
    continue
  fi
  echo "===================================================================" | tee -a "$LOG_DIR/progress.log"
  echo "[run_batch] Phase2: $name (GPU=$GPU)" | tee -a "$LOG_DIR/progress.log"
  echo "===================================================================" | tee -a "$LOG_DIR/progress.log"
  $PYTHON -m survot_rank.cli train --config "$cfg" --set "gpu=$GPU" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  echo "[run_batch] $name done at $(date)" | tee -a "$LOG_DIR/progress.log"
done

echo "[run_batch] === 全量实验完成 === " | tee -a "$LOG_DIR/progress.log"
echo "[run_batch] 日志: $LOG_DIR" | tee -a "$LOG_DIR/progress.log"
