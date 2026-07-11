#!/bin/bash
# Batch run all experiments sequentially
# Usage: bash scripts/run_all_experiments.sh

set -e  # Exit on error

# Use trisurv environment which has torch and compatible numpy
export PATH="/home/ubuntu/.conda/envs/trisurv/bin:$PATH"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/results/batch_runs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

log_info "Starting batch experiments at $(date)"
log_info "Log directory: $LOG_DIR"

# ============================================
# V45 Base Experiments (6 configs)
# ============================================
V45_CONFIGS=(
    "configs/v45_blca.yaml"
    "configs/v45_brca.yaml"
    "configs/v45_stad.yaml"
    "configs/v45_coadread.yaml"
    "configs/v45_hnsc.yaml"
    "configs/v45_best_blca.yaml"
)

# ============================================
# Ablation Experiments (10 configs)
# ============================================
ABL_CONFIGS=(
    "configs/ablation/abl_00_baseline.yaml"
    "configs/ablation/abl_01_clinical.yaml"
    "configs/ablation/abl_02_unified.yaml"
    "configs/ablation/abl_03_disentangle.yaml"
    "configs/ablation/abl_04_sinkhorn.yaml"
    "configs/ablation/abl_05_crossmodal.yaml"
    "configs/ablation/abl_06_adaptive_iters.yaml"
    "configs/ablation/abl_07_learnable_weights.yaml"
    "configs/ablation/abl_08_all_on.yaml"
    "configs/ablation/abl_09_all_on_learnable.yaml"
)

# All configs
ALL_CONFIGS=("${V45_CONFIGS[@]}" "${ABL_CONFIGS[@]}")

TOTAL=${#ALL_CONFIGS[@]}
CURRENT=0

# Results tracking
RESULTS_FILE="$LOG_DIR/results_summary.txt"
echo "Batch Run Results Summary" > "$RESULTS_FILE"
echo "Started: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Function to run a single experiment
run_experiment() {
    local config="$1"
    local exp_name=$(basename "$config" .yaml)
    local log_file="$LOG_DIR/${exp_name}.log"
    
    CURRENT=$((CURRENT + 1))
    local progress="[$CURRENT/$TOTAL]"
    
    log_info "$progress Running: $exp_name"
    
    # Check if results already exist
    local results_dir=$(grep -A2 "^train:" "$config" | grep "results_dir:" | awk '{print $2}')
    if [ -d "$PROJECT_DIR/$results_dir" ]; then
        # Check if we have all 5 folds
        if [ -f "$PROJECT_DIR/$results_dir"/*/SURVIVAL_LOG.csv ] 2>/dev/null || \
           [ -f "$PROJECT_DIR/$results_dir"/*/*/SURVIVAL_LOG.csv ] 2>/dev/null; then
            log_warn "$progress Skipping $exp_name (results exist)"
            echo "$progress $exp_name: SKIPPED (results exist)" >> "$RESULTS_FILE"
            return 0
        fi
    fi
    
    # Run the experiment
    local start_time=$(date +%s)
    
    set +e  # Don't exit on error within experiment
    $PYTHON -m survot_rank.cli train --config "$config" 2>&1 | tee "$log_file"
    local exit_code=$?
    set -e
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    
    if [ $exit_code -eq 0 ]; then
        log_success "$progress Completed: $exp_name (${duration_min} min)"
        echo "$progress $exp_name: SUCCESS (${duration_min} min)" >> "$RESULTS_FILE"
    else
        log_error "$progress Failed: $exp_name (exit code: $exit_code)"
        echo "$progress $exp_name: FAILED (exit code: $exit_code)" >> "$RESULTS_FILE"
    fi
    
    return $exit_code
}

# Main loop
log_info "========================================"
log_info "Starting $TOTAL experiments..."
log_info "========================================"

failed=0
for config in "${ALL_CONFIGS[@]}"; do
    run_experiment "$config" || failed=$((failed + 1))
done

# Final summary
log_info "========================================"
log_info "Batch run completed at $(date)"
log_info "========================================"

if [ $failed -eq 0 ]; then
    log_success "All $TOTAL experiments completed successfully!"
else
    log_warn "$failed experiment(s) failed. Check $LOG_DIR for details."
fi

echo "" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "Completed: $(date)" >> "$RESULTS_FILE"
echo "Total: $TOTAL, Failed: $failed" >> "$RESULTS_FILE"

cat "$RESULTS_FILE"

# Keep terminal open for viewing
echo ""
echo "Log files saved to: $LOG_DIR"
