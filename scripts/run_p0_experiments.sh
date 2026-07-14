#!/bin/bash
# P0 实验一键运行脚本
# 只跑 fold 0 和 fold 2，各 30 epoch
# 用法: bash scripts/run_p0_experiments.sh

set -e

PROJECT_ROOT="/home/ubuntu/SurvOT-Rank"
CONDA_ENV="trisurv"
GPU=0

activate() {
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
}

run_experiment() {
    local config=$1
    local exp_name=$(basename "$config" .yaml)
    echo "=============================================="
    echo "[P0] Running: $exp_name"
    echo "=============================================="
    
    cd $PROJECT_ROOT
    activate
    
    # 只跑 fold 0 和 fold 2
    python -m survot_rank.cli train \
        --config "$config" \
        --set "gpu=$GPU" \
        --set "num_workers=4" \
        -- --k_start 0 --k_end 1  # fold 0
    
    python -m survot_rank.cli train \
        --config "$config" \
        --set "gpu=$GPU" \
        --set "num_workers=4" \
        -- --k_start 2 --k_end 3  # fold 2
    
    echo "[P0] Completed: $exp_name"
}

# P0-1: v45 全 8 损失 + 分箱 B 对照
run_experiment "configs/p0_experiments/v45_baseline_globalbin_blca.yaml"

# P0-2: v45_norank 固定 seed 复核
run_experiment "configs/p0_experiments/v45_norank_seed3_blca.yaml"
run_experiment "configs/p0_experiments/v45_norank_seed5_blca.yaml"

# P0-3: v50_norank 固定 seed 复核
run_experiment "configs/p0_experiments/v50_norank_seed3_blca.yaml"
run_experiment "configs/p0_experiments/v50_norank_seed5_blca.yaml"

# P0-4: v50 损失消融（四档：stripped → spec_only → spec_cover → full）
run_experiment "configs/p0_experiments/v50_ablation_only_ot_eventsurv_blca.yaml"  # stripped
run_experiment "configs/p0_experiments/v50_ablation_spec_only_blca.yaml"          # +spec
run_experiment "configs/p0_experiments/v50_ablation_spec_cover_blca.yaml"         # +cover
run_experiment "configs/p0_experiments/v50_ablation_full_blca.yaml"               # +compete (full)

echo "=============================================="
echo "[P0] All experiments completed!"
echo "共 18 次训练 × 2 折 = 36 次运行"
echo "=============================================="
