#!/bin/bash
# ==============================================================================
# 一键验证脚本：验证 "eps 间断 bug + batch=4 早峰" 两处修复是否生效
#
# 修复内容（已在代码/配置中落地）：
#   1. Sinkhorn eps 调度改为单调（epoch0=0.10软 → 单调收紧到0.05），
#      消除 epoch0 假峰（RG-ET / SPT / FET / DCT / CATET 五个方法）。
#   2. 梯度累积 grad_accum_steps=8 → 有效 batch=32，稳住 NLL 梯度。
#   3. warmup_epochs=3 + grad_clip_norm=1.0 + opt=adamW + reg=5e-4。
#   4. 全程不早停，跑满 30 epoch（base config 无 early_stop_* 键）。
#
# 只跑 fold 0 和 fold 2（节省时间）。5 个方法 × 2 折 = 10 次运行。
#
# 用法:  bash scripts/run_fix_verify_fold02.sh
#
# 验证要点（跑完看 results/<method>/epoch_curve_fold{0,2}.csv）：
#   - train_cindex 应随 epoch 稳步上升到 0.6+（而不是塌到 0.4）
#   - val_cindex 峰值应后移到中后段（而不是 epoch 0-2）
# ==============================================================================

set -e

PROJECT_ROOT="/home/ubuntu/SurvOT-Rank"
CONDA_ENV="trisurv"
GPU=0

activate() {
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
}

run_experiment() {
    local config=$1
    local exp_name
    exp_name=$(basename "$config" .yaml)
    echo "=============================================="
    echo "[FIX-VERIFY] Running: $exp_name  (fold 0 + fold 2)"
    echo "=============================================="

    cd "$PROJECT_ROOT"
    activate

    # fold 0
    python -m survot_rank.cli train \
        --config "$config" \
        --set "gpu=$GPU" \
        --set "num_workers=4" \
        -- --k_start 0 --k_end 1

    # fold 2
    python -m survot_rank.cli train \
        --config "$config" \
        --set "gpu=$GPU" \
        --set "num_workers=4" \
        -- --k_start 2 --k_end 3

    echo "[FIX-VERIFY] Completed: $exp_name"
}

# 五个受 eps bug 影响、需要重跑验证的方法（base config，无早停）
run_experiment "configs/rank_guided_event_transport_blca.yaml"
run_experiment "configs/stagewise_prognostic_transport_blca.yaml"
run_experiment "configs/faithful_evidence_transport_blca.yaml"
run_experiment "configs/distributional_counterfactual_transport_blca.yaml"
run_experiment "configs/censoring_aware_temporal_evidence_transport_blca.yaml"

echo "=============================================="
echo "[FIX-VERIFY] All done!  5 方法 × 2 折 = 10 次运行"
echo "查看逐 epoch 曲线: results/<method>/epoch_curve_fold0.csv 和 _fold2.csv"
echo "=============================================="
