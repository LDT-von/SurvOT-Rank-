#!/usr/bin/env bash
# ============================================================
# P0 优先级实验一键顺序运行脚本（仅 fold0 + fold2，30ep，节省时间）
#
# 对应 docs/NEXT_STEPS_2026-07-14.md 的 P0 三项：
#   P0-1  v45 全8损失 + 分箱B  vs  v45_norank(关rankevent)  —— 拆分"分箱"和
#         "关损失"各自的贡献
#   P0-2  v50_norank 固定 seed=3 和 seed=5  —— 确认 last5=0.6572 不是运气
#   P0-3  v50 的 spec/cover/compete 三项系统消融（rankevent 已锁定关闭）——
#         定位"时间局部机制"里具体哪个子正则在起作用
#
# 用法：bash scripts/run_p0_experiments.sh
#
# 注意：
#   - 只跑 fold0 和 fold2（--k_start N --k_end N+1 各跑一次），不是完整5-fold。
#     fold0 代表"较易折"，fold2 代表"最难折"，两端能覆盖大部分信息，成本约
#     为完整5-fold的2/5。
#   - 每组实验跑完立即写入 EXPERIMENT_SUMMARY.md 对应小节，不等全部跑完再写，
#     防止中断丢失结果。
#   - 全部使用分箱 B（当前代码默认全局分箱，无需额外开关）。
# ============================================================
set -euo pipefail

SURVOT_DIR="/home/ubuntu/SurvOT-Rank"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
GPU=0
LOG_BASE="/data1/sweep_results_30ep/_logs/p0_experiments"
SUMMARY_FILE="$SURVOT_DIR/EXPERIMENT_SUMMARY.md"
FOLDS=(0 2)

mkdir -p "$LOG_BASE"
cd "$SURVOT_DIR"

# ============================================================
# 辅助函数：跑一个 config 在指定 fold 上，记录 best/last5
# ============================================================
run_fold() {
    local TAG="$1" CFG="$2" RESDIR="$3" SEED="$4" FOLD="$5"
    shift 5
    local EXTRA_ARGS=("$@")

    local FOLD_TAG="${TAG}_fold${FOLD}_seed${SEED}"
    local ABS_RESDIR="$SURVOT_DIR/$RESDIR/fold${FOLD}_seed${SEED}"

    echo ""
    echo "===== [$FOLD_TAG] Start: $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "  Config: $CFG | Fold: $FOLD | Seed: $SEED"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m survot_rank.cli train \
        --config "$CFG" \
        --set results_dir="$RESDIR/fold${FOLD}_seed${SEED}" \
        -- --k_start "$FOLD" --k_end $((FOLD + 1)) --seed "$SEED" --max_epochs 30 "${EXTRA_ARGS[@]}" \
        >> "$LOG_BASE/${FOLD_TAG}.log" 2>&1

    local RC=$?
    echo "  Exit code: $RC"
    echo "===== [$FOLD_TAG] Done: $(date '+%Y-%m-%d %H:%M:%S') ====="

    if [ $RC -ne 0 ]; then
        echo "  [ERROR] $FOLD_TAG failed, see $LOG_BASE/${FOLD_TAG}.log"
        return 1
    fi

    # 提取 best / last5，回显到 stdout 供人工/后续脚本读取
    local CSV
    CSV=$(find "$ABS_RESDIR" -name "epoch_curve_fold${FOLD}.csv" 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        local BEST_LINE BEST_EPOCH BEST_VAL LAST5
        BEST_LINE=$(tail -n +2 "$CSV" | sort -t, -k2 -nr | head -1)
        BEST_EPOCH=$(echo "$BEST_LINE" | cut -d, -f1 | xargs)
        BEST_VAL=$(echo "$BEST_LINE"    | cut -d, -f2 | xargs)
        LAST5=$(tail -n 5 "$CSV" | cut -d, -f2 | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "—"}')
        echo "  [$FOLD_TAG] best=$BEST_VAL@ep$BEST_EPOCH  last5=$LAST5"
        echo "$FOLD_TAG,$FOLD,$SEED,$BEST_EPOCH,$BEST_VAL,$LAST5" >> "$LOG_BASE/p0_results.csv"
    else
        echo "  [WARN] 找不到 epoch_curve_fold${FOLD}.csv in $ABS_RESDIR"
    fi
}

echo "combo_tag,fold,seed,best_epoch,best,last5" > "$LOG_BASE/p0_results.csv"

# ============================================================
# P0-1: v45 全8损失(分箱B) vs v45_norank(关rankevent, 分箱B)
#       两者除 rankevent 4 项开关外完全一致（同 opt/reg/lr/seed）
# ============================================================
echo ""
echo "########################################"
echo "# P0-1: 分箱B下 关rankevent 的净效果"
echo "########################################"

for FOLD in "${FOLDS[@]}"; do
    run_fold "p01_v45_full8loss" \
        "configs/fix/v45_baseline_globalbin_blca.yaml" \
        "results/p0/v45_baseline_globalbin" \
        3 "$FOLD"

    run_fold "p01_v45_norank" \
        "configs/fix/v45_norank_blca.yaml" \
        "results/p0/v45_norank_seed3" \
        3 "$FOLD"
done

# ============================================================
# P0-2: v50_norank 固定 seed=3 和 seed=5（对照 seed=22646 的探索性结果）
# ============================================================
echo ""
echo "########################################"
echo "# P0-2: v50_norank 固定 seed 复核"
echo "########################################"

for SEED in 3 5; do
    for FOLD in "${FOLDS[@]}"; do
        run_fold "p02_v50_norank" \
            "configs/fix/v50_norank_blca.yaml" \
            "results/p0/v50_norank_seedcheck" \
            "$SEED" "$FOLD"
    done
done

# ============================================================
# P0-3: v50 的 spec/cover/compete 系统消融
#       rankevent 4 项始终锁定为 0（沿用 v50_norank 配方）
#       固定 seed=3，四档：仅OT+event_surv / +spec / +spec+cover / 全开(7损失)
# ============================================================
echo ""
echo "########################################"
echo "# P0-3: V50 时间局部机制(spec/cover/compete)消融"
echo "########################################"

declare -A P03_COMBOS=(
    ["stripped"]="--lambda_timelocal_spec 0.0 --lambda_timelocal_cover 0.0 --lambda_compete_reg 0.0"
    ["spec_only"]="--lambda_timelocal_spec 0.01 --lambda_timelocal_cover 0.0 --lambda_compete_reg 0.0"
    ["spec_cover"]="--lambda_timelocal_spec 0.01 --lambda_timelocal_cover 0.01 --lambda_compete_reg 0.0"
    ["full"]="--lambda_timelocal_spec 0.01 --lambda_timelocal_cover 0.01 --lambda_compete_reg 0.001"
)

for COMBO in stripped spec_only spec_cover full; do
    EXTRA="${P03_COMBOS[$COMBO]}"
    for FOLD in "${FOLDS[@]}"; do
        # shellcheck disable=SC2086
        run_fold "p03_v50_${COMBO}" \
            "configs/fix/v50_norank_blca.yaml" \
            "results/p0/v50_ablation_${COMBO}" \
            3 "$FOLD" $EXTRA
    done
done

# ============================================================
# 汇总：把 p0_results.csv 整理成 markdown 表格并追加到 SUMMARY
# ============================================================
echo ""
echo "########################################"
echo "# 汇总结果 -> EXPERIMENT_SUMMARY.md"
echo "########################################"

{
    echo ""
    echo "## P0 实验结果 (fold0+fold2, 30ep, $(date '+%Y-%m-%d'))"
    echo ""
    echo "> 脚本: scripts/run_p0_experiments.sh | 仅 fold0/fold2 (节省时间)，非完整5-fold"
    echo ""
    echo "| combo | fold | seed | best_epoch | best | last5 |"
    echo "|---|:---:|:---:|:---:|:---:|:---:|"
    tail -n +2 "$LOG_BASE/p0_results.csv" | while IFS=',' read -r tag fold seed ep best last5; do
        echo "| $tag | $fold | $seed | $ep | $best | $last5 |"
    done
} >> "$SUMMARY_FILE"

echo ""
echo "===== Push to main ====="
cd "$SURVOT_DIR"
git add EXPERIMENT_SUMMARY.md scripts/run_p0_experiments.sh 2>/dev/null || true
git commit -m "feat: P0 实验结果 (v45分箱对照 + v50固定seed + v50损失消融, fold0/fold2)" || echo "  (nothing to commit)"
git push origin main 2>&1 || echo "  [WARN] push failed"

echo ""
echo "===== ALL P0 EXPERIMENTS DONE @ $(date '+%Y-%m-%d %H:%M:%S') ====="
echo "结果csv: $LOG_BASE/p0_results.csv"
