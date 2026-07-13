#!/usr/bin/env bash
# ============================================================
# 修复版 config 5-fold seed=None 排队脚本
# 接在正在跑的 V51/V60 后面，一夜跑完批次1+批次2
#
# 用法：bash scripts/queue_fix_5fold.sh
# ============================================================
set -euo pipefail

SURVOT_DIR="/home/ubuntu/SurvOT-Rank"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
GPU=0
WORKERS=4
LOG_BASE="/data1/sweep_results_30ep/_logs"
SUMMARY_FILE="$SURVOT_DIR/EXPERIMENT_SUMMARY.md"

mkdir -p "$LOG_BASE"
cd "$SURVOT_DIR"

# ============================================================
# 等待正在运行的 V51/V60 结束
# ============================================================
echo "=== $(date '+%H:%M:%S') 检查后台进程 ==="
# 等待 V51 (slimbridge) / V60 (v60_ot_event) 的 train_runner 进程结束
while pgrep -f "slimbridge|v60_ot_event" > /dev/null 2>&1; do
    PROCS=$(pgrep -fc "slimbridge|v60_ot_event" 2>/dev/null || echo 0)
    echo "  $(date '+%H:%M:%S') - $PROCS V51/V60 进程仍在运行, 等待 60s..."
    sleep 60
done
echo "  $(date '+%H:%M:%S') - 所有后台进程已结束, 开始排队"

# ============================================================
# 辅助函数：解析结果并写入 SUMMARY
# ============================================================
write_summary_entry() {
    local TAG="$1" RESDIR="$2"
    local SECTION_TITLE="### $TAG — 5-fold seed=random"

    # 确保节标题不重复（追加模式）
    if grep -q "$SECTION_TITLE" "$SUMMARY_FILE" 2>/dev/null; then
        echo "  [skip] $TAG already in SUMMARY, skipping duplicate entry"
        return
    fi

    {
        echo ""
        echo "$SECTION_TITLE"
        echo ""
        echo "| Fold | Best Epoch | val_cidx (best) | val_cidx (last5) | train_cidx (last) |"
        echo "|:----:|:----------:|:---------------:|:----------------:|:-----------------:|"

        local FOLD_BESTS=()
        local FOLD_LAST5S=()
        local FOLD_EPOCHS=()
        local FOLD_TRAINS=()

        for f in 0 1 2 3 4; do
            # 在 results_dir 下递归找 epoch_curve_fold{f}.csv
            local CSV
            CSV=$(find "$RESDIR" -name "epoch_curve_fold${f}.csv" 2>/dev/null | head -1)
            if [ -z "$CSV" ]; then
                echo "| $f | — | — | — | — |"
                continue
            fi

            # best val_cindex: 跳过 header, 按第2列降序取第1行
            local BEST_LINE
            BEST_LINE=$(tail -n +2 "$CSV" | sort -t, -k2 -nr | head -1)
            local BEST_EPOCH BEST_VAL
            BEST_EPOCH=$(echo "$BEST_LINE" | cut -d, -f1 | xargs)
            BEST_VAL=$(echo "$BEST_LINE"    | cut -d, -f2 | xargs)

            # last 5: 取最后5行的第2列求均值
            local LAST5
            LAST5=$(tail -n 5 "$CSV" | cut -d, -f2 | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "—"}')

            # 最后一行 train cindex (第3列)
            local LAST_TRAIN
            LAST_TRAIN=$(tail -1 "$CSV" | cut -d, -f3 | xargs)

            echo "| $f | $BEST_EPOCH | $BEST_VAL | $LAST5 | $LAST_TRAIN |"

            FOLD_BESTS+=("$BEST_VAL")
            FOLD_LAST5S+=("$LAST5")
            FOLD_EPOCHS+=("$BEST_EPOCH")
            FOLD_TRAINS+=("$LAST_TRAIN")
        done

        # 均值 & std
        echo ""
        # best 均值 (仅供参考, peak-picking 有乐观偏差)
        local MEAN_BEST STD_BEST
        MEAN_BEST=$(printf '%s\n' "${FOLD_BESTS[@]}" | awk '{sum+=$1; n++} END {printf "%.4f", sum/n}')
        STD_BEST=$(printf '%s\n' "${FOLD_BESTS[@]}" | awk -v m="$MEAN_BEST" '{sum+=($1-m)^2; n++} END {printf "%.4f", sqrt(sum/n)}')

        # last5 均值 (主要判据)
        local MEAN_LAST5 STD_LAST5
        MEAN_LAST5=$(printf '%s\n' "${FOLD_LAST5S[@]}" | awk '{sum+=$1; n++} END {printf "%.4f", sum/n}')
        STD_LAST5=$(printf '%s\n' "${FOLD_LAST5S[@]}" | awk -v m="$MEAN_LAST5" '{sum+=($1-m)^2; n++} END {printf "%.4f", sqrt(sum/n)}')

        # train mean
        local MEAN_TRAIN
        MEAN_TRAIN=$(printf '%s\n' "${FOLD_TRAINS[@]}" | awk '{sum+=$1; n++} END {printf "%.4f", sum/n}')
        local GAP
        GAP=$(awk "BEGIN {printf \"%.4f\", $MEAN_TRAIN - $MEAN_LAST5}")

        echo "**5-fold mean (best):** $MEAN_BEST ± $STD_BEST  (仅供参考, peak-picking +0.04~0.07 乐观偏差)"
        echo ""
        echo "**5-fold mean (last5):** $MEAN_LAST5 ± $STD_LAST5  ← 主要判据"
        echo ""
        echo "**train-val gap:** $GAP"

        # 判据检查
        local CHECKS=""
        if [ "$(awk "BEGIN {print ($MEAN_LAST5 >= 0.65)}")" = "1" ]; then
            CHECKS="$CHECKS ✅ last5≥0.65"
        else
            CHECKS="$CHECKS ❌ last5<0.65"
        fi
        local IBS_OK="?"
        local IBS_FILE
        IBS_FILE=$(find "$RESDIR" -name "summary.csv" 2>/dev/null | head -1)
        if [ -n "$IBS_FILE" ]; then
            local IBS_MEAN
            IBS_MEAN=$(grep "val_IBS" "$IBS_FILE" | head -1 | grep -oP '[\d.]+' | head -1)
            if [ -n "$IBS_MEAN" ] && [ "$(awk "BEGIN {print ($IBS_MEAN < 0.30)}")" = "1" ]; then
                IBS_OK="IBS=$IBS_MEAN<0.30 ✅"
            else
                IBS_OK="IBS>0.30 ❌"
            fi
        fi
        CHECKS="$CHECKS | $IBS_OK"

        if [ "$(awk "BEGIN {print ($GAP < 0.15)}")" = "1" ]; then
            CHECKS="$CHECKS | ✅ gap<0.15"
        else
            CHECKS="$CHECKS | ❌ gap≥0.15"
        fi

        echo ""
        echo "**判据:** $CHECKS"
        echo ""
    } >> "$SUMMARY_FILE"
    echo "  [OK] Summary entry written for $TAG"
}

# ============================================================
# 核心：跑一个 config
# ============================================================
run_one() {
    local TAG="$1" CFG="$2" RESDIR="$3" EXTRA_CLI="$4"

    echo ""
    echo "===== [$TAG] Start: $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "  Config:   $CFG"
    echo "  Results:  $RESDIR"

    local SEED=$RANDOM
    echo "  Seed:     $SEED (labeled as seed=random)"

    # 确保 results_dir 使用绝对路径
    local ABS_RESDIR="$SURVOT_DIR/$RESDIR"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m survot_rank.cli train \
        --config "$CFG" \
        --set results_dir="$RESDIR" \
        -- --k_start 0 --k_end 5 --seed "$SEED" $EXTRA_CLI \
        >> "$LOG_BASE/${TAG}_5fold.log" 2>&1

    local RC=$?
    echo "  Exit code: $RC"
    echo "===== [$TAG] Done:  $(date '+%Y-%m-%d %H:%M:%S') ====="

    if [ $RC -eq 0 ]; then
        write_summary_entry "$TAG" "$ABS_RESDIR"
    else
        echo "  [ERROR] $TAG failed with code $RC, skipping SUMMARY entry"
        {
            echo ""
            echo "### $TAG — 5-fold seed=random (FAILED, exit=$RC)"
            echo ""
        } >> "$SUMMARY_FILE"
    fi
}

# ============================================================
# 批次 1 — 必跑 (3 configs, V45/V45v2/V50 关 rankevent)
# ============================================================
echo ""
echo "========================================"
echo " 批次 1/2 — 损失黑名单修复 (3 configs)"
echo "========================================"

run_one "v45_norank"     "configs/fix/v45_norank_blca.yaml"     "results/v45_norank_5fold_noseed"     ""
run_one "v45v2_norank"   "configs/fix/v45v2_norank_blca.yaml"   "results/v45v2_norank_5fold_noseed"   ""
run_one "v50_norank"     "configs/fix/v50_norank_blca.yaml"     "results/v50_norank_5fold_noseed"     ""

# ============================================================
# 批次 2 — 时间允许再跑 (4 configs, 修复版基线方法)
# ============================================================
echo ""
echo "========================================"
echo " 批次 2/2 — 修复版基线方法 (4 configs)"
echo "========================================"

run_one "rg_et_fix"      "configs/fix/rank_guided_event_transport_fix_blca.yaml" \
    "results/rank_guided_event_transport_fix_5fold_noseed" ""

run_one "catet_fix"      "configs/fix/censoring_aware_temporal_evidence_transport_fix_blca.yaml" \
    "results/censoring_aware_temporal_evidence_transport_fix_5fold_noseed" ""

# DCT: 覆盖 max_epochs 60→30
run_one "dct_fix"        "configs/fix/distributional_counterfactual_transport_fix_blca.yaml" \
    "results/distributional_counterfactual_transport_fix_5fold_noseed" "--max_epochs 30"

run_one "faithful_fix"   "configs/fix/faithful_evidence_transport_fix_blca.yaml" \
    "results/faithful_evidence_transport_fix_5fold_noseed" ""

# ============================================================
# 汇总 & 推送
# ============================================================
echo ""
echo "===== All $(date '+%Y-%m-%d %H:%M:%S') — 运行 honest_report ====="

REPORT_OUT="results/norank_5fold_noseed_report.md"
CUDA_VISIBLE_DEVICES="" "$PYTHON" robust_eval/honest_report.py \
    --dirs \
        results/v45_norank_5fold_noseed \
        results/v45v2_norank_5fold_noseed \
        results/v50_norank_5fold_noseed \
        results/rank_guided_event_transport_fix_5fold_noseed \
        results/censoring_aware_temporal_evidence_transport_fix_5fold_noseed \
        results/distributional_counterfactual_transport_fix_5fold_noseed \
        results/faithful_evidence_transport_fix_5fold_noseed \
    --labels v45_norank v45v2_norank v50_norank rg_et_fix catet_fix dct_fix faithful_fix \
    --strategy last_k_mean \
    --out "$REPORT_OUT" \
    >> "$LOG_BASE/honest_report_5fold.log" 2>&1 || echo "  [WARN] honest_report failed"

echo "===== Push to main ====="
cd "$SURVOT_DIR"
git add EXPERIMENT_SUMMARY.md "$REPORT_OUT" scripts/queue_fix_5fold.sh 2>/dev/null || true
git commit -m "feat: 修复版 config 5-fold seed=random 全队列结果" || echo "  (nothing to commit)"
git push origin main 2>&1 || echo "  [WARN] push failed"

echo ""
echo "===== DONE @ $(date '+%Y-%m-%d %H:%M:%S') ====="
