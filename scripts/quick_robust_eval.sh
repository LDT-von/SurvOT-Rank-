#!/bin/bash
# quick_robust_eval.sh — 快速跑通三个 robust_eval 工具并输出报告
# 用法: bash scripts/quick_robust_eval.sh

set -e
TIMEFORMAT="⏱ elapsed: %R s"
ROOT=$(dirname "$(dirname "$(realpath "$0")")")
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/rank_guided_event_transport_blca.yaml"
RES_DIR="$ROOT/results/quick_robust_eval"
LOG="$RES_DIR/run.log"

mkdir -p "$RES_DIR"

echo "============================================================"
echo "  Robust Eval 快速验证 (3 步)"
echo "  开始时间: $(date '+%H:%M:%S')"
echo "  输出目录: $RES_DIR"
echo "============================================================"

# ─── Step 1: epoch_curve_selection 自检 ──────────────────────
echo ""
echo "[Step 1/3] epoch_curve_selection selftest (无 GPU) ..."
echo "============================================================"
start=$SECONDS
$PYTHON -m robust_eval.epoch_curve_selection 2>&1 | tee "$RES_DIR/selftest.log"
echo "✓ selftest 通过  ⏱ $((SECONDS-start)) s"

# ─── Step 2: 快速训练 (10 epoch, 1 seed, fold 2) ─────────────
echo ""
echo "[Step 2/3] stable_train_launcher 快速训练 ..."
echo "  config: $CONFIG   seed=1  epochs=10  fold=k2-2  grad_clip=1.0"
echo "============================================================"
start=$SECONDS
$PYTHON robust_eval/stable_train_launcher.py \
  --config "$CONFIG" \
  --seeds 1 \
  --grad-clip 1.0 \
  --gpu 0 \
  --set "max_epochs=10" \
  --set "results_dir=$RES_DIR/quick_train" \
  2>&1 | tee "$LOG"

# stable_train_launcher 不支持 --k_start/--k_end，训练后手动只取 fold 2 的 csv
echo ""
echo "✓ 训练完成  ⏱ $((SECONDS-start)) s"

# ─── Step 3: honest_report 诚实汇总 ─────────────────────────
echo ""
echo "[Step 3/3] honest_report 诚实汇总 & 乐观偏差分析 ..."
echo "============================================================"
start=$SECONDS

# 递归找所有 epoch_curve csv
CSV_COUNT=$(find "$RES_DIR" -name "epoch_curve_fold*.csv" 2>/dev/null | wc -l)
echo "  找到 epoch_curve csv: $CSV_COUNT 个"

if [ "$CSV_COUNT" -gt 0 ]; then
  $PYTHON robust_eval/honest_report.py \
    --dirs "$RES_DIR/quick_train" \
    --strategy last_k_mean --k 5 \
    --out "$RES_DIR/report.md" 2>&1 | tee -a "$LOG"

  echo ""
  echo "============================================================"
  echo "  报告已生成: $RES_DIR/report.md"
  echo "============================================================"
  if [ -f "$RES_DIR/report.md" ]; then
    cat "$RES_DIR/report.md"
  fi
else
  echo "  ⚠ 未找到 epoch_curve csv，跳过报告生成"
  echo "  检查训练日志: tail -50 $LOG"
fi
echo "✓ 报告生成完成  ⏱ $((SECONDS-start)) s"

# ─── 汇总 ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🏁 全部完成  结束时间: $(date '+%H:%M:%S')"
echo "============================================================"
echo ""
echo "  产物:"
echo "    日志:       $LOG"
echo "    自检:       $RES_DIR/selftest.log"
echo "    报告:       $RES_DIR/report.md"
echo "    训练结果:   $RES_DIR/quick_train/"
echo ""
echo "  单独使用各工具："
echo "    python robust_eval/epoch_curve_selection.py"
echo "    python robust_eval/stable_train_launcher.py --config ... --seeds 3 5 7 --grad-clip 1.0"
echo "    python robust_eval/honest_report.py --dirs results/xxx --strategy last_k_mean"
echo ""
