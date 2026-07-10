#!/usr/bin/env bash
# 顺序运行 configs/ablation/ 下的全部消融配置（OTEHV2RankEventV2 系统消融）。
#
# 用法：
#   bash scripts/run_ablation.sh            # 跑全部消融，GPU=0 seed=3
#   GPU=1 SEED=5 bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh abl_00_baseline abl_08_all_on   # 只跑指定几个
#
# 先用 tools/gen_ablation_configs.py 生成配置；跑完后用你的汇总脚本
# （如 tools/aggregate_cross_cancer.py）比较各消融 results/ablation/<name>/ 下的
# summary.csv，定位哪个新能力有用、哪个拖累。
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-3}"
ABL_DIR="configs/ablation"

if [[ ! -d "$ABL_DIR" ]]; then
  echo "[run_ablation] 找不到 $ABL_DIR，先运行: python tools/gen_ablation_configs.py" >&2
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  NAMES=("$@")
else
  NAMES=()
  for f in "$ABL_DIR"/*.yaml; do
    NAMES+=("$(basename "$f" .yaml)")
  done
fi

for name in "${NAMES[@]}"; do
  cfg="$ABL_DIR/$name.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[run_ablation] 跳过：$cfg 不存在" >&2
    continue
  fi
  echo "==================================================================="
  echo "[run_ablation] 开始消融: $name (GPU=$GPU SEED=$SEED)"
  echo "==================================================================="
  python -m survot_rank.cli train --config "$cfg" --set "gpu=$GPU" --set "seed=$SEED"
done

echo "[run_ablation] 全部消融完成。结果在 results/ablation/ 下。"
