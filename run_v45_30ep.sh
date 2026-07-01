#!/bin/bash
# ============================================================================
# 45_otehv2_rankevent (V45) 独立运行脚本
#
# 只依赖本文件夹（model.py / backbone.py / paths.py / args.py / train.py）
# + 外部一份 SlotSPE/ 基座（数据集/loss/底层网络层），不再需要 common/ 或
# 其他数字编号的实验文件夹。
#
# 用法:
#   bash run_v45_30ep.sh [GPU]
#   nohup bash run_v45_30ep.sh 0 > v45_standalone.log 2>&1 &
#
# 如果 SlotSPE/ 不在本文件夹的兄弟目录，请先设置:
#   export SLOTSPE_DIR=/path/to/SlotSPE
# ============================================================================

set -uo pipefail

export PYTHON="${PYTHON:-python}"
GPU=${1:-0}

# 按需修改这两个路径
DATA_ROOT="${DATA_ROOT:-/data/CPathPatchFeature}"
DATA_PATH="${DATA_PATH:-../SlotSPE/dataset_csv}"
RESULT_DIR="${RESULT_DIR:-./results_v45_standalone}"

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR"

echo "[$(date)] 45_otehv2_rankevent (V45) 独立运行  GPU=$GPU"
echo "[$(date)] DATA_ROOT=$DATA_ROOT"
echo "[$(date)] DATA_PATH=$DATA_PATH"
echo "[$(date)] RESULT_DIR=$RESULT_DIR"

$PYTHON -u train.py \
  --n_classes 4 --num_patches 2048 --encoding_dim 1024 \
  --max_epochs 30 --batch_size 4 --seed 3 --study blca \
  --rna_format Pathways --label_col survival_months_dss --bag_loss nll_surv \
  --signature combine --slot_num_wsi 8 --slot_num_omics 8 \
  --slot_iters 5 --temperature 0.01 --topk_ratio 0.25 \
  --top_k_method parallel_topk_st --k_start 0 --k_end 5 \
  --lr 5e-4 --gpu "$GPU" --num_workers 4 \
  --data_root_dir "$DATA_ROOT" --data_path "$DATA_PATH" \
  --results_dir "$RESULT_DIR" --specific_simple "v45_standalone_30ep" \
  --otehv2_eps 0.05 --otehv2_iter 50 --otehv2_warmup 5 \
  --otehv2_num_events 24 --otehv2_heads 4 --otehv2_layers 4 \
  --otehv2_dropout 0.1 \
  --lambda_otehv2_ot 0.06 --lambda_otehv2_div 0.01 \
  --lambda_otehv2_event_surv 0.25 --lambda_otehv2_recon 0.2 \
  --lambda_rankevent_per_event 0.15 --lambda_rankevent_rank 0.15 \
  --lambda_rankevent_global_cons 0.02 --lambda_rankevent_gate_ent 0.005 \
  --rankevent_eps_start 0.10 --rankevent_eps_end 0.05 \
  --rankevent_eps_anneal_epochs 12 --rankevent_global_init -2.0

echo "[$(date)] DONE"
