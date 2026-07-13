#!/usr/bin/env bash
# ============================================================
# 排队脚本：依次运行所有未测方法的 fold2 30epoch 快速验证
# 每个实验跑完自动开始下一个，无需人工干预
#
# 用法：bash scripts/queue_fold2.sh
# 注意：#5 (RG-ET+PCGrad) 需先在 train_runner 中集成 pcgrad_backward
# ============================================================
set -euo pipefail

PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
WORKERS=4

# ---- 排队列表 ----
# 格式: "tag:config_or_method:extra_args"
# config_or_method: SurvOT-Rank config 文件路径 (relative to SurvOT-Rank)
# extra_args:     额外的 --set 参数; 以 "NS:" 开头表示从 newSlotSPE 用 --newslot_method 运行

QUEUE=(
  # 1 -- SurvOT-Rank
  "rg_et:configs/rank_guided_event_transport_blca.yaml:"
  # 2 -- SurvOT-Rank
  "v50:configs/v50_blca.yaml:"
  # 3 -- SurvOT-Rank
  "cate_t:configs/censoring_aware_temporal_evidence_transport_blca.yaml:"
  # 4 -- SurvOT-Rank
  "dct:configs/distributional_counterfactual_transport_blca.yaml:"
  # 5 -- SurvOT-Rank, 需先在 train_runner 中集成 robust_eval/pcgrad.py
  "rg_et_pcgrad:configs/rank_guided_event_transport_blca.yaml:"
  # 6 -- SurvOT-Rank, newSlotSPE V2: 关 rankevent, 仅 4-loss (5-fold 0.7100)
  "v2_norank:configs/v2_norank_blca.yaml:"
  # 7 -- SurvOT-Rank, newSlotSPE V4a: 关 rankevent + AdamW wd=5e-4
  "v4a_adamw:configs/v2_norank_blca.yaml:--set opt=adamW --set reg=0.0005"
  # 8 -- newSlotSPE, ot_v3 (FINAL_SUMMARY #1, val_cindex=0.7282)
  "ot_v3:NS:ot_v3"
  # 9 -- V45 损失子集扫描 curated (10 组, fold2, 30ep, ~2h)
  "loss_sweep_v45:configs/v45_blca.yaml:loss_sweep_v45"
  # 10 -- V50 损失子集扫描 curated (10 组, fold2, 30ep, ~2h)
  "loss_sweep_v50:configs/v50_blca.yaml:loss_sweep_v50"
)

TOTAL=${#QUEUE[@]}
echo "========================================"
echo " Fold2 quick eval queue — $TOTAL jobs"
echo " GPU=$GPU  workers=$WORKERS  epochs=30"
echo "========================================"

for i in "${!QUEUE[@]}"; do
  IFS=":" read -r TAG CFG EXTRA <<< "${QUEUE[$i]}"
  JOB_NUM=$((i + 1))

  echo ""
  echo "===== [$JOB_NUM/$TOTAL] $TAG ====="

  if [ "$CFG" = "NS" ]; then
    # --- newSlotSPE 方法 ---
    METHOD="$EXTRA"
    RESDIR="/data1/sweep_results_30ep/${METHOD}_fold2_30ep"
    LOGDIR="/data1/sweep_results_30ep/_logs"
    mkdir -p "$LOGDIR"

    echo "Method: $METHOD (newSlotSPE)"
    echo "Start:  $(date '+%H:%M:%S')"

    cd /home/ubuntu/newSlotSPE
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u common/train_runner.py \
      --newslot_method "$METHOD" \
      --specific_simple "runall_${METHOD}_fold2_30ep" \
      --data_root_dir "/data/CPathPatchFeature" \
      --data_path "./SlotSPE/dataset_csv" \
      --results_dir "$RESDIR" \
      --n_classes 4 --num_patches 2048 --encoding_dim 1024 \
      --max_epochs 30 --batch_size 4 --study blca \
      --rna_format Pathways --label_col survival_months_dss \
      --bag_loss nll_surv --alpha_surv 0.15 \
      --signature combine \
      --slot_num_wsi 8 --slot_num_omics 8 \
      --slot_iters 5 --temperature 0.01 \
      --topk_ratio 0.25 --top_k_method parallel_topk_st \
      --k_start 2 --k_end 3 --seed 3 \
      --lr 5e-4 --gpu 0 --num_workers 0 \
      --otehv2_eps 0.05 --otehv2_iter 50 --otehv2_warmup 5 \
      --otehv2_num_events 24 --otehv2_heads 4 --otehv2_layers 4 \
      --otehv2_dropout 0.1 \
      --lambda_otehv2_ot 0.06 --lambda_otehv2_div 0.01 \
      --lambda_otehv2_event_surv 0.25 --lambda_otehv2_recon 0.2 \
      >> "$LOGDIR/${METHOD}_fold2.log" 2>&1

    cd - > /dev/null
  elif [ "$EXTRA" = "loss_sweep_v45" ]; then
    # --- V45 损失子集扫描 curated (10 组, ~2h) ---
    echo "Mode:  V45 loss sweep curated (10 groups, fold2, 30ep)"
    echo "Start: $(date '+%H:%M:%S')"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" robust_eval/loss_group_sweep.py \
      --method otehv2_rankevent \
      --config "$CFG" \
      --preset curated \
      --epochs 30 \
      --fold 2 \
      --seed 3 \
      --gpu 0 \
      --python "$PYTHON" \
      --yes
  elif [ "$EXTRA" = "loss_sweep_v50" ]; then
    # --- V50 损失子集扫描 curated (10 组, ~2h) ---
    echo "Mode:  V50 loss sweep curated (10 groups, fold2, 30ep)"
    echo "Start: $(date '+%H:%M:%S')"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" robust_eval/loss_group_sweep.py \
      --method otehv2_timelocal_competing \
      --config "$CFG" \
      --preset curated \
      --epochs 30 \
      --fold 2 \
      --seed 3 \
      --gpu 0 \
      --python "$PYTHON" \
      --yes
  else
    # --- SurvOT-Rank config ---
    echo "Config: $CFG"
    echo "Start:  $(date '+%H:%M:%S')"

    if [ -z "$EXTRA" ]; then
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m survot_rank.cli train \
        --config "$CFG" \
        --set "gpu=0" --set "num_workers=$WORKERS" \
        -- --k_start 2 --k_end 3
    else
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m survot_rank.cli train \
        --config "$CFG" \
        --set "gpu=0" --set "num_workers=$WORKERS" $EXTRA \
        -- --k_start 2 --k_end 3
    fi
  fi

  echo "Done:   $(date '+%H:%M:%S')"
done

echo ""
echo "===== All $TOTAL jobs finished ====="
echo "Run: python robust_eval/honest_report.py  (to generate reports)"
