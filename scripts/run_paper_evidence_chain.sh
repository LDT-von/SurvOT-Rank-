#!/usr/bin/env bash
# Paper-evidence ablation chain (DCT v3.8.2) — 四步一键运行
# Step 1: post-hoc audit (0 训练) → Step 2: 机制消融 (12 jobs)
# Step 3: 跨癌种原型迁移 (6 jobs) → Step 4: 汇总
set -uo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-/home/ubuntu/.conda/envs/trisurv/bin/python}"
GPU="${GPU:-0}"

# 与 run_dct_v382_final_cross_cancer.py 的 FINAL_OVERRIDES 完全一致
SETS=(
  survot_method=dct_v382_prognostic_transport_reconstruction
  max_epochs=50
  dct_v382_warmup_epochs=5
  dct_v382_ramp_epochs=10
  dct_v382_lambda_mgptr=0.05
  dct_v382_distill_weight=0.5
  dct_v382_adaptive_aux_weights=false
  dct_v38_lambda_direction=0.05
  dct_v38_lambda_dose=0.03
  dct_v38_lambda_reconfiguration=0.02
  fit_bins_on_train=true
  binning_mode=global_qcut
  dct_slot_init_mode=deterministic
  event_stratified_batches=true
  event_sampling_fraction=0.0
  dct_lambda_ipcw_rank=0.1
  dct_ipcw_rank_memory_size=64
  dct_lambda_etar=0.0
  dct_lambda_listwise=0.0
  dct_mix_ratio=1.0
  num_patches=2048
  batch_size=8
  which_splits=5fold_uni2h
  on_missing_wsi=error
  wsi_encoder=uni2-h
  encoding_dim=1536
  data_root_dir=/data1/TCGA-UNI2-h-features
)

set_args() {
  for s in "${SETS[@]}"; do
    printf -- '--set %s ' "$s"
  done
}

CANCERS=(ucec kirc blca hnsc skcm lusc)
FOLDS=(0 1 2 3 4)

echo "==================== Step 1/4: post-hoc audit (6 cancers × 5 folds) ===================="
for cancer in "${CANCERS[@]}"; do
  cfg="configs/distributional_counterfactual_transport_${cancer}.yaml"
  for fold in "${FOLDS[@]}"; do
    ckpt="$(find "results/dct_v3.8.2/robust/fixed_full/${cancer}" -name "model_best_s${fold}.pth" 2>/dev/null | head -1)"
    if [ -z "${ckpt}" ]; then
      echo "[skip] no checkpoint: ${cancer} fold${fold}"
      continue
    fi
    outdir="results/dct_v3.8.2_paper_evidence/audit/${cancer}_fold${fold}"
    echo "--- audit ${cancer} fold${fold} ---"
    "$PY" scripts/audit_dct_v382.py audit --config "$cfg" --checkpoint "$ckpt" \
      --fold "$fold" --output-dir "$outdir" --gpu "$GPU" $(set_args) || true
    "$PY" scripts/audit_dct_v382.py sweep --config "$cfg" --checkpoint "$ckpt" \
      --fold "$fold" --output-dir "$outdir" --gpu "$GPU" $(set_args) || true
  done
done

echo "==================== Step 2/4: mechanism ablations (12 jobs, ~9 GPU-day) ===================="
"$PY" scripts/run_dct_v382_paper_evidence.py run --python "$PY" --gpu "$GPU" \
  --cancers ucec,blca,lusc --folds 1

echo "==================== Step 3/4: cross-cancer prototype transfer (6 jobs, ~7 GPU-day) ===================="
"$PY" scripts/run_dct_v382_cross_cancer_prototype.py run --python "$PY" --gpu "$GPU" \
  --pairs blca->kirc,blca->ucec,blca->lusc

echo "==================== Step 4/4: summarize ===================="
"$PY" scripts/summarize_paper_evidence.py
if [ -f PAPER_EVIDENCE_LEDGER.md ]; then
  echo "--- PAPER_EVIDENCE_LEDGER.md ---"
  cat PAPER_EVIDENCE_LEDGER.md
fi

echo "==================== ALL DONE ===================="
