#!/bin/bash
set -e
PYTHON_BIN=/home/ubuntu/.conda/envs/trisurv/bin/python
export CUDA_VISIBLE_DEVICES=0

cd /data1/SurvOT-Rank

echo "========== STEP 1: BLCA robust base+full fold0 =========="
"$PYTHON_BIN" scripts/run_dct_v38_transport_consistency.py run \
  --python "$PYTHON_BIN" \
  --protocols robust \
  --variants base,full \
  --cancers blca \
  --folds 0 \
  --max-epochs 20 \
  --gpu 0 --num-workers 4

echo ""
echo "========== STEP 2: 3-cancer direction vs base (12 folds) =========="
"$PYTHON_BIN" scripts/run_dct_v38_transport_consistency.py run \
  --python "$PYTHON_BIN" \
  --protocols robust \
  --variants direction,base \
  --cancers brca,blca,luad \
  --folds 0,2 \
  --max-epochs 20 \
  --gpu 0 --num-workers 4

echo ""
echo "========== ALL DONE =========="
