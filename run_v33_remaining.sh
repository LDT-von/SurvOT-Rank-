#!/bin/bash
# DCT v3.3 Score-First — 剩余癌种: BRCA, LUAD, LUSC, HNSC
# 每癌种 5 折顺序执行

cd /home/ubuntu/SurvOT-Rank
PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python
LOGFILE=logs/v33_remaining.log
truncate -s 0 $LOGFILE

CANCERS=(brca luad lusc hnsc)

for cancer in "${CANCERS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting v3.3 Score-First: $cancer" | tee -a $LOGFILE
    RESULT_DIR="results/dct_v3.3_score_first_${cancer}"
    CONFIG="configs/distributional_counterfactual_transport_${cancer}.yaml"

    for fold in 0 1 2 3 4; do
        echo "[$(date +%H:%M:%S)] ${cancer} fold${fold} starting..." | tee -a $LOGFILE
        $PYTHON -m survot_rank.cli train \
            --config $CONFIG \
            --set k_start=$fold \
            --set k_end=$((fold+1)) \
            --set max_epochs=50 \
            --set batch_size=8 \
            --set lr=0.0005 \
            --set alpha_surv=0.15 \
            --set dct_lambda_ipcw_rank=0.10 \
            --set dct_lambda_etar=0.0 \
            --set dct_lambda_listwise=0.0 \
            --set dct_lambda_ot=0.0 \
            --set dct_lambda_rank=0.0 \
            --set dct_lambda_anchor=0.0 \
            --set dct_lambda_stage_risk=0.0 \
            --set dct_lambda_coordinate=0.0 \
            --set results_dir=$RESULT_DIR \
            --set specific_simple="distributional_counterfactual_transport_score_first_${cancer}" \
            --set gpu=0 \
            --set num_workers=4 >> $LOGFILE 2>&1
        echo "[$(date +%H:%M:%S)] ${cancer} fold${fold} done" | tee -a $LOGFILE
    done
    echo "[$(date +%H:%M:%S)] ${cancer} ALL 5 folds completed!" | tee -a $LOGFILE
done
echo "[$(date +%H:%M:%S)] v3.3 ALL DONE!" | tee -a $LOGFILE
