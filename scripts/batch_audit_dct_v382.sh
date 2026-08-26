#!/bin/bash
# Batch audit script for DCT v3.8.2 cross-cancer models

STUDIES="hnsc skcm kirc lusc ucec"

for STUDY in $STUDIES; do
    echo "=========================================="
    echo "Auditing $STUDY"
    echo "=========================================="
    
    # Create study-specific config
    sed "s/STADY/$STUDY/g; s/STUDY/$STUDY/g" configs/dct_v382_fixed_full_audit_template.yaml > configs/dct_v382_fixed_full_${STUDY}_audit.yaml
    
    CHECKPOINT_DIR="results/dct_v3.8.2/robust/fixed_full/${STUDY}/${STUDY}/SurvOTRank_dct_v382_prognostic_transport_reconstruction"
    RESULT_DIR="results/dct_v3.8.2_paper_evidence/audit/${STUDY}_fold0"
    
    # Find the checkpoint directory
    CHECKPOINT_DIR=$(find results/dct_v3.8.2 -path "*${STUDY}*" -name "*dct_v382_robust_fixed_full_${STUDY}*" -type d | head -1)
    
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "Warning: No checkpoint directory found for $STUDY, skipping..."
        continue
    fi
    
    for fold in 0 1 2 3 4; do
        CHECKPOINT="${CHECKPOINT_DIR}/model_best_s${fold}.pth"
        if [ -f "$CHECKPOINT" ]; then
            echo "Running audit for $STUDY fold $fold..."
            
            # Run audit
            /home/ubuntu/.conda/envs/trisurv/bin/python scripts/audit_dct_v382.py audit \
                --config configs/dct_v382_fixed_full_${STUDY}_audit.yaml \
                --checkpoint "$CHECKPOINT" \
                --fold $fold \
                --output-dir results/dct_v3.8.2_paper_evidence/audit/${STUDY}_fold${fold} \
                --gpu 0 2>&1 | grep -E '(correct_rate|above_margin_rate|n_cases|Fold|error|Error)' || true
            
            # Run sweep
            /home/ubuntu/.conda/envs/trisurv/bin/python scripts/audit_dct_v382.py sweep \
                --config configs/dct_v382_fixed_full_${STUDY}_audit.yaml \
                --checkpoint "$CHECKPOINT" \
                --fold $fold \
                --alphas "0.0,0.25,0.5,0.75,1.0" \
                --output-dir results/dct_v3.8.2_paper_evidence/audit/${STUDY}_fold${fold} \
                --gpu 0 2>&1 | grep -E '(monotone_rate|error|Error)' || true
        else
            echo "Warning: Checkpoint not found: $CHECKPOINT"
        fi
    done
done

echo ""
echo "=========================================="
echo "All audits complete!"
echo "=========================================="
