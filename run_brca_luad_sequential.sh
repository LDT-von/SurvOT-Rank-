#!/bin/bash
# Train BRCA then LUAD sequentially on GPU 0.
# Both use the same v45v2 config style; each takes ~1.5-3 hr.
set -e
cd /home/ubuntu/SurvOT-Rank

TS() { date "+[%Y-%m-%d %H:%M:%S]"; }

echo "$(TS) BRCA 5-fold start"
/home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
  --config configs/v45v2_brca_clinical.yaml \
  >> logs/v45v2_brca_clinical.log 2>&1
echo "$(TS) BRCA 5-fold done"

echo "$(TS) LUAD 5-fold start"
/home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
  --config configs/v45v2_luad_clinical.yaml \
  >> logs/v45v2_luad_clinical.log 2>&1
echo "$(TS) LUAD 5-fold done"

echo "$(TS) ALL done"