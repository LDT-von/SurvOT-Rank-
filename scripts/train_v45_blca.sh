#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/v45_blca.yaml}"
GPU="${GPU:-0}"
SEED="${SEED:-3}"

python -m survot_rank.cli train --config "$CONFIG" --set "gpu=$GPU" --set "seed=$SEED"

