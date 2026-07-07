#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/ensemble_v45.sh DIR_SEED3 DIR_SEED5 [DIR_MORE ...]"
  exit 2
fi

python -m survot_rank.cli ensemble --dirs "$@"

