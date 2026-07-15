#!/usr/bin/env bash
# One-command runner for DCT v3 after pulling the repository.
#
# Default:
#   bash scripts/run_dct_v3.sh
#
# Useful modes:
#   MODE=doctor bash scripts/run_dct_v3.sh   # check repo/data wiring only
#   MODE=smoke  bash scripts/run_dct_v3.sh   # fold0, 1 epoch, quick runtime check
#   MODE=verify bash scripts/run_dct_v3.sh   # fold0 + fold2, normal DCT v3 config
#   MODE=full   bash scripts/run_dct_v3.sh   # full 5-fold, normal DCT v3 config
#   MODE=fix    bash scripts/run_dct_v3.sh   # 60-epoch fix config, fold2 by default
#
# Optional overrides:
#   GPU=1 MODE=verify bash scripts/run_dct_v3.sh
#   PYTHON=/path/to/python MODE=full bash scripts/run_dct_v3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-verify}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CONFIG="${CONFIG:-configs/distributional_counterfactual_transport_blca.yaml}"
FIX_CONFIG="${FIX_CONFIG:-configs/fix/distributional_counterfactual_transport_fix_blca.yaml}"
PYTHON="${PYTHON:-python}"

cd "${PROJECT_ROOT}"

run_train() {
  local config="$1"
  local k_start="$2"
  local k_end="$3"
  shift 3

  echo "[DCT-v3] config=${config} folds=${k_start}..$((k_end - 1)) gpu=${GPU}"
  "${PYTHON}" -m survot_rank.cli train \
    --config "${config}" \
    --set "gpu=${GPU}" \
    --set "num_workers=${NUM_WORKERS}" \
    "$@" \
    -- --k_start "${k_start}" --k_end "${k_end}"
}

case "${MODE}" in
  doctor)
    "${PYTHON}" -m survot_rank.cli doctor
    ;;
  smoke)
    run_train "${CONFIG}" 0 1 --set "max_epochs=1"
    ;;
  verify)
    run_train "${CONFIG}" 0 1
    run_train "${CONFIG}" 2 3
    ;;
  full)
    run_train "${CONFIG}" 0 5
    ;;
  fix)
    run_train "${FIX_CONFIG}" 2 3
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use doctor, smoke, verify, full, or fix." >&2
    exit 2
    ;;
esac

echo "[DCT-v3] Done. Check results/distributional_counterfactual_transport_blca/ or the fix results directory."
