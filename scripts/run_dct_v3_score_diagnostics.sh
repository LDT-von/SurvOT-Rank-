#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
VARIANT="${2:-all}"
FOLDS="${FOLDS:-0,2,3}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/diagnostics/dct_v3_score_blca.yaml}"
RESULTS_ROOT="results/dct_v3_score_first_diagnostics"
cd "$(dirname "$0")/.."

variants_for() {
  if [[ "$1" == "all" ]]; then printf '%s\n' full nll_only unweighted_rank legacy_six_loss; else printf '%s\n' "$1"; fi
}
override_for() {
  case "$1" in
    full) ;;
    nll_only) printf '%s\n' 'dct_lambda_ipcw_rank=0.0' ;;
    unweighted_rank) printf '%s\n' 'dct_lambda_ipcw_rank=0.0' 'dct_lambda_rank=0.05' ;;
    legacy_six_loss) printf '%s\n' \
      'dct_lambda_ipcw_rank=0.0' \
      'dct_lambda_ot=0.06' \
      'dct_lambda_rank=0.05' \
      'dct_lambda_anchor=0.03' \
      'dct_lambda_stage_risk=0.05' \
      'dct_lambda_coordinate=0.01' ;;
    *) echo "Unknown variant: $1" >&2; exit 2 ;;
  esac
}
run_variant() {
  local variant="$1" fold_list="$2" smoke="${3:-false}"
  IFS=',' read -ra fold_array <<< "$fold_list"
  for fold_text in "${fold_array[@]}"; do
    local fold="${fold_text// /}"
    local end_fold=$((fold + 1))
    local args=( -m survot_rank.cli train --config "$CONFIG" --set "gpu=$GPU" --set "num_workers=$NUM_WORKERS" --set "results_dir=$RESULTS_ROOT/$variant" --set "specific_simple=dct_v3_score_first_$variant" )
    while IFS= read -r override; do
      if [[ -n "$override" ]]; then args+=( --set "$override" ); fi
    done < <(override_for "$variant")
    if [[ "$smoke" == "true" ]]; then args+=( --set "max_epochs=1" ); fi
    args+=( -- --k_start "$fold" --k_end "$end_fold" )
    echo "Running $variant, fold $fold"
    "$PYTHON_BIN" "${args[@]}"
  done
}
case "$MODE" in
  doctor) exec "$PYTHON_BIN" -m survot_rank.cli doctor ;;
  smoke) run_variant full "${FOLDS%%,*}" true ;;
  run) while IFS= read -r variant; do run_variant "$variant" "$FOLDS"; done < <(variants_for "$VARIANT") ;;
  summarize) ;;
  *) echo "Usage: $0 [doctor|smoke|run|summarize] [full|nll_only|unweighted_rank|legacy_six_loss|all]" >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" scripts/summarize_dct_v3_score_diagnostics.py --root "$RESULTS_ROOT" --expected-folds "$FOLDS"
