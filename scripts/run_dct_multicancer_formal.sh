#!/usr/bin/env bash
# Linux wrapper for the cross-platform Python launcher.
# Examples:
#   bash scripts/run_dct_multicancer_formal.sh doctor
#   bash scripts/run_dct_multicancer_formal.sh smoke --cancers brca,lusc
#   CANCERS=brca,lusc GPU=1 bash scripts/run_dct_multicancer_formal.sh run
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" scripts/run_dct_multicancer_formal.py "$@"
