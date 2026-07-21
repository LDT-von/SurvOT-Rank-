#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python3}" scripts/run_dct_v35_screen.py "$@"
