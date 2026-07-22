#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python3}" scripts/run_dct_etar_20260723.py "$@"
