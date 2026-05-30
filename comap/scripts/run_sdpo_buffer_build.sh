#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift 1 ;;
  esac
done

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
if [[ -n "$CONDA_ENV" ]]; then
  conda run -n "$CONDA_ENV" python -m bridge.build_sdpo_buffer "${ARGS[@]}"
else
  python -m bridge.build_sdpo_buffer "${ARGS[@]}"
fi
