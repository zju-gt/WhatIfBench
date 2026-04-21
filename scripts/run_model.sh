#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_KEY="${API_KEY:-${OPENROUTER_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
MODEL="${1:-}"

if [[ -z "${API_KEY}" ]]; then
  echo "Set OPENROUTER_API_KEY or API_KEY." >&2
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/run_model.sh MODEL" >&2
  exit 1
fi

python3 "${ROOT_DIR}/code/src/main.py" \
  run \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --dataset "mvp/data/benchmark_candidates.json" \
  --model "${MODEL}" \
  --output-dir "result"
